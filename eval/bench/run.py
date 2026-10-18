"""Run a launcher to measure replay, place it on SLO tiers, check answer quality or record.

Search, public development checks and verification share this executor. Timing
keeps one server and one excluded warmup per operation; quality grades answers without timing.
SLO timing is bounded by a time budget and request cap and scored against fixed tiers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Iterable
from contextlib import AsyncExitStack
from pathlib import Path
from uuid import uuid4

import yaml
from datasets import load_dataset, load_from_disk

from ..accuracy import grade
from ..baselines import FIXED_ENV, FIXED_OPTIONS, resolve_launch
from ..config import WORKLOADS, duration
from ..config import model as get_model
from ..launch import ENDPOINT, Launch
from ..models import prepare_asset
from ..request_policy import RequestPolicy, matches_workload, request_fields
from .clients import ChatClient
from .metrics import summarize
from .progress import Progress, publish
from .replay import ReplayError, failure_records, replay, session_limit
from .serve import Server, StartupError
from .slo import interleave, prompt_bins
from .slo import score as slo_score
from .watchdog import PerformanceCutoff, Watchdog

REQUEST_CAP = 600
WARMUP = {
    "messages": [{"role": "user", "content": "Hello."}],
    "max_tokens": 16,
    "timeout_seconds": 300,
}


class _QualityCutoff(Exception):
    """The remaining answers cannot reach the fixed minimum correct count."""


async def execute(
    rows: Iterable[dict],
    workload: str,
    policy: RequestPolicy,
    *,
    server: Server,
    mode: str = "timing",
    concurrency: int | None = None,
    request_timeout: float = 3600,
    output: Path | None = None,
    servers: dict[str, dict] | None = None,
    tokenizer=None,
    budget: float | None = None,
    request_cap: float | None = None,
    progress: Path | None = None,
    reference: dict | None = None,
    min_correct: int | None = None,
) -> dict | list[dict]:
    """Own the server, share scheduling and select only the required evidence.

    A failed warmup, transport or stream raises; no partial successful score is
    returned. Quality answers are graded when completed on the main thread,
    preserving checker signal deadlines. Quality rows are independent single-turn questions;
    answers unfinished at the quality ``budget`` count as wrong.

    SLO mode replays rows in length-bin order. No request starts after ``budget``
    seconds and single-stream requests are cut at ``request_cap``; cut and failed
    single-stream requests remain evidence. S-N/A-N failures still fail the run.
    """
    rows = list(rows)
    if mode in ("timing", "slo") and workload not in ("S-N", "A-N"):
        concurrency = 1
    if mode == "quality":
        workload = "S-1"
        concurrency = -1 if concurrency is None else concurrency
        # Timing-only prompts have no answer key to grade.
        rows = [row for row in rows if row.get("evaluation")]
        if not rows:
            raise ValueError("quality needs rows with an answer key")
    hardware = None
    if progress is not None and mode in ("timing", "slo") and getattr(server, "launch", None):
        from ..verify.machine import machine

        task = Path("/var/lib/local-eval/task.json")
        hardware = machine(json.loads(task.read_text()) if task.exists() else {})["fingerprint"]
    journal = (
        Progress(
            progress,
            {
                "rows": rows,
                "workload": workload,
                "mode": mode,
                "concurrency": concurrency,
                "policy": policy.fixed(workload),
                "budget": budget,
                "request_cap": request_cap,
                "launch": server.launch.prepare() if getattr(server, "launch", None) else None,
                "hardware": hardware,
                "warmup": WARMUP if mode in ("timing", "slo") else None,
                "reference": reference,
                "min_correct": min_correct,
            },
        )
        if progress is not None
        else None
    )
    if journal and (journal.folder / "result.json").exists():
        saved_result = json.loads((journal.folder / "result.json").read_text())
        if mode == "timing" and not saved_result.get("cutoff"):
            saved_result.update(summarize(workload, saved_result["records"]))
        return saved_result
    if mode == "slo" and workload.startswith("S"):
        rows = interleave(rows)
    if mode == "record":
        from .live import live
        from .proxy import Proxy

        workers = 1 if concurrency is None else len(rows) if concurrency == -1 else concurrency
        if workers < 1:
            raise ValueError("record concurrency must be positive or -1")
        output = output or Path("runs") / str(uuid4())
        selected = servers or {
            "default": {
                "server": server,
                "endpoint": server.endpoint,
                "policy": policy,
                "model": get_model(policy.model).name,
                "engine": policy.baseline,
                "checkpoint": server.launch.checkpoint if server.launch else None,
            }
        }
        async with AsyncExitStack() as resources:
            for entry in selected.values():
                await resources.enter_async_context(
                    entry.get("server") or Server(None, entry["endpoint"])
                )
            proxy = await resources.enter_async_context(Proxy(selected, output))
            return await live(
                rows, get_model(policy.model).name, workload, proxy, output, concurrency=workers
            )
    single = workload not in ("S-N", "A-N")
    cap = request_cap if mode == "slo" and single else None
    # All admitted sessions can queue silently behind one engine. Allow one
    # request budget per admitted session for both reads and total completion.
    n = session_limit(len(rows), workload, concurrency) if workload in ("S-N", "A-N") else 1
    total_timeout = request_timeout * session_limit(
        len(rows), workload, concurrency if mode == "quality" or n > 1 else None
    )
    if cap is not None:
        # The replay's own cap must cut a request first, keeping its evidence.
        total_timeout = max(total_timeout, cap + 60)
    saved = [journal.saved(i) for i in range((len(rows) + n - 1) // n)] if journal else []
    committed = {}
    items = {}
    grades = journal.folder / "answers.json" if journal and mode == "quality" else None
    if grades and grades.exists():
        items = json.loads(grades.read_text())
    cutoff = None

    def completed(index, selected, records):
        committed[index] = records
        if mode != "quality":
            return
        answered = {record["row_id"]: record for record in records if not record.get("cut")}
        for row in selected:
            if row["id"] not in items:
                record = answered.get(row["id"])
                items[row["id"]] = bool(
                    record
                    and record["finish_reason"] == "stop"
                    and grade(row["evaluation"], record.get("raw_text", record["text"]))
                )
        if grades:
            publish(grades, items)
        if min_correct is not None and sum(items.values()) + len(rows) - len(items) < min_correct:
            raise _QualityCutoff("remaining answers cannot meet the required accuracy")

    if saved and all(unit is not None for unit in saved):
        records = [record for unit in saved for record in unit]
        if mode == "quality":
            for index, unit in enumerate(saved):
                try:
                    completed(index, rows[index : index + 1], unit)
                except _QualityCutoff:
                    cutoff = {"reason": "remaining answers cannot meet the required accuracy"}
                    break
        if mode != "quality":
            warmup = json.loads((journal.folder / "warmup.json").read_text())
    else:
        async with server as running:
            async with ChatClient(
                running.endpoint,
                policy,
                served_model=running.served_model,
                timeout=request_timeout,
                total_timeout=total_timeout,
                mode="timing" if mode == "slo" else mode,
                tokenizer=tokenizer,
            ) as client:
                if mode in ("timing", "slo"):
                    try:
                        async with asyncio.timeout(WARMUP["timeout_seconds"]):
                            warmup = await client.complete(
                                policy.request(
                                    WARMUP["messages"],
                                    workload="S-1",
                                    output_limit=WARMUP["max_tokens"],
                                    served_model=running.served_model,
                                )
                            )
                    except TimeoutError as error:
                        raise StartupError(
                            f"warmup did not finish within {WARMUP['timeout_seconds']:g}s",
                            running.endpoint,
                            getattr(server, "log", None),
                        ) from error
                    warmup.update(row_id="warmup", request_id="warmup")
                    if not warmup["ok"]:
                        raise ReplayError(warmup)
                    if journal:
                        publish(journal.folder / "warmup.json", warmup)
                # Only SLO timing and the quality deadline bound a replay.
                bounds = {
                    key: value
                    for key, value in (
                        ("budget", budget if mode in ("slo", "quality") else None),
                        ("request_cap", cap),
                        ("strict", False if mode == "slo" and single else None),
                    )
                    if value is not None
                }

                watchdog = None
                if reference is not None or mode == "slo" or (journal and mode == "timing"):
                    from .slo import ladder, table

                    floor = ladder(workload, table())[1][0]["throughput"] if mode == "slo" else None
                    watchdog = Watchdog(workload, reference=reference, floor=floor)
                    client.observer = watchdog
                try:
                    async with asyncio.TaskGroup() as tasks:
                        monitor = tasks.create_task(watchdog.run(journal)) if watchdog else None
                        try:
                            records = await replay(
                                rows,
                                workload,
                                client,
                                concurrency=concurrency,
                                progress=journal,
                                quality=mode == "quality",
                                on_complete=completed,
                                **bounds,
                            )
                        finally:
                            if monitor:
                                monitor.cancel()
                except* PerformanceCutoff as failure:
                    cutoff = failure.exceptions[0].detail
                except* _QualityCutoff:
                    cutoff = {"reason": "remaining answers cannot meet the required accuracy"}
                if cutoff:
                    records = [record for index in sorted(committed) for record in committed[index]]
                    if journal:
                        publish(journal.folder / "cutoff.json", cutoff)
    if mode == "slo":
        result = {
            "records": records,
            "warmup": warmup,
            "rows": len(rows),
            "concurrency": n,
            **({"cutoff": cutoff} if cutoff else {}),
        }
        if journal:
            publish(journal.folder / "result.json", result)
        return result
    if mode == "quality":
        items = {row["id"]: items.get(row["id"], False) for row in rows}
        result = {
            "ok": True,
            "accuracy": sum(items.values()) / len(items),
            "items": items,
            "records": records,
            **({"cutoff": cutoff} if cutoff else {}),
        }
    else:
        result = {
            **summarize(workload, records),
            "records": records,
            "warmup": warmup,
            "concurrency": concurrency or (8 if workload in ("S-N", "A-N") else 1),
        }
    if cutoff and mode == "timing":
        result.update(valid=False, errors=[cutoff["reason"]], cutoff=cutoff)
    if journal:
        publish(journal.folder / "result.json", result)
    return result


async def measure(
    rows: Iterable[dict],
    workload: str,
    policy: RequestPolicy,
    *,
    server: Server,
    concurrency: int | None = None,
    request_timeout: float = 3600,
    tokenizer=None,
    progress: Path | None = None,
    reference: dict | None = None,
) -> dict:
    """Timing entry point, with one short warmup excluded from measured evidence."""
    return await execute(
        rows,
        workload,
        policy,
        server=server,
        concurrency=concurrency,
        request_timeout=request_timeout,
        tokenizer=tokenizer,
        progress=progress,
        reference=reference,
    )


async def measure_slo(
    rows: Iterable[dict],
    workload: str,
    policy: RequestPolicy,
    *,
    server: Server,
    budget: float | None,
    request_cap: float | None = REQUEST_CAP,
    request_timeout: float = 3600,
    tokenizer=None,
    concurrency: int | None = None,
    progress: Path | None = None,
) -> dict:
    """SLO entry point: one excluded warmup, then bounded replay evidence."""
    return await execute(
        rows,
        workload,
        policy,
        server=server,
        mode="slo",
        request_timeout=request_timeout,
        tokenizer=tokenizer,
        budget=budget,
        request_cap=request_cap,
        concurrency=concurrency,
        progress=progress,
    )


async def quality(
    rows: Iterable[dict],
    policy: RequestPolicy,
    *,
    server: Server,
    concurrency: int = -1,
    request_timeout: float = 3600,
    tokenizer=None,
    budget: float | None = None,
    progress: Path | None = None,
    min_correct: int | None = None,
) -> dict:
    """Grade standalone questions; by default admit the complete sample at once."""
    return await execute(
        rows,
        "S-1",
        policy,
        server=server,
        mode="quality",
        concurrency=concurrency,
        request_timeout=request_timeout,
        tokenizer=tokenizer,
        budget=budget,
        progress=progress,
        min_correct=min_correct,
    )


def recording_servers(specs: dict[str, dict], model: str, workload: str, folder: Path) -> dict:
    """Resolve recording endpoints and reuse mapped scripts, weights and policies."""
    servers = {}
    for name, spec in specs.items():
        selected = get_model(spec.get("model", model))
        script = Path(spec["launch"]).resolve() if spec.get("launch") else None
        engine = spec.get("engine", "vllm")
        if script and script.parent.parent.name == "baselines":
            engine = script.parent.name
        checkpoint = spec.get("checkpoint")
        environment = dict(spec.get("env", {}))
        if script and checkpoint is None:
            if script.parent.name == "vllm" and script.parent.parent.name == "record":
                checkpoint = prepare_asset(
                    selected.checkpoint, folder / "weights" / name
                ).checkpoint
            else:
                engine, checkpoint = resolve_launch(script, selected.name, workload)
        launch = (
            Launch(
                script,
                checkpoint,
                tuple(spec.get("args", ())),
                environment,
                fixed_options=FIXED_OPTIONS.get(engine, frozenset()),
                fixed_env=FIXED_ENV.get(engine, frozenset()),
            )
            if script
            else None
        )
        endpoint = spec.get("endpoint", ENDPOINT)
        servers[name] = {
            "server": Server(
                launch,
                endpoint,
                cwd=script.parent if script else None,
                log=folder / f"{name}-server.log",
            ),
            "endpoint": endpoint,
            "model": selected.name,
            "engine": engine,
            "checkpoint": checkpoint,
            "args": spec.get("args", []),
            "context_window": spec.get("context_window", 131072),
            "env": environment,
            "policy": request_fields(selected.name, engine),
        }
    return servers


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", help="benchmark model key or name")
    parser.add_argument("workload", nargs="?", choices=WORKLOADS)
    parser.add_argument("launch", type=Path, nargs="?", help="baseline or candidate launch.sh path")
    parser.add_argument("--params", type=Path, help="JSON file with external launch args and env")
    parser.add_argument("--resume", type=Path, help="resume this operation directory")
    parser.add_argument("--mode", choices=("timing", "slo", "quality", "record"), default="timing")
    parser.add_argument(
        "--servers", type=Path, help="recording YAML with servers and optional row routes"
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="raw recording source root, or a prepared HF replay/quality dataset",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help="active rows/sessions; -1 admits all; timing uses workload defaults, quality -1, record 1",
    )

    def seconds(value: str) -> int:
        return duration(int(value) if value.isdigit() else value)

    parser.add_argument(
        "--budget", type=seconds, help="SLO: stop starting requests after this time, e.g. 1h"
    )
    parser.add_argument(
        "--request-cap",
        type=seconds,
        default=REQUEST_CAP,
        help="SLO: cut a single-stream request after this time (default 10m)",
    )
    args = parser.parse_args(argv)
    if args.resume:
        if args.model or args.workload or args.launch:
            parser.error("--resume takes only an existing operation directory")
        folder = args.resume.resolve()
        args = argparse.Namespace(**json.loads((folder / "operation.json").read_text()))
        for key in ("launch", "params", "servers", "data"):
            if getattr(args, key):
                setattr(args, key, Path(getattr(args, key)))
    else:
        if not args.model or not args.workload:
            parser.error("provide MODEL WORKLOAD LAUNCH, or --resume FOLDER")
        base = (
            Path("/workspace/output/experiments")
            if Path("/var/lib/local-eval/task.json").exists()
            else Path("runs")
        )
        folder = (base / str(uuid4())).resolve()
        folder.mkdir(parents=True)
        for key in ("launch", "params", "servers", "data"):
            if getattr(args, key):
                setattr(args, key, getattr(args, key).resolve())
        if (
            args.concurrency is None
            and args.mode in ("timing", "slo")
            and args.workload in ("S-N", "A-N")
        ):
            task = Path("/var/lib/local-eval/task.json")
            args.concurrency = (
                json.loads(task.read_text())["evaluation"]["concurrency"] if task.exists() else 8
            )
        publish(
            folder / "operation.json",
            {
                key: str(v) if isinstance(v, Path) else v
                for key, v in vars(args).items()
                if key != "resume"
            },
        )
    print(f"Operation folder: {folder}", flush=True)
    if args.servers and args.mode != "record":
        parser.error("--servers is a recording option")
    if args.launch is None and args.servers is None:
        parser.error("provide a launch script or --servers")
    selected = get_model(args.model)
    script = args.launch.resolve() if args.launch else None
    servers = None
    tokenizer = None
    if args.mode == "record":
        from .sources import raw_items

        rows = raw_items(selected.name, args.workload, args.data)
        if args.servers:
            configuration = yaml.safe_load(args.servers.read_text())
            specs = configuration["servers"]
            for spec in specs.values():
                if spec.get("launch"):
                    spec["launch"] = str(args.servers.resolve().parent / spec["launch"])
            for row in rows:
                if row["id"] in configuration.get("routes", {}):
                    row["server"] = configuration["routes"][row["id"]]
        else:
            params = json.loads(args.params.read_text()) if args.params else {}
            specs = {"default": {"launch": str(script), **params}}
        servers = recording_servers(specs, selected.name, args.workload, folder)
        first = next(iter(servers.values()))
        server, engine, checkpoint = first["server"], first["engine"], first["checkpoint"]
    else:
        source = args.data or (
            Path("/var/lib/local-eval/quality")
            if args.mode == "quality"
            else (
                Path("/inputs/public")
                if Path("/var/lib/local-eval/task.json").exists()
                else Path("data/dataset")
            )
        )
        scoped = Path(source).joinpath("state.json").exists()
        if scoped:
            rows = load_from_disk(str(source)).to_list()
        elif args.mode == "quality" and Path(source).joinpath("questions/state.json").exists():
            rows = load_from_disk(str(Path(source) / "questions")).to_list()
        else:
            rows = load_dataset(
                str(source),
                "quality" if args.mode == "quality" else "workloads",
                split="questions" if args.mode == "quality" else "records",
            ).to_list()
        rows = [
            row
            for row in rows
            if ("model" not in row or get_model(row["model"]).name == selected.name)
            and (args.mode == "quality" or matches_workload(row, args.workload, scoped=scoped))
        ]
        engine, checkpoint = resolve_launch(script, selected.name, args.workload)
        params = json.loads(args.params.read_text()) if args.params else {}
        launch = Launch(
            script,
            checkpoint,
            tuple(params.get("args", ())),
            dict(params.get("env", {})),
            fixed_options=FIXED_OPTIONS.get(engine, frozenset()),
            fixed_env=FIXED_ENV.get(engine, frozenset()),
        )
        server = Server(launch, ENDPOINT, cwd=script.parent, log=folder / "server.log")
        if script.name == "server.launch":
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file("/models/tokenizer/tokenizer.json")
    try:
        report = asyncio.run(
            execute(
                rows,
                args.workload,
                request_fields(selected.name, engine),
                server=server,
                mode=args.mode,
                concurrency=args.concurrency,
                output=folder,
                servers=servers,
                tokenizer=tokenizer,
                budget=args.budget,
                request_cap=args.request_cap,
                progress=folder / "replay" if args.mode != "record" else None,
            )
        )
    except* ReplayError as errors:
        failure = folder / "failure.json"
        failure.write_text(
            json.dumps(
                {"ok": False, "mode": args.mode, "records": failure_records(errors)}, indent=2
            )
            + "\n"
        )
        print(f"Replay failure: {failure.resolve()}", file=sys.stderr)
        raise
    if args.mode == "record":
        print(f"Recorded rows: {(folder / 'rows.jsonl').resolve()}")
        return
    if args.mode == "quality":
        report = {"accuracy": report["accuracy"]}
    elif args.mode == "slo":
        bins = prompt_bins(rows) if args.workload.startswith("S") else None
        scored = slo_score(
            args.workload,
            report["records"],
            rows=rows,
            bins=bins,
            concurrency=args.concurrency,
        )
        report = {
            "model": selected.name,
            "launch": str(script),
            "slo": scored,
            "score": scored["rewards"]["score"],
            "rewards": scored["rewards"],
            "records": report["records"],
            "warmup": report["warmup"],
            **({"cutoff": report["cutoff"]} if report.get("cutoff") else {}),
        }
        print(f"SLO tier: {scored['tier'] or 'none'}; score {scored['score']:.3f}/100")
    else:
        report.update(
            model=selected.name,
            baseline=engine,
            checkpoint=checkpoint,
            launch=str(script),
            parameters=params,
        )
    output = folder / "report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    label = {"quality": "Accuracy", "slo": "SLO"}.get(args.mode, "Measurement")
    print(f"{label} report: {output.resolve()}")


if __name__ == "__main__":
    main()
