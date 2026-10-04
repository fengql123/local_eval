"""Run a launcher and replay prepared rows for timing or answer quality.

Search, public development checks and verification share this executor. Timing
includes one excluded warmup; quality grades authored S answers without timing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

from ..accuracy import grade
from ..baselines import resolve_launch
from ..config import WORKLOADS
from ..config import model as get_model
from ..launch import ENDPOINT, Launch
from ..request_policy import RequestPolicy, pool_of, request_fields
from .clients import ChatClient
from .metrics import summarize
from .replay import ReplayError, replay
from .serve import Server


async def execute(
    rows: Iterable[dict],
    workload: str,
    policy: RequestPolicy,
    *,
    server: Server,
    mode: str = "timing",
    concurrency: int | None = None,
    request_timeout: float = 3600,
) -> dict:
    """Own the server, share scheduling and select only the required evidence.

    A failed warmup, transport or stream raises; no partial successful score is
    returned. Quality grading runs synchronously after cleanup so checker signal
    deadlines run on the main thread. Quality rows are authored S questions.
    """
    rows = list(rows)
    if mode == "quality":
        workload = "S-1"
        concurrency = -1 if concurrency is None else concurrency
    async with server:
        async with ChatClient(
            server.endpoint,
            policy,
            served_model=server.served_model,
            timeout=request_timeout,
            mode=mode,
        ) as client:
            if mode == "timing":
                warmup = {
                    **await client.request(rows[0]["requests"][0], workload),
                    "row_id": rows[0]["id"],
                }
                if not warmup["ok"]:
                    raise ReplayError(warmup)
            records = await replay(rows, workload, client, concurrency=concurrency)
    if mode == "quality":
        items = {
            row["id"]: grade(row["evaluation"], record["text"])
            for row, record in zip(rows, records, strict=True)
        }
        return {
            "ok": True,
            "accuracy": sum(items.values()) / len(items),
            "items": items,
            "records": records,
        }
    return {**summarize(workload, records), "records": records, "warmup": warmup}


async def measure(
    rows: Iterable[dict],
    workload: str,
    policy: RequestPolicy,
    *,
    server: Server,
    concurrency: int | None = None,
    request_timeout: float = 3600,
) -> dict:
    """Timing entry point, with one warmup excluded from measured evidence."""
    return await execute(
        rows,
        workload,
        policy,
        server=server,
        concurrency=concurrency,
        request_timeout=request_timeout,
    )


async def quality(
    rows: Iterable[dict],
    policy: RequestPolicy,
    *,
    server: Server,
    concurrency: int = -1,
    request_timeout: float = 3600,
) -> dict:
    """Grade S questions; by default admit the complete sample at once."""
    return await execute(
        rows,
        "S-1",
        policy,
        server=server,
        mode="quality",
        concurrency=concurrency,
        request_timeout=request_timeout,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="benchmark model key or name")
    parser.add_argument("workload", choices=WORKLOADS)
    parser.add_argument("launch", type=Path, help="baseline or candidate launch.sh path")
    parser.add_argument("--params", type=Path, help="JSON file with external launch args and env")
    parser.add_argument("--mode", choices=("timing", "quality"), default="timing")
    parser.add_argument(
        "--concurrency",
        type=int,
        help="active rows/sessions; -1 admits all rows, omitted uses workload defaults",
    )
    args = parser.parse_args(argv)
    selected = get_model(args.model)
    source = (
        Path("data") / ("S" if args.mode == "quality" else pool_of(args.workload)) / "rows.jsonl"
    )
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows = [row for row in rows if row["model"] in (selected.key, selected.name)]
    script = args.launch.resolve()
    engine, checkpoint = resolve_launch(script, selected.key, args.workload)
    params = json.loads(args.params.read_text()) if args.params else {}
    launch = Launch(script, checkpoint, tuple(params.get("args", ())), dict(params.get("env", {})))
    folder = Path("bench") / str(uuid4())
    folder.mkdir(parents=True)
    server = Server(
        launch,
        ENDPOINT,
        cwd=script.parent,
        log=folder / "server.log",
    )
    report = asyncio.run(
        execute(
            rows,
            args.workload,
            request_fields(selected.key, engine),
            server=server,
            mode=args.mode,
            concurrency=args.concurrency,
        )
    )
    if args.mode == "quality":
        report = {"accuracy": report["accuracy"]}
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
    label = "Accuracy" if args.mode == "quality" else "Measurement"
    print(f"{label} report: {output.resolve()}")


if __name__ == "__main__":
    main()
