"""Native Harbor verification: prepare, review, quality, then reference and measurement.

SLO tasks have no reference: after quality, one bounded timing run places the
candidate on fixed tiers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tomllib
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

import httpx
from datasets import load_from_disk
from tokenizers import Tokenizer

from ..baselines import FIXED_ENV, FIXED_OPTIONS
from ..bench import slo
from ..bench.metrics import gains, score_gains, summarize
from ..bench.replay import ReplayError
from ..bench.run import measure, measure_slo, quality
from ..bench.serve import Server, StartupError, process_command
from ..config import model as get_model
from ..launch import ENDPOINT, Launch
from ..request_policy import RequestPolicy, request_fields
from . import judge, profile
from .fingerprint import installed_identity
from .machine import machine

VERIFIER = Path("/logs/verifier")
STATE = Path("/var/lib/local-eval/verification")
RUBRIC = Path("/workspace/.harbor/review-rubric")
REVIEW = Path("/workspace/output/review")
REFERENCE = Path("/workspace/runtime/reference")
CANDIDATE_UID = 1004
REFERENCE_UID = 1005
RUNTIME_ERRORS = (
    OSError,
    TimeoutError,
    httpx.HTTPError,
    RuntimeError,
    ValueError,
    subprocess.SubprocessError,
)


class Infrastructure(RuntimeError):
    """An environment failure cannot be scored as candidate output."""


def runtime_environment(account: str = "verifier") -> dict[str, str]:
    """Separate runtime caches and device tools, without reviewer credentials."""
    shared = (
        "PATH",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "TRITON_LIBCUDA_PATH",
        "CUDA_HOME",
        "ROCM_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "UV_PYTHON_INSTALL_DIR",
        "UV_PYTHON_PREFERENCE",
    )
    home = f"/workspace/.harbor/scratch/{account}"
    runtime = f"/workspace/runtime/{'candidate' if account == 'verifier' else account}"
    return {
        **{key: os.environ[key] for key in shared if key in os.environ},
        "HOME": home,
        "USER": account,
        "LOGNAME": account,
        "TMPDIR": f"{runtime}/tmp",
        "XDG_CACHE_HOME": f"{runtime}/cache",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


@dataclass(frozen=True)
class Backend:
    """Request policy and a fresh, owned server for each evaluation stage."""

    name: str
    policy: RequestPolicy
    server: Callable[[str], AbstractAsyncContextManager[Server]]
    tokenizer: object | None = None
    progress: Path | None = None


def _leaves(error: BaseException):
    if isinstance(error, BaseExceptionGroup):
        for nested in error.exceptions:
            yield from _leaves(nested)
    else:
        yield error


def _errors(error: BaseException) -> list[str]:
    return [
        f"{item}: {item.record['errors']}"
        if isinstance(item, ReplayError)
        else f"{type(item).__name__}: {item}"
        for item in _leaves(error)
    ]


async def probe(
    backend: Backend,
    rows: list[dict],
    *,
    budget: float | None = None,
    min_correct: int | None = None,
) -> dict:
    """Replay and grade the supplied quality rows without timing or warmup.

    Answers unfinished at ``budget`` seconds count as wrong.
    """
    try:
        return await quality(
            rows,
            backend.policy,
            server=backend.server("quality"),
            concurrency=-1,
            tokenizer=backend.tokenizer,
            budget=budget,
            min_correct=min_correct,
            progress=backend.progress / "quality" if backend.progress else None,
        )
    except* RUNTIME_ERRORS as failure:
        errors = list(_leaves(failure))
    return {
        "ok": False,
        "items": {},
        "records": [error.record for error in errors if isinstance(error, ReplayError)],
        "errors": [f"{type(error).__name__}: {error}" for error in errors],
        "infrastructure": any(
            isinstance(error, (Infrastructure, OSError, subprocess.SubprocessError))
            for error in errors
        ),
        "stage": "startup"
        if any(isinstance(error, StartupError) for error in errors)
        else "quality",
    }


async def timed(
    task: dict, rows: list[dict], backend: Backend, *, reference: dict | None = None
) -> dict:
    try:
        return await measure(
            rows,
            task["workload"],
            backend.policy,
            server=backend.server("timed"),
            tokenizer=backend.tokenizer,
            concurrency=task["evaluation"].get("concurrency", 8),
            # Match Harbor's 48-hour timing step for slow, valid generations.
            request_timeout=172800,
            progress=backend.progress / "timing" if backend.progress else None,
            reference=reference,
        )
    except* RUNTIME_ERRORS as error:
        result = {
            "valid": False,
            "errors": _errors(error),
            "infrastructure": error.subgroup((Infrastructure, OSError, subprocess.SubprocessError))
            is not None,
        }
    return result


async def slo_timed(task: dict, rows: list[dict], backend: Backend) -> dict:
    """One bounded SLO timing run; failed single-stream requests stay evidence."""
    evaluation = task["evaluation"]
    try:
        result = {
            "valid": True,
            **await measure_slo(
                rows,
                task["workload"],
                backend.policy,
                server=backend.server("slo"),
                budget=evaluation["slo_budget_seconds"],
                request_cap=evaluation["slo_request_cap_seconds"],
                tokenizer=backend.tokenizer,
                concurrency=evaluation.get("concurrency", 8),
                progress=backend.progress / f"slo-{task.get('repeat', 0)}"
                if backend.progress
                else None,
            ),
        }
    except* RUNTIME_ERRORS as error:
        result = {
            "valid": False,
            "errors": _errors(error),
            "infrastructure": error.subgroup((Infrastructure, OSError, subprocess.SubprocessError))
            is not None,
        }
    return result


async def pre_run(
    rows: list[dict], bound: dict, candidate: Backend, *, budget: float | None = None
) -> dict:
    """Run once against the previously frozen offline lower bound."""
    result = {
        "quality": {
            "profile": bound,
            "candidate": await probe(
                candidate, rows, budget=budget, min_correct=bound["min_correct"]
            ),
        }
    }
    compared = result["quality"]
    if not compared["candidate"]["ok"]:
        return {
            **result,
            "ok": None if compared["candidate"].get("infrastructure") else False,
            "stage": compared["candidate"].get("stage", "quality"),
            "reason": "candidate quality run failed",
        }
    compared.update(judge.quality_result(rows, bound, compared["candidate"]["items"]))
    return {
        **result,
        "ok": compared["ok"],
        "stage": "quality",
        "reason": "pre-run gate passed"
        if compared["ok"]
        else "candidate accuracy is below the offline lower bound",
    }


def zero_rewards(workload: str) -> dict:
    """Candidate failures receive one zero performance reward."""
    return {"score": 0.0}


def rewards(workload: str, reference: dict, candidate: dict) -> dict:
    """Normalize the geometric mean of relevant speedups to one 0–100 score."""
    return {"score": score_gains(gains(workload, reference, candidate))}


def _save(out: Path, report: dict) -> None:
    if "rewards" in report:
        report["score"] = report["rewards"]["score"]
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if "rewards" in report:
        (out / "reward.json").write_text(json.dumps(report["rewards"], indent=2) + "\n")
    text = [
        f"# {report['model']} — {report['workload']}",
        "",
        f"Outcome: **{report['outcome']}**.",
        f"Pre-run stage: `{report['pre_run']['stage']}`.",
        f"Timed benchmark skipped: `{report['bench']['skipped']}`.",
        "",
    ]
    if "reason" in report:
        text.append(report["reason"])
    if "slo" in report:
        placed = report["slo"]
        text.append(
            f"SLO tier: **{placed['tier'] or 'none'}** (score {placed['score']:.3f}/100, "
            f"coverage {placed['coverage']['fraction']:.0%})."
        )
    if "rewards" in report:
        text.extend(f"- {key}: {value}" for key, value in report["rewards"].items())
    text.extend(["", "Full quality and timing evidence: [report.json](report.json).", ""])
    (out / "report.md").write_text("\n".join(text))


async def evaluate(
    task: dict,
    rows: list[dict],
    reference: dict,
    candidate: Backend,
    *,
    out: Path,
    prepared: dict,
    reviewed: dict,
    quality_gate: dict,
) -> dict:
    """Time a fresh candidate after the private quality gate and reference timing."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward.json").unlink(missing_ok=True)
    gate = {**quality_gate, "prepare": prepared, "rules": reviewed}
    reference_bench = {
        **summarize(task["workload"], reference["bench"]["records"]),
        "records": reference["bench"]["records"],
        "warmup": reference["bench"]["warmup"],
    }
    report = {
        "model": task["model"],
        "workload": task["workload"],
        "reference": reference["baseline"],
        "reference_checkpoint": reference["checkpoint"],
        "candidate": candidate.name,
        "candidate_checkpoint": task["target"]["source"],
        "machine": machine(task),
        "pre_run": gate,
        "bench": {"skipped": True, "reference": reference_bench},
    }
    if gate["ok"] is not True:
        report["outcome"] = "candidate_failure" if gate["ok"] is False else "infrastructure_error"
        report["reason"] = gate["reason"]
        if gate["ok"] is False:
            report["rewards"] = zero_rewards(task["workload"])
    else:
        report["bench"]["skipped"] = False
        result = report["bench"]["candidate"] = await timed(
            task, rows, candidate, reference=reference_bench
        )
        if result["valid"]:
            report["outcome"] = "passed"
            report["gains"] = gains(task["workload"], reference_bench, result)
            report["rewards"] = rewards(task["workload"], reference_bench, result)
        else:
            infrastructure = result.get("infrastructure", False)
            report["outcome"] = "infrastructure_error" if infrastructure else "candidate_failure"
            report["reason"] = f"candidate timing failed: {result['errors']}"
            if result.get("cutoff"):
                report["termination"] = {"kind": "performance_cutoff", **result["cutoff"]}
            if not infrastructure:
                report["rewards"] = zero_rewards(task["workload"])
    _save(out, report)
    return report


async def slo_measure(
    task: dict,
    rows: list[dict],
    candidate: Backend,
    *,
    out: Path,
    prepared: dict,
    reviewed: dict,
    quality_gate: dict,
) -> dict:
    """Time a fresh candidate after the quality gate and place it on the SLO tiers."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward.json").unlink(missing_ok=True)
    gate = {**quality_gate, "prepare": prepared, "rules": reviewed}
    table = slo.table()
    report = {
        "model": task["model"],
        "workload": task["workload"],
        "type": "SLO",
        "candidate": candidate.name,
        "candidate_checkpoint": task["target"]["source"],
        "machine": machine(task),
        "pre_run": gate,
        "bench": {"skipped": True},
    }
    if gate["ok"] is not True:
        report["outcome"] = "candidate_failure" if gate["ok"] is False else "infrastructure_error"
        report["reason"] = gate["reason"]
        if gate["ok"] is False:
            report["rewards"] = slo.zero(task["workload"], table)
        _save(out, report)
        return report
    report["bench"]["skipped"] = False
    runs = report["bench"]["candidate"] = []
    for repeat in range(task["evaluation"].get("slo_repeats", 1)):
        runs.append(await slo_timed({**task, "repeat": repeat}, rows, candidate))
        if not runs[-1]["valid"] or runs[-1].get("cutoff"):
            break
    failed = next((run for run in runs if not run["valid"]), None)
    if failed is not None:
        infrastructure = failed.get("infrastructure", False)
        report["outcome"] = "infrastructure_error" if infrastructure else "candidate_failure"
        report["reason"] = f"candidate timing failed: {failed['errors']}"
        if not infrastructure:
            report["rewards"] = slo.zero(task["workload"], table)
    else:
        records = [
            {**record, "repeat": number}
            for number, run in enumerate(runs)
            for record in run["records"]
        ]
        scored = slo.score(
            task["workload"],
            records,
            rows=rows,
            repeats=task["evaluation"].get("slo_repeats", 1),
            value=table,
            bins=slo.prompt_bins(rows) if task["workload"].startswith("S") else None,
            concurrency=task["evaluation"].get("concurrency", 8),
        )
        report["outcome"] = "passed"
        report["slo"] = scored
        report["rewards"] = scored["rewards"]
        cutoffs = [run["cutoff"] for run in runs if run.get("cutoff")]
        if cutoffs:
            report["termination"] = {"kind": "performance_cutoff", **cutoffs[-1]}
    _save(out, report)
    return report


async def _process(
    command: list[str],
    *,
    cwd: Path,
    environment: dict,
    log: Path,
    timeout: float,
    user: int | None = None,
) -> int:
    """Keep trusted control separate from submitted subprocesses."""
    command, environment = process_command(command, environment, user)
    with log.open("wb") as output:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            start_new_session=True,
            umask=0o022,
        )
        try:
            async with asyncio.timeout(timeout):
                await process.communicate()
            return process.returncode
        finally:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            await asyncio.sleep(0.25)
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()


def reference_checkpoint(task: dict, selected: dict) -> str:
    source = task["assets"]["baselines"][selected["baseline"]]
    return (
        "/models/target"
        if source == task["target"]["source"]
        else str(REFERENCE / "weights" / "target")
    )


async def install_reference(task: dict, selected: dict, out: Path) -> str:
    """Install the selected baseline and acquire its independently mapped checkpoint."""
    from .access import reference_workspace

    reference_workspace()
    metadata = REFERENCE / "task.json"
    metadata.write_text(
        json.dumps(
            {
                "model": task["model"],
                "device_category": task["device_category"],
                "assets": task["assets"],
            }
        )
        + "\n"
    )
    commands = (
        (
            "install",
            [
                sys.executable,
                "-I",
                "-m",
                "eval.baselines",
                "--task",
                str(metadata),
                "--baseline",
                selected["baseline"],
                "--workspace",
                str(REFERENCE / "baselines"),
            ],
        ),
    )
    for name, command in commands:
        code = await _process(
            command,
            cwd=REFERENCE,
            environment=runtime_environment("reference"),
            log=out / f"reference-{name}.log",
            timeout=task["evaluation"]["install_timeout_seconds"],
            user=REFERENCE_UID,
        )
        if code:
            raise Infrastructure(f"reference {name} exited {code}; see reference-{name}.log")
    checkpoint = reference_checkpoint(task, selected)
    if checkpoint != "/models/target":
        from ..models import prepare_asset
        from .access import readonly

        source = task["assets"]["baselines"][selected["baseline"]]
        await asyncio.to_thread(prepare_asset, source, Path(checkpoint))
        readonly(Path(checkpoint))
    return checkpoint


def complete(path: Path, result: dict) -> None:
    """Only successful work receives an atomic completion artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    partial.replace(path)


async def prepare_reference(task: dict, submission: Path, out: Path) -> dict:
    """Acquire trusted reference dependencies online after candidate quality passes."""
    if (STATE / "reference.json").is_file():
        return {"ok": True, "stage": "reference-prepare", "reason": "reference already measured"}
    from .access import readonly

    out.mkdir(parents=True, exist_ok=True)
    selected = json.loads(Path(task["reference"]).read_text())
    result = {"baseline": selected["baseline"], "stage": "reference-prepare"}
    try:
        result["checkpoint"] = await install_reference(task, selected, out)
        readonly(REFERENCE / "baselines")
        result.update(ok=True, reason="reference installation complete")
    except RUNTIME_ERRORS as error:
        result.update(ok=None, reason=f"{type(error).__name__}: {error}")
    finally:
        await stop_runtime_users()
    return result


async def reference(task: dict, submission: Path, out: Path) -> dict:
    """Time the reference with its mapped checkpoint, preserving the canonical target."""
    marker = STATE / "reference.json"
    if marker.is_file():
        return json.loads(marker.read_text())
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    selected = json.loads(Path(task["reference"]).read_text())
    result["baseline"] = selected["baseline"]
    try:
        result["checkpoint"] = task["assets"]["baselines"][selected["baseline"]]
        # Dependencies and weights were acquired by the preceding online step.
        checkpoint = reference_checkpoint(task, selected)
        backend = _backend(
            task,
            selected["baseline"],
            selected,
            out,
            checkpoint=checkpoint,
        )
        result["bench"] = await timed(task, _rows(task["inputs"]["private"]), backend)
        if not result["bench"]["valid"]:
            raise Infrastructure(f"reference timing failed: {result['bench']['errors']}")
        result.update(ok=True, stage="reference", reason="reference timing complete")
    except RUNTIME_ERRORS as error:
        result.update(ok=None, stage="reference", reason=f"{type(error).__name__}: {error}")
    finally:
        await stop_runtime_users()
    if result["ok"] is True:
        complete(marker, result)
    return result


async def _prepare_script(task: dict, submission: Path, name: str, log: Path) -> dict:
    environment = runtime_environment()
    try:
        code = await _process(
            ["/bin/bash", str((submission / f"{name}.sh").resolve())],
            cwd=submission,
            environment=environment,
            log=log,
            timeout=task["evaluation"]["install_timeout_seconds"],
            user=CANDIDATE_UID,
        )
    except TimeoutError:
        return {"ok": False, "exit_code": "timeout", "log": str(log)}
    return {"ok": code == 0, "exit_code": code, "log": str(log)}


async def stop_runtime_users() -> None:
    """Remove detached runtime children before reviewer credentials arrive."""
    for _ in range(20):
        alive = []
        for status in Path("/proc").glob("[0-9]*/status"):
            try:
                fields = dict(line.split(":", 1) for line in status.read_text().splitlines())
            except (FileNotFoundError, ProcessLookupError):
                continue
            if fields["State"].strip().startswith("Z"):
                continue
            if int(fields["Uid"].split()[0]) in (CANDIDATE_UID, REFERENCE_UID):
                alive.append(int(status.parent.name))
        if not alive:
            return
        for pid in alive:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        await asyncio.sleep(0.05)
    raise Infrastructure("candidate/reference processes survived verification cleanup")


def validate_installation(submission: Path) -> str:
    identity = installed_identity(submission)
    marker = STATE / "installation.json"
    previous = json.loads(marker.read_text()) if marker.exists() else None
    if previous is not None and previous["identity"] != identity:
        history = STATE / "superseded" / previous["identity"]
        history.mkdir(parents=True, exist_ok=True)
        for path in (STATE / "quality.json", STATE / "operations/candidate", REVIEW):
            if path.exists():
                destination = history / path.name
                if destination.exists():
                    shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
                shutil.move(str(path), destination)
        REVIEW.mkdir(mode=0o2750)
        from .access import ACCOUNTS

        os.chown(REVIEW, *ACCOUNTS["judge"])
    complete(marker, {"identity": identity})
    return identity


async def prepare_submission(task: dict, submission: Path, out: Path) -> dict:
    """Reproduce the candidate, end its processes, then protect inputs for review."""
    from .access import reset_runtime, seal_submission

    marker = STATE / "prepare.json"
    if marker.is_file():
        judge.write_review(task, submission, RUBRIC)
        return json.loads(marker.read_text())
    out.mkdir(parents=True, exist_ok=True)
    result = {"static": judge.static_check(submission)}
    if not result["static"]["ok"]:
        return {
            **result,
            "ok": False,
            "stage": "static",
            "reason": "submission scripts are missing or external",
        }
    # Keep enough private space to report an installer that exhausts sandbox disk.
    STATE.mkdir(parents=True, exist_ok=True)
    reserve = STATE / "report-space.reserve"
    reserve.write_bytes(bytes(64 * 1024**2))
    try:
        for name in ("install",):
            done = STATE / f"{name}.json"
            if done.is_file():
                result[name] = json.loads(done.read_text())
            else:
                try:
                    result[name] = await _prepare_script(
                        task, submission, name, out / f"{name}.log"
                    )
                except RUNTIME_ERRORS as error:
                    result[name] = {"ok": None, "error": f"{type(error).__name__}: {error}"}
                if result[name]["ok"] is True:
                    complete(done, result[name])
            if result[name]["ok"] is not True:
                return {
                    **result,
                    "ok": result[name]["ok"],
                    "stage": name,
                    "reason": f"{name} did not pass",
                }
    finally:
        try:
            await stop_runtime_users()
        finally:
            reserve.unlink(missing_ok=True)
    launch = Path(task["candidate"]["launch"])
    if not launch.is_file() or not os.access(launch, os.X_OK):
        return {
            **result,
            "ok": False,
            "stage": "install",
            "reason": "install.sh must provide executable /workspace/server.launch",
        }
    result["implementation"] = validate_installation(submission)
    seal_submission(submission)
    reset_runtime("verifier")
    judge.write_review(task, submission, RUBRIC)
    result.update(ok=True, stage="prepare", reason="submission prepared and protected for review")
    complete(marker, result)
    return result


async def quality_gate(task: dict, out: Path) -> dict:
    """Check hidden answers against the canonical checkpoint's published quality range.

    Reference formats do not change the candidate checkpoint or its quality
    requirement. The canonical target remains fixed and read-only.
    """
    marker = STATE / "quality.json"
    rows = _rows(str(Path(task["inputs"]["quality"]) / "questions"))
    try:
        bound = profile.read(
            Path(task["inputs"]["quality"]),
            get_model(task["model"]).checkpoint,
            request_fields(task["model"]),
            rows,
        )
    except (FileNotFoundError, ValueError) as error:
        return {"ok": None, "stage": "quality_profile", "reason": str(error)}
    if marker.is_file():
        previous = json.loads(marker.read_text())
        if previous["quality"]["profile"] == bound:
            return previous
    result = await pre_run(
        rows,
        bound,
        _backend(task, "candidate", {}, out, checkpoint="/models/target"),
        # Tasks rendered before the quality deadline existed keep waiting as before.
        budget=task.get("evaluation", {}).get("quality_timeout_seconds"),
    )
    if result["ok"] is True:
        complete(marker, result)
    return result


def save_gate(out: Path, report: dict) -> None:
    """A failed candidate stops later Harbor steps; infrastructure has no reward."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    reward = out / "reward.json"
    reward.unlink(missing_ok=True)
    if report["ok"] is not None:
        reward.write_text(json.dumps({"reward": int(report["ok"])}) + "\n")


async def review_rules(submission: Path) -> dict:
    """Run native Rewardkit as Harbor's judge user, without private task metadata."""
    marker = REVIEW / "review.json"
    if marker.is_file():
        previous = json.loads(marker.read_text())
        if previous["ok"] is not None:
            return previous
    options = tomllib.loads((RUBRIC / "rules.toml").read_text())["judge"]
    required = [REVIEW / "reward.json", REVIEW / "reward-details.json"]
    for path in required:
        path.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-I",
        "-m",
        "eval.verify.review",
        str(RUBRIC),
        "--workspace",
        str(submission),
        "--output",
        str(REVIEW / "reward.json"),
    ]
    try:
        code = await _process(
            command,
            cwd=Path(os.environ["HOME"]),
            environment={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            log=REVIEW / "rules.log",
            timeout=options["timeout"],
        )
    except TimeoutError:
        return {"ok": None, "error": "Rewardkit rules reviewer timed out"}
    if code != 0:
        return {"ok": None, "exit_code": code, "error": "Rewardkit rules reviewer failed"}
    if not all(path.is_file() for path in required):
        return {"ok": None, "error": "Rewardkit returned no complete rules result"}
    return judge.rules_result(*(json.loads(path.read_text()) for path in required))


def _reference_env(values: dict[str, str]) -> dict[str, str]:
    """Relocate searched runtime paths into the reference user's fresh workspace."""
    source = "/workspace/runtime/search"
    return {
        key: str(REFERENCE) + value[len(source) :]
        if value == source or value.startswith(source + "/")
        else value
        for key, value in values.items()
    }


def _backend(
    task: dict, name: str, params: dict, out: Path, *, checkpoint: str | None = None
) -> Backend:
    """Launch restricted runtimes; the controller owns requests, timing and evidence."""
    candidate = name == "candidate"
    environment = runtime_environment() if candidate else runtime_environment("reference")
    user = CANDIDATE_UID if candidate else REFERENCE_UID
    script = (
        Path(task["candidate"]["launch"])
        if candidate
        else (REFERENCE / "baselines" / task["device_category"] / name / "launch.sh")
    )

    @asynccontextmanager
    async def server(stage: str):
        launch = Launch(
            script,
            checkpoint,
            tuple(params.get("args", ())),
            dict(params.get("env", {})) if candidate else _reference_env(params.get("env", {})),
            fixed_options=FIXED_OPTIONS.get(name, frozenset()),
            fixed_env=FIXED_ENV.get(name, frozenset())
            | {"HOME", "USER", "LOGNAME", "TMPDIR", "XDG_CACHE_HOME"},
        )
        try:
            from .access import reset_runtime

            reset_runtime("verifier" if candidate else "reference")
            async with Server(
                launch,
                ENDPOINT,
                cwd=script.parent,
                health_timeout=task["evaluation"]["health_timeout_seconds"],
                log=out / f"{name}-{stage}.log",
                environment=environment,
                user=user,
            ) as owned:
                yield owned
        finally:
            await stop_runtime_users()

    return Backend(
        name,
        request_fields(task["model"], name),
        server,
        # Suite references use native accepted-token counters. Submitted servers
        # must additionally expose IDs and raw text for fixed-tokenizer checks.
        Tokenizer.from_file("/models/tokenizer/tokenizer.json") if candidate else None,
        STATE / "operations" / name,
    )


def _rows(folder: str) -> list[dict]:
    return load_from_disk(folder).to_list()


def collect_submission(
    publish: Path = Path("/var/lib/local-eval/submission"),
    workspace: Path = Path("/workspace/output/submission"),
) -> None:
    """Export only the authored submission; installations and experiments stay separate."""
    # Harbor collect hooks are best-effort: a failed export must leave no stale artifact.
    shutil.rmtree(publish, ignore_errors=True)
    staged = publish.with_suffix(".partial")
    shutil.rmtree(staged, ignore_errors=True)
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        workspace,
        staged,
        symlinks=True,
    )
    staged.replace(publish)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=Path("/var/lib/local-eval/task.json"))
    parser.add_argument(
        "--stage",
        choices=(
            "handoff",
            "prepare",
            "review",
            "quality",
            "reference-prepare",
            "reference",
            "measure",
            "slo",
        ),
    )
    parser.add_argument("--collect-submission", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.collect_submission:
        collect_submission()
        return 0
    if args.stage is None:
        parser.error("provide --stage")
    out = VERIFIER
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward.json").unlink(missing_ok=True)
    if args.stage == "review":
        result = asyncio.run(review_rules(Path("/workspace")))
        (REVIEW / "review.json").write_text(json.dumps(result, indent=2) + "\n")
        for source in REVIEW.iterdir():
            if source.is_file():
                name = f"rules-{source.name}" if source.name.startswith("reward") else source.name
                shutil.copyfile(source, out / name)
        shutil.copyfile(RUBRIC / "rules.toml", out / "rules-rubric.toml")
        shutil.copyfile(RUBRIC / "prompt.txt", out / "rules-prompt.md")
        save_gate(out, result)
        return int(result["ok"] is None)

    task = json.loads(args.task.read_text())
    submission = Path(task["candidate"]["submission"])
    if args.stage in ("handoff", "prepare", "quality", "reference-prepare", "reference"):
        if args.stage == "handoff":
            result = judge.static_check(submission)
        elif args.stage == "quality":
            result = asyncio.run(quality_gate(task, out))
        else:
            operation = {
                "prepare": prepare_submission,
                "reference-prepare": prepare_reference,
                "reference": reference,
            }[args.stage]
            result = asyncio.run(operation(task, submission, out))
        save_gate(out, result)
        return int(result["ok"] is None)

    evidence = {
        "out": out,
        "prepared": json.loads((STATE / "prepare.json").read_text()),
        "reviewed": json.loads((REVIEW / "review.json").read_text()),
        "quality_gate": json.loads((STATE / "quality.json").read_text()),
    }
    candidate = _backend(task, "candidate", {}, out, checkpoint="/models/target")
    rows = _rows(task["inputs"]["private"])
    if args.stage == "slo":
        report = asyncio.run(slo_measure(task, rows, candidate, **evidence))
    else:
        reference_result = json.loads((STATE / "reference.json").read_text())
        report = asyncio.run(evaluate(task, rows, reference_result, candidate, **evidence))
    print(f"Verifier outcome: {report['outcome']}; report: {out / 'report.json'}")
    return int(report["outcome"] == "infrastructure_error")


if __name__ == "__main__":
    raise SystemExit(main())
