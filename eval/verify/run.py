"""Native Harbor verification steps: root preparation, judge review, root measurement."""

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
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..baselines import resolve_launch
from ..bench.replay import ReplayError
from ..bench.run import measure, quality
from ..bench.serve import Server, StartupError
from ..launch import ENDPOINT, Launch
from ..request_policy import RequestPolicy, request_fields
from . import judge

ZERO = {"throughput": 0.0, "ttft": 0.0}
VERIFIER = Path("/logs/verifier")
STATE = Path("/root/eval/verification")
RUBRIC = Path("/workspace/review-rubric")
REVIEW = Path("/workspace/review-output")
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


def runtime_environment() -> dict[str, str]:
    """Root runtime context with device tools and without reviewer credentials."""
    shared = (
        "PATH",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "ROCM_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
    )
    return {
        **{key: os.environ[key] for key in shared if key in os.environ},
        "HOME": "/root",
        "USER": "root",
        "LOGNAME": "root",
        "TMPDIR": "/tmp/local-eval/root",
        "XDG_CACHE_HOME": "/tmp/local-eval/root/cache",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


@dataclass(frozen=True)
class Backend:
    """Request policy and a fresh, owned server for each named evaluation stage."""

    name: str
    requests: RequestPolicy | Callable[[], RequestPolicy]
    server: Callable[[str], AbstractAsyncContextManager[Server]]

    @property
    def policy(self) -> RequestPolicy:
        """Resolve a candidate's current request policy after its installer finishes."""
        return self.requests() if callable(self.requests) else self.requests


def _leaves(error: BaseException):
    if isinstance(error, BaseExceptionGroup):
        for nested in error.exceptions:
            yield from _leaves(nested)
    else:
        yield error


async def probe(backend: Backend, rows: list[dict]) -> dict:
    """Run the shared quality replay with every S row eligible immediately."""
    try:
        return await quality(rows, backend.policy, server=backend.server("quality"), concurrency=-1)
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


async def pre_run(
    rows: list[dict],
    reference: Backend,
    candidate: Backend,
) -> dict:
    """Gate timing on paired quality after native preparation and review passed."""
    result = {}
    quality = result["quality"] = {"reference": await probe(reference, rows)}
    if not quality["reference"]["ok"]:
        return {
            **result,
            "ok": None,
            "stage": quality["reference"].get("stage", "quality"),
            "reason": "reference quality run failed",
        }
    quality["candidate"] = await probe(candidate, rows)
    if not quality["candidate"]["ok"]:
        return {
            **result,
            "ok": None if quality["candidate"].get("infrastructure") else False,
            "stage": quality["candidate"].get("stage", "quality"),
            "reason": "candidate quality run failed",
        }
    quality.update(judge.paired(rows, quality["reference"]["items"], quality["candidate"]["items"]))
    return {
        **result,
        "ok": quality["ok"],
        "stage": "quality",
        "reason": "pre-run gate passed"
        if quality["ok"]
        else "candidate quality is significantly worse",
    }


def _errors(error: BaseException) -> list[str]:
    """Keep request protocol context when a TaskGroup reports nested failures."""
    if isinstance(error, BaseExceptionGroup):
        return [detail for child in error.exceptions for detail in _errors(child)]
    if isinstance(error, ReplayError):
        return [f"{error}: {error.record['errors']}"]
    return [f"{type(error).__name__}: {error}"]


def rewards(reference: dict, candidate: dict) -> dict:
    """Twice as fast earns 50; equal or slower performance earns zero."""
    gains = {
        "throughput": candidate["throughput"] / reference["throughput"],
        "ttft": reference["ttft_mean"] / candidate["ttft_mean"],
    }
    return {key: round(100 * max(0, 1 - 1 / gain), 2) for key, gain in gains.items()}


def _save(out: Path, report: dict) -> None:
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
    if "rewards" in report:
        text.extend(f"- {key}: {value}" for key, value in report["rewards"].items())
    text.extend(["", "Full quality and timing evidence: [report.json](report.json).", ""])
    (out / "report.md").write_text("\n".join(text))


async def evaluate(
    task: dict,
    rows: list[dict],
    quality_rows: list[dict],
    reference: Backend,
    candidate: Backend,
    *,
    out: Path,
    prepared: dict,
    reviewed: dict,
) -> dict:
    """Measure an already prepared/reviewed submission, using fresh owned servers."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward.json").unlink(missing_ok=True)
    gate = {**prepared, "rules": reviewed}
    if prepared["ok"] is True and reviewed["ok"] is True:
        gate.update(await pre_run(quality_rows, reference, candidate))
    elif prepared["ok"] is True:
        gate.update(ok=reviewed["ok"], stage="rules", reason="rules did not pass")
    report = {
        "model": task["model"],
        "workload": task["workload"],
        "reference": reference.name,
        "candidate": candidate.name,
        "pre_run": gate,
        "bench": {"skipped": True},
    }
    if gate["ok"] is not True:
        report["outcome"] = "candidate_failure" if gate["ok"] is False else "infrastructure_error"
        report["reason"] = gate["reason"]
        if gate["ok"] is False:
            report["rewards"] = dict(ZERO)
        _save(out, report)
        return report

    report["bench"]["skipped"] = False
    for side, backend in (("reference", reference), ("candidate", candidate)):
        try:
            result = await measure(
                rows, task["workload"], backend.policy, server=backend.server("timed")
            )
        except* RUNTIME_ERRORS as error:
            result = {
                "valid": False,
                "errors": _errors(error),
                "infrastructure": error.subgroup(
                    (Infrastructure, OSError, subprocess.SubprocessError)
                )
                is not None,
            }
        report["bench"][side] = result
        if not result["valid"]:
            infrastructure = side == "reference" or result.get("infrastructure", False)
            report["outcome"] = "infrastructure_error" if infrastructure else "candidate_failure"
            report["reason"] = f"{side} timing failed: {result['errors']}"
            if not infrastructure:
                report["rewards"] = dict(ZERO)
            _save(out, report)
            return report
    report["outcome"] = "passed"
    report["rewards"] = rewards(report["bench"]["reference"], report["bench"]["candidate"])
    _save(out, report)
    return report


@contextmanager
def offline():
    """Keep quality/timed traffic on loopback in this fresh container only."""
    rule = ["OUTPUT", "1", "!", "-o", "lo", "-j", "REJECT"]
    added = []
    try:
        for tool in ("iptables", "ip6tables"):
            try:
                subprocess.run([tool, "-I", *rule], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as error:
                raise Infrastructure(f"could not enforce offline serving with {tool}") from error
            added.append(tool)
        yield
    finally:
        for tool in reversed(added):
            subprocess.run(
                [tool, "-D", *rule[:1], *rule[2:]], check=True, capture_output=True, text=True
            )


async def _process(
    command: list[str], *, cwd: Path, environment: dict, log: Path, timeout: float
) -> int:
    """Inherit Harbor's selected step user and clean up only the owned process group."""
    with log.open("wb") as output:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            start_new_session=True,
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


async def _prepare_script(
    submission: Path, name: str, checkpoint: str, log: Path, *, timeout: float
) -> dict:
    environment = {
        **runtime_environment(),
        "CHECKPOINT": checkpoint,
        "WEIGHTS": str(submission.parent / "weights"),
        "RUNTIME": str(submission.parent / "runtime"),
    }
    try:
        code = await _process(
            ["/bin/bash", str((submission / name).resolve())],
            cwd=submission,
            environment=environment,
            log=log,
            timeout=timeout,
        )
    except TimeoutError:
        return {"ok": False, "exit_code": "timeout", "log": str(log)}
    return {"ok": code == 0, "exit_code": code, "log": str(log)}


async def convert(submission: Path, checkpoint: str, log: Path, *, timeout: float = 14400) -> dict:
    """Create fresh output directories and run optional submitted conversion as root."""
    for folder in (
        submission.parent / "weights",
        submission.parent / "runtime",
        Path(runtime_environment()["XDG_CACHE_HOME"]),
    ):
        folder.mkdir(parents=True, exist_ok=True)
    if not (submission / "convert_weight.sh").is_file():
        return {"ok": True, "skipped": True}
    return await _prepare_script(submission, "convert_weight.sh", checkpoint, log, timeout=timeout)


async def install(submission: Path, checkpoint: str, log: Path, *, timeout: float = 14400) -> dict:
    """Reproduce dependencies using Harbor's root verifier identity."""
    return await _prepare_script(submission, "install.sh", checkpoint, log, timeout=timeout)


def save_gate(out: Path, report: dict) -> None:
    """A failed candidate stops later Harbor steps; infrastructure has no reward."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    reward = out / "reward.json"
    reward.unlink(missing_ok=True)
    if report["ok"] is not None:
        reward.write_text(json.dumps({"reward": int(report["ok"])}) + "\n")


async def prepare_submission(task: dict, submission: Path, out: Path) -> dict:
    """Root checks, converts and installs; the image prepares readable review inputs."""
    out.mkdir(parents=True, exist_ok=True)
    result = {"static": judge.static_check(submission)}
    if not result["static"]["ok"]:
        return {
            **result,
            "ok": False,
            "stage": "static",
            "reason": "submission scripts are missing or external",
        }
    for name, operation in (("convert", convert), ("install", install)):
        try:
            result[name] = await operation(
                submission,
                task["assets"]["source"]["checkpoint"],
                out / f"{name}.log",
                timeout=task["evaluation"]["install_timeout_seconds"],
            )
        except RUNTIME_ERRORS as error:
            result[name] = {"ok": None, "error": f"{type(error).__name__}: {error}"}
        if result[name]["ok"] is not True:
            return {
                **result,
                "ok": result[name]["ok"],
                "stage": name,
                "reason": f"{name} did not pass",
            }
    judge.write_review(task, submission, RUBRIC)
    code = await _process(
        ["/bin/bash", "/opt/eval/images/prepare.sh", str(submission), str(RUBRIC), str(REVIEW)],
        cwd=submission,
        environment=runtime_environment(),
        log=out / "access.log",
        timeout=60,
    )
    result["access"] = {"exit_code": code}
    return {
        **result,
        "ok": True if code == 0 else False if code == 65 else None,
        "stage": "prepare",
        "reason": "submission prepared" if code == 0 else "review inputs could not be prepared",
    }


async def review_rules(submission: Path) -> dict:
    """Run native Rewardkit as Harbor's judge user, without private task metadata."""
    options = tomllib.loads((RUBRIC / "rules.toml").read_text())["judge"]
    required = [REVIEW / "reward.json", REVIEW / "reward-details.json"]
    for path in required:
        path.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-I",
        "-m",
        "rewardkit",
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


def _backend(
    task: dict, name: str, params: dict, out: Path, *, reference: dict | None = None
) -> Backend:
    """Launch both reference and candidate as the root verifier, after preparation."""
    environment = runtime_environment()
    if name == "candidate":
        script = Path(task["candidate"]["launch"])
        cwd = script.parent
    else:
        script = Path("/opt/eval/images") / task["category"] / "baselines" / name / "launch.sh"
        cwd = Path(environment["HOME"])

    def resolved():
        return resolve_launch(
            script, task["model"], task["workload"], task=task, reference=reference or params
        )

    @asynccontextmanager
    async def server(stage: str):
        _engine, checkpoint = resolved()
        launch = Launch(
            script,
            checkpoint,
            tuple(params.get("args", ())),
            dict(params.get("env", {})),
            fixed_env=frozenset({"HOME", "USER", "LOGNAME"}),
        )
        # Server exits before the network guard, including on startup failure.
        with offline():
            async with Server(
                launch,
                ENDPOINT,
                cwd=cwd,
                health_timeout=task["evaluation"]["health_timeout_seconds"],
                log=out / f"{name}-{stage}.log",
                environment=environment,
            ) as owned:
                yield owned

    return Backend(name, lambda: request_fields(task["model"], resolved()[0]), server)


def _rows(folder: str) -> list[dict]:
    return [json.loads(line) for line in (Path(folder) / "rows.jsonl").read_text().splitlines()]


def collect_submission(publish: Path = Path("/logs/artifacts")) -> None:
    """Stop coding-owned processes and leave only the declared source artifact."""
    result = subprocess.run(["pkill", "-KILL", "-u", "agent"])
    if result.returncode not in (0, 1):
        raise Infrastructure(f"could not stop coding processes: {result.returncode}")
    for path in publish.iterdir():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=Path("/root/eval/task.json"))
    parser.add_argument("--stage", choices=("handoff", "prepare", "review", "measure"))
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
        result = asyncio.run(review_rules(Path("/workspace/submission")))
        (REVIEW / "review.json").write_text(json.dumps(result, indent=2) + "\n")
        # Flat native logs can be moved by Harbor across container/host UIDs.
        # Keep the private reviewer workspace and its ownership out of the export.
        for source in REVIEW.iterdir():
            if source.is_file():
                name = f"rules-{source.name}" if source.name.startswith("reward") else source.name
                shutil.copyfile(source, out / name)
        shutil.copyfile(RUBRIC / "rules.toml", out / "rules-rubric.toml")
        shutil.copyfile(RUBRIC / "prompt.txt", out / "rules-prompt.md")
        save_gate(out, result)
        return int(result["ok"] is None)

    task = json.loads(args.task.read_text())
    submission = Path(task["candidate"]["launch"]).parent
    if args.stage == "handoff":
        result = judge.static_check(submission)
        save_gate(out, result)
        return 0
    if args.stage == "prepare":
        result = asyncio.run(prepare_submission(task, submission, out))
        STATE.mkdir(parents=True, exist_ok=True)
        (STATE / "prepare.json").write_text(json.dumps(result, indent=2) + "\n")
        save_gate(out, result)
        return int(result["ok"] is None)

    selected = json.loads(Path(task["reference"]).read_text())
    reference = _backend(task, selected["baseline"], selected, out)
    candidate = _backend(task, "candidate", {}, out, reference=selected)
    report = asyncio.run(
        evaluate(
            task,
            _rows(task["inputs"]["private"]),
            _rows(task["inputs"]["quality"]),
            reference,
            candidate,
            out=out,
            prepared=json.loads((STATE / "prepare.json").read_text()),
            reviewed=json.loads((REVIEW / "review.json").read_text()),
        )
    )
    print(f"Verifier outcome: {report['outcome']}; report: {out / 'report.json'}")
    return int(report["outcome"] == "infrastructure_error")


if __name__ == "__main__":
    raise SystemExit(main())
