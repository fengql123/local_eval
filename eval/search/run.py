"""Publish the chosen mapped baseline and adjustable launch parameters.

Search may instead report that no mapped baseline serves the model on this
device, which makes the pair an SLO task on FreeToken's checkpoint. The verdict
keeps bounded log tails from every attempt for audit.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path

from ..baselines import FIXED_ENV, FIXED_OPTIONS, RECIPES
from ..launch import Launch

STAGES = ("precheck", "install", "acquire", "launch", "serve")
SEARCH_UID = 1002
LOG_ROOTS = (Path("/home/search"), Path("/workspace"))
LOG_LIMIT = 64 * 1024
MAX_ATTEMPTS = 32


def _tail(source: str, destination: Path, *, uid: int, roots: tuple[Path, ...]) -> str:
    """Copy the end of a search-owned regular file without following a final link."""
    path = Path(source).resolve()
    if not any(path.is_relative_to(root) for root in roots):
        raise ValueError(f"log {source} is outside {', '.join(map(str, roots))}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        status = os.fstat(handle.fileno())
        if not stat.S_ISREG(status.st_mode) or status.st_uid != uid:
            raise ValueError(f"log {source} is not a regular file owned by search")
        handle.seek(max(0, status.st_size - LOG_LIMIT))
        destination.write_bytes(handle.read(LOG_LIMIT))
    destination.chmod(0o644)
    return destination.name


def verdict(
    task: dict,
    chosen: dict,
    output: Path,
    *,
    uid: int = SEARCH_UID,
    roots: tuple[Path, ...] = LOG_ROOTS,
) -> dict:
    """Validate an unsupported verdict; every mapped baseline needs a failed attempt."""
    mapped = task["assets"]["baselines"]
    attempts = chosen.get("attempts")
    if not isinstance(attempts, list) or not 0 < len(attempts) <= MAX_ATTEMPTS:
        raise ValueError(f"an unsupported verdict needs 1-{MAX_ATTEMPTS} attempts")
    if not all(isinstance(attempt, dict) for attempt in attempts):
        raise ValueError("each attempt must be an object")
    tried = {attempt.get("baseline") for attempt in attempts}
    if tried != set(mapped):
        raise ValueError(f"attempts must cover each mapped baseline {sorted(mapped)} only")
    folder = output.parent / "attempts"
    folder.mkdir(exist_ok=True)
    published = []
    for number, attempt in enumerate(attempts):
        if attempt.get("stage") not in STAGES:
            raise ValueError(f"attempt stage must be one of {', '.join(STAGES)}")
        error = attempt.get("error")
        if not isinstance(error, str) or not error.strip():
            raise ValueError("each attempt needs a non-empty error")
        entry = {key: attempt[key] for key in ("baseline", "stage")} | {"error": error[:4096]}
        log = attempt.get("log")
        entry["log"] = None
        if log is not None:
            name = f"{number:02d}-{attempt['baseline']}-{attempt['stage']}.log"
            try:
                entry["log"] = "attempts/" + _tail(str(log), folder / name, uid=uid, roots=roots)
            except (OSError, ValueError) as failure:
                # Missing evidence is reported; the stated error remains the verdict.
                entry["log_error"] = f"{type(failure).__name__}: {failure}"
        published.append(entry)
    return {"supported": False, "attempts": published}


def finalize(task: dict, selection: Path, *, recipes: Path = RECIPES, **options) -> dict:
    """Publish only a mapped baseline and adjustable parameters, never local weights."""
    chosen = json.loads(selection.read_text())
    if not isinstance(chosen, dict):
        raise TypeError("selection.json must contain an object")
    supported = chosen.get("supported", True)
    if type(supported) is not bool:
        raise TypeError("supported must be true or false")
    output = Path(task["reference"])
    if not supported:
        reference = verdict(task, chosen, output, **options)
    else:
        baseline = chosen["baseline"]
        if baseline not in task["assets"]["baselines"]:
            raise ValueError(f"baseline {baseline!r} is not mapped for {task['model']}")
        asset = task["assets"]["baselines"][baseline]
        params = chosen.get("parameters", chosen)
        script = recipes / task["device_category"] / baseline / "launch.sh"
        launch = Launch(
            script,
            asset,
            tuple(params.get("args", ())),
            dict(params.get("env", {})),
            fixed_options=FIXED_OPTIONS.get(baseline, frozenset()),
            fixed_env=FIXED_ENV.get(baseline, frozenset()) | {"HOME", "USER", "LOGNAME"},
        )
        launch.prepare()
        reference = {
            "baseline": baseline,
            "args": list(launch.args),
            "env": launch.env,
        }
    output.write_text(json.dumps(reference, indent=2) + "\n")
    output.chmod(0o644)
    return reference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=Path("/var/lib/local-eval/task.json"))
    parser.add_argument("--stage", choices=("collect", "gate"), required=True)
    args = parser.parse_args(argv)
    task = json.loads(args.task.read_text())
    report_path = Path(task["reference"]).with_name("report.json")
    if args.stage == "collect":
        # A cancelled attempt can leave outputs before the native artifact handoff.
        for output in (report_path, Path(task["reference"])):
            output.unlink(missing_ok=True)
        attempts = Path(task["reference"]).with_name("attempts")
        if attempts.is_dir():
            for path in attempts.iterdir():
                path.unlink()
        # Native Harbor collects artifacts before its verifier and owns cleanup.
        selection = Path("/workspace/output/selection.json")
        try:
            reference = finalize(task, selection)
        except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
            if isinstance(error, FileNotFoundError) and error.filename != str(selection):
                # A missing protected image asset is infrastructure, not search output.
                raise
            report = {"ok": False, "error": str(error)}
        else:
            report = {"ok": True, "reference": reference}
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        return 0

    report = json.loads(report_path.read_text())
    out = Path("/logs/verifier")
    out.mkdir(parents=True, exist_ok=True)
    reward = out / "reward.json"
    reward.unlink(missing_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    reward.write_text(json.dumps({"reward": int(report["ok"])}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
