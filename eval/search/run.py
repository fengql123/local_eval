"""Publish the chosen mapped baseline and adjustable launch parameters."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ..launch import Launch


def finalize(task: dict, selection: Path, *, images: Path = Path("/opt/eval/images")) -> dict:
    """Resolve weights from trusted task metadata, never the agent's weight path.

    Search writes a baseline and args/env, or a bench report with parameters.
    Physical checkpoints are mounted at /models/reference in subsequent jobs.
    The selected scripts remain the protected versions supplied by the image.
    """
    chosen = json.loads(selection.read_text())
    baseline = chosen["baseline"]
    if baseline not in task["assets"]["baselines"]:
        raise ValueError(f"baseline {baseline!r} is not mapped for {task['model']}")
    asset = task["assets"]["baselines"][baseline]
    params = chosen.get("parameters", chosen)
    script = images / task["category"] / "baselines" / baseline / "launch.sh"
    launch = Launch(
        script,
        asset["checkpoint"],
        tuple(params.get("args", ())),
        dict(params.get("env", {})),
        fixed_env=frozenset({"HOME", "USER", "LOGNAME"}),
    )
    launch.prepare()
    reference = {
        "category": task["category"],
        "model": task["model"],
        "baseline": baseline,
        "checkpoint": "/models/reference" if asset["path"] is not None else asset["checkpoint"],
        "args": list(launch.args),
        "env": launch.env,
    }
    for name in ("reference", "public_reference"):
        output = Path(task[name])
        output.write_text(json.dumps(reference, indent=2) + "\n")
        output.chmod(0o644)
    return reference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=Path("/root/eval/task.json"))
    parser.add_argument("--stage", choices=("collect", "gate"), required=True)
    args = parser.parse_args(argv)
    task = json.loads(args.task.read_text())
    report_path = Path(task["reference"]).with_name("report.json")
    if args.stage == "collect":
        # A cancelled attempt can leave outputs in this persistent root bind.
        for output in (report_path, Path(task["reference"]), Path(task["public_reference"])):
            output.unlink(missing_ok=True)
        # Native single-step Harbor collects artifacts before its verifier.
        # Stop search-owned servers, then publish the trusted reference once.
        cleanup = subprocess.run(["pkill", "-KILL", "-u", "search"])
        if cleanup.returncode not in (0, 1):
            raise RuntimeError(f"could not stop search processes: {cleanup.returncode}")
        selection = Path("/home/search/selection.json")
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
