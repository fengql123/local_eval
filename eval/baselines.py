"""Select engine checkpoints from the device/model mapping table."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import TABLES
from .config import model as get_model

RECIPES = Path("/opt/eval/baselines")
WORKSPACE = Path("/workspace/runtime/search/baselines")

FIXED_OPTIONS = {
    "llama.cpp": frozenset(
        {"-m", "-a", "--alias", "-mu", "--model-url", "-hf", "--hf-repo", "-hff", "--hf-file"}
    ),
    "vllm": frozenset({"--served-model-names"}),
    "ktransformers": frozenset({"--model-path", "--kt-weight-path"}),
    "ktransformers-dsv4": frozenset({"--model-path", "--kt-weight-path"}),
}
FIXED_ENV = {
    "ollama": frozenset({"OLLAMA_HOST", "OLLAMA_MODELS", "OLLAMA_NOPRUNE", "OLLAMA_NO_CLOUD"}),
}


@dataclass(frozen=True)
class BaselineMapping:
    category: str
    model: str
    baseline: str
    checkpoint: str

    @property
    def launch_script(self) -> Path:
        return Path("images") / self.category / "baselines" / self.baseline / "launch.sh"


def mappings(category: str, model: str) -> tuple[BaselineMapping, ...]:
    name = get_model(model).name
    with (TABLES / "baseline_map.csv").open(newline="") as handle:
        return tuple(
            BaselineMapping(**row)
            for row in csv.DictReader(handle)
            if row["category"] == category and row["model"] == name
        )


def mapping(category: str, model: str, baseline: str) -> BaselineMapping:
    return next(item for item in mappings(category, model) if item.baseline == baseline)


def needed(category: str, models: tuple[str, ...]) -> tuple[str, ...]:
    """List each engine once, in the selected models' table order."""
    return tuple(
        dict.fromkeys(item.baseline for name in models for item in mappings(category, name))
    )


def install(
    task: dict,
    names: tuple[str, ...] | None = None,
    *,
    recipes: Path = RECIPES,
    workspace: Path = WORKSPACE,
    environment: dict[str, str] | None = None,
) -> None:
    """Install small trusted recipes in the current phase's writable workspace.

    The caller already runs as search, agent or verifier. No model bytes are
    copied, and a completed installation is reused when a local step resumes.
    """
    for name in task["assets"]["baselines"] if names is None else names:
        destination = workspace / task["device_category"] / name
        complete = destination / ".installed"
        if complete.is_file():
            continue
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        recipe = recipes / task["device_category"] / name
        for source in (recipe / "install.sh", recipe / "launch.sh", *recipe.glob("*.patch")):
            target = destination / source.name
            shutil.copyfile(source, target)
            target.chmod(0o644 if source.suffix == ".patch" else 0o755)
        with (destination / "install.log").open("wb") as log:
            result = subprocess.run(
                ["/bin/bash", str(destination / "install.sh")],
                cwd=destination,
                env=environment,
                stdout=log,
                stderr=log,
            )
        if result.returncode:
            detail = (destination / "install.log").read_text(errors="replace")
            raise RuntimeError(f"{name} installation exited {result.returncode}:\n{detail}")
        complete.touch()


def resolve_launch(
    script: Path,
    model: str,
    workload: str,
    *,
    task: dict | None = None,
    reference: dict | None = None,
) -> tuple[str, str | None]:
    """Recognize authored baseline folders; completed candidate scripts own their weights.

    Public experiments can supply their acquired path through CHECKPOINT. Otherwise
    a baseline's checkpoint source comes from its model/category mapping. Trusted
    verification passes an acquired local path directly to Launch.
    """
    script = script.resolve()
    if script.parent.parent.name == "baselines":
        engine, category = script.parent.name, script.parent.parent.parent.name
    elif script.parent.parent.parent.name == "baselines":
        engine, category = script.parent.name, script.parent.parent.name
    else:
        return "candidate", "/models/target" if script.name == "server.launch" else None
    checkpoint = os.environ.get("CHECKPOINT")
    if checkpoint is None:
        checkpoint = (
            task["assets"]["baselines"][engine]
            if task is not None
            else mapping(category, model, engine).checkpoint
        )
    return engine, checkpoint


def main(argv: list[str] | None = None) -> int:
    """Install selected recipes when an agent or trusted verifier requests them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=Path("/var/lib/local-eval/task.json"))
    parser.add_argument("--baseline")
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--recipes", type=Path, default=RECIPES)
    args = parser.parse_args(argv)
    install(
        json.loads(args.task.read_text()),
        (args.baseline,) if args.baseline else None,
        workspace=args.workspace,
        recipes=args.recipes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
