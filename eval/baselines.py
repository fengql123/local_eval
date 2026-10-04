"""Select engine checkpoints from the device/model mapping table."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .config import TABLES
from .config import model as get_model
from .launch import Launch


@dataclass(frozen=True)
class BaselineMapping:
    category: str
    model: str
    baseline: str
    source_model_checkpoint: str
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


def resolve_launch(
    script: Path,
    model: str,
    workload: str,
    *,
    task: dict | None = None,
    reference: dict | None = None,
) -> tuple[str, str | None]:
    """Resolve a template within the selected model/workload task context.

    A complete candidate owns its weights and needs no mapping. A baseline folder
    identifies its engine/category; a copied template uses the task's selected
    reference. Public replay reads that selection from its working directory's
    reference.json. Private verification passes the protected task/selection.
    """
    needs_checkpoint = Launch(script).needs_checkpoint
    if script.parent.parent.name == "baselines":
        engine, category = script.parent.name, script.parents[2].name
    elif needs_checkpoint:
        reference = reference or json.loads(Path("reference.json").read_text())
        engine = reference["baseline"]
        category = task["category"] if task is not None else reference["category"]
    else:
        return "candidate", None
    if not needs_checkpoint:
        return engine, None
    if reference is not None and reference.get("baseline") == engine and "checkpoint" in reference:
        return engine, reference["checkpoint"]
    if task is not None:
        return engine, task["assets"]["baselines"][engine]["checkpoint"]
    if reference is not None:
        return engine, reference["checkpoint"]
    baseline = mapping(category, model, engine)
    checkpoint = (
        "/models/source"
        if baseline.checkpoint == baseline.source_model_checkpoint
        else baseline.checkpoint
    )
    mounted = Path("/models/baselines") / engine
    if checkpoint != "/models/source" and mounted.is_dir():
        checkpoint = str(mounted)
    return engine, checkpoint
