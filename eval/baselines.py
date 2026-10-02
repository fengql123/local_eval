"""Select engine checkpoints from the device/model mapping table."""

from __future__ import annotations

import csv
from dataclasses import dataclass

from .config import TABLES
from .config import model as get_model


@dataclass(frozen=True)
class BaselineMapping:
    category: str
    model: str
    baseline: str
    source_model_checkpoint: str
    checkpoint: str


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
