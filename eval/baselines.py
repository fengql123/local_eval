"""Baseline source, installer paths and launch metadata; execute nothing."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .config import TABLES
from .config import model as get_model


@dataclass(frozen=True)
class BaselineLaunch:
    argv: tuple[str, ...]
    env: dict[str, str]
    served_model: str


@dataclass(frozen=True)
class BaselineRecipe:
    baseline: str
    category: str
    repository: str
    code_ref: str
    prefix: Path
    environment: str
    install_script: Path
    launch_argv: tuple[str, ...]
    launch_env: dict[str, str]
    status: str
    doc: str
    note: str

    def launch(
        self,
        checkpoint: str,
        port: int,
        *,
        args: tuple[str, ...] = (),
        env: dict[str, str] | None = None,
        prefix: Path | str | None = None,
    ) -> BaselineLaunch:
        """Compose catalog defaults plus caller arguments and environment overrides."""
        if self.status == "planned":
            raise ValueError(f"{self.baseline} on {self.category} is planned, not supported")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("checkpoint must be a nonempty model string or local checkpoint path")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("port must be an integer between 1 and 65535")
        values = {"prefix": str(prefix or self.prefix), "checkpoint": checkpoint, "port": str(port)}
        argv = tuple(part.format_map(values) for part in self.launch_argv) + tuple(args)
        settings = {key: value.format_map(values) for key, value in self.launch_env.items()}
        settings.update(env or {})
        return BaselineLaunch(argv, settings, checkpoint if self.baseline == "ollama" else "served")


@dataclass(frozen=True)
class BaselineMapping:
    category: str
    model: str
    baseline: str
    checkpoint_kind: str
    checkpoint: str
    file_pattern: str
    default_args: tuple[str, ...]
    default_env: dict[str, str]
    doc: str


def _rows(name: str) -> tuple[dict[str, str], ...]:
    with (TABLES / name).open(newline="") as handle:
        return tuple(csv.DictReader(handle))


def categories() -> tuple[dict[str, str], ...]:
    return _rows("categories.csv")


def recipes(category: str | None = None) -> tuple[BaselineRecipe, ...]:
    rows = _rows("baselines.csv")
    return tuple(
        BaselineRecipe(
            **{
                **row,
                "prefix": Path(row["prefix"]),
                "install_script": Path(row["install_script"]),
                "launch_argv": tuple(json.loads(row["launch_argv"])),
                "launch_env": json.loads(row["launch_env"]),
            }
        )
        for row in rows
        if category is None or row["category"] == category
    )


def recipe(baseline: str, category: str) -> BaselineRecipe:
    for item in recipes(category):
        if item.baseline == baseline:
            return item
    raise ValueError(f"no baseline recipe for {baseline!r} on {category!r}")


def mappings(category: str, model: str) -> tuple[BaselineMapping, ...]:
    name = get_model(model).name
    return tuple(
        BaselineMapping(
            **{
                **row,
                "default_args": tuple(json.loads(row["default_args"])),
                "default_env": json.loads(row["default_env"]),
            }
        )
        for row in _rows("baseline_map.csv")
        if row["category"] == category and row["model"] == name
    )


def mapping(category: str, model: str, baseline: str) -> BaselineMapping:
    for item in mappings(category, model):
        if item.baseline == baseline:
            return item
    raise ValueError(
        f"no checkpoint mapping for {get_model(model).name} on {baseline!r} in {category!r}"
    )


def needed(category: str, models: tuple[str, ...]) -> tuple[BaselineRecipe, ...]:
    """Deduplicate usable engines; every selected model must have a usable mapping."""
    available = {item.baseline: item for item in recipes(category) if item.status != "planned"}
    offered = set()
    for name in models:
        supported = {
            item.baseline for item in mappings(category, name) if item.baseline in available
        }
        if not supported:
            raise ValueError(
                f"no usable baseline mapping for {get_model(name).name} in {category!r}"
            )
        offered.update(supported)
    return tuple(item for item in available.values() if item.baseline in offered)


def validate_catalog() -> None:
    """Check our authored tables' cross references; this never reads the user dataset."""
    category_rows = categories()
    known_categories = {row["category"] for row in category_rows}
    if "" in known_categories or len(known_categories) != len(category_rows):
        raise ValueError("empty or duplicate machine category")
    items = recipes()
    keys = {(item.category, item.baseline) for item in items}
    if len(keys) != len(items):
        raise ValueError("duplicate baseline recipe")
    for item in items:
        if item.category not in known_categories:
            raise ValueError(f"unknown category {item.category!r}")
        if item.environment not in ("venv", "native") or item.status not in (
            "pending-assets",
            "planned",
        ):
            raise ValueError(f"invalid recipe status or environment for {item.baseline}")
        expected_script = (
            Path("images") / item.category / "baselines" / item.baseline / "install.sh"
        )
        if item.install_script != expected_script or not item.prefix.is_absolute():
            raise ValueError(f"invalid installer path or prefix for {item.baseline}")
        if item.status != "planned" and (not item.launch_argv or not item.code_ref):
            raise ValueError(f"incomplete launch metadata for {item.baseline}")
        if (
            item.environment == "venv"
            and item.status != "planned"
            and "{prefix}/.venv/bin/" not in item.launch_argv[0]
        ):
            raise ValueError(f"{item.baseline} must launch within its own venv")
    seen = set()
    for row in _rows("baseline_map.csv"):
        key = (row["category"], row["model"], row["baseline"])
        if key in seen or (row["category"], row["baseline"]) not in keys:
            raise ValueError(f"duplicate or unresolved mapping {key}")
        seen.add(key)
        get_model(row["model"])
        if row["checkpoint_kind"] not in ("hf", "gguf", "ollama", "planned"):
            raise ValueError(f"invalid checkpoint kind in {key}")
        if not row["checkpoint"] or (row["checkpoint_kind"] == "gguf" and not row["file_pattern"]):
            raise ValueError(f"missing checkpoint or GGUF file selection in {key}")
        args, env = json.loads(row["default_args"]), json.loads(row["default_env"])
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ValueError(f"invalid default argv in {key}")
        if not isinstance(env, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in env.items()
        ):
            raise ValueError(f"invalid default env in {key}")
