"""Validated run settings and task selection; reading configuration has no side effects."""

from __future__ import annotations

import copy
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SEED = 20260921
OUTPUT_ALLOWANCE = 32768
A_N_SESSIONS = 8
QUALITY_SUBSET = 24
POOLS = {"S": ("S-1", "S-N"), "A": ("A-1", "A-N"), "A-MA": ("A-MA",)}
WORKLOAD_POOL = {workload: pool for pool, workloads in POOLS.items() for workload in workloads}
WORKLOADS = tuple(WORKLOAD_POOL)
_CHECKOUT_TABLES = Path(__file__).resolve().parents[1] / "tables"
_PACKAGED_TABLES = Path(__file__).resolve().parent / "_tables"
TABLES = _PACKAGED_TABLES if _PACKAGED_TABLES.is_dir() else _CHECKOUT_TABLES


@dataclass(frozen=True)
class Model:
    key: str
    name: str
    source: str

    @property
    def slug(self) -> str:
        return self.name.lower().replace(".", "").replace(" ", "-")


MODELS = {
    item.key: item
    for item in (
        Model("N1", "Qwen3.8-Flash-Next", "Qwen/Qwen3.8-Flash-Next"),
        Model("N2", "GLM-5.3-Flash", "zai-org/GLM-5.3-Flash"),
        Model("N3", "DeepSeek-V4-Flash-0731", "deepseek-ai/DeepSeek-V4-Flash-0731"),
        Model("N4", "DeepSeek-V4.1-Flash", "deepseek-ai/DeepSeek-V4.1-Flash"),
        Model("N5", "MiniMax-M3", "MiniMaxAI/MiniMax-M3"),
        Model("C1", "Qwen3.8-27B", "Qwen/Qwen3.8-27B"),
        Model("C2", "Qwen3.6-35B-A3B", "Qwen/Qwen3.6-35B-A3B"),
    )
}


def model(value: str) -> Model:
    """Resolve a catalog key, display name, slug, or Hugging Face model string."""
    for item in MODELS.values():
        if value in (item.key, item.name, item.slug, item.source):
            return item
    raise ValueError(f"unknown model {value!r}")


@dataclass(frozen=True)
class Task:
    task_id: str
    model: str
    workload: str
    type: str


@dataclass(frozen=True)
class TaskSelection:
    models: tuple[str, ...] | None = None
    workloads: tuple[str, ...] | None = None
    types: tuple[str, ...] | None = ("SOTA",)


@dataclass(frozen=True)
class AgentConfig:
    name: str
    model: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    auth: str = "login"
    env: tuple[str, ...] = ()
    budget_seconds: int = 7200


@dataclass(frozen=True)
class SearchConfig:
    mode: str = "combined"
    auth: str = "login"


@dataclass(frozen=True)
class EvaluationConfig:
    install_timeout_seconds: int = 14400
    health_timeout_seconds: int = 3600


@dataclass(frozen=True)
class RunConfig:
    """Future runner input. Paths are interpreted beside this configuration file."""

    dataset: str | None = None
    gpus: tuple[str, ...] | None = None
    tasks: TaskSelection = field(default_factory=TaskSelection)
    agent: AgentConfig | None = None
    search: SearchConfig = field(default_factory=SearchConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    out: Path = Path("runs")


def _object(value: Any, name: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {name} field(s): {', '.join(sorted(unknown))}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _choice(value: Any, name: str, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}")
    return value


def _strings(value: Any, name: str, *, all_allowed: bool = True) -> tuple[str, ...] | None:
    if value == "all" and all_allowed:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list" + (" or 'all'" if all_allowed else ""))
    items = tuple(_text(v, name) for v in value)
    if len(set(items)) != len(items):
        raise ValueError(f"{name} contains duplicates")
    return items


def duration(value: Any, name: str = "duration") -> int:
    """Parse a positive integer number of seconds or a duration such as ``2h``."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    if isinstance(value, str) and (match := re.fullmatch(r"([1-9][0-9]*)(s|m|h)", value)):
        return int(match[1]) * {"s": 1, "m": 60, "h": 3600}[match[2]]
    raise ValueError(f"{name} must be positive seconds or an integer duration ending in s, m, or h")


def parse_config(value: Any, *, base: Path | str = ".") -> RunConfig:
    """Validate authored settings without loading datasets, probing hardware, or running tools."""
    raw = _object(
        value, "run", {"dataset", "gpus", "tasks", "agent", "search", "evaluation", "out"}
    )
    dataset = raw.get("dataset")
    if dataset is not None:
        dataset = _text(dataset, "dataset")
    task_raw = _object(raw.get("tasks", {}), "tasks", {"models", "workloads", "types"})
    names = _strings(task_raw.get("models", "all"), "tasks.models")
    names = tuple(model(n).name for n in names) if names is not None else None
    if names is not None and len(set(names)) != len(names):
        raise ValueError("tasks.models selects the same model more than once through aliases")
    workloads = _strings(task_raw.get("workloads", "all"), "tasks.workloads")
    if workloads is not None and set(workloads) - set(WORKLOADS):
        raise ValueError(
            f"unknown workload(s): {', '.join(sorted(set(workloads) - set(WORKLOADS)))}"
        )
    types = _strings(task_raw.get("types", ["SOTA"]), "tasks.types")
    if types is not None and set(types) - {"SOTA", "NEW"}:
        raise ValueError("tasks.types must select SOTA or NEW")
    gpus_raw = raw.get("gpus", "all")
    gpus = None
    if gpus_raw != "all":
        if not isinstance(gpus_raw, list) or not gpus_raw:
            raise ValueError("gpus must be 'all' or a nonempty list of GPU indices or UUIDs")
        gpus = tuple(
            str(v)
            if isinstance(v, int) and not isinstance(v, bool) and v >= 0
            else _text(v, "gpus")
            for v in gpus_raw
        )
        if len(set(gpus)) != len(gpus):
            raise ValueError("gpus contains duplicates")
    agent = None
    if "agent" in raw:
        a = _object(raw["agent"], "agent", {"name", "model", "kwargs", "auth", "env", "budget"})
        kwargs = a.get("kwargs", {})
        if not isinstance(kwargs, dict) or not all(isinstance(k, str) for k in kwargs):
            raise ValueError("agent.kwargs must be a mapping with string keys")
        env_raw = a.get("env", [])
        env = () if env_raw == [] else _strings(env_raw, "agent.env", all_allowed=False)
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in env):
            raise ValueError("agent.env contains an invalid environment variable name")
        agent = AgentConfig(
            _text(a.get("name"), "agent.name"),
            _text(a.get("model"), "agent.model"),
            copy.deepcopy(kwargs),
            _choice(a.get("auth", "login"), "agent.auth", ("login", "api_key")),
            env,
            duration(a.get("budget", "2h"), "agent.budget"),
        )
    s = _object(raw.get("search", {}), "search", {"mode", "auth"})
    search = SearchConfig(
        _choice(s.get("mode", "combined"), "search.mode", ("combined", "single")),
        _choice(s.get("auth", "login"), "search.auth", ("login", "api_key")),
    )
    e = _object(raw.get("evaluation", {}), "evaluation", {"install_timeout", "health_timeout"})
    evaluation = EvaluationConfig(
        duration(e.get("install_timeout", "4h"), "evaluation.install_timeout"),
        duration(e.get("health_timeout", "1h"), "evaluation.health_timeout"),
    )
    out = Path(_text(raw.get("out", "runs"), "out"))
    if not out.is_absolute():
        out = Path(base).resolve() / out
    return RunConfig(
        dataset, gpus, TaskSelection(names, workloads, types), agent, search, evaluation, out
    )


def load_config(path: Path | str) -> RunConfig:
    """Read YAML and resolve output paths relative to its directory."""
    path = Path(path)
    return parse_config(yaml.safe_load(path.read_text()), base=path.parent)


def tasks() -> tuple[Task, ...]:
    """Read the single authoritative task catalog bundled with the package."""
    with (TABLES / "tasks.csv").open(newline="") as handle:
        catalog = tuple(Task(**row) for row in csv.DictReader(handle))
    seen = set()
    for task in catalog:
        if not task.task_id or task.task_id in seen:
            raise ValueError(f"empty or duplicate task identity {task.task_id!r}")
        seen.add(task.task_id)
        if model(task.model).name != task.model:
            raise ValueError(f"task {task.task_id} must use its canonical model name")
        if task.workload not in WORKLOADS or task.type not in {"SOTA", "NEW"}:
            raise ValueError(f"invalid workload or type for task {task.task_id}")
    return catalog


def select_tasks(
    selection: TaskSelection, catalog: tuple[Task, ...] | None = None
) -> tuple[Task, ...]:
    """Select the intersection of model, workload, and type filters in catalog order."""
    names = None if selection.models is None else {model(n).name for n in selection.models}
    selected = tuple(
        task
        for task in (tasks() if catalog is None else catalog)
        if (names is None or task.model in names)
        and (selection.workloads is None or task.workload in selection.workloads)
        and (selection.types is None or task.type in selection.types)
    )
    if not selected:
        raise ValueError("task filters select no tasks")
    return selected
