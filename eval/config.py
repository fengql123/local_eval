"""Run settings and task selection; reading configuration has no side effects."""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SEED = 20260921
OUTPUT_ALLOWANCE = 32768
A_N_SESSIONS = 8
CALIBRATION_SUBSET = 24
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
    model: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    auth: str = "login"
    env: tuple[str, ...] = ()
    budget_seconds: int = 7200


@dataclass(frozen=True)
class SearchConfig:
    mode: str = "combined"
    agent: AgentConfig = field(default_factory=lambda: AgentConfig("claude-code"))


@dataclass(frozen=True)
class EvaluationConfig:
    install_timeout_seconds: int = 14400
    health_timeout_seconds: int = 3600


@dataclass(frozen=True)
class RunConfig:
    """Runner input. Paths are interpreted beside this configuration file."""

    dataset: str | None = None
    gpus: tuple[str, ...] | None = None
    tasks: TaskSelection = field(default_factory=TaskSelection)
    agent: AgentConfig | None = None
    search: SearchConfig = field(default_factory=SearchConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    out: Path = Path("runs")


def _selection(value):
    return None if value is None or value == "all" else tuple(value)


def duration(value: int | str) -> int:
    """Convert seconds or a duration such as 2h; native parsing errors propagate."""
    return (
        int(value[:-1]) * {"s": 1, "m": 60, "h": 3600}[value[-1]]
        if isinstance(value, str)
        else value
    )


def _agent(value: dict | None) -> AgentConfig | None:
    if value is None:
        return None
    budget = duration(value.pop("budget", "2h"))
    value["env"] = tuple(value.get("env", ()))
    return AgentConfig(**value, budget_seconds=budget)


def parse_config(value: Any, *, base: Path | str = ".") -> RunConfig:
    """Map YAML settings to the run dataclasses without acquisition or probes."""
    raw = copy.deepcopy(value)
    selected = TaskSelection(**raw.pop("tasks", {}))
    names = _selection(selected.models)
    selected = TaskSelection(
        tuple(model(name).name for name in names) if names is not None else None,
        _selection(selected.workloads),
        _selection(selected.types),
    )
    gpu_ids = _selection(raw.pop("gpus", "all"))
    gpu_ids = tuple(map(str, gpu_ids)) if gpu_ids is not None else None
    agent = _agent(raw.pop("agent", None))
    search = raw.pop("search", {})
    search_agent = _agent(search.pop("agent", {"name": "claude-code"}))
    search = SearchConfig(**search, agent=search_agent)
    evaluation = raw.pop("evaluation", {})
    evaluation = EvaluationConfig(
        duration(evaluation.pop("install_timeout", "4h")),
        duration(evaluation.pop("health_timeout", "1h")),
        **evaluation,
    )
    out = Path(raw.pop("out", "runs"))
    if not out.is_absolute():
        out = Path(base).resolve() / out
    return RunConfig(
        **raw,
        gpus=gpu_ids,
        tasks=selected,
        agent=agent,
        search=search,
        evaluation=evaluation,
        out=out,
    )


def load_config(path: Path | str) -> RunConfig:
    """Read YAML and resolve output paths relative to its directory."""
    path = Path(path)
    return parse_config(yaml.safe_load(path.read_text()), base=path.parent)


def tasks() -> tuple[Task, ...]:
    """Read the single authoritative task catalog bundled with the package."""
    with (TABLES / "tasks.csv").open(newline="") as handle:
        catalog = tuple(Task(**row) for row in csv.DictReader(handle))
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
