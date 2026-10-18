"""Run settings and task selection; reading configuration has no side effects."""

from __future__ import annotations

import copy
import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

OUTPUT_ALLOWANCE = 32768
A_N_SESSIONS = 8
POOLS = {"S": ("S-1", "S-N"), "A": ("A-1", "A-N"), "A-MA": ("A-MA",)}
WORKLOAD_POOL = {workload: pool for pool, workloads in POOLS.items() for workload in workloads}
WORKLOADS = tuple(WORKLOAD_POOL)
JUDGE_ENV = {
    "claude-code": (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "REWARDKIT_FORCE_SUBSCRIPTION",
    ),
    "codex": ("OPENAI_API_KEY", "CODEX_AUTH_JSON", "REWARDKIT_FORCE_SUBSCRIPTION"),
    "fx": ("AI_GATEWAY_API_KEY", "VERCEL_OIDC_TOKEN"),
}
_CHECKOUT_TABLES = Path(__file__).resolve().parents[1] / "tables"
_PACKAGED_TABLES = Path(__file__).resolve().parent / "_tables"
TABLES = _PACKAGED_TABLES if _PACKAGED_TABLES.is_dir() else _CHECKOUT_TABLES


@dataclass(frozen=True)
class Model:
    name: str
    checkpoint: str

    @property
    def slug(self) -> str:
        return self.name.lower().replace(".", "").replace(" ", "-")


def models() -> tuple[Model, ...]:
    """Read the model catalog and its fixed candidate checkpoints."""
    with (TABLES / "models.csv").open(newline="") as handle:
        return tuple(Model(row["model"], row["checkpoint"]) for row in csv.DictReader(handle))


def model(value: str) -> Model:
    """Resolve a catalog key, display name, slug, or Hugging Face model string."""
    for item in models():
        if value in (item.name, item.slug, item.checkpoint):
            return item
    raise ValueError(f"unknown model {value!r}")


@dataclass(frozen=True)
class Task:
    """A catalog serving task, classified after reference resolution."""

    task_id: str
    model: str
    workload: str
    type: str | None = None
    device_category: str | None = None


@dataclass(frozen=True)
class TaskSelection:
    models: tuple[str, ...] | None = None
    workloads: tuple[str, ...] | None = None


@dataclass(frozen=True)
class AgentConfig:
    name: str
    model: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    budget_seconds: int = 7200


def reviewer_env(agent: AgentConfig) -> tuple[str, ...]:
    """Select native credential inputs that Harbor forwards to the verifier."""
    return tuple(name for name in JUDGE_ENV[agent.name] if name in os.environ)


@dataclass(frozen=True)
class SearchConfig:
    agent: AgentConfig = field(default_factory=lambda: AgentConfig("claude-code"))
    reference: Path | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    concurrency: int | dict[str, int] = field(
        default_factory=lambda: {"S-N": 32, "A-N": 8}, kw_only=True
    )
    install_timeout_seconds: int = 14400
    health_timeout_seconds: int = 3600
    agent: AgentConfig = field(
        default_factory=lambda: AgentConfig("claude-code", budget_seconds=1800)
    )
    # Quality answers unfinished at this deadline count as wrong.
    quality_timeout_seconds: int = 18000
    # SLO timing has no reference to finish first, so it is bounded instead.
    slo_budget_seconds: int = 21600
    slo_request_cap_seconds: int = 600
    slo_repeats: int = 1

    def replay_concurrency(self, workload: str) -> int:
        """Official N for this workload; public experiments may override it."""
        if workload not in ("S-N", "A-N"):
            return 1
        return (
            self.concurrency.get(workload, 32 if workload == "S-N" else 8)
            if isinstance(self.concurrency, dict)
            else self.concurrency
        )


@dataclass(frozen=True)
class EnvironmentConfig:
    """Device category and native provider resources, without host inspection."""

    cpus: int = 8
    memory_mb: int = 65536
    provider: str = "docker"
    import_path: str | None = None
    device_category: str = "nvidia-cuda-x86"
    storage_mb: int = 524288
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    """Runner input. Paths are interpreted beside this configuration file."""

    dataset: str | None = None
    concurrency: int = 1
    max_retries: int = 3
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    tasks: TaskSelection = field(default_factory=TaskSelection)
    agent: AgentConfig | None = None
    agents: dict[str, AgentConfig] = field(default_factory=dict)
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


def _agent(value: dict | None, *, default_budget: str = "2h") -> AgentConfig | None:
    if value is None:
        return None
    value = dict(value)
    budget = duration(value.pop("budget", default_budget))
    return AgentConfig(**value, budget_seconds=budget)


def parse_config(value: Any, *, base: Path | str = ".") -> RunConfig:
    """Map YAML settings to the run dataclasses without acquisition or probes."""
    raw = copy.deepcopy(value)
    selected = TaskSelection(**raw.pop("tasks", {}))
    names = _selection(selected.models)
    selected = TaskSelection(
        tuple(model(name).name for name in names) if names is not None else None,
        _selection(selected.workloads),
    )
    environment_value = raw.pop("environment", {})
    environment = EnvironmentConfig(**environment_value)
    if environment.provider != "docker" and "gpu_ids" in environment.kwargs:
        raise ValueError("local GPU IDs cannot select cloud GPUs; use provider GPU kwargs")
    agent = _agent(raw.pop("agent", None))
    agents = {key: _agent(value) for key, value in raw.pop("agents", {}).items()}
    if agents:
        if agent is not None or any(
            not key or not key.replace("-", "").replace("_", "").isalnum() for key in agents
        ):
            raise ValueError(
                "use agent or agents; agent labels may contain letters, numbers, _ and -"
            )
        agent = next(iter(agents.values()))
    search = raw.pop("search", {})
    if search.get("reference") is not None:
        search["reference"] = (Path(base).resolve() / search["reference"]).resolve()
    search_agent = _agent(search.pop("agent", {"name": "claude-code"}))
    search = SearchConfig(**search, agent=search_agent)
    evaluation = raw.pop("evaluation", {})
    evaluation_agent = _agent(
        evaluation.pop("agent", {"name": "claude-code"}), default_budget="30m"
    )
    evaluation = EvaluationConfig(
        duration(evaluation.pop("install_timeout", "4h")),
        duration(evaluation.pop("health_timeout", "1h")),
        agent=evaluation_agent,
        quality_timeout_seconds=duration(evaluation.pop("quality_timeout", "5h")),
        slo_budget_seconds=duration(evaluation.pop("slo_budget", "6h")),
        slo_request_cap_seconds=duration(evaluation.pop("slo_request_cap", "10m")),
        **evaluation,
    )
    if evaluation.agent.name not in JUDGE_ENV:
        raise ValueError(f"unknown Rewardkit agent judge: {evaluation.agent.name}")
    if type(evaluation.slo_repeats) is not int or evaluation.slo_repeats < 1:
        raise ValueError("evaluation.slo_repeats must be a positive integer")
    limits = evaluation.concurrency
    limits = limits if isinstance(limits, dict) else {"S-N": limits, "A-N": limits}
    if set(limits) - {"S-N", "A-N"} or any(
        type(n) is not int or n < 8 or n % 8 for n in limits.values()
    ):
        raise ValueError(
            "evaluation.concurrency must be a positive multiple of 8 or an S-N/A-N mapping"
        )
    out = Path(raw.pop("out", "runs"))
    if not out.is_absolute():
        out = Path(base).resolve() / out
    concurrency = raw.pop("concurrency", 1)
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    max_retries = raw.pop("max_retries", 3)
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError("max_retries must be a nonnegative integer")
    config = RunConfig(
        **raw,
        concurrency=concurrency,
        max_retries=max_retries,
        environment=environment,
        tasks=selected,
        agent=agent,
        agents=agents,
        search=search,
        evaluation=evaluation,
        out=out,
    )
    configured_agents = {
        "search.agent": search.agent,
        "evaluation.agent": evaluation.agent,
        **({f"agents.{key}": item for key, item in agents.items()} if agents else {"agent": agent}),
    }
    for location, item in configured_agents.items():
        if item is not None:
            effort = item.kwargs.get("reasoning_effort")
            if not isinstance(effort, str) or not effort.strip():
                raise ValueError(
                    f"{location}.kwargs.reasoning_effort must be explicitly set to a non-empty string"
                )
    return config


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
    """Select the intersection of model and workload filters in catalog order.

    Every catalog row is a serving task. Search determines whether a reference
    serves it on the device and classifies it as SOTA or SLO afterwards.
    """
    names = None if selection.models is None else {model(n).name for n in selection.models}
    selected = tuple(
        task
        for task in (tasks() if catalog is None else catalog)
        if (names is None or task.model in names)
        and (selection.workloads is None or task.workload in selection.workloads)
    )
    if not selected:
        raise ValueError("task filters select no tasks")
    return selected
