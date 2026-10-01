"""Collect coding-agent/harness usage from Harbor evidence.

Serving tokens are benchmark evidence and never enter this ledger. Search and
judge usage can use the same collector with separate roles. No price lookup or
paid call happens here: estimates require an explicit model rate table.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

Role = Literal["implementation", "search", "judge"]
CostKind = Literal["reported", "estimated", "mixed", "unknown"]


def _count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("usage token counts must be nonnegative integers or None")
    return value


def _money(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("USD costs/rates must be finite nonnegative numbers or None")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("USD costs/rates must be finite nonnegative numbers or None")
    return value


def _complete_sum(values: Iterable[int | float | None]) -> int | float | None:
    values = tuple(values)
    return sum(values) if values and all(value is not None for value in values) else None


@dataclass(frozen=True)
class TokenPrices:
    """Explicit USD per million tokens, with cache separate from uncached input."""

    input: float
    cached_input: float
    output: float

    def __post_init__(self) -> None:
        for value in (self.input, self.cached_input, self.output):
            if _money(value) is None:
                raise ValueError("all three token prices are required")


@dataclass(frozen=True)
class Usage:
    """Input includes cached tokens; total_tokens never adds cache twice."""

    input_tokens: int | None = None
    cached_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    cost_kind: CostKind = "unknown"

    def __post_init__(self) -> None:
        for value in (self.input_tokens, self.cached_tokens, self.output_tokens):
            _count(value)
        _money(self.cost_usd)
        if (
            self.input_tokens is not None
            and self.cached_tokens is not None
            and self.cached_tokens > self.input_tokens
        ):
            raise ValueError("cached tokens are a subset of input tokens")
        if self.cost_kind not in ("reported", "estimated", "mixed", "unknown"):
            raise ValueError("unrecognized cost kind")
        if (self.cost_usd is None) != (self.cost_kind == "unknown"):
            raise ValueError("unknown cost must be None; known cost needs its source kind")

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Attempt:
    """One immutable Harbor trial, including its failed/interrupted usage."""

    attempt_id: str
    task_name: str
    trial_name: str
    role: Role
    agent_name: str | None
    agent_version: str | None
    model: str | None
    provider: str | None
    usage: Usage
    usage_source: str
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.attempt_id, str) or not self.attempt_id:
            raise ValueError("a stable Harbor trial/attempt ID is required")
        if self.role not in ("implementation", "search", "judge"):
            raise ValueError("role must be implementation, search, or judge")
        if not isinstance(self.usage, Usage):
            raise ValueError("attempt usage must be a Usage value")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if not all(isinstance(source, str) for source in self.evidence):
            raise ValueError("evidence sources must be strings")


def _estimate(usage: Usage, prices: TokenPrices | None) -> Usage:
    if prices is None or usage.cost_usd is not None:
        return usage
    if usage.input_tokens is None or usage.output_tokens is None:
        return usage
    if usage.cached_tokens is None and prices.input != prices.cached_input:
        return usage
    cache = usage.cached_tokens or 0
    cost = (
        (usage.input_tokens - cache) * prices.input
        + cache * prices.cached_input
        + usage.output_tokens * prices.output
    ) / 1_000_000
    return Usage(usage.input_tokens, usage.cached_tokens, usage.output_tokens, cost, "estimated")


def _context_usage(context: Mapping[str, Any]) -> Usage:
    cost = _money(context.get("cost_usd"))
    return Usage(
        _count(context.get("n_input_tokens")),
        _count(context.get("n_cache_tokens")),
        _count(context.get("n_output_tokens")),
        cost,
        "reported" if cost is not None else "unknown",
    )


def _aggregate(usages: Iterable[Usage]) -> Usage:
    usages = tuple(usages)
    cost = _complete_sum(usage.cost_usd for usage in usages)
    kinds = {usage.cost_kind for usage in usages}
    kind = "unknown" if cost is None else next(iter(kinds)) if len(kinds) == 1 else "mixed"
    return Usage(
        _complete_sum(usage.input_tokens for usage in usages),
        _complete_sum(usage.cached_tokens for usage in usages),
        _complete_sum(usage.output_tokens for usage in usages),
        cost,
        kind,
    )


def from_harbor_result(
    result: Mapping[str, Any],
    *,
    trajectory: Mapping[str, Any] | None = None,
    role: Role = "implementation",
    prices: Mapping[str, TokenPrices] | None = None,
    evidence: Iterable[str] = (),
) -> Attempt:
    """Read Harbor TrialResult/AgentContext and ATIF aggregate metric shapes.

    A top-level agent_result takes precedence over step_results, matching Harbor.
    Missing fields may fall back to ATIF final_metrics; aggregate values and
    per-model breakdowns are never added together. Costs read from Harbor/ATIF
    are labelled reported by that evidence, not asserted to be provider invoices.
    Mixed-model estimates require per-model usage and prices for every model.
    """
    info = result.get("agent_info") or {}
    trajectory_agent = (trajectory or {}).get("agent") or {}
    model = (info.get("model_info") or {}).get("name") or trajectory_agent.get("model_name")
    if result.get("agent_result") is not None:
        contexts = [result["agent_result"]]
        source = "agent_result"
    else:
        # A missing step result means unknown usage, not proof of zero calls.
        # Preserve its position so successful sibling steps cannot turn an
        # incomplete multi-step trial into a complete total.
        contexts = [step.get("agent_result") or {} for step in result.get("step_results") or ()]
        source = "step_results" if contexts else "missing"
    usages = []
    for context in contexts:
        usage = _context_usage(context)
        by_model = context.get("model_usage")
        if by_model:
            parts = [
                _estimate(_context_usage(values), (prices or {}).get(name))
                for name, values in by_model.items()
            ]
            # A reported aggregate wins. The breakdown fills only absent fields.
            fallback = _aggregate(parts)
            for field in ("input_tokens", "cached_tokens", "output_tokens"):
                aggregate_value = getattr(usage, field)
                breakdown_value = getattr(fallback, field)
                if (
                    aggregate_value is not None
                    and breakdown_value is not None
                    and aggregate_value != breakdown_value
                ):
                    raise ValueError(f"model usage breakdown disagrees with aggregate {field}")
            usage = Usage(
                usage.input_tokens if usage.input_tokens is not None else fallback.input_tokens,
                usage.cached_tokens if usage.cached_tokens is not None else fallback.cached_tokens,
                usage.output_tokens if usage.output_tokens is not None else fallback.output_tokens,
                usage.cost_usd if usage.cost_usd is not None else fallback.cost_usd,
                usage.cost_kind if usage.cost_usd is not None else fallback.cost_kind,
            )
        else:
            usage = _estimate(usage, (prices or {}).get(model))
        usages.append(usage)
    usage = _aggregate(usages)
    final = (trajectory or {}).get("final_metrics") or {}
    if final:
        fallback_cost = _money(final.get("total_cost_usd"))
        fallback = Usage(
            _count(final.get("total_prompt_tokens")),
            _count(final.get("total_cached_tokens")),
            _count(final.get("total_completion_tokens")),
            fallback_cost,
            "reported" if fallback_cost is not None else "unknown",
        )
        missing = any(
            getattr(usage, key) is None and getattr(fallback, key) is not None
            for key in ("input_tokens", "cached_tokens", "output_tokens", "cost_usd")
        )
        # A saved reported cost has precedence over a local estimate.
        use_fallback_cost = fallback.cost_usd is not None and usage.cost_kind in (
            "estimated",
            "mixed",
            "unknown",
        )
        usage = Usage(
            usage.input_tokens if usage.input_tokens is not None else fallback.input_tokens,
            usage.cached_tokens if usage.cached_tokens is not None else fallback.cached_tokens,
            usage.output_tokens if usage.output_tokens is not None else fallback.output_tokens,
            fallback.cost_usd if use_fallback_cost else usage.cost_usd,
            fallback.cost_kind if use_fallback_cost else usage.cost_kind,
        )
        if missing or use_fallback_cost:
            source = (
                "trajectory.final_metrics"
                if source == "missing"
                else f"{source}+trajectory.final_metrics"
            )
        # ATIF totals may include subagents. Without a model breakdown, do not
        # apply a single-model price to a trajectory aggregate.
    return Attempt(
        str(result.get("id") or ""),
        result.get("task_name", ""),
        result.get("trial_name", ""),
        role,
        info.get("name") or trajectory_agent.get("name"),
        info.get("version") or trajectory_agent.get("version"),
        model,
        (info.get("model_info") or {}).get("provider"),
        usage,
        source,
        tuple(evidence),
    )


def merge_attempts(existing: Iterable[Attempt], incoming: Iterable[Attempt]) -> tuple[Attempt, ...]:
    """Deduplicate identical evidence; retain retries with distinct trial IDs.

    Changed evidence for an already finalized ID raises: callers must collect
    each immutable attempt once it finishes or immediately before its deletion.
    """
    merged: dict[str, Attempt] = {}
    for attempt in (*existing, *incoming):
        previous = merged.get(attempt.attempt_id)
        if previous is not None and previous != attempt:
            raise ValueError(f"conflicting immutable evidence for attempt {attempt.attempt_id}")
        merged[attempt.attempt_id] = attempt
    return tuple(merged.values())


def read_ledger(path: str | Path) -> tuple[Attempt, ...]:
    """Load saved attempts. A missing ledger starts empty, not with zero costs."""
    path = Path(path)
    if not path.exists():
        return ()
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError("unsupported cost ledger schema version")
    attempts = []
    for item in data["attempts"]:
        attempts.append(
            Attempt(
                **{**item, "usage": Usage(**item["usage"]), "evidence": tuple(item["evidence"])}
            )
        )
    return merge_attempts((), attempts)


def write_ledger(path: str | Path, attempts: Iterable[Attempt]) -> None:
    """Atomically persist a complete ledger outside disposable trial folders.

    This is a single-writer API. The foreground runner owns serialization; it
    must not place its persistent ledger inside a trial directory it deletes.
    """
    path = Path(path)
    data = {"schema_version": 1, "attempts": [asdict(a) for a in merge_attempts((), attempts)]}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def update_ledger(path: str | Path, attempts: Iterable[Attempt]) -> tuple[Attempt, ...]:
    """Merge new finalized attempts and save them before trial cleanup."""
    merged = merge_attempts(read_ledger(path), attempts)
    write_ledger(path, merged)
    return merged


def collect_trial(
    trial_directory: str | Path,
    *,
    trajectory_path: str | Path | None = None,
    role: Role = "implementation",
    prices: Mapping[str, TokenPrices] | None = None,
) -> Attempt:
    """Read trial/result.json, with an optional explicit ATIF trajectory path."""
    result_path = Path(trial_directory) / "result.json"
    result = json.loads(result_path.read_text())
    trajectory = None
    evidence = [str(result_path)]
    if trajectory_path is not None:
        trajectory_path = Path(trajectory_path)
        trajectory = json.loads(trajectory_path.read_text())
        evidence.append(str(trajectory_path))
    return from_harbor_result(
        result, trajectory=trajectory, role=role, prices=prices, evidence=evidence
    )


def summarize(attempts: Iterable[Attempt], *, role: Role = "implementation") -> dict[str, Any]:
    """Return complete totals and known subtotals for one cost role only."""
    if role not in ("implementation", "search", "judge"):
        raise ValueError("role must be implementation, search, or judge")
    selected = tuple(a for a in merge_attempts((), attempts) if a.role == role)
    usage = _aggregate(a.usage for a in selected)
    known = [a.usage.cost_usd for a in selected if a.usage.cost_usd is not None]
    return {
        "role": role,
        "attempts": len(selected),
        **asdict(usage),
        "total_tokens": usage.total_tokens,
        "known_cost_usd": sum(known),
        "unknown_cost_attempts": sum(a.usage.cost_usd is None for a in selected),
        "cost_kinds": {
            kind: sum(a.usage.cost_kind == kind for a in selected)
            for kind in ("reported", "estimated", "mixed", "unknown")
        },
    }
