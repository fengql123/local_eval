"""Timing arithmetic over timed replay evidence, without I/O or answer grading.

All timestamps are seconds from one monotonic clock. Callers pass only the timed
requests: startup and warmup evidence belongs outside this API. Token counts come
from the server, never from the text shown to a user.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
from math import ceil, exp, fsum, log
from typing import Any


def union_seconds(intervals: Iterable[tuple[float, float]]) -> float:
    """Return the duration covered by intervals; overlaps count once."""
    total, previous_end = 0.0, None
    for start, end in sorted(intervals):
        if previous_end is None or start > previous_end:
            total += end - start
        elif end > previous_end:
            total += end - previous_end
        previous_end = end if previous_end is None else max(previous_end, end)
    return total


def _p95(values: list[float]) -> float:
    """Nearest-rank percentile, including the slowest request in small samples."""
    return sorted(values)[ceil(0.95 * len(values)) - 1]


def reward_metrics(workload: str) -> tuple[str, ...]:
    """Parallel workloads score throughput; serial workloads also score TTFT."""
    return ("throughput",) if workload in ("S-N", "A-N") else ("throughput", "ttft")


def gains(workload: str, reference: Mapping, candidate: Mapping) -> dict[str, float]:
    """Throughput and inverse mean-TTFT ratios; values above one improve on reference."""
    values = {"throughput": candidate["throughput"] / reference["throughput"]}
    if "ttft" in reward_metrics(workload):
        before, after = reference["ttft_mean"], candidate["ttft_mean"]
        values["ttft"] = before / after if after else float("inf") if before else 1.0
    return values


def combined_gain(values: Mapping[str, float]) -> float:
    """Equal multiplicative weight for the workload's scored speedups."""
    if any(value <= 0 for value in values.values()):
        return 0.0
    return exp(fsum(log(value) for value in values.values()) / len(values))


def score_gains(values: Mapping[str, float]) -> float:
    """One 0–100 SOTA score: zero at parity/regression, 50 at 2x, 75 at 4x."""
    gain = combined_gain(values)
    return round(100 * (1 - 1 / gain), 4) if gain > 1 else 0.0


def numbers(workload: str, requests: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize timed requests; failed requests and zero durations have no score.

    Each request has ``ok``, ``t_send``, ``t_first``, ``t_last``, ``t_end``,
    ``output_tokens`` and optional ``cached_tokens``. Timing and count fields are
    consumed directly from replay evidence. Diagnostic unions/counts describe
    successful requests; a failed request invalidates the whole run's score.
    """
    rows = list(requests)
    good = [row for row in rows if row["ok"]]
    errors = [
        f"request[{index}]: request did not finish successfully"
        for index, row in enumerate(rows)
        if not row["ok"]
    ]

    batches = {row.get("batch", 0) for row in good}
    decode = sum(
        union_seconds(
            (row["t_first"], row["t_last"]) for row in good if row.get("batch", 0) == batch
        )
        for batch in batches
    )
    busy = sum(
        union_seconds((row["t_send"], row["t_end"]) for row in good if row.get("batch", 0) == batch)
        for batch in batches
    )
    tokens = sum(row["output_tokens"] for row in good)
    if not rows:
        errors.append("no timed requests")
    elif good:
        if decode == 0 and "ttft" in reward_metrics(workload):
            errors.append("decode duration is zero; generation throughput is undefined")
        if busy == 0:
            errors.append("busy duration is zero; busy throughput is undefined")
    valid = not errors
    cached_counts = [row.get("cached_tokens") for row in good]
    tpot = [
        (row["t_last"] - row["t_first"]) / (row["output_tokens"] - 1)
        for row in good
        if row["output_tokens"] > 1
    ]
    gaps = [row.get("stream_gap_max") for row in good]
    report = {
        "workload": workload,
        "requests": len(rows),
        "ok": len(good),
        "valid": valid,
        "errors": errors,
        "output_tokens": tokens,
        "cached_tokens": (
            sum(c for c in cached_counts if c is not None)
            if cached_counts and all(c is not None for c in cached_counts)
            else None
        ),
        "decode_s": decode,
        "busy_s": busy,
        "throughput": tokens / (busy if workload in ("S-N", "A-N") else decode) if valid else None,
        "decode_throughput": tokens / decode if valid and decode else None,
        "busy_throughput": tokens / busy if valid else None,
        "tpot_mean": statistics.fmean(tpot) if valid and tpot else None,
        "stream_gap_max": max(gaps) if valid and all(gap is not None for gap in gaps) else None,
    }
    if "ttft" in reward_metrics(workload):
        ttft = [row["t_first"] - row["t_send"] for row in good]
        report.update(
            ttft_mean=statistics.fmean(ttft) if valid else None,
            ttft_p95=_p95(ttft) if valid else None,
        )
    return report


def summarize(workload: str, requests: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The shared measurement entry point for every workload."""
    return numbers(workload, requests)
