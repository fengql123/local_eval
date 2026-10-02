"""Timing arithmetic over timed replay evidence, without I/O or answer grading.

All timestamps are seconds from one monotonic clock. Callers pass only the timed
requests: startup and warmup evidence belongs outside this API. Token counts come
from the server, never from the text shown to a user.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
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


def numbers(workload: str, requests: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize timed requests; failed requests and zero durations have no score.

    Each request has ``ok``, ``t_send``, ``t_first``, ``t_last``, ``t_end``,
    ``output_tokens`` and optional ``cached_tokens``. Timing and count fields are
    consumed directly from replay evidence. Diagnostic unions/counts describe
    successful requests; a failed request invalidates the whole run's score.
    """
    rows = list(requests)
    good = [
        (
            row["t_send"],
            row["t_first"],
            row["t_last"],
            row["t_end"],
            row["output_tokens"],
            row.get("cached_tokens"),
        )
        for row in rows
        if row["ok"]
    ]
    errors = [
        f"request[{index}]: request did not finish successfully"
        for index, row in enumerate(rows)
        if not row["ok"]
    ]

    decode = union_seconds((first, last) for _, first, last, _, _, _ in good)
    busy = union_seconds((send, end) for send, _, _, end, _, _ in good)
    tokens = sum(row[4] for row in good)
    if not rows:
        errors.append("no timed requests")
    elif good:
        if decode == 0:
            errors.append("decode duration is zero; generation throughput is undefined")
        if busy == 0:
            errors.append("busy duration is zero; busy throughput is undefined")
    valid = not errors
    cached_counts = [row[5] for row in good]
    return {
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
        "throughput": tokens / decode if valid else None,
        "busy_throughput": tokens / busy if valid else None,
        "ttft_mean": statistics.fmean(first - send for send, first, _, _, _, _ in good)
        if valid
        else None,
    }


def summarize(workload: str, requests: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The shared measurement entry point for every workload."""
    return numbers(workload, requests)
