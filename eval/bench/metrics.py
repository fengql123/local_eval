"""Timing arithmetic over timed replay evidence, without I/O or answer grading.

All timestamps are seconds from one monotonic clock. Callers pass only the timed
requests: startup and warmup evidence belongs outside this API. Token counts come
from the server, never from the text shown to a user.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any


def _seconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("timestamps must be finite numbers in seconds")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("timestamps must be finite numbers in seconds")
    return result


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("token counts must be nonnegative integers")
    return value


def union_seconds(intervals: Iterable[tuple[float, float]]) -> float:
    """Return the duration covered by intervals; overlaps count once.

    Zero-width intervals are valid. Reversed and nonfinite intervals raise
    ``ValueError`` rather than producing a plausible measurement.
    """
    ordered = []
    for start, end in intervals:
        start, end = _seconds(start), _seconds(end)
        if end < start:
            raise ValueError("interval end precedes its start")
        ordered.append((start, end))
    total, previous_end = 0.0, None
    for start, end in sorted(ordered):
        if previous_end is None or start > previous_end:
            total += end - start
        elif end > previous_end:
            total += end - previous_end
        previous_end = end if previous_end is None else max(previous_end, end)
    return total


def numbers(workload: str, requests: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize timed requests and explicitly invalidate incomplete evidence.

    Each request has ``ok``, ``t_send``, ``t_first``, ``t_last``, ``t_end``,
    ``output_tokens`` and optional ``cached_tokens``. A caller may label evidence
    with ``phase='timed'``; any other phase is rejected. Failed/nonstreamed
    requests, malformed counts/times, or zero overall durations make ``valid``
    false and all three score fields ``None``. Diagnostic unions/counts describe
    only the successfully validated requests; they never represent a partial
    run as a successful score.
    """
    rows = list(requests)
    good: list[tuple[float, float, float, float, int, int | None]] = []
    errors: list[str] = []
    for index, row in enumerate(rows):
        try:
            if row.get("phase", "timed") != "timed":
                raise ValueError("only timed requests belong in metric inputs")
            if row.get("ok") is not True:
                raise ValueError("request did not finish successfully")
            send, first, last, end = (
                _seconds(row.get(key)) for key in ("t_send", "t_first", "t_last", "t_end")
            )
            if not send <= first <= last <= end:
                raise ValueError("timestamps must satisfy send <= first <= last <= end")
            tokens = _count(row.get("output_tokens"))
            if tokens == 0:
                raise ValueError("streamed request must report at least one output token")
            if first == last and tokens > 1:
                raise ValueError("multi-token response has zero decode duration")
            cached = row.get("cached_tokens")
            if cached is not None:
                cached = _count(cached)
            good.append((send, first, last, end, tokens, cached))
        except (AttributeError, ValueError) as exc:
            errors.append(f"request[{index}]: {exc}")

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
