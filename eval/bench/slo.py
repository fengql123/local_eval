"""Score whole replay units on four fixed SLO tiers, normalized to 0–100.

S-1, A-1 and A-MA give TTFT and generation throughput equal geometric weight.
S-N and A-N score throughput including prefill. Every planned graph or fixed N
batch contributes equally, including incomplete and unstarted units at zero.
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..config import TABLES
from .metrics import numbers
from .replay import session_limit


def table(path: Path | None = None) -> dict:
    """Read the fixed tier table bundled with the package."""
    return json.loads((TABLES / "slo.json" if path is None else path).read_text())


def ladder(workload: str, value: dict) -> tuple[str, list[dict]]:
    """Select balanced TTFT/throughput tiers or throughput-only tiers."""
    if workload in value["single_stream"]["workloads"]:
        return "balanced", value["single_stream"]["tiers"]
    return "throughput", value[workload]["tiers"]


def failed(record: Mapping[str, Any]) -> bool:
    """Only complete, successful requests are eligible for performance credit."""
    return bool(record.get("cut") or record.get("errors")) or not record.get("ok")


def wilson(passes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson interval for whole-unit tier attainment, reported as a diagnostic."""
    if total == 0:
        return 0.0, 1.0
    z = statistics.NormalDist().inv_cdf(0.5 + confidence / 2)
    p = passes / total
    scale = 1 + z * z / total
    center = (p + z * z / (2 * total)) / scale
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale
    return max(0.0, center - half), min(1.0, center + half)


def prompt_bins(rows: Iterable[dict]) -> dict[str, str]:
    """Group supplied prefill payload sizes in doubling byte ranges."""
    bins = {}
    for row in rows:
        size = max(
            len(request["messages"].encode()) + len((request["tools"] or "").encode())
            for request in row["requests"]
        )
        bins[row["id"]] = f"B{max(1, size).bit_length():02d}"
    return bins


def interleave(rows: Iterable[dict]) -> list[dict]:
    """Cycle through prefill-size groups for bounded single-turn measurements."""
    rows = list(rows)
    groups: dict[str, list[dict]] = {}
    bins = prompt_bins(rows)
    for row in rows:
        groups.setdefault(bins[row["id"]], []).append(row)
    layers = itertools.zip_longest(*(groups[key] for key in sorted(groups)))
    return [row for layer in layers for row in layer if row is not None]


def zero(workload: str, value: dict | None = None) -> dict:
    """The final Harbor reward for a rejected candidate."""
    return {"score": 0.0}


def _units(workload, records, rows, repeats, concurrency):
    parallel = workload in ("S-N", "A-N")
    n = session_limit(len(rows), workload, concurrency) if parallel and rows else 1
    grouped = defaultdict(list)
    for record in records:
        key = record.get("batch", 0) if parallel else record["row_id"]
        grouped[record.get("repeat", 0), key].append(record)
    units = []
    for repeat in range(repeats):
        for index, offset in enumerate(range(0, len(rows), n)):
            group = rows[offset : offset + n]
            ids = [row["id"] for row in group]
            items = grouped[repeat, index if parallel else ids[0]]
            required = {(row["id"], request["id"]) for row in group for request in row["requests"]}
            observed = {(r["row_id"], r["request_id"]) for r in items}
            complete = observed == required and len(items) == len(required)
            units.append(_unit(workload, repeat, index, ids, items, complete))
    return units


def _unit(workload, repeat, index, ids, records, complete):
    complete = complete and bool(records) and not any(failed(r) for r in records)
    measured = numbers(workload, records) if complete else None
    complete = complete and measured["valid"]
    return {
        "repeat": repeat,
        "unit": index,
        "row_ids": ids,
        "complete": bool(complete),
        "throughput": measured["throughput"] if complete else None,
        "ttft_mean": measured.get("ttft_mean") if complete else None,
    }


def score(
    workload: str,
    records: Iterable[Mapping[str, Any]],
    *,
    rows: Iterable[dict],
    repeats: int = 1,
    value: dict | None = None,
    bins: Mapping[str, str] | None = None,
    concurrency: int | None = None,
) -> dict:
    """Score complete graphs/batches against every supplied row and request ID.

    A successful graph prefix cannot earn credit without the remaining requests.
    Pass the operation's actual N. Original records stay available for auditing.
    """
    value = table() if value is None else value
    kind, tiers = ladder(workload, value)
    records, rows = list(records), list(rows)
    if workload.startswith("S"):
        rows = interleave(rows)
    units = _units(workload, records, rows, repeats, concurrency)
    summary = []
    for tier in tiers:
        credits, latency, rates, passing = [], [], [], []
        grouped = defaultdict(list)
        for unit in units:
            rate = min(1.0, unit["throughput"] / tier["throughput"]) if unit["complete"] else 0.0
            if kind == "balanced":
                ttft = unit["ttft_mean"]
                start = (
                    (min(1.0, tier["ttft_s"] / ttft) if ttft else 1.0) if unit["complete"] else 0.0
                )
                credit = math.sqrt(start * rate)
                met = start == rate == 1
                latency.append(start)
            else:
                credit, met = rate, rate == 1
            rates.append(rate)
            credits.append(credit)
            passing.append(met)
            if bins and kind == "balanced" and unit["row_ids"]:
                label = bins.get(unit["row_ids"][0])
                if label is not None:
                    grouped[label].append(met)
        passes = sum(passing)
        count = len(units)
        low, high = wilson(passes, count)
        attained = passes / count if count else 0.0
        summary.append(
            {
                **tier,
                "score": statistics.fmean(credits) if credits else 0.0,
                "throughput_credit": statistics.fmean(rates) if rates else 0.0,
                **(
                    {"ttft_credit": statistics.fmean(latency) if latency else 0.0}
                    if kind == "balanced"
                    else {}
                ),
                "passes": passes,
                "units": count,
                "attainment": attained,
                "interval": [low, high],
                "met": count > 0 and passes >= math.ceil(value["attainment"] * count - 1e-9),
                "uncertain": low < value["attainment"] <= high,
                "bins": {
                    label: {
                        "passes": sum(items),
                        "units": len(items),
                        "attainment": sum(items) / len(items),
                    }
                    for label, items in sorted(grouped.items())
                },
            }
        )
    met = list(itertools.takewhile(lambda tier: tier["met"], summary))
    total = round(100 * statistics.fmean(tier["score"] for tier in summary), 4)
    started = len({(r.get("repeat", 0), r["row_id"]) for r in records})
    completed = sum(len(unit["row_ids"]) for unit in units if unit["complete"])
    planned = len(rows) * repeats
    complete_units = [unit for unit in units if unit["complete"]]
    return {
        "workload": workload,
        "kind": kind,
        "attainment_required": value["attainment"],
        "coverage": {
            "rows": len(rows),
            "repeats": repeats,
            "started": started,
            "fraction": started / planned if planned else 0.0,
            "completed": completed,
            "complete_fraction": completed / planned if planned else 0.0,
        },
        "requests": {
            "scored": len(records),
            "failed": sum(failed(record) for record in records),
            "capped": sum(record.get("cut") == "cap" for record in records),
            "cut_at_budget": sum(record.get("cut") == "budget" for record in records),
        },
        "throughput": statistics.fmean(unit["throughput"] for unit in complete_units)
        if complete_units
        else 0.0,
        "detail": units,
        "tiers": summary,
        "tier_met": len(met),
        "tier": met[-1]["id"] if met else None,
        "score": total,
        "rewards": {"score": round(total, 4)},
    }
