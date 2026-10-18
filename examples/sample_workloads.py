"""Tool-aware sampling helpers and an audit of the approved dataset memberships.

Sampling is maintainer work. Evaluation only filters the resulting memberships.
Run ``python -m examples.sample_workloads data/dataset --check`` to compare the
dataset with its sampling record without drawing a new sample.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import quantiles

import pyarrow.parquet as pq


def tool_tiers(rows, field, *, parts=3):
    """Keep zero separate; use inclusive quantiles of positive waits per model."""
    positive = sorted(row[field] for row in rows if row[field] > 0)
    cuts = quantiles(positive, n=parts, method="inclusive") if len(positive) > 1 else []
    tiers = {
        row["id"]: 0 if row[field] == 0 else 1 + bisect.bisect_left(cuts, row[field])
        for row in rows
    }
    return tiers, {"field": field, "positive_quantile_cuts_seconds": cuts}


def coarse_bin(row):
    """Public compute bins are coarser than the private token/round bins."""
    result = (
        int(row["n"] > 32),
        2 if row["peakprompt"] is None else int(row["peakprompt"] > 32768),
        int(row["out"] > 32768),
    )
    return result + (int(row["width"] > 1),) if row["pool"] == "A-MA" else result


def stratify(rows, bin_of, target, seed_text, *, field, protected=None, parts=3):
    """Cover compute bins and wait tiers, then fill toward population proportions.

    Tool tiers are balancing constraints, not another Cartesian dimension.
    Draws within a chosen bin/tier are uniform and seeded. Forced coverage and
    adaptive balancing mean exact marginal inclusion probabilities are not known.
    """
    rows = sorted(rows, key=lambda row: row["index"])
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big"))
    tiers, timing = tool_tiers(rows, field, parts=parts)
    population = Counter(tiers.values())
    reasons = {name: list(value) for name, value in (protected or {}).items()}
    chosen = [row for row in rows if row["id"] in reasons]
    selected = {row["id"] for row in chosen}
    chosen_tiers = Counter(tiers[row["id"]] for row in chosen)
    groups = defaultdict(list)
    for row in rows:
        if row["id"] not in selected:
            groups[bin_of(row)].append(row)
    draws = Counter()

    def pick(candidates, reason):
        available = {tiers[row["id"]] for row in candidates}
        tier = max(
            available,
            key=lambda t: (
                (len(chosen) + 1) * population[t] / len(rows) - chosen_tiers[t],
                t,
            ),
        )
        members = sorted(
            (row for row in candidates if tiers[row["id"]] == tier),
            key=lambda row: row["id"],
        )
        row = rng.choice(members)
        chosen.append(row)
        selected.add(row["id"])
        chosen_tiers[tier] += 1
        draws[bin_of(row)] += 1
        reasons[row["id"]] = [reason]

    for bucket in sorted(groups):
        pick(groups[bucket], "compute_bin_coverage")
    for tier in sorted(population):
        if chosen_tiers[tier] == 0:
            pick([row for row in rows if tiers[row["id"]] == tier], "tool_tier_coverage")
    if not len(chosen) <= target <= len(rows):
        raise ValueError(f"{seed_text}: {len(chosen)} coverage rows do not fit budget {target}")
    while len(chosen) < target:
        available = {
            bucket: [row for row in members if row["id"] not in selected]
            for bucket, members in groups.items()
        }
        bucket = max(
            (b for b, members in available.items() if members),
            key=lambda b: (len(groups[b]) / (draws[b] + 1), b),
        )
        pick(available[bucket], "balanced_fill")
    chosen.sort(key=lambda row: row["index"])
    bins = []
    for bucket in sorted({bin_of(row) for row in rows}):
        members = [row for row in rows if bin_of(row) == bucket]
        bins.append(
            {
                "bin": list(bucket),
                "population": len(members),
                "selected": sum(row["id"] in selected for row in members),
            }
        )
    return chosen, {
        "seed": seed_text,
        "eligible_rows": len(rows),
        "compute_bins": bins,
        "tool_time": {
            **timing,
            "population": dict(sorted(population.items())),
            "selected": dict(sorted(chosen_tiers.items())),
        },
        "selected": [
            {
                "id": row["id"],
                "compute_bin": list(bin_of(row)),
                "tool_tier": tiers[row["id"]],
                "reasons": reasons[row["id"]],
                "requests": row["n"],
                "peak_input_tokens": row["peakprompt"],
                "output_tokens": row["out"],
                "width": row["width"],
                "tool_total_seconds": row["effective_tool"],
                "tool_critical_seconds": row["effective_critical"],
                "longest_tool_seconds": row["maxwait"],
            }
            for row in chosen
        ],
        "inclusion_probabilities": "Not claimed for this constrained, adaptively balanced design.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    audit = json.loads((args.dataset / "sampling.json").read_text())
    expected = {(r["model"], r["id"], r["split"]): r["workloads"] for r in audit["memberships"]}
    actual = {}
    counts = Counter()
    for path in sorted((args.dataset / "workloads").glob("*.parquet")):
        for row in pq.read_table(path, columns=["id", "model", "public", "private"]).to_pylist():
            counts["records"] += 1
            assert not (row["public"] and row["private"])
            for split in ("public", "private"):
                if row[split]:
                    actual[row["model"], row["id"], split] = row[split]
                    counts[split] += 1
    assert actual == expected, "dataset memberships differ from the approved selection"
    print(json.dumps(dict(counts), indent=2))


if __name__ == "__main__":
    main()
