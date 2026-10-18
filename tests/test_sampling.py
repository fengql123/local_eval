"""Tool-aware selection: two normal populations and two allocation boundaries."""

import copy

import pytest

from examples.sample_workloads import stratify, tool_tiers


def rows(count=32, *, branched=False):
    return [
        {
            "id": f"session-{i:02d}",
            "index": i,
            "bucket": i % 3,
            "n": i + 1,
            "peakprompt": 100 * (i + 1),
            "out": 10 * (i + 1),
            "width": 1 if branched and i % 4 == 0 else 2 if branched else 1,
            "effective_tool": float(i + 1),
            "effective_critical": 0.0 if branched and i % 4 == 0 else float(i + 1),
            "maxwait": float(i + 1),
        }
        for i in range(count)
    ]


def test_an_covers_bins_and_waits_with_a_reproducible_fixed_batch_budget():
    population = rows()
    snapshot = copy.deepcopy(population)
    options = dict(field="effective_tool", protected={"session-31": ["longest_wait"]})
    chosen, report = stratify(population, lambda row: (row["bucket"],), 16, "an", **options)
    assert len(chosen) == 16 and len(chosen) % 8 == 0
    assert "session-31" in {row["id"] for row in chosen}
    assert all(bucket["selected"] > 0 for bucket in report["compute_bins"])
    assert set(report["tool_time"]["selected"]) == {1, 2, 3}
    assert (chosen, report) == stratify(
        population, lambda row: (row["bucket"],), 16, "an", **options
    )
    assert population == snapshot


def test_ma_covers_zero_and_positive_critical_waits_alongside_compute_bins():
    population = rows(branched=True)
    chosen, report = stratify(
        population,
        lambda row: (row["bucket"], row["width"]),
        17,
        "ma",
        field="effective_critical",
    )
    assert len(chosen) == 17  # A-MA is serial between graphs; no eight-row rounding.
    assert set(report["tool_time"]["selected"]) == {0, 1, 2, 3}
    assert all(bucket["selected"] > 0 for bucket in report["compute_bins"])
    assert any(row["width"] == 1 for row in chosen)


def test_zero_or_tied_waits_leave_only_actual_tiers():
    population = rows(4)
    for value, expected in ((0.0, 0), (5.0, 1)):
        for row in population:
            row["effective_tool"] = value
        tiers, _ = tool_tiers(population, "effective_tool")
        assert set(tiers.values()) == {expected}


def test_insufficient_budget_fails_instead_of_silently_dropping_a_bin():
    with pytest.raises(ValueError, match="do not fit budget"):
        stratify(rows(8), lambda row: (row["index"],), 4, "small", field="effective_tool")
