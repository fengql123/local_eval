"""SLO unit scoring: two common cases and two incomplete/unequal-graph edges."""

import pytest

from eval.bench.slo import ladder, score, table, zero


def stream(row, ttft=1, rate=10, *, request=0, start=0, **changes):
    first = start + ttft
    last = first + 100 / rate
    return {
        "ok": True,
        "errors": [],
        "row_id": row,
        "request_id": request,
        "t_send": start,
        "t_first": first,
        "t_last": last,
        "t_end": last,
        "output_tokens": 100,
        **changes,
    }


def rows(*ids):
    return [
        {
            "id": name,
            "requests": [{"id": request, "messages": "x", "tools": None} for request in requests],
        }
        for name, requests in ids
    ]


def one_tier():
    value = table()
    value["single_stream"]["tiers"] = [{"id": "T1", "name": "test", "ttft_s": 1, "throughput": 10}]
    for workload in ("S-N", "A-N"):
        value[workload]["tiers"] = [{"id": "T1", "name": "test", "throughput": 10}]
    return value


@pytest.mark.parametrize("workload", ["S-1", "A-1", "A-MA"])
def test_balanced_credit_and_four_fixed_tiers(workload):
    result = score(
        workload, [stream("a", ttft=1.25, rate=4.5)], rows=rows(("a", [0])), value=one_tier()
    )
    assert result["score"] == pytest.approx(60)
    assert result["rewards"] == {"score": 60}
    assert result["tiers"][0]["ttft_credit"] == pytest.approx(0.8)
    assert result["tiers"][0]["throughput_credit"] == pytest.approx(0.45)
    assert result["tier_met"] == 0
    fixed = table()
    assert [len(ladder(w, fixed)[1]) for w in ("S-1", "A-1", "A-MA", "S-N", "A-N")] == [4] * 5
    records = [stream(str(i), ttft=1 if i < 9 else 4) for i in range(10)]
    placed = score(
        workload, records, rows=rows(*[(str(i), [0]) for i in range(10)]), value=one_tier()
    )
    assert placed["tiers"][0]["passes"] == 9 and placed["tiers"][0]["met"]
    assert placed["score"] == pytest.approx(95)


@pytest.mark.parametrize("workload", ["S-N", "A-N"])
def test_fixed_batches_include_prefill_and_keep_unstarted_batches_in_denominator(workload):
    # Two overlapping requests: 200 output tokens / 20 active seconds = 10 tok/s.
    records = [stream(name, ttft=10, rate=10, batch=0) for name in ("a", "b")]
    expected = rows(*[(name, [0]) for name in ("a", "b", "c", "d")])
    result = score(workload, records, rows=expected, concurrency=2, value=one_tier())
    assert result["throughput"] == pytest.approx(10)
    assert result["score"] == pytest.approx(50)
    assert result["coverage"]["started"] == result["coverage"]["completed"] == 2
    assert result["coverage"]["complete_fraction"] == 0.5
    assert result["tiers"][0]["units"] == 2 and not result["tiers"][0]["met"]


def test_caps_budget_cuts_and_missing_graph_suffixes_cannot_earn_credit():
    expected = rows(("a", [0, 1]), ("b", [0]), ("c", [0]), ("d", [0]))
    records = [stream("a"), stream("b", cut="cap", ok=False), stream("c", cut="budget", ok=False)]
    result = score("A-1", records, rows=expected, value=one_tier())
    assert result["rewards"] == zero("A-1") == {"score": 0}
    assert result["coverage"] == {
        "rows": 4,
        "repeats": 1,
        "started": 3,
        "fraction": 0.75,
        "completed": 0,
        "complete_fraction": 0.0,
    }
    assert result["requests"] == {"scored": 3, "failed": 2, "capped": 1, "cut_at_budget": 1}
    assert len(result["detail"]) == 4 and not any(unit["complete"] for unit in result["detail"])


@pytest.mark.parametrize("workload", ["A-1", "A-MA"])
def test_long_and_short_graphs_receive_equal_weight(workload):
    expected = rows(("long", list(range(10))), ("short", [0]))
    records = [stream("long", request=i, start=20 * i) for i in range(10)]
    records.append(stream("short", ttft=4, start=220))
    result = score(workload, records, rows=expected, value=one_tier())
    assert result["score"] == pytest.approx(75)
    assert [unit["throughput"] for unit in result["detail"]] == pytest.approx([10, 10])
    assert result["coverage"]["complete_fraction"] == 1
