"""Two common cost cases and two edge cases using the Phase 1 collector."""

import json
from pathlib import Path

import pytest

from eval.costs import (
    TokenPrices,
    collect_trial,
    from_harbor_result,
    merge_attempts,
    read_ledger,
    summarize,
    update_ledger,
)

FIXTURE = Path(__file__).parent / "fixtures" / "harbor_trial"


def result():
    return json.loads((FIXTURE / "result.json").read_text())


def test_saved_harbor_usage_counts_cache_once_and_preserves_unknown_cost():
    attempt = collect_trial(FIXTURE)
    assert attempt.usage.input_tokens == 100
    assert attempt.usage.cached_tokens == 40
    assert attempt.usage.total_tokens == 110
    assert attempt.usage.cost_usd is None
    assert attempt.usage.cost_kind == "unknown"


def test_explicit_price_estimate_survives_ledger_round_trip(tmp_path):
    attempt = from_harbor_result(result(), prices={"fixture-coding-model": TokenPrices(2, 0.2, 8)})
    assert attempt.usage.cost_kind == "estimated"
    assert attempt.usage.cost_usd == pytest.approx(0.000208)
    ledger = tmp_path / "costs.json"
    update_ledger(ledger, [attempt])
    assert read_ledger(ledger) == (attempt,)
    assert summarize(read_ledger(ledger))["total_tokens"] == 110


def test_duplicate_collection_is_deduplicated_but_retry_is_retained():
    first = result()
    first["agent_result"]["cost_usd"] = 0.5
    retry = result()
    retry["id"] = "retry-2"
    retry["trial_name"] = "retry-2"
    retry["agent_result"]["cost_usd"] = 0.75
    attempt = from_harbor_result(first)
    attempts = merge_attempts([attempt], [attempt, from_harbor_result(retry)])
    report = summarize(attempts)
    assert report["attempts"] == 2
    assert report["total_tokens"] == 220
    assert report["cost_usd"] == 1.25


def test_cached_tokens_cannot_exceed_total_input_tokens():
    evidence = result()
    evidence["agent_result"]["n_cache_tokens"] = 101
    with pytest.raises(ValueError, match="subset of input"):
        from_harbor_result(evidence)
