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
    evidence = result()
    evidence["agent_result"]["model_usage"] = {
        "fixture-coding-model": {
            "n_input_tokens": 60,
            "n_cache_tokens": 20,
            "n_output_tokens": 6,
        },
        "other-model": {"n_input_tokens": 40, "n_cache_tokens": 20, "n_output_tokens": 4},
    }
    attempt = from_harbor_result(
        evidence,
        prices={
            "fixture-coding-model": TokenPrices(2, 0.2, 8),
            "other-model": TokenPrices(4, 0.4, 16),
        },
    )
    assert attempt.usage.cost_kind == "estimated"
    assert attempt.usage.cost_usd == pytest.approx(0.000284)
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


def test_finalized_attempt_id_cannot_be_reused_with_different_evidence():
    evidence = result()
    attempt = from_harbor_result(evidence)
    evidence["agent_result"]["n_output_tokens"] = 11
    with pytest.raises(ValueError, match="conflicting immutable evidence"):
        merge_attempts([attempt], [from_harbor_result(evidence)])
