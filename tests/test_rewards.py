"""Two common score cases and two boundaries for the single Harbor reward."""

import pytest

from eval.verify.run import gains, rewards, zero_rewards


@pytest.mark.parametrize("workload", ["S-N", "A-N"])
def test_parallel_workloads_only_score_throughput(workload):
    reference, candidate = {"throughput": 10}, {"throughput": 20}
    assert gains(workload, reference, candidate) == {"throughput": 2}
    assert rewards(workload, reference, candidate) == {"score": 50}
    assert zero_rewards(workload) == {"score": 0}


@pytest.mark.parametrize("workload", ["S-1", "A-1", "A-MA"])
def test_serial_workloads_give_both_metrics_equal_geometric_weight(workload):
    reference = {"throughput": 10, "ttft_mean": 2}
    assert rewards(workload, reference, {"throughput": 20, "ttft_mean": 1}) == {"score": 50}
    # A 4x gain in either component gives the same 2x combined gain.
    assert rewards(workload, reference, {"throughput": 40, "ttft_mean": 2}) == {"score": 50}
    assert rewards(workload, reference, {"throughput": 10, "ttft_mean": 0.5}) == {"score": 50}
    assert zero_rewards(workload) == {"score": 0}


def test_parity_regression_and_offsetting_speedups_earn_no_improvement_points():
    reference = {"throughput": 10, "ttft_mean": 2}
    for candidate in (
        reference,
        {"throughput": 5, "ttft_mean": 4},
        {"throughput": 20, "ttft_mean": 4},
    ):
        assert rewards("A-1", reference, candidate) == {"score": 0}


def test_instant_first_output_and_zero_output_keep_scores_bounded():
    assert rewards(
        "S-1", {"throughput": 10, "ttft_mean": 2}, {"throughput": 20, "ttft_mean": 0}
    ) == {"score": 100}
    assert rewards(
        "S-1", {"throughput": 10, "ttft_mean": 0}, {"throughput": 0, "ttft_mean": 0}
    ) == {"score": 0}
    reference = {"throughput": 10, "ttft_mean": 0}
    assert rewards("S-1", reference, reference) == {"score": 0}
