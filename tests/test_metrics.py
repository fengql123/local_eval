"""Two common timing cases and two invalid-evidence cases; no replay runner."""

import pytest

from eval.bench.metrics import numbers


def timed(send=0, first=1, last=4, end=6, tokens=10):
    return dict(
        ok=True,
        t_send=send,
        t_first=first,
        t_last=last,
        t_end=end,
        output_tokens=tokens,
        cached_tokens=None,
    )


@pytest.mark.parametrize("workload", ["S-N", "A-N"])
def test_overlapping_requests_count_elapsed_time_once(workload):
    report = numbers(
        workload,
        [{**timed(), "stream_gap_max": 2}, {**timed(2, 3, 5, 7, 20), "stream_gap_max": 1}],
    )
    assert report["valid"]
    assert (report["decode_s"], report["busy_s"]) == (4, 7)
    assert report["throughput"] == pytest.approx(30 / 7)
    assert report["decode_throughput"] == 7.5
    assert report["busy_throughput"] == pytest.approx(30 / 7)
    assert "ttft_mean" not in report and "ttft_p95" not in report
    assert not any(key.startswith("completion_") for key in report)
    assert report["tpot_mean"] == pytest.approx((3 / 9 + 2 / 19) / 2)
    assert report["stream_gap_max"] == 2
    assert report["cached_tokens"] is None


@pytest.mark.parametrize("workload", ["S-1", "A-1", "A-MA"])
def test_serial_requests_exclude_idle_gaps(workload):
    report = numbers(workload, [timed(end=5), timed(6, 7, 9, 10, 20)])
    assert report["valid"]
    assert (report["decode_s"], report["busy_s"]) == (5, 9)
    assert report["throughput"] == 6
    assert report["busy_throughput"] == pytest.approx(30 / 9)
    assert report["ttft_mean"] == 1
    assert report["ttft_p95"] == 1
    assert not any(key.startswith("completion_") for key in report)
    # Historical records have no intermediate chunk timestamps.
    assert report["stream_gap_max"] is None


def test_failed_request_invalidates_the_whole_run():
    report = numbers("S-N", [timed(), {**timed(), "ok": False}])
    assert not report["valid"]
    assert (report["requests"], report["ok"]) == (2, 1)
    assert report["throughput"] is None
    assert report["busy_throughput"] is None
    assert "ttft_mean" not in report
    assert not any(key.startswith("completion_") for key in report)
    assert report["tpot_mean"] is None
    assert report["stream_gap_max"] is None


def test_zero_decode_window_does_not_produce_a_score():
    report = numbers("S-1", [timed(first=1, last=1, tokens=1)])
    assert not report["valid"]
    assert report["decode_s"] == 0
    assert report["throughput"] is None
    assert "zero" in report["errors"][0]
    assert report["ttft_mean"] is None and report["ttft_p95"] is None

    parallel = numbers("S-N", [timed(first=1, last=1, tokens=1)])
    assert parallel["valid"] and parallel["throughput"] == pytest.approx(1 / 6)
    assert parallel["decode_throughput"] is None
