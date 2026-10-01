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


def test_overlapping_requests_count_elapsed_time_once():
    report = numbers("S-N", [timed(), timed(2, 3, 5, 7, 20)])
    assert report["valid"]
    assert (report["decode_s"], report["busy_s"]) == (4, 7)
    assert report["throughput"] == 7.5
    assert report["busy_throughput"] == pytest.approx(30 / 7)
    assert report["ttft_mean"] == 1
    assert report["cached_tokens"] is None


def test_serial_requests_exclude_idle_gaps():
    report = numbers("S-1", [timed(end=5), timed(6, 7, 9, 10, 20)])
    assert report["valid"]
    assert (report["decode_s"], report["busy_s"]) == (5, 9)
    assert report["throughput"] == 6
    assert report["busy_throughput"] == pytest.approx(30 / 9)
    assert report["ttft_mean"] == 1


def test_failed_request_invalidates_the_whole_run():
    report = numbers("S-N", [timed(), {**timed(), "ok": False}])
    assert not report["valid"]
    assert (report["requests"], report["ok"]) == (2, 1)
    assert report["throughput"] is None
    assert report["busy_throughput"] is None
    assert report["ttft_mean"] is None


def test_zero_decode_window_does_not_produce_a_score():
    report = numbers("S-1", [timed(first=1, last=1, tokens=1)])
    assert not report["valid"]
    assert report["decode_s"] == 0
    assert report["throughput"] is None
    assert "zero" in report["errors"][0]
