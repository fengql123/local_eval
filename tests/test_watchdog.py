"""Two performance decisions and two timing/accounting boundaries."""

import pytest

from eval.bench.watchdog import PerformanceCutoff, Watchdog


def fixture(workload, **kwargs):
    now = [0.0]
    watchdog = Watchdog(workload, clock=lambda: now[0], **kwargs)
    record = {"t_send": 0.0, "t_first": None, "token_ids": []}
    watchdog.start(record)
    return watchdog, record, now


def test_sustained_regression_stops_and_reports_the_reference_threshold():
    monitor, record, now = fixture("S-N", reference={"throughput": 10})
    record["token_ids"] = list(range(600))
    record["t_first"] = 1.0
    now[0] = 119
    monitor.update(record)
    assert monitor.check() is None
    now[0] = 120
    with pytest.raises(PerformanceCutoff) as failure:
        monitor.check()
    assert failure.value.detail["observed"] == 0.5
    assert failure.value.detail["window_seconds"] == 120


def test_better_ttft_can_outweigh_lower_decode_throughput():
    monitor, record, now = fixture("A-1", reference={"throughput": 10, "ttft_mean": 4})
    now[0] = 1
    record["t_first"] = 1
    monitor.update(record)
    now[0] = 121
    record["token_ids"] = list(range(960))
    monitor.update(record)
    detail = monitor.check()
    assert detail["throughput"] == 8
    assert detail["observed"] > 1


def test_tool_only_idle_and_failed_transports_do_not_trigger_performance_rejection():
    monitor, record, now = fixture("A-N", floor=10)
    now[0] = 1
    record["token_ids"] = [1]
    monitor.finish(record)
    now[0] = 1000
    assert monitor.check() is None and monitor.elapsed == 1
    monitor.start(record)
    now[0] = 1300
    monitor.failed = True
    assert monitor.check() is None


def test_parallel_accounting_sums_accepted_ids_and_ignores_chunk_or_usage_padding():
    monitor, a, now = fixture("S-N", floor=10)
    b = {"t_send": 0.0, "t_first": 1, "token_ids": []}
    monitor.start(b)
    now[0] = 120
    a.update(token_ids=list(range(600)), t_first=1, output_tokens=100000)
    b["token_ids"] = list(range(600))
    monitor.update(a)
    monitor.update(b)
    detail = monitor.check()
    assert detail["throughput"] == 10
    assert monitor.elapsed == 120


@pytest.mark.parametrize("threshold", [{"reference": {"throughput": 10}}, {"floor": 10}])
def test_fast_batch_is_not_cut_off_when_its_last_requests_drain_slowly(threshold):
    monitor, record, now = fixture("S-N", **threshold)
    now[0] = 120
    record.update(token_ids=list(range(3600)), t_first=1)
    monitor.update(record)
    assert monitor.check()["throughput"] == 30
    now[0] = 240
    record["token_ids"] = list(range(4200))
    monitor.update(record)
    detail = monitor.check()
    assert detail["throughput"] == 5
    assert detail["overall_throughput"] == 17.5
    now[0] = 720
    record["token_ids"] = list(range(6600))
    monitor.update(record)
    with pytest.raises(PerformanceCutoff):
        monitor.check()
