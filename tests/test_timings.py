"""Phase work excludes pauses; interrupted provider operations stay incomplete."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from eval import timings


def interval(start, end):
    origin = datetime(2026, 10, 17, tzinfo=UTC)
    return {
        "started_at": (origin + timedelta(seconds=start)).isoformat(),
        "finished_at": (origin + timedelta(seconds=end)).isoformat() if end is not None else None,
    }


def native_trial(tmp_path, name, **result):
    path = tmp_path / name
    path.mkdir()
    value = {"id": name, "config": {"task": {"path": "/tasks/model_A-1"}}, **result}
    (path / "result.json").write_text(json.dumps(value))
    return path


def test_phase_work_separates_setup_agent_and_verifier_steps_across_retries(tmp_path):
    first = native_trial(
        tmp_path,
        "first",
        **interval(0, 100),
        environment_setup=interval(0, 30),
        agent_execution=interval(30, 80),
        step_results=[{"step_name": "prepare", "verifier": interval(80, 100)}],
    )
    second = native_trial(
        tmp_path,
        "resumed",
        **interval(43200, 43270),
        environment_setup=interval(43200, 43210),
        agent_execution=interval(43210, 43240),
        step_results=[{"step_name": "review", "verifier": interval(43240, 43270)}],
    )
    result = timings.collect([first, second])["model_A-1"]
    assert result["attempts"] == 2 and result["incomplete_attempts"] == 0
    seconds = result["recorded_seconds"]
    assert seconds["elapsed"] == 170  # The twelve-hour gap is not work.
    assert seconds["environment_setup"] == 40 and seconds["agent_execution"] == 80
    assert seconds["steps"] == {"prepare": 20, "review": 30}
    assert seconds["agent_setup"] is None  # Missing historical data is not zero.


def test_provider_journal_records_real_durations_and_failure(tmp_path, monkeypatch):
    path = native_trial(tmp_path, "attempt", **interval(0, 100))
    ticks = iter((100, 102, 200, 205))
    monkeypatch.setattr(timings.time, "monotonic", lambda: next(ticks))
    with timings.record(path, "restore.pack"):
        pass
    with pytest.raises(RuntimeError, match="lost transport"):
        with timings.record(path, "restore.upload", bytes=1024):
            raise RuntimeError("lost transport")
    events = [json.loads(line) for line in (path / "timings.jsonl").read_text().splitlines()]
    assert events[1]["seconds"] == 2 and events[1]["failed"] is False
    assert events[3]["seconds"] == 5 and events[3]["failed"] is True
    result = timings.collect([path])["model_A-1"]
    assert result["provider_seconds"] == {"restore.pack": 2, "restore.upload": 5}
    assert result["recorded_seconds"]["elapsed"] == 100  # Components are not added twice.


def test_imported_copy_of_an_attempt_does_not_double_its_duration(tmp_path):
    first = native_trial(tmp_path, "original", id="same-native-id", **interval(0, 60))
    copied = native_trial(tmp_path, "imported", id="same-native-id", **interval(0, 60))
    result = timings.collect([first, copied])["model_A-1"]
    assert result["attempts"] == 1 and result["recorded_seconds"]["elapsed"] == 60


def test_hard_interruption_keeps_unknown_duration_and_unfinished_journal(tmp_path):
    path = native_trial(tmp_path, "interrupted", **interval(0, None))
    event = {
        "event": "started",
        "component": "restore.upload",
        "started_at": "2026-10-17T00:00:00Z",
    }
    (path / "timings.jsonl").write_text(json.dumps(event) + '\n{"event": "fin')
    result = timings.collect([path])["model_A-1"]
    assert result["incomplete_attempts"] == 1
    assert result["recorded_seconds"]["elapsed"] is None
    assert result["provider_seconds"] == {} and result["unfinished_provider_operations"] == 1
