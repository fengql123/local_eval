"""Recorded phase work and provider overhead, in seconds; missing times stay unknown."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


@contextmanager
def record(trial: Path, component: str, **details):
    """Journal a provider operation independently of overlapping agent/replay work."""
    path = trial / "timings.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"component": component, "started_at": datetime.now(UTC).isoformat(), **details}

    def append(value):
        with path.open("a") as handle:
            handle.write(json.dumps(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    append({**event, "event": "started"})
    started = time.monotonic()
    try:
        yield
    finally:
        append(
            {
                **event,
                "event": "finished",
                "finished_at": datetime.now(UTC).isoformat(),
                "seconds": time.monotonic() - started,
                "failed": sys.exc_info()[0] is not None,
            }
        )


def _seconds(interval: dict | None) -> float | None:
    if not interval or not interval.get("started_at") or not interval.get("finished_at"):
        return None
    return (
        datetime.fromisoformat(interval["finished_at"])
        - datetime.fromisoformat(interval["started_at"])
    ).total_seconds()


def _sum(values):
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _components(path: Path) -> tuple[dict, int]:
    totals = defaultdict(float)
    pending = set()
    if path.exists():
        with path.open() as handle:
            for line in handle:
                if not line.endswith("\n"):
                    break  # An interrupted append is not a completed duration.
                event = json.loads(line)
                key = (event["component"], event["started_at"])
                if event["event"] == "started":
                    pending.add(key)
                else:
                    pending.discard(key)
                    totals[event["component"]] += event["seconds"]
    return dict(totals), len(pending)


def collect(trials) -> dict[str, dict]:
    """Sum recorded attempts per task; never present paused calendar time as work.

    Recorded seconds include retries. Incomplete attempts make these lower bounds;
    provider components overlap native intervals and must not be added to them.
    """
    grouped = defaultdict(list)
    seen = set()
    for trial in trials:
        path = trial / "result.json"
        result = (
            json.loads(path.read_text())
            if path.exists()
            else {"config": json.loads((trial / "config.json").read_text())}
        )
        identity = result.get("id") or str(trial.resolve())
        if identity in seen:
            continue
        seen.add(identity)
        task = Path(result["config"]["task"]["path"]).name
        grouped[task].append((trial, result))
    summaries = {}
    fields = ("environment_setup", "agent_setup", "agent_execution", "verifier")
    for task, attempts in grouped.items():
        native = [result for _, result in attempts]
        steps = defaultdict(list)
        components = defaultdict(float)
        unfinished = 0
        for trial, result in attempts:
            for step in result.get("step_results") or ():
                steps[step["step_name"]].append(_seconds(step.get("verifier")))
            values, pending = _components(trial / "timings.jsonl")
            unfinished += pending
            for component, seconds in values.items():
                components[component] += seconds
        summaries[task] = {
            "attempts": len(attempts),
            "incomplete_attempts": sum(_seconds(result) is None for result in native),
            "recorded_seconds": {
                "elapsed": _sum(_seconds(result) for result in native),
                **{
                    field: _sum(_seconds(result.get(field)) for result in native)
                    for field in fields
                },
                "steps": {name: _sum(values) for name, values in steps.items()},
            },
            "provider_seconds": dict(components),
            "unfinished_provider_operations": unfinished,
        }
    return summaries
