"""Durable replay units: one session, one N batch, or one quality answer."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


def publish(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w") as handle:
        json.dump(value, handle)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def _identity(definition: dict) -> str:
    # A startup deadline does not change the requests or measured replay units.
    if warmup := definition.get("warmup"):
        definition = {
            **definition,
            "warmup": {key: value for key, value in warmup.items() if key != "timeout_seconds"},
        }
    return hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()


class Timer:
    def __init__(self):
        self.started = None
        self.running = 0
        self.prior_elapsed = 0.0

    @property
    def elapsed(self) -> float:
        return self.prior_elapsed + (time.monotonic() - self.started if self.started else 0)

    def enter(self):
        if not self.running:
            self.started = time.monotonic()
        self.running += 1

    def leave(self):
        self.running -= 1
        if not self.running:
            self.prior_elapsed = self.elapsed
            self.started = None


class Progress(Timer):
    """Append diagnostics; commit complete units atomically and reuse only those."""

    def __init__(self, folder: Path, definition: dict):
        super().__init__()
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        identity = _identity(definition)
        config = self.folder / "config.json"
        extended_budget = False
        previous_budget = None
        if config.exists():
            saved_definition = json.loads(config.read_text())
            saved_definition.pop("identity")
            previous_budget = saved_definition.get("budget")
            extended_budget = (
                definition.get("mode") == "quality"
                and previous_budget is not None
                and definition.get("budget") is not None
                and definition["budget"] > previous_budget
            )
            if extended_budget:
                saved_definition["budget"] = definition["budget"]
            if _identity(saved_definition) != identity:
                raise ValueError("replay inputs or configuration changed; start a new operation")
        self.units = self.folder / "units"
        self.units.mkdir(exist_ok=True)
        self.committed = 0.0
        checkpoint = self.folder / "checkpoint.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            self.prior_elapsed = (
                saved["elapsed_seconds"]
                if definition.get("mode") == "quality"
                else saved.get("committed_seconds", saved["elapsed_seconds"])
            )
            self.committed = saved.get("committed_seconds", self.prior_elapsed)
        if extended_budget:
            self._extend_quality_budget(previous_budget, definition["budget"])
        publish(config, {"identity": identity, **definition})

    def _extend_quality_budget(self, previous: float, current: float) -> None:
        """Keep finished answers and elapsed work; retry only deadline-cut answers."""
        cut_units = {}
        for path in self.units.glob("*.json"):
            records = json.loads(path.read_text())
            if any(record.get("cut") == "budget" for record in records):
                cut_units[path] = {record["row_id"] for record in records}
        if cut_units:
            removed = set().union(*cut_units.values())
            answers = self.folder / "answers.json"
            if answers.exists():
                publish(
                    answers,
                    {
                        key: value
                        for key, value in json.loads(answers.read_text()).items()
                        if key not in removed
                    },
                )
            # Clear derived grades before removing units, so interruption cannot
            # retain a deadline's false grade after its partial record disappears.
            for name in ("result.json", "cutoff.json"):
                (self.folder / name).unlink(missing_ok=True)
            for path in cut_units:
                path.unlink()
        self.event(
            "quality_budget_extended",
            previous=previous,
            budget=current,
            restarted_units=sorted(int(path.stem) for path in cut_units),
        )
        self.checkpoint()

    def event(self, kind: str, **details) -> None:
        with (self.folder / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({"event": kind, "at": time.time(), **details}) + "\n")
            handle.flush()

    def saved(self, unit: int) -> list[dict] | None:
        path = self.units / f"{unit:06}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def commit(self, unit: int, records: list[dict]) -> None:
        publish(self.units / f"{unit:06}.json", records)
        self.committed = self.elapsed
        self.event("unit_completed", unit=unit, requests=len(records))
        self.checkpoint()

    def checkpoint(self) -> None:
        publish(
            self.folder / "checkpoint.json",
            {
                "elapsed_seconds": self.elapsed,
                "committed_seconds": self.committed,
                "completed_units": [int(p.stem) for p in sorted(self.units.glob("*.json"))],
            },
        )

    async def heartbeat(self) -> None:
        import asyncio

        while True:
            await asyncio.sleep(5)
            self.checkpoint()
