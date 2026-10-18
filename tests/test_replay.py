"""Prepared DAG scheduling and cancellation. Durable batching has its own four tests."""

import asyncio
import copy
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from eval.bench.replay import ReplayError, failure_records, replay


def recordings(pool):
    source = Path(
        os.environ.get("LOCAL_EVAL_DATA", Path(__file__).resolve().parents[1] / "data/dataset")
    )
    return [
        r
        for r in pq.read_table(source / "workloads/Qwen3.6-35B-A3B.parquet").to_pylist()
        if any(
            w in r["public"]
            for w in {"S": ("S-1", "S-N"), "A": ("A-1", "A-N"), "A-MA": ("A-MA",)}[pool]
        )
    ]


def test_real_recorded_graphs_preserve_payloads_and_parent_dependencies():
    async def scenario(pool, workload):
        row = min(recordings(pool), key=lambda x: len(x["requests"]))
        # This case tests captured payloads and DAG edges; tool scheduling is below.
        for request in row["requests"]:
            request["tool"] = 0
        original = copy.deepcopy(row)
        done = set()

        class Client:
            async def request(self, request, _):
                assert set(request["parents"]) <= done
                await asyncio.sleep(0)
                done.add(request["id"])
                return {"ok": True, "errors": [], "text": "fresh"}

        records = await replay([row], workload, Client())
        assert row == original
        assert [r["request_id"] for r in records] == [r["id"] for r in row["requests"]]

    for pool, workload in (("A", "A-1"), ("A-MA", "A-MA")):
        asyncio.run(scenario(pool, workload))


def test_serial_sessions_finish_before_the_next_starts():
    async def scenario():
        selected = recordings("S")[:2]
        started = []

        class Client:
            async def request(self, request, _):
                started.append(request["messages"])
                await asyncio.sleep(0)
                return {"ok": True, "errors": []}

        records = await replay(selected, "S-1", Client())
        assert [r["row_id"] for r in records] == [r["id"] for r in selected]
        assert started == [r["requests"][0]["messages"] for r in selected]

    asyncio.run(scenario())


def test_a_failed_stream_cancels_its_graph_siblings():
    row = {"id": "fork", "requests": [{"id": i, "parents": []} for i in range(2)]}
    cancelled = []

    async def scenario():
        class Client:
            async def request(self, request, _):
                if request["id"] == 0:
                    await asyncio.sleep(0)
                    return {"ok": False, "errors": ["broken stream"]}
                try:
                    await asyncio.sleep(10)
                finally:
                    cancelled.append(1)

        with pytest.raises(ExceptionGroup) as failure:
            await replay([row], "A-MA", Client())
        assert failure.group_contains(ReplayError)
        assert failure_records(failure.value)[0]["errors"] == ["broken stream"]
        assert cancelled == [1]

    asyncio.run(scenario())


def test_quality_can_admit_all_independent_rows():
    async def scenario():
        selected = recordings("S")[:8]
        ready = asyncio.Event()
        count = 0

        class Client:
            async def request(self, request, _):
                nonlocal count
                count += 1
                if count == 8:
                    ready.set()
                await asyncio.wait_for(ready.wait(), 1)
                return {"ok": True, "errors": []}

        assert len(await replay(selected, "S-1", Client(), quality=True, concurrency=-1)) == 8

    asyncio.run(scenario())


def tool_case(monkeypatch, workload, parents):
    """Observe scheduling waits without spending the recorded wall time."""
    original_sleep = asyncio.sleep
    waits, done = [], set()

    async def wait(seconds):
        if seconds:
            waits.append(seconds)
        await original_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", wait)
    requests = [
        {"id": i, "parents": dependencies, "tool": i / 10} for i, dependencies in enumerate(parents)
    ]

    class Client:
        async def request(self, request, _):
            assert set(request["parents"]) <= done
            done.add(request["id"])
            return {"ok": True, "errors": []}

    rows = [{"id": str(i), "requests": requests} for i in range(8 if workload == "A-N" else 1)]
    asyncio.run(replay(rows, workload, Client()))
    return waits


def test_a_n_replays_waits_in_each_of_its_n_sessions(monkeypatch):
    assert tool_case(monkeypatch, "A-N", [[], [0]]) == [0.1] * 8


def test_branched_a_ma_waits_after_forks_and_joins(monkeypatch):
    assert sorted(tool_case(monkeypatch, "A-MA", [[], [0], [0], [1, 2]])) == [0.1, 0.2, 0.3]


def test_linear_a_ma_skips_waits_even_with_a_redundant_ancestor_edge(monkeypatch):
    assert tool_case(monkeypatch, "A-MA", [[], [0], [0, 1]]) == []


def test_serial_and_single_turn_workloads_ignore_tool_metadata(monkeypatch):
    assert tool_case(monkeypatch, "A-1", [[], [0]]) == []
    assert tool_case(monkeypatch, "S-1", [[], [0]]) == []
