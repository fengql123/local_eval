"""Two scheduling and two cancellation cases using the supplied local recordings.

Set LOCAL_EVAL_DATA to their root, or keep them in sibling
edge-harness-v2/local_eval_data. Prepared rows are consumed without schema audits.
"""

import asyncio
import copy
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from eval.bench.replay import ReplayError, replay


def recordings(pool):
    default = Path(__file__).resolve().parents[2] / "edge-harness-v2" / "local_eval_data"
    source = Path(os.environ.get("LOCAL_EVAL_DATA", default))
    return pq.read_table(source / pool / "public-00000-of-00001.parquet").to_pylist()


class ControlledClient:
    def __init__(self, requests):
        self.started = {request["id"]: asyncio.Event() for request in requests}
        self.release = {request["id"]: asyncio.Event() for request in requests}
        self.done = set()
        self.parents_at_start = {}
        self.clock = 0

    async def request(self, request, workload):
        request_id = request["id"]
        self.parents_at_start[request_id] = set(self.done)
        self.started[request_id].set()
        await self.release[request_id].wait()
        self.done.add(request_id)
        self.clock += 1
        return {"ok": True, "errors": [], "t_end": self.clock, "text": "server reply"}


def test_recorded_chain_fork_and_join_wait_for_parents_and_tool_delays():
    async def exercise(row, workload):
        client = ControlledClient(row["requests"])
        delays = []

        async def sleep(delay):
            delays.append((delay, set(client.done)))

        task = asyncio.create_task(replay([row], workload, client, sleep=sleep))
        if workload == "A-MA":
            for request_id in [0, 1]:
                await asyncio.wait_for(client.started[request_id].wait(), 1)
                client.release[request_id].set()
            for request_id in [2, 3, 4]:
                await asyncio.wait_for(client.started[request_id].wait(), 1)
            assert not client.started[12].is_set()
            for request_id in [2, 6, 8, 11]:
                client.release[request_id].set()
                await asyncio.wait_for(client.started[request_id].wait(), 1)
            await asyncio.sleep(0)
            assert 11 in client.done
            assert not client.started[12].is_set()
            client.release[3].set()
            await asyncio.sleep(0)
            assert not client.started[12].is_set()
            client.release[4].set()
            await asyncio.wait_for(client.started[12].wait(), 1)
            assert {3, 4, 11} <= client.parents_at_start[12]
        for event in client.release.values():
            event.set()
        records = await asyncio.wait_for(task, 1)
        assert [record["request_id"] for record in records] == [
            request["id"] for request in row["requests"]
        ]
        assert all(record["row_id"] == row["id"] for record in records)
        for request in row["requests"]:
            assert set(request["parents"]) <= client.parents_at_start[request["id"]]
            if request["tool"]:
                assert any(
                    delay == request["tool"] and set(request["parents"]) <= done
                    for delay, done in delays
                )

    async def main():
        await exercise(recordings("A")[0], "A-N")
        await exercise(recordings("A-MA")[0], "A-MA")

    asyncio.run(main())


def test_workload_session_limits_preserve_requests_and_a1_omits_delays():
    async def exercise(workload, rows, limit):
        before = copy.deepcopy(rows)
        originals = {id(request) for row in rows for request in row["requests"]}
        released = asyncio.Event()
        batch_ready = asyncio.Event()
        active, peak, started = 0, 0, 0
        delays = []

        class Client:
            async def request(self, request, received_workload):
                nonlocal active, peak, started
                assert received_workload == workload
                assert id(request) in originals
                active += 1
                started += 1
                peak = max(peak, active)
                if started == limit:
                    batch_ready.set()
                await released.wait()
                active -= 1
                return {"ok": True, "text": "fresh answer"}

        async def sleep(delay):
            delays.append(delay)

        task = asyncio.create_task(replay(rows, workload, Client(), sleep=sleep))
        await asyncio.wait_for(batch_ready.wait(), 1)
        await asyncio.sleep(0)
        assert started == limit
        released.set()
        records = await asyncio.wait_for(task, 1)
        assert peak == limit
        assert rows == before
        assert [(record["row_id"], record["request_id"]) for record in records] == [
            (row["id"], request["id"]) for row in rows for request in row["requests"]
        ]
        assert all(record["text"] == "fresh answer" for record in records)
        if workload == "A-1":
            assert delays == []
        if workload == "A-MA":
            assert delays == [
                request["tool"] for row in rows for request in row["requests"] if request["tool"]
            ]

    async def main():
        singles = recordings("S")[:10]
        chains = recordings("A")[:2]
        await exercise("S-1", singles, 1)
        await exercise("S-N", singles, 10)
        await exercise("A-N", singles, 8)
        await exercise("A-1", chains, 1)
        await exercise("A-MA", chains, 1)

    asyncio.run(main())


def two_requests():
    return [{"id": "session", "requests": [{"id": i, "parents": [], "tool": 0}]} for i in range(2)]


def test_failed_stream_cancels_blocked_sibling_and_retains_failure_evidence():
    async def exercise():
        blocked = asyncio.Event()
        cancelled = asyncio.Event()

        class Client:
            async def request(self, request, workload):
                if request["id"] == 0:
                    await blocked.wait()
                    return {"ok": False, "errors": ["missing final usage"], "text": "partial"}
                blocked.set()
                try:
                    await asyncio.Future()
                finally:
                    cancelled.set()

        with pytest.raises(ExceptionGroup) as failure:
            await replay(two_requests(), "S-N", Client())
        error = failure.value.exceptions[0].exceptions[0]
        assert isinstance(error, ReplayError)
        assert error.record == {
            "ok": False,
            "errors": ["missing final usage"],
            "text": "partial",
            "row_id": "session",
            "request_id": 0,
        }
        assert cancelled.is_set()

    asyncio.run(exercise())


def test_caller_cancellation_stops_all_started_requests():
    async def exercise():
        both_started = asyncio.Event()
        active = set()
        cancelled = set()

        class Client:
            async def request(self, request, workload):
                active.add(request["id"])
                if len(active) == 2:
                    both_started.set()
                try:
                    await asyncio.Future()
                finally:
                    active.remove(request["id"])
                    cancelled.add(request["id"])

        task = asyncio.create_task(replay(two_requests(), "S-N", Client()))
        await asyncio.wait_for(both_started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert active == set()
        assert cancelled == {0, 1}

    asyncio.run(exercise())
