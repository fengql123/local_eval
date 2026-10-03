"""Replay prepared request graphs and collect timing evidence without grading.

Rows are independent sessions. ``S-1``, ``A-1`` and ``A-MA`` run sessions one at
a time; ``S-N`` runs all rows together and ``A-N`` runs at most eight sessions.
Within a session, requests wait for every parent and then their recorded tool
delay, except for ``A-1``, which omits tool delays. Branches can overlap freely.

The caller owns the client's async context. Task groups cancel unfinished work
on a failed stream or cancellation; only a completed replay returns evidence.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence


class ReplayError(RuntimeError):
    """A failed protocol response, with its evidence retained for diagnosis."""

    def __init__(self, record: dict):
        self.record = record
        super().__init__(f"request {record['row_id']}/{record['request_id']} failed")


async def replay(rows: Sequence[dict], workload: str, client, *, sleep=asyncio.sleep) -> list[dict]:
    """Send original request records; return evidence in row/request order."""
    session_limit = {"S-1": 1, "S-N": len(rows), "A-1": 1, "A-N": 8, "A-MA": 1}[workload]
    slots = asyncio.Semaphore(session_limit)
    completed: list[list[dict]] = [[] for _ in rows]

    async def session(row: dict, index: int):
        async with slots:
            requests = row["requests"]
            finished = {request["id"]: asyncio.Event() for request in requests}
            results = {}

            async def send(request: dict):
                for parent in request["parents"]:
                    await finished[parent].wait()
                if workload != "A-1" and request["tool"]:
                    await sleep(request["tool"])
                result = await client.request(request, workload)
                record = {**result, "row_id": row["id"], "request_id": request["id"]}
                if not record["ok"]:
                    raise ReplayError(record)
                results[request["id"]] = record
                finished[request["id"]].set()

            async with asyncio.TaskGroup() as group:
                for request in requests:
                    group.create_task(send(request))
            completed[index] = [results[request["id"]] for request in requests]

    async with asyncio.TaskGroup() as group:
        for index, row in enumerate(rows):
            group.create_task(session(row, index))
    return [record for session_results in completed for record in session_results]
