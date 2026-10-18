"""Replay prepared request graphs with workload defaults or explicit concurrency.

Rows are independent sessions. S-1, A-1 and A-MA run sessions serially.
S-N and A-N use fixed batches of N sessions. Official verification requires a
multiple of eight; public experiments may choose any concurrency.
S-N sends its independent requests together; A-N follows each graph's parents.
The next batch starts only after the current batch finishes. Quality answers
are independent and may all be admitted together. A-N retains recorded tool
waits. A-MA retains them only in sessions with concurrent subagent branches;
linear A-MA sessions, S workloads and A-1 omit them.

The caller owns the client's async context. Task groups cancel unfinished work
on a failed stream or cancellation; only a completed replay returns evidence.

SLO timing bounds a replay instead. After ``budget`` seconds no request starts
and requests in flight are cut; ``request_cap`` cuts one long request. A cut
request keeps the evidence that arrived, marked ``cut`` ("budget" or "cap").
Without ``strict``, failed requests are returned as evidence instead of raised.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import suppress

import httpx

from .progress import Progress, Timer


def tool_waits(workload: str, requests: Sequence[dict]) -> bool:
    """Wait only where recorded tool time changes concurrent scheduling.

    A frontier wider than one identifies a non-linear session, independent of
    the harness's delegation tool names. Redundant ancestor edges do not turn
    a linear session into a concurrent one. Requests are in topological order.
    """
    if workload == "A-N":
        return True
    if workload != "A-MA":
        return False
    frontier = set()
    for request in requests:
        frontier.difference_update(request["parents"])
        frontier.add(request["id"])
        if len(frontier) > 1:
            return True
    return False


class ReplayError(RuntimeError):
    """A failed protocol response, with its evidence retained for diagnosis."""

    def __init__(self, record: dict):
        self.record = record
        reason = "; ".join(record["errors"])
        super().__init__(f"request {record['row_id']}/{record['request_id']} failed: {reason}")


def failure_records(error: ReplayError | BaseExceptionGroup) -> list[dict]:
    """Extract matching request evidence from nested TaskGroup failures."""
    if isinstance(error, ReplayError):
        return [error.record]
    return [record for child in error.exceptions for record in failure_records(child)]


def session_limit(row_count: int, workload: str, concurrency: int | None) -> int:
    """Resolve the admitted rows, shared by scheduling and queue-time budgeting."""
    if concurrency == 0 or (concurrency is not None and concurrency < -1):
        raise ValueError("concurrency must be positive or -1")
    defaults = {"S-1": 1, "S-N": 8, "A-1": 1, "A-N": 8, "A-MA": 1}
    limit = defaults[workload] if concurrency is None else concurrency
    return row_count if limit == -1 else min(row_count, limit)


async def _requests(
    rows: Sequence[dict],
    workload: str,
    client,
    *,
    concurrency: int | None = None,
    budget: float | None = None,
    request_cap: float | None = None,
    strict: bool = True,
    progress: Progress | None = None,
    unit: int = 0,
) -> list[dict]:
    """Send original request records; return evidence in row/request order."""
    slots = asyncio.Semaphore(session_limit(len(rows), workload, concurrency))
    results: list[dict] = [{} for _ in rows]
    inflight: dict[tuple[int, int], dict] = {}
    bounded = budget is not None or request_cap is not None

    def evidence(result: dict, index: int, request: dict, **extra) -> dict:
        record = {**result, **extra, "row_id": rows[index]["id"], "request_id": request["id"]}
        if bounded:
            # Allowance checks retain the recorded output count.
            record["out"] = request["out"]
        return record

    async def session(row: dict, index: int):
        async with slots:
            requests = row["requests"]
            finished = {request["id"]: asyncio.Event() for request in requests}
            wait_tools = tool_waits(workload, requests)

            async def send(request: dict):
                for parent in request["parents"]:
                    await finished[parent].wait()
                delay = request.get("tool", 0) if wait_tools else 0
                if delay:
                    if progress:
                        progress.event(
                            "tool_wait",
                            unit=unit,
                            row_id=row["id"],
                            request_id=request["id"],
                            seconds=delay,
                        )
                    await asyncio.sleep(delay)
                if progress:
                    progress.event(
                        "request_started", unit=unit, row_id=row["id"], request_id=request["id"]
                    )
                if not bounded:
                    record = evidence(await client.request(request, workload), index, request)
                else:
                    partial = inflight[index, request["id"]] = {}
                    try:
                        async with asyncio.timeout(request_cap):
                            result = await client.request(request, workload, result=partial)
                        record = evidence(result, index, request)
                    except TimeoutError:
                        errors = partial.get("errors", []) + client.partial_errors(
                            partial, request["out"]
                        )
                        record = evidence(partial, index, request, cut="cap", errors=errors)
                    except (httpx.HTTPError, ValueError) as error:
                        if strict:
                            raise
                        # A broken stream is a failed request, not a broken replay.
                        errors = [*partial.get("errors", []), f"{type(error).__name__}: {error}"]
                        record = evidence(partial, index, request, ok=False, errors=errors)
                    del inflight[index, request["id"]]
                if record["errors"] if record.get("cut") else not record["ok"]:
                    if strict:
                        raise ReplayError(record)
                results[index][request["id"]] = record
                if progress:
                    progress.event(
                        "request_finished",
                        unit=unit,
                        row_id=row["id"],
                        request_id=request["id"],
                        ok=record.get("ok"),
                        output_tokens=record.get("output_tokens"),
                    )
                finished[request["id"]].set()

            async with asyncio.TaskGroup() as group:
                for request in requests:
                    group.create_task(send(request))

    async def admit():
        async with asyncio.TaskGroup() as group:
            for index, row in enumerate(rows):
                group.create_task(session(row, index))

    if budget is None:
        await admit()
    else:
        running = asyncio.ensure_future(admit())
        try:
            await asyncio.wait({running}, timeout=budget)
        finally:
            if not running.done():
                running.cancel()
                with suppress(asyncio.CancelledError):
                    await running
        if not running.cancelled():
            running.result()
        for (index, request_id), partial in inflight.items():
            request = next(item for item in rows[index]["requests"] if item["id"] == request_id)
            errors = partial.get("errors", []) + client.partial_errors(partial, request["out"])
            record = evidence(partial, index, request, cut="budget", errors=errors)
            if errors and strict:
                raise ReplayError(record)
            results[index][request_id] = record
    return [
        results[index][request["id"]]
        for index, row in enumerate(rows)
        for request in row["requests"]
        if request["id"] in results[index]
    ]


async def replay(
    rows: Sequence[dict],
    workload: str,
    client,
    *,
    concurrency: int | None = None,
    budget: float | None = None,
    request_cap: float | None = None,
    strict: bool = True,
    progress: Progress | None = None,
    quality: bool = False,
    on_complete=None,
) -> list[dict]:
    """Replay and commit independent units; N batches have a completion barrier.

    Public and private rows use the same implementation. An interrupted unit is
    rerun in full. Completed quality answers and timed batches are never retried.
    The caller owns one server and one excluded warmup for the whole replay.
    """
    rows = list(rows)
    parallel = workload in ("S-N", "A-N") and not quality
    n = session_limit(len(rows), workload, concurrency) if parallel else 1
    units = [rows[i : i + n] for i in range(0, len(rows), n)]
    records: list[list[dict]] = [[] for _ in units]
    slots = asyncio.Semaphore(session_limit(len(rows), workload, concurrency) if quality else 1)
    clock = progress or Timer()

    async def run_unit(index, selected):
        async with slots:
            saved = progress.saved(index) if progress else None
            if saved is not None:
                records[index] = saved
                if on_complete:
                    on_complete(index, selected, saved)
                return
            remaining = None if budget is None else budget - clock.elapsed
            if remaining is not None and remaining <= 0:
                return
            if progress:
                progress.event("unit_started", unit=index, rows=[r["id"] for r in selected])
            clock.enter()
            try:
                result = await _requests(
                    selected,
                    workload,
                    client,
                    concurrency=n if parallel else 1,
                    budget=remaining,
                    request_cap=request_cap,
                    strict=strict,
                    progress=progress,
                    unit=index,
                )
            finally:
                clock.leave()
            result = [{**record, "batch": index} for record in result]
            if progress:
                progress.commit(index, result)
            records[index] = result
            if on_complete:
                on_complete(index, selected, result)

    heartbeat = asyncio.create_task(progress.heartbeat()) if progress else None
    try:
        async with asyncio.TaskGroup() as group:
            for index, selected in enumerate(units):
                group.create_task(run_unit(index, selected))
    finally:
        if heartbeat:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
            progress.checkpoint()
    return [record for unit in records for record in unit]
