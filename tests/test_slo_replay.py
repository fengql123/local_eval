"""Bounded SLO replay over real HTTP streams: caps, budgets and partial evidence."""

import asyncio
import json

import pytest
from http_server import event, headers, server

from eval.bench.clients import ChatClient
from eval.bench.replay import ReplayError, replay
from eval.bench.run import execute
from eval.bench.serve import Server, StartupError
from eval.bench.slo import score
from eval.request_policy import request_fields

POLICY = request_fields("Qwen3.6-35B-A3B")


def row(name, out=16):
    message = json.dumps([{"role": "user", "content": name}])
    request = {"id": 0, "messages": message, "tools": None, "out": out, "parents": [], "tool": 0}
    return {"id": name, "requests": [request]}


async def stream(writer, tokens, *, stall=0.0, finish=True):
    await headers(writer)
    for token in range(1, tokens + 1):
        writer.write(event({"content": f"t{token}", "token_ids": [token]}))
        await writer.drain()
        await asyncio.sleep(0.01)
    if stall:
        await asyncio.sleep(stall)
    if finish:
        writer.write(event(finish="stop") + event(count=tokens) + b"data: [DONE]\n\n")
        await writer.drain()


async def respond(reader, writer, body):
    name = body["messages"][-1]["content"]
    if name == "broken":
        # A server that drops the connection mid-stream fails only this request.
        await headers(writer)
        writer.write(b"data: {not json}\n\n")
        await writer.drain()
    elif name == "slow":
        await stream(writer, 3, stall=1.5, finish=False)
    elif name == "stall":
        await stream(writer, 2, stall=1.5, finish=False)
    else:
        await stream(writer, 4)


def test_cap_and_budget_cut_requests_but_keep_their_evidence():
    async def run():
        rows = [row(name) for name in ("fast-0", "slow", "broken", "fast-1", "stall", "never")]
        async with server(respond) as (url, requests):
            async with ChatClient(url, POLICY, timeout=10) as client:
                records = await replay(
                    rows, "S-1", client, request_cap=0.6, budget=1.0, strict=False
                )
        return rows, records, requests

    rows, records, requests = asyncio.run(run())
    assert [record["row_id"] for record in records] == [
        "fast-0",
        "slow",
        "broken",
        "fast-1",
        "stall",
    ]
    assert len(requests) == 5
    fast, slow, broken, _, stall = records
    assert fast["ok"] and fast["output_tokens"] == 4 and fast["out"] == 16
    assert slow["cut"] == "cap" and slow["errors"] == [] and slow["token_ids"] == [1, 2, 3]
    assert slow["t_first"] is not None and slow["t_end"] - slow["t_send"] >= 0.6
    assert not broken["ok"] and "JSONDecodeError" in broken["errors"][-1]
    assert stall["cut"] == "budget" and stall["token_ids"] == [1, 2]
    result = score("S-1", records, rows=rows)
    assert result["coverage"]["started"] == 5 and result["coverage"]["fraction"] == 5 / 6
    assert result["requests"] == {"scored": 5, "failed": 3, "capped": 1, "cut_at_budget": 1}
    assert result["coverage"]["completed"] == 2
    assert not result["detail"][1]["complete"]


def test_parallel_workloads_stay_strict_and_measure_within_the_window():
    async def run(names, budget):
        async with server(respond) as (url, _):
            async with ChatClient(url, POLICY, timeout=10) as client:
                return await replay([row(name) for name in names], "S-N", client, budget=budget)

    with pytest.raises(ExceptionGroup) as failure:
        asyncio.run(run(("fast-0", "broken", *[f"fast-{i}" for i in range(2, 8)]), 2))
    assert failure.group_contains(json.JSONDecodeError)
    records = asyncio.run(run(("fast-0", "stall", *[f"fast-{i}" for i in range(1, 7)]), 0.5))
    assert [record.get("cut") for record in records] == [None, "budget", *[None] * 6]
    result = score("S-N", records, rows=[row(record["row_id"]) for record in records])
    # A cut fixed-N batch earns no credit, even when other requests finished.
    assert result["throughput"] == 0 and result["requests"]["cut_at_budget"] == 1
    assert result["score"] == 0
    assert sum(len(record.get("token_ids", ())) for record in records) == 30


def test_cut_evidence_is_checked_against_the_allowance_and_tokenizer():
    class Tokenizer:
        def get_vocab_size(self, with_added_tokens=True):
            return 100

        def decode(self, ids, skip_special_tokens=False):
            return "".join(f"t{token}" for token in ids)

    client = ChatClient("http://127.0.0.1:1", POLICY, tokenizer=Tokenizer())
    partial = {"t_first": 1.0, "token_ids": [1, 2], "raw_text": "t1t2"}
    assert client.partial_errors(partial, 16) == []
    # One character may still be in flight when a stream is cut.
    assert client.partial_errors({**partial, "raw_text": "t1t"}, 16) == []
    assert "exceed allowance" in client.partial_errors(partial, 1)[0]
    assert "decoding" in client.partial_errors({**partial, "raw_text": "other text"}, 16)[0]
    assert "outside" in client.partial_errors({**partial, "token_ids": [500]}, 16)[0]
    assert client.partial_errors({"t_first": None}, 16) == []

    async def fail():
        class Failing:
            async def request(self, request, workload, *, result=None):
                result.update(ok=False, errors=["invalid accepted token IDs"])
                await asyncio.sleep(5)

            def partial_errors(self, result, allowance):
                return []

        await replay([row("x")], "S-1", Failing(), request_cap=0.05)

    # Strict replays fail on cut evidence that was already invalid.
    with pytest.raises(ExceptionGroup) as failure:
        asyncio.run(fail())
    assert failure.group_contains(ReplayError)


def test_silent_warmup_fails_at_its_own_deadline(tmp_path, monkeypatch):
    from eval.bench import run

    monkeypatch.setitem(run.WARMUP, "timeout_seconds", 0.05)

    async def scenario():
        async def respond(reader, writer, body):
            await headers(writer)
            await reader.read()

        async with server(respond) as (url, requests):
            with pytest.raises(StartupError, match="warmup did not finish within 0.05s") as failure:
                await execute(
                    [row("a")],
                    "S-1",
                    POLICY,
                    server=Server(None, url, health_timeout=1),
                    mode="slo",
                    progress=tmp_path,
                    budget=2,
                    request_cap=1,
                )
            assert isinstance(failure.value.__cause__, TimeoutError)
            assert len(requests) == 1
            assert requests[0][POLICY.max_tokens] == 16
            assert not (tmp_path / "units/000000.json").exists()

    asyncio.run(scenario())
