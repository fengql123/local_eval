"""Completed-unit cutoffs and reproduced-installation approvals."""

import asyncio
import json
import os

from http_server import event, headers, server

from eval.bench import run
from eval.bench.serve import Server
from eval.bench.watchdog import Watchdog
from eval.request_policy import request_fields
from eval.verify import access
from eval.verify import run as verifier


def rows():
    return [
        {
            "id": str(i),
            "evaluation": '{"method":"letter","answer":"B"}',
            "requests": [
                {
                    "id": 0,
                    "parents": [],
                    "out": 4,
                    "tools": None,
                    "messages": json.dumps([{"role": "user", "content": str(i)}]),
                }
            ],
        }
        for i in range(4)
    ]


def test_quality_stops_when_passing_is_impossible_and_reuses_the_failed_result(tmp_path):
    async def scenario():
        async def respond(reader, writer, body):
            await headers(writer)
            if body["messages"][0]["content"] == "0":
                writer.write(event({"content": "C"}) + event(finish="stop") + b"data: [DONE]\n\n")
                await writer.drain()
            else:
                await reader.read()

        async with server(respond) as (endpoint, received):

            async def check():
                return await run.quality(
                    rows(),
                    request_fields("Qwen3.6-35B-A3B"),
                    server=Server(None, endpoint),
                    budget=10,
                    min_correct=4,
                    progress=tmp_path,
                )

            result = await asyncio.wait_for(check(), 2)
            assert (
                result["cutoff"]["reason"] == "remaining answers cannot meet the required accuracy"
            )
            assert len(result["records"]) == 1 and result["accuracy"] == 0
            assert json.loads((tmp_path / "answers.json").read_text()) == {"0": False}
            sent = len(received)
            assert await check() == result
            assert len(received) == sent

    asyncio.run(scenario())


def test_slo_cutoff_keeps_only_completed_unit_credit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run, "Watchdog", lambda workload, **kw: Watchdog(workload, window=0.05, **kw)
    )

    async def scenario():
        async def respond(reader, writer, body):
            await headers(writer)
            writer.write(event({"content": "B", "token_ids": [1]}))
            await writer.drain()
            if body["messages"] == run.WARMUP["messages"] or body["messages"][0]["content"] == "0":
                await asyncio.sleep(0.002)
                writer.write(
                    event({"content": "B", "token_ids": [1, 1, 1]})
                    + event(finish="stop")
                    + event(count=4)
                    + b"data: [DONE]\n\n"
                )
                await writer.drain()
            else:
                await reader.read()

        async with server(respond) as (endpoint, _):
            result = await asyncio.wait_for(
                run.execute(
                    rows(),
                    "S-1",
                    request_fields("Qwen3.6-35B-A3B"),
                    server=Server(None, endpoint),
                    mode="slo",
                    budget=20,
                    progress=tmp_path,
                ),
                5,
            )
            assert result["cutoff"]["reason"] == (
                "recent and overall throughput stayed below the lowest SLO tier"
            )
            assert result["cutoff"]["overall_observed"] < result["cutoff"]["threshold"]
            assert {r["row_id"] for r in result["records"]} == {"0"}
            from eval.bench.slo import score

            placed = score("S-1", result["records"], rows=rows())
            assert 0 < placed["score"] <= 25
            assert placed["coverage"]["complete_fraction"] == 0.25

    asyncio.run(scenario())


def test_rebuilt_installation_keeps_approvals_until_source_changes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "server.py").write_text("original")
    state, review = tmp_path / "state", tmp_path / "review"
    state.mkdir()
    review.mkdir()
    monkeypatch.setattr(verifier, "STATE", state)
    monkeypatch.setattr(verifier, "REVIEW", review)
    monkeypatch.setattr(access, "ACCOUNTS", {"judge": (os.getuid(), os.getgid())})
    original = verifier.validate_installation(source)
    (state / "quality.json").write_text('{"ok":true}')
    (review / "review.json").write_text('{"ok":true}')
    operation = state / "operations/candidate/timing"
    operation.mkdir(parents=True)
    (operation / "result.json").write_text('{"valid":true}')
    (source / "server.py").touch()
    assert verifier.validate_installation(source) == original
    assert (state / "quality.json").exists() and (review / "review.json").exists()
    (source / "server.py").write_text("changed")
    assert verifier.validate_installation(source) != original
    assert not (state / "quality.json").exists()
    assert not (operation / "result.json").exists()
    assert not (review / "review.json").exists()
    assert (state / "superseded" / original / "quality.json").exists()
