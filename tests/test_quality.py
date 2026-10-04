"""Two ordinary quality checks and two quality gate failure cases."""

import asyncio
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from http_server import event, headers, server

from eval.bench.run import quality
from eval.bench.serve import Server
from eval.request_policy import request_fields
from eval.verify.judge import paired
from eval.verify.run import Backend, pre_run, probe


def sample():
    default = Path(__file__).resolve().parents[2] / "edge-harness-v2" / "local_eval_data"
    source = Path(os.environ.get("LOCAL_EVAL_DATA", default))
    return [
        row
        for row in pq.read_table(source / "S" / "public-00000-of-00001.parquet").to_pylist()
        if row["model"] == "C2"
    ][:2]


def gold(row):
    rule = json.loads(row["evaluation"])
    if rule["method"] == "math_verify":
        return rf"\boxed{{{rule['answer']}}}"
    return f"The correct answer is ({rule['answer']})"


async def reply(writer, text):
    await headers(writer)
    writer.write(
        event({"content": text}) + event(finish="stop") + event(count=3) + b"data: [DONE]\n\n"
    )
    await writer.drain()


def test_quality_batches_all_rows_without_usage_warmup_or_timing():
    selected = sample()
    policy = request_fields("C2")

    async def scenario():
        admitted = 0
        all_ready = asyncio.Event()

        async def respond(_reader, writer, body):
            nonlocal admitted
            admitted += 1
            if admitted == len(selected):
                all_ready.set()
            await asyncio.wait_for(all_ready.wait(), 2)
            row = next(
                row
                for row in selected
                if json.loads(row["requests"][0]["messages"]) == body["messages"]
            )
            await headers(writer)
            writer.write(event({"content": gold(row)}) + event(finish="stop") + b"data: [DONE]\n\n")
            await writer.drain()

        async with server(respond) as (endpoint, received):
            result = await quality(
                selected, policy, server=Server(None, endpoint, health_timeout=2)
            )
        assert result["ok"] and result["accuracy"] == 1
        assert result["items"] == {row["id"]: True for row in selected}
        assert len(received) == len(selected)
        assert set(result) == {"ok", "accuracy", "items", "records"}
        for record in result["records"]:
            assert not any(key.startswith("t_") for key in record)
            assert "output_tokens" not in record and "cached_tokens" not in record
            assert record["finish_reason"] == "stop"

    asyncio.run(scenario())


def test_probe_batches_supplied_questions_with_complete_grade_evidence():
    selected = sample()
    policy = request_fields("C2")
    answers = {row["requests"][0]["messages"]: gold(row) for row in selected}

    async def scenario():
        submitted = 0
        batch_ready = asyncio.Event()
        stages = []

        async def respond(_reader, writer, body):
            nonlocal submitted
            submitted += 1
            if submitted == len(selected):
                batch_ready.set()
            # Both requests must reach the backend before either response.
            await asyncio.wait_for(batch_ready.wait(), 2)
            key = json.dumps(body["messages"], ensure_ascii=False)
            answer = next(
                value for raw, value in answers.items() if json.loads(raw) == json.loads(key)
            )
            await reply(writer, answer)

        async with server(respond) as (endpoint, received):

            def owned(stage):
                stages.append(stage)
                return Server(None, endpoint, health_timeout=2)

            result = await probe(Backend("reference", policy, owned), selected)
        assert stages == ["quality"]
        assert result["ok"] and result["items"] == {row["id"]: True for row in selected}
        assert len(received) == len(selected)  # No quality warmup or timed replay.
        expected = [
            policy.request(
                json.loads(row["requests"][0]["messages"]),
                workload="S-1",
                output_limit=row["requests"][0]["out"],
            )
            for row in selected
        ]
        assert sorted(map(json.dumps, received)) == sorted(map(json.dumps, expected))
        assert {record["row_id"] for record in result["records"]} == set(result["items"])
        assert all(record["ok"] and record["text"] for record in result["records"])
        assert paired(selected, result["items"], result["items"])["mcnemar"]["p"] == 1

    asyncio.run(scenario())


def test_exact_quality_threshold_requires_every_selected_item():
    selected = [{"id": f"question-{number}"} for number in range(20)]
    reference = {row["id"]: True for row in selected}
    five_losses = {key: number >= 5 for number, key in enumerate(reference)}
    rejected = paired(selected, reference, five_losses)
    assert not rejected["ok"]
    assert rejected["mcnemar"] == {
        "items": 20,
        "lost": 5,
        "won": 0,
        "p": pytest.approx(1 / 32),
        "ref_accuracy": 1,
        "accuracy": 0.75,
    }
    four_losses = {key: number >= 4 for number, key in enumerate(reference)}
    allowed = paired(selected, reference, four_losses)
    assert allowed["ok"] and allowed["mcnemar"]["p"] == pytest.approx(1 / 16)
    incomplete = paired(
        selected, reference, {key: value for key, value in five_losses.items() if value}
    )
    assert not incomplete["ok"] and len(incomplete["missing"]["candidate"]) == 5
    extra = paired(selected, reference, {**reference, "unselected": True})
    assert not extra["ok"] and extra["unexpected"]["candidate"] == ["unselected"]
    assert not paired([], {}, {})["ok"]


def test_invalid_candidate_is_rejected_and_invalid_reference_has_no_verdict():
    selected = sample()
    policy = request_fields("C2")

    async def scenario():
        async def valid(_reader, writer, body):
            row = next(
                row
                for row in selected
                if json.loads(row["requests"][0]["messages"]) == body["messages"]
            )
            await reply(writer, gold(row))

        async def incomplete(_reader, writer, body):
            if body["messages"] == json.loads(selected[0]["requests"][0]["messages"]):
                await headers(writer)
                writer.write(b"data: [DONE]\n\n")
                await writer.drain()
            else:
                await reply(writer, gold(selected[1]))

        async with server(valid) as (good_url, _), server(incomplete) as (bad_url, _):
            good = Backend("valid", policy, lambda _stage: Server(None, good_url, health_timeout=2))
            bad = Backend("invalid", policy, lambda _stage: Server(None, bad_url, health_timeout=2))
            failed_candidate = await pre_run(selected, good, bad)
            assert failed_candidate["ok"] is False and failed_candidate["stage"] == "quality"
            failed = failed_candidate["quality"]["candidate"]
            assert not failed["ok"] and failed["records"]
            invalid = next(record for record in failed["records"] if not record["ok"])
            assert "missing completion finish marker" in invalid["errors"]
            failed_reference = await pre_run(selected, bad, good)
            assert failed_reference["ok"] is None and failed_reference["stage"] == "quality"
            assert "candidate" not in failed_reference["quality"]

    asyncio.run(scenario())
