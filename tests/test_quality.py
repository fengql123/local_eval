"""Two ordinary quality checks and two quality gate failure cases."""

import asyncio
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from http_server import event, headers, server

from eval.bench.run import quality
from eval.bench.serve import Server
from eval.request_policy import request_fields
from eval.verify.judge import quality_result
from eval.verify.run import Backend, pre_run, probe


def sample():
    source = Path(
        os.environ.get("LOCAL_EVAL_DATA", Path(__file__).resolve().parents[1] / "data/dataset")
    )
    return [
        row
        for row in pq.read_table(source / "quality/questions.parquet").to_pylist()
        if row["category"] == "math"
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


def test_quality_batches_all_rows_without_usage_warmup_or_timing(tmp_path, monkeypatch):
    selected = sample()
    policy = request_fields("Qwen3.6-35B-A3B")

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
            # A queued reply may stay completely silent past one request's
            # budget while a healthy serialized backend serves another row.
            await asyncio.sleep(0.2 if row == selected[0] else 0.4)
            writer.write(event({"content": gold(row)}) + event(finish="stop") + b"data: [DONE]\n\n")
            await writer.drain()

        async with server(respond) as (endpoint, received):
            result = await quality(
                selected,
                policy,
                server=Server(None, endpoint, health_timeout=2),
                request_timeout=0.3,
                progress=tmp_path,
            )
            assert "replay_version" not in json.loads((tmp_path / "config.json").read_text())
            assert (
                await quality(
                    selected,
                    policy,
                    server=Server(None, endpoint, health_timeout=2),
                    request_timeout=0.3,
                    progress=tmp_path,
                )
                == result
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
    policy = request_fields("Qwen3.6-35B-A3B")
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
        assert quality_result(selected, bound(selected, 2), result["items"])["ok"]

    asyncio.run(scenario())


def bound(rows, minimum):
    return {
        "id": "frozen-profile",
        "question_ids": [row["id"] for row in rows],
        "min_correct": minimum,
    }


def test_frozen_lower_bound_requires_every_selected_item_and_allows_higher_accuracy():
    selected = [{"id": f"question-{number}"} for number in range(20)]
    reference = {row["id"]: True for row in selected}
    five_losses = {key: number >= 5 for number, key in enumerate(reference)}
    rejected = quality_result(selected, bound(selected, 16), five_losses)
    assert not rejected["ok"]
    assert rejected["bound"] == {
        "items": 20,
        "correct": 15,
        "min_correct": 16,
        "accuracy": 0.75,
        "profile_id": "frozen-profile",
    }
    four_losses = {key: number >= 4 for number, key in enumerate(reference)}
    assert quality_result(selected, bound(selected, 16), four_losses)["ok"]
    assert quality_result(selected, bound(selected, 16), reference)["ok"]
    incomplete = quality_result(
        selected, bound(selected, 16), {key: value for key, value in five_losses.items() if value}
    )
    assert not incomplete["ok"] and len(incomplete["missing"]) == 5
    extra = quality_result(selected, bound(selected, 16), {**reference, "unselected": True})
    assert not extra["ok"] and extra["unexpected"] == ["unselected"]
    assert not quality_result([], bound([], 0), {})["ok"]


def test_invalid_candidate_is_rejected_and_invalid_reference_has_no_verdict():
    selected = sample()
    policy = request_fields("Qwen3.6-35B-A3B")

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

        async with server(incomplete) as (bad_url, _):
            bad = Backend("invalid", policy, lambda _stage: Server(None, bad_url, health_timeout=2))
            failed_candidate = await pre_run(selected, bound(selected, 2), bad)
            assert failed_candidate["ok"] is False and failed_candidate["stage"] == "quality"
            failed = failed_candidate["quality"]["candidate"]
            assert not failed["ok"] and failed["records"]
            invalid = next(record for record in failed["records"] if not record["ok"])
            assert "missing completion finish marker" in invalid["errors"]
            failed_reference = await probe(bad, selected)
            assert failed_reference["ok"] is False and failed_reference["stage"] == "quality"
            assert failed_reference["errors"]

    asyncio.run(scenario())
