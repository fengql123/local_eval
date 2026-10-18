"""Two completed offline profiles and two incomplete or incompatible preparations."""

import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from http_server import event, headers, server
from test_quality_suite import questions

from eval.bench.serve import Server
from eval.request_policy import request_fields
from eval.verify import profile


@pytest.fixture(autouse=True)
def grading(monkeypatch):
    # Profile tests exercise HTTP batching and persistence; real checkers have their own tests.
    monkeypatch.setattr("eval.accuracy.grade", lambda rule, text: text == "B")


def test_real_batches_prepare_32_rounds_and_resume_without_repeating_completed_work(tmp_path):
    async def scenario():
        received = active = peak = 0
        attempts = {}
        later_round = asyncio.Event()

        async def respond(reader, writer, body):
            nonlocal received, active, peak
            question = body["messages"][0]["content"]
            round = attempts.get(question, 0)
            attempts[question] = round + 1
            received += 1
            active += 1
            peak = max(peak, active)
            if question == "0" and round == 1:
                later_round.set()
            if question == "0" and round == 0:
                await asyncio.wait_for(later_round.wait(), timeout=5)
            correct = 16 if round == 0 else len(questions())
            answer = "B" if int(body["messages"][0]["content"]) < correct else "C"
            await headers(writer)
            finish = "length" if body["messages"][0]["content"] == "0" else "stop"
            writer.write(event({"content": answer}) + event(finish=finish) + b"data: [DONE]\n\n")
            await writer.drain()
            active -= 1

        async with server(respond) as (endpoint, _):
            inputs = dict(
                rows=questions(),
                source="nvidia/Qwen3.6-35B-A3B-NVFP4",
                policy=request_fields("Qwen3.6-35B-A3B"),
                server=Server(None, endpoint, health_timeout=2),
                output=tmp_path / "rounds",
                implementation={"engine": "trusted fixture"},
                tokenizer_identity="fixed-tokenizer",
                concurrency=4,
            )
            result = await profile.build(**inputs)
            assert received == 32 * len(questions()) and result["min_correct"] == 15
            assert later_round.is_set() and 1 < peak <= 4
            assert result["rounds"][0]["correct"] == 15 and len(result["rounds"]) == 32
            assert result["lower_rank"] == 1
            assert result["rounds"][0]["truncated"] == ["0"]
            assert result["rounds"][0]["items"]["0"] is False
            repeated = await profile.build(**inputs)
            assert repeated == result and received == 32 * len(questions())
            pilot = await profile.build(
                **{**inputs, "output": tmp_path / "pilot", "pilot": True, "concurrency": -1}
            )
            assert len(pilot["rounds"]) == 2 and "min_correct" not in pilot
            assert not (tmp_path / "pilot/profile.json").exists()
            with pytest.raises(ValueError, match="pilot cannot"):
                profile.validate_profile(
                    pilot,
                    "nvidia/Qwen3.6-35B-A3B-NVFP4",
                    request_fields("Qwen3.6-35B-A3B"),
                    questions(),
                )

    asyncio.run(scenario())


def test_saved_profile_matches_checkpoint_questions_and_authored_policy(tmp_path):
    policy = request_fields("Qwen3.6-35B-A3B")
    rows = questions()
    saved = {
        "id": profile.identity("nvidia/Qwen3.6-35B-A3B-NVFP4", policy, rows),
        "checkpoint": "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "model": policy.model,
        "policy": policy.fixed("S-1"),
        "tokenizer_identity": "tokenizer",
        "checkpoint_revision": "revision",
        "questions_identity": profile.questions_identity(rows),
        "question_ids": [row["id"] for row in rows],
        "rounds": [
            {
                "round": number,
                "correct": 18,
                "items": {row["id"]: index < 18 for index, row in enumerate(rows)},
            }
            for number in range(32)
        ],
        "lower_rank": 1,
        "min_correct": 18,
    }
    from datasets import Dataset
    from huggingface_hub import DatasetCard

    from eval import quality

    Dataset.from_list(rows).to_parquet(str(tmp_path / "questions.parquet"))
    card = DatasetCard("---\nconfigs: []\n---\n# Fixture\n")
    card.data["configs"] = [
        {
            "config_name": "quality",
            "data_files": [{"split": "questions", "path": "questions.parquet"}],
        }
    ]
    card.save(tmp_path / "README.md")
    original = tmp_path / "result.json"
    original.write_text(json.dumps(saved))
    table = tmp_path / "quality.csv"
    quality.publish(original, str(tmp_path), table=table)
    bound = quality.model_range(policy.model, rows, table=table)
    assert bound["min_correct"] == 18 and bound["max_correct"] == 18
    assert bound["rounds"] == [18] * 32
    timing_only = json.loads(json.dumps(rows))
    timing_only[0]["requests"][0]["tool"] = 123
    assert quality.model_range(policy.model, timing_only, table=table) == bound
    for count, rank, minimum in ((31, 1, 18), (32, 5, 18), (32, 1, 17)):
        changed = {
            **saved,
            "rounds": saved["rounds"][:count],
            "lower_rank": rank,
            "min_correct": minimum,
        }
        with pytest.raises(ValueError, match="does not match"):
            profile.validate_profile(changed, "nvidia/Qwen3.6-35B-A3B-NVFP4", policy, rows)
    saved["rounds"][0]["truncated"] = ["0"]
    with pytest.raises(ValueError, match="does not match"):
        profile.validate_profile(saved, "nvidia/Qwen3.6-35B-A3B-NVFP4", policy, rows)


def test_interrupted_round_preserves_answers_but_cannot_publish_a_bound(tmp_path):
    async def scenario():
        received = 0
        healthy = False

        async def respond(reader, writer, body):
            nonlocal received
            received += 1
            await headers(writer)
            if (
                not healthy
                and received > len(questions())
                and body["messages"][0]["content"] == "1"
            ):
                async with asyncio.timeout(2):
                    while not (tmp_path / "round-001/000.json").exists():
                        await asyncio.sleep(0.01)
                writer.write(b"data: [DONE]\n\n")
            else:
                writer.write(event({"content": "B"}) + event(finish="stop") + b"data: [DONE]\n\n")
            await writer.drain()

        async with server(respond) as (endpoint, _):
            inputs = dict(
                rows=questions(),
                source="nvidia/Qwen3.6-35B-A3B-NVFP4",
                policy=request_fields("Qwen3.6-35B-A3B"),
                server=Server(None, endpoint, health_timeout=2),
                output=tmp_path,
                implementation={},
                tokenizer_identity="fixed-tokenizer",
                concurrency=2,
            )
            with pytest.raises(ExceptionGroup):
                await profile.build(**inputs)
            assert (tmp_path / "round-000.json").exists()
            assert (tmp_path / "round-001/000.json").is_file()
            assert not (tmp_path / "round-001/001.json").exists()
            assert not (tmp_path / "profile.json").exists()
            await asyncio.sleep(0.05)
            before = received
            cached = sum(
                len(json.loads(p.read_text())["records"]) for p in tmp_path.glob("round-*.json")
            )
            cached += len(list(tmp_path.glob("round-*/*.json")))
            healthy = True
            result = await profile.build(**inputs)
            assert len(result["rounds"]) == 32
            assert received - before == 32 * len(questions()) - cached
            assert not list(tmp_path.glob("round-*/*.json"))

    asyncio.run(scenario())


def test_incompatible_question_set_cannot_reuse_completed_rounds(tmp_path):
    (tmp_path / "preparation.json").write_text(json.dumps({"checkpoint": "another/checkpoint"}))

    @asynccontextmanager
    async def forbidden():
        pytest.fail("incompatible preparation must fail before starting the server")
        yield

    with pytest.raises(ValueError, match="different preparation"):
        asyncio.run(
            profile.build(
                questions(),
                "nvidia/Qwen3.6-35B-A3B-NVFP4",
                request_fields("Qwen3.6-35B-A3B"),
                server=forbidden(),
                output=tmp_path,
                implementation={},
                tokenizer_identity="fixed-tokenizer",
            )
        )
