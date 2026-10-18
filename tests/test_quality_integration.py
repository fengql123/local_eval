"""The table gates a hidden suite, with cached answers and terminal wrong answers."""

import asyncio
import json

import pytest
from datasets import Dataset
from http_server import event, headers, server
from huggingface_hub import DatasetCard
from test_quality_suite import questions

from eval import quality
from eval import run as coordinator
from eval.bench.serve import Server
from eval.config import model
from eval.request_policy import request_fields
from eval.verify import profile
from eval.verify import run as verifier


def prepared(tmp_path, monkeypatch):
    rows = questions()
    suite = tmp_path / "dataset"
    suite.mkdir()
    Dataset.from_list(rows).to_parquet(str(suite / "questions.parquet"))
    card = DatasetCard("---\nconfigs: []\n---\n# Fixture\n")
    card.data["configs"] = [
        {
            "config_name": "quality",
            "data_files": [{"split": "questions", "path": "questions.parquet"}],
        }
    ]
    card.save(suite / "README.md")
    view = tmp_path / "run/input/quality"
    quality.map_suite(str(suite), view)
    policy = request_fields("Qwen3.6-35B-A3B")
    source = model(policy.model).checkpoint
    bound = {
        "id": profile.identity(source, policy, rows),
        "model": policy.model,
        "checkpoint": source,
        "policy": policy.fixed("S-1"),
        "tokenizer_identity": "fixed",
        "questions_identity": profile.questions_identity(rows),
        "question_ids": [r["id"] for r in rows],
        "lower_rank": 1,
        "min_correct": 18,
        "rounds": [
            {"round": i, "correct": 18, "items": {row["id"]: n < 18 for n, row in enumerate(rows)}}
            for i in range(32)
        ],
    }
    raw = tmp_path / "offline-profile.json"
    raw.write_text(json.dumps(bound))
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(verifier, "STATE", state)
    monkeypatch.setattr(quality, "TABLES", tmp_path)
    monkeypatch.setattr("eval.bench.run.grade", lambda rule, text: text == "B")
    task = {
        "model": policy.model,
        "target": {"source": source},
        "inputs": {"quality": str(view)},
    }
    return task, suite, raw, policy


def test_coordinator_reads_the_model_table_before_allocating_verification(tmp_path, monkeypatch):
    task, suite, raw, _ = prepared(tmp_path, monkeypatch)
    folder = tmp_path / "run"
    env = folder / "tasks/evaluation/task/environment"
    env.mkdir(parents=True)
    (env / "task.json").write_text(json.dumps(task))
    (folder / "prepared.json").write_text(
        json.dumps({"quality": {"source": task["inputs"]["quality"]}})
    )
    with pytest.raises(FileNotFoundError):
        asyncio.run(coordinator._quality_ready(folder, (env.parent,), {}))
    quality.publish(raw, str(suite))
    asyncio.run(coordinator._quality_ready(folder, (env.parent,), {}))
    assert not (suite / "profiles").exists()


def test_verification_only_uses_hidden_questions_and_reuses_a_finished_gate(tmp_path, monkeypatch):
    task, suite, raw, policy = prepared(tmp_path, monkeypatch)
    quality.publish(raw, str(suite))

    async def scenario():
        async def respond(reader, writer, body):
            await headers(writer)
            writer.write(event({"content": "B"}) + event(finish="stop") + b"data: [DONE]\n\n")
            await writer.drain()

        async with server(respond) as (endpoint, received):
            backend = verifier.Backend(
                "candidate",
                policy,
                lambda _: Server(None, endpoint),
                progress=verifier.STATE / "operations/candidate",
            )
            monkeypatch.setattr(verifier, "_backend", lambda *a, **k: backend)
            first = await verifier.quality_gate(task, tmp_path / "out")
            second = await verifier.quality_gate(task, tmp_path / "out")
        assert first == second and first["ok"]
        assert len(received) == 30
        assert set(first["quality"]["candidate"]["items"]) == {r["id"] for r in questions()}

    asyncio.run(scenario())


def test_missing_or_changed_quality_range_does_not_authorize_measurement(tmp_path, monkeypatch):
    task, suite, raw, _ = prepared(tmp_path, monkeypatch)
    missing = asyncio.run(verifier.quality_gate(task, tmp_path / "out"))
    assert missing["ok"] is None and missing["stage"] == "quality_profile"
    table = quality.publish(raw, str(suite))
    table.write_text(table.read_text().replace("18,18,", "17,18,"))
    invalid = asyncio.run(verifier.quality_gate(task, tmp_path / "out"))
    assert invalid["ok"] is None


def test_completed_incorrect_answers_are_kept_when_the_gate_is_repeated(tmp_path, monkeypatch):
    task, suite, raw, policy = prepared(tmp_path, monkeypatch)
    quality.publish(raw, str(suite))

    async def scenario():
        async def respond(reader, writer, body):
            await headers(writer)
            writer.write(event({"content": "C"}) + event(finish="stop") + b"data: [DONE]\n\n")
            await writer.drain()

        async with server(respond) as (endpoint, received):
            backend = verifier.Backend(
                "candidate",
                policy,
                lambda _: Server(None, endpoint),
                progress=verifier.STATE / "operations/candidate",
            )
            monkeypatch.setattr(verifier, "_backend", lambda *a, **k: backend)
            for _ in range(2):
                result = await verifier.quality_gate(task, tmp_path / "out")
                assert result["ok"] is False
        assert len(received) == 30

    asyncio.run(scenario())
