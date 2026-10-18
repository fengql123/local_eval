"""Trusted reference transport and strict submitted-server evidence."""

import asyncio

import pytest
from http_server import event, headers, server
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from eval.bench.clients import ChatClient
from eval.verify import run as verifier


@pytest.mark.parametrize(
    "name,evidence,count,expected_error",
    [
        ("candidate", True, 1, None),
        ("ollama", False, 1, None),
        ("candidate", False, 1, "missing accepted token IDs"),
        ("ollama", False, 0, "no completion tokens generated"),
    ],
)
def test_reference_and_submission_evidence(
    monkeypatch, tmp_path, name, evidence, count, expected_error
):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "B": 1}, unk_token="[UNK]"))
    monkeypatch.setattr(verifier.Tokenizer, "from_file", lambda _: tokenizer)
    task = {
        "model": "Qwen3.8-27B",
        "device_category": "nvidia-cuda-x86",
        "candidate": {"launch": str(tmp_path / "server.launch")},
    }
    backend = verifier._backend(task, name, {}, tmp_path, checkpoint="/models/target")

    async def respond(reader, writer, body):
        await headers(writer)
        delta = {"content": "B"}
        if evidence:
            delta.update(token_ids=[1], raw_text="B")
        writer.write(event(delta))
        writer.write(event(finish="stop"))
        writer.write(event(count=count))
        writer.write(b"data: [DONE]\n\n")
        await writer.drain()

    async def run():
        async with server(respond) as (endpoint, _):
            async with ChatClient(endpoint, backend.policy, tokenizer=backend.tokenizer) as client:
                return await client.complete({"model": "served", "stream": True})

    result = asyncio.run(run())
    assert result["ok"] is (expected_error is None)
    if expected_error:
        assert expected_error in result["errors"]
    else:
        assert result["output_tokens"] == 1
        assert result["t_first"] is not None
