"""Two accepted-token streams and two inconsistent evidence cases."""

import asyncio

from http_server import event, headers, server
from tokenizers import AddedToken, Tokenizer, decoders, models

from eval.bench.clients import ChatClient
from eval.request_policy import request_fields


def tokenizer():
    value = Tokenizer(
        models.BPE(
            {"a": 0, "b": 1, "ab": 2, "<think>": 3, "</think>": 4, "<eos>": 5, "x": 6}, [("a", "b")]
        )
    )
    value.decoder = decoders.ByteLevel()
    value.add_special_tokens(
        [AddedToken(text, special=True) for text in ("<think>", "</think>", "<eos>")]
    )
    return value


async def exchange(frames, *, mode="timing"):
    async def respond(reader, writer, body):
        await headers(writer)
        writer.write(b"".join(frames) + b"data: [DONE]\n\n")
        await writer.drain()

    seen = []

    async def capture(value):
        seen.append(value)

    async with server(respond) as (endpoint, _):
        async with ChatClient(
            endpoint, request_fields("Qwen3.6-35B-A3B"), tokenizer=tokenizer(), mode=mode
        ) as client:
            return await client.complete({"max_tokens": 8}, on_event=capture), seen


def test_hidden_reasoning_control_tokens_and_eos_survive_before_parser_fields():
    ids = [3, 6, 4, 2, 5]
    result, seen = asyncio.run(
        exchange(
            [
                event({"role": "assistant"}),
                event({"token_ids": [3]}),
                event({"token_ids": [6, 4], "reasoning_content": "x"}),
                event({"token_ids": [2], "content": "ab"}),
                event({"token_ids": [5], "raw_text": "<think>x</think>ab<eos>"}),
                event(finish="stop"),
                event(count=5),
            ]
        )
    )
    assert result["ok"] and result["token_ids"] == ids and result["output_tokens"] == 5
    assert result["reasoning"] == "x" and result["text"] == "ab"
    assert result["raw_text"] == tokenizer().decode(ids, skip_special_tokens=False)
    assert [
        token
        for value in seen
        for choice in value["choices"]
        for token in choice["delta"].get("token_ids", [])
    ] == ids


def test_identical_decoded_text_keeps_actual_segmentation_and_quality_has_no_metrics():
    for ids in ([2], [0, 1]):
        result, _ = asyncio.run(
            exchange(
                [
                    event({"token_ids": ids, "raw_text": "ab", "content": "ab"}),
                    event(finish="stop"),
                    event(count=len(ids)),
                ],
                mode="quality",
            )
        )
        assert result["ok"] and result["token_ids"] == ids
        assert "output_tokens" not in result and not any(key.startswith("t_") for key in result)
    assert tokenizer().encode("ab").ids == [2]  # Re-tokenizing text would miscount [0, 1].


def test_inflated_usage_and_out_of_vocabulary_ids_are_rejected():
    for ids, count, error in (
        ([2], 8, "usage does not match"),
        ([999], 1, "outside the fixed tokenizer"),
    ):
        result, _ = asyncio.run(
            exchange(
                [
                    event({"token_ids": ids, "raw_text": "ab", "content": "ab"}),
                    event(finish="stop"),
                    event(count=count),
                ]
            )
        )
        assert not result["ok"] and any(error in item for item in result["errors"])


def test_stripped_control_syntax_or_missing_ids_cannot_pass_consistency_checks():
    for delta, count, error in (
        (
            {"token_ids": [3, 6, 4], "raw_text": "x", "content": "x"},
            3,
            "does not match fixed-tokenizer",
        ),
        ({"content": "ab"}, 1, "missing accepted token IDs"),
    ):
        result, _ = asyncio.run(exchange([event(delta), event(finish="stop"), event(count=count)]))
        assert not result["ok"] and any(error in item for item in result["errors"])
