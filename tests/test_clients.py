"""Two real streaming cases and two failure/cleanup cases."""

import asyncio
import json
from time import perf_counter

import httpx
import pytest
from http_server import event, headers, server

from eval.bench.clients import ChatClient
from eval.request_policy import request_fields


def recorded(out=8):
    return {
        "id": 7,
        "messages": json.dumps(
            [
                {"role": "assistant", "reasoning_content": "earlier", "content": ""},
                {"role": "user", "content": "Read 文件", "name": "user-a"},
            ]
        ),
        "tools": json.dumps(
            [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
        ),
        "out": out,
    }


def test_recorded_messages_tools_and_policy_survive_fragmented_sse():
    async def run():
        role_sent = None

        async def respond(reader, writer, body):
            nonlocal role_sent
            await headers(writer)
            writer.write(b": keepalive\n\n" + event({"role": "assistant"}))
            # A tool-call ID describes the envelope, not generated output.
            writer.write(event({"tool_calls": [{"index": 0, "id": "call_1"}]}))
            await writer.drain()
            role_sent = perf_counter()
            await asyncio.sleep(0.02)
            # SSE joins data fields, while HTTP can split in the middle of UTF-8.
            data = json.dumps(
                {"choices": [{"delta": {"reasoning_content": "思考"}}]}, ensure_ascii=False
            ).replace('"choices": ', '"choices":\ndata: ')
            frame = f"data: {data}\n\n".encode()
            split = frame.index("思".encode()) + 1
            writer.write(frame[:split])
            await writer.drain()
            await asyncio.sleep(0.01)
            writer.write(frame[split:])
            writer.write(
                event(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path":'},
                            }
                        ]
                    }
                )
            )
            writer.write(
                event(
                    {
                        "content": "Done",
                        "tool_calls": [{"index": 0, "function": {"arguments": '"a.py"}'}}],
                    }
                )
            )
            writer.write(
                event(finish="tool_calls") + event(count=8, cached=3) + b"data: [DONE]\n\n"
            )
            await writer.drain()

        request = recorded()
        async with server(respond) as (endpoint, requests):
            async with ChatClient(endpoint, request_fields("Qwen3.6-35B-A3B", "ollama")) as client:
                result = await client.request(request, "A-N")
        assert result["ok"], result["errors"]
        assert requests[0]["messages"] == json.loads(request["messages"])
        assert requests[0]["tools"] == json.loads(request["tools"])
        assert requests[0]["think"] is True
        assert requests[0]["ignore_eos"] is True
        assert requests[0]["max_tokens"] == 8
        assert result["text"] == "Done"
        assert result["reasoning"] == "思考"
        assert result["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
            }
        ]
        assert (result["request_id"], result["output_tokens"], result["cached_tokens"]) == (7, 8, 3)
        assert (
            result["t_send"] < role_sent < result["t_first"] <= result["t_last"] < result["t_end"]
        )
        assert result["output_events"] == 3
        assert result["stream_gap_max"] >= 0

    asyncio.run(run())


def test_finish_boundary_covers_invisible_forced_tokens_and_s_allowance():
    async def run():
        visible_sent = []
        finish_sent = []

        async def respond(reader, writer, body):
            await headers(writer)
            writer.write(event({"content": "x"}))
            await writer.drain()
            visible_sent.append(perf_counter())
            await asyncio.sleep(0.04)
            writer.write(event({"content": "y"}))
            await writer.drain()
            writer.write(event(finish="length"))
            await writer.drain()
            finish_sent.append(perf_counter())
            await asyncio.sleep(0.02)
            writer.write(event(count=8) + b"data: [DONE]\n\n")
            await writer.drain()

        async with server(respond) as (endpoint, requests):
            async with ChatClient(
                endpoint, request_fields("Qwen3.6-35B-A3B"), served_model="candidate"
            ) as client:
                forced = await client.request(recorded(), "A-MA")
                single = await client.request(recorded(32768), "S-1")
        assert forced["ok"] and single["ok"]
        assert forced["text"] == "xy" and forced["output_tokens"] == 8
        assert forced["t_last"] >= finish_sent[0] > visible_sent[0]
        assert forced["t_last"] - forced["t_first"] >= 0.03
        assert forced["t_end"] - forced["t_last"] >= 0.015
        assert forced["output_events"] == 2 and forced["stream_gap_max"] >= 0.03
        assert requests[0]["ignore_eos"] is True
        assert "ignore_eos" not in requests[1]
        assert requests[1]["max_tokens"] == 32768
        assert requests[1]["model"] == "candidate"

        async def answer_without_usage(reader, writer, body):
            await headers(writer)
            writer.write(
                event({"content": "Final answer: B."}) + event(finish="stop") + b"data: [DONE]\n\n"
            )
            await writer.drain()

        async with server(answer_without_usage) as (endpoint, _):
            async with ChatClient(
                endpoint, request_fields("Qwen3.6-35B-A3B"), mode="quality"
            ) as client:
                quality = await client.request(recorded(), "S-1")
        assert quality["ok"] and quality["text"] == "Final answer: B."
        assert "output_tokens" not in quality and "cached_tokens" not in quality
        assert "output_events" not in quality and "stream_gap_max" not in quality
        assert not any(key.startswith("t_") for key in quality)

    asyncio.run(run())


def test_incomplete_error_and_usage_only_streams_have_no_valid_score():
    async def run():
        complete = event({"content": "x"}) + event(finish="stop")
        cases = [
            (complete + event(count=8), "missing [DONE]"),
            (complete + b"data: [DONE]\n\n", "missing completion token usage"),
            (b'data: {"error":{"message":"engine failed"}}\n\ndata: [DONE]\n\n', "server error"),
            (
                event({"role": "assistant"})
                + event({"tool_calls": [{"index": 0, "id": "call_1", "type": "function"}]})
                + event(finish="length")
                + event(count=8)
                + b"data: [DONE]\n\n",
                "TTFT is unavailable",
            ),
        ]
        for payload, message in cases:

            async def respond(reader, writer, body, payload=payload):
                await headers(writer)
                writer.write(payload)
                await writer.drain()

            async with server(respond) as (endpoint, _):
                async with ChatClient(endpoint, request_fields("Qwen3.6-35B-A3B")) as client:
                    result = await client.request(recorded(), "A-N")
            assert not result["ok"]
            assert any(message in error for error in result["errors"])

            if message in ("missing [DONE]", "server error"):
                async with server(respond) as (endpoint, _):
                    async with ChatClient(
                        endpoint, request_fields("Qwen3.6-35B-A3B"), mode="quality"
                    ) as client:
                        quality = await client.request(recorded(), "S-1")
                assert not quality["ok"]
                assert any(message in error for error in quality["errors"])

    asyncio.run(run())


def test_cancellation_and_http_failure_release_the_connection():
    async def run():
        started = asyncio.Event()
        disconnected = asyncio.Event()

        async def respond(reader, writer, body):
            await headers(writer)
            writer.write(event({"content": "waiting"}))
            await writer.drain()
            started.set()
            await reader.read()
            disconnected.set()

        async with server(respond) as (endpoint, _):
            async with ChatClient(endpoint, request_fields("Qwen3.6-35B-A3B")) as client:
                task = asyncio.create_task(client.request(recorded(), "A-N"))
                await asyncio.wait_for(started.wait(), 2)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                await asyncio.wait_for(disconnected.wait(), 2)
            async with ChatClient(
                endpoint, request_fields("Qwen3.6-35B-A3B"), timeout=0.05, total_timeout=0.3
            ) as client:
                with pytest.raises(TimeoutError):
                    await client.request(recorded(), "A-N")

        heartbeat_disconnected = asyncio.Event()
        heartbeats = []

        async def heartbeat(reader, writer, body):
            await headers(writer)
            eof = asyncio.create_task(reader.read())
            while not (await asyncio.wait({eof}, timeout=0.005))[0]:
                writer.write(b": keepalive\n\n")
                await writer.drain()
                heartbeats.append(perf_counter())
            await eof
            heartbeat_disconnected.set()

        async with server(heartbeat) as (endpoint, _):
            async with ChatClient(
                endpoint, request_fields("Qwen3.6-35B-A3B"), timeout=0.05, total_timeout=0.1
            ) as client:
                with pytest.raises(TimeoutError):
                    await client.request(recorded(), "A-N")
                await asyncio.wait_for(heartbeat_disconnected.wait(), 2)
        assert len(heartbeats) >= 2

        async def failed(reader, writer, body):
            writer.write(
                b"HTTP/1.1 503 Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            await writer.drain()

        async with server(failed) as (endpoint, _):
            async with ChatClient(endpoint, request_fields("Qwen3.6-35B-A3B")) as client:
                with pytest.raises(httpx.HTTPStatusError, match="503"):
                    await client.request(recorded(), "A-N")

    asyncio.run(run())
