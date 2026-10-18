"""Local HTTP/SSE peer used by the Phase 3 client and replay integration tests."""

import asyncio
import json
from contextlib import asynccontextmanager


@asynccontextmanager
async def server(respond):
    """Yield a base URL and captured bodies; respond(reader, writer, body) streams a reply."""
    requests = []

    async def connection(reader, writer):
        headers = (await reader.readuntil(b"\r\n\r\n")).decode()
        try:
            if headers.startswith("GET /v1/models "):
                writer.write(
                    b'HTTP/1.1 200 OK\r\nContent-Length: 28\r\nConnection: close\r\n\r\n{"data": [{"id": "served"}]}'
                )
                await writer.drain()
            elif headers.startswith("GET "):
                writer.write(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
                await writer.drain()
            else:
                length = next(
                    int(line.split(":", 1)[1])
                    for line in headers.split("\r\n")
                    if line.lower().startswith("content-length:")
                )
                body = json.loads(await reader.readexactly(length))
                requests.append(body)
                await respond(reader, writer, body)
        finally:
            writer.close()
            await writer.wait_closed()

    async with await asyncio.start_server(connection, "127.0.0.1", 0) as listener:
        yield f"http://127.0.0.1:{listener.sockets[0].getsockname()[1]}", requests


async def headers(writer):
    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
    await writer.drain()


def event(delta=None, *, finish=None, count=None, cached=None):
    value = {
        "choices": []
        if count is not None
        else [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
    }
    if count is not None:
        value["usage"] = {
            "completion_tokens": count,
            "prompt_tokens_details": {"cached_tokens": cached},
        }
    return f"data: {json.dumps(value, ensure_ascii=False)}\n\n".encode()
