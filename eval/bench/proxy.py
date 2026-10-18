"""Route native harness sessions to pooled canonical Chat Completions servers."""

import asyncio
import json
from collections import defaultdict
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import httpx
from aiohttp import web

from .clients import ChatClient
from .protocols import Emitter, chat, sse


class Proxy:
    """One session prefix per row; one shared HTTP client per configured server."""

    def __init__(self, servers: dict[str, dict], output: Path):
        self.servers, self.output = servers, output
        self.sessions = {}
        self.calls, self.active = defaultdict(list), defaultdict(set)

    def register(self, session: str, server_id: str, workload: str):
        """Register on the proxy's event loop before starting row workers."""
        self.sessions[session] = (server_id, workload)

    async def __aenter__(self):
        self.loop = asyncio.get_running_loop()
        self.resources = await AsyncExitStack().__aenter__()
        self.clients = {
            name: await self.resources.enter_async_context(
                ChatClient(server["endpoint"], server["policy"], mode="quality")
            )
            for name, server in self.servers.items()
        }
        app = web.Application()
        app.router.add_route("*", "/{session}/{tail:.*}", self.handle)
        self.runner = web.AppRunner(app, handler_cancellation=True)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.bridges, self.bridge_lock = set(), asyncio.Lock()
        return self

    async def __aexit__(self, *exc):
        await self.runner.cleanup()
        await self.resources.__aexit__(*exc)

    async def container_url(self, gateway):
        """Expose the same session routes on a native task's Docker bridge."""
        async with self.bridge_lock:
            if gateway not in self.bridges:
                await web.TCPSite(self.runner, gateway, self.port).start()
                self.bridges.add(gateway)
        return f"http://{gateway}:{self.port}"

    async def finish(self, session):
        """Return row evidence after its remaining HTTP handlers have settled."""
        while self.active[session]:
            await asyncio.gather(*self.active[session], return_exceptions=True)
        return list(self.calls[session])

    @asynccontextmanager
    async def capture(self, session, raw, api="chat", on_event=None):
        server_id, workload = self.sessions[session]
        server = self.servers[server_id]
        call = dict(
            api=api,
            raw_request=raw,
            server={
                "id": server_id,
                **{key: server[key] for key in ("endpoint", "model", "engine")},
                "checkpoint": server.get("checkpoint"),
                "context_window": server.get("context_window", 131072),
                "args": server.get("args", []),
                "env_keys": sorted(server.get("env", {})),
            },
            request=None,
            t_send=perf_counter(),
            t_end=None,
            error=None,
            abandoned=False,
        )
        self.calls[session].append(call)
        reply = {}
        directory = self.output / session
        directory.mkdir(parents=True, exist_ok=True)
        try:
            try:
                call["request"] = {
                    **chat(raw, api),
                    **server["policy"].fixed(workload),
                    **server["policy"].overrides,
                    "model": "served",
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                await self.clients[server_id].complete(
                    call["request"], result=reply, on_event=on_event
                )
            finally:
                call.update(
                    answer=dict(
                        role="assistant",
                        content=reply.get("text", ""),
                        reasoning_content=reply.get("reasoning", ""),
                        tool_calls=reply.get("tool_calls", []),
                    ),
                    usage=reply.get("usage"),
                    finish=reply.get("finish_reason"),
                    t_end=perf_counter(),
                )
                call.update({key: reply[key] for key in ("token_ids", "raw_text") if key in reply})
            if not reply["ok"]:
                raise ValueError("; ".join(reply["errors"]))
            if (reply["usage"] or {}).get("completion_tokens") is None:
                raise ValueError("missing server completion-token usage")
            yield call
        except (asyncio.CancelledError, ConnectionResetError):
            call["abandoned"] = True
            raise
        except Exception as error:
            # Rendering is inside this context so a failed native reply is never exported.
            call["error"] = str(error)
            raise
        finally:
            with (directory / "trace.jsonl").open("a") as trace:
                trace.write(json.dumps(call, ensure_ascii=False) + "\n")

    async def handle(self, request):
        session, path = request.match_info["session"], request.match_info["tail"]
        if session not in self.sessions:
            raise web.HTTPNotFound()
        if request.method == "GET" and path.endswith("models"):
            return web.json_response(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "served",
                            "object": "model",
                            "owned_by": "local",
                            "max_model_len": self.servers[self.sessions[session][0]].get(
                                "context_window", 131072
                            ),
                        }
                    ],
                }
            )
        api = {"chat/completions": "chat", "responses": "responses", "messages": "messages"}.get(
            path.removeprefix("v1/")
        )
        if request.method != "POST" or api is None:
            raise web.HTTPNotFound()
        task = asyncio.current_task()
        self.active[session].add(task)
        response = None
        emitter = None

        async def forward(event):
            nonlocal response
            if response is None:
                response = web.StreamResponse(
                    headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
                )
                await response.prepare(request)
                if emitter:
                    await response.write(emitter.start())
            await response.write(emitter.chunk(event) if emitter else sse(None, event))

        try:
            raw = await request.json()
            emitter = Emitter(api, raw) if api != "chat" else None
            streaming = raw.get("stream", False)
            async with self.capture(session, raw, api, forward if streaming else None) as call:
                if streaming:
                    await response.write(
                        emitter.finish(call["answer"], call["usage"], call["finish"])
                        if emitter
                        else b"data: [DONE]\n\n"
                    )
                    await response.write_eof()
                    return response
                if emitter:
                    return web.json_response(
                        emitter.reply(call["answer"], call["usage"], call["finish"])
                    )
                return web.json_response(
                    {
                        "id": "chatcmpl_" + uuid4().hex,
                        "object": "chat.completion",
                        "model": "served",
                        "choices": [
                            {"index": 0, "message": call["answer"], "finish_reason": call["finish"]}
                        ],
                        "usage": call["usage"],
                    }
                )
        except (httpx.HTTPError, ValueError, TimeoutError) as error:
            if response is not None:
                await response.write(
                    emitter.failure(str(error))
                    if emitter
                    else sse(None, {"error": {"type": "api_error", "message": str(error)}})
                )
                return response
            return web.json_response({"error": {"message": str(error)}}, status=502)
        finally:
            self.active[session].discard(task)
