"""Run fresh native harness sessions in a fixed recording worker pool."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import signal
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from uuid import uuid4

import httpx

from ..config import OUTPUT_ALLOWANCE
from ..config import model as get_model
from .harnesses import harness, profile, sub_agents
from .protocols import text as _text
from .sources import prepared, source_command


def _turn(message):
    calls = []
    for call in message.get("tool_calls", []):
        function = call["function"]
        arguments = function["arguments"]
        # Truncated tool arguments are legitimate stream evidence, not parseable JSON.
        try:
            arguments = json.dumps(json.loads(arguments), sort_keys=True)
        except ValueError:
            pass
        calls.append((function["name"], arguments))
    return message["role"], _text(message.get("content")).strip(), tuple(calls)


def _words(message):
    text = " ".join(
        [
            _text(message.get("content")),
            message.get("reasoning_content", ""),
            *[
                c["function"]["name"] + " " + c["function"]["arguments"]
                for c in message.get("tool_calls", [])
            ],
        ]
    )
    words = re.findall(r"[A-Za-z0-9]+", re.sub(r'\\u[0-9a-fA-F]{4}|\\[nrt"\\/]', " ", text))
    return {" ".join(words[i : i + 8]) for i in range(len(words) - 7)}


def _delegations(answer):
    """Native Hermes task indices identify even one-word child prompts."""
    prompts = {}
    for tool in answer.get("tool_calls", []):
        function = tool["function"]
        if function["name"] != "delegate_task":
            continue
        try:
            arguments = json.loads(function["arguments"])
        except ValueError:
            continue  # A malformed model tool call cannot establish a branch.
        if not isinstance(arguments, dict):
            continue
        tasks = arguments.get("tasks", [arguments])
        if not isinstance(tasks, list):
            continue
        for index, task in enumerate(tasks):
            goal = task.get("goal") if isinstance(task, dict) else None
            if isinstance(goal, str) and goal.strip():
                prompts[(tool["id"], index)] = goal.strip()
    return prompts


def _session_title(call):
    """Recognize the native Hermes title request, which is outside the task."""
    raw = call.get("raw_request", {})
    output = raw.get("response_format", {})
    return (
        output.get("type") == "json_schema"
        and output.get("json_schema", {}).get("name") == "session_title"
        and not raw.get("tools")
        and not call["request"].get("tools")
        and any(
            message["role"] == "system"
            and _text(message.get("content")).startswith("You name chat sessions.")
            for message in raw.get("messages", [])
        )
    )


def graph(calls):
    """Infer continuation/content dependencies and preserve independent branches."""
    calls = sorted((call for call in calls if not call["abandoned"]), key=lambda c: c["t_send"])
    if not calls:
        raise ValueError("no completed request reached the server")
    for call in calls:
        if call["error"]:
            raise ValueError(call["error"])
        if (call["usage"] or {}).get("completion_tokens") is None:
            raise ValueError("missing server completion-token usage")
    # Native harnesses may update system/developer budgets between turns. Match
    # the conversation, while preserving the full original input for replay.
    histories = [
        [
            (i, _turn(m))
            for i, m in enumerate(c["request"]["messages"])
            if m["role"] not in ("system", "developer")
        ]
        for c in calls
    ]
    turns = [[turn for _, turn in history] for history in histories]
    answers = [_turn(c["answer"]) for c in calls]
    delegations = [_delegations(c["answer"]) for c in calls]
    branches = []
    prefixes, children, ancestors, tips = [None] * len(calls), defaultdict(list), [], set()
    requests, extra, first = [], [], calls[0]["t_send"]
    for index, call in enumerate(calls):
        ended = [i for i in range(index) if calls[i]["t_end"] <= call["t_send"]]
        search = (tips | {prefixes[i] for i in tips}) & set(ended)
        candidates = []
        for parent in search:
            count = len(turns[parent])
            if (
                call["request"].get("tools") != calls[parent]["request"].get("tools")
                or turns[index][:count] != turns[parent]
                or (len(turns[index]) > count and turns[index][count][0] != "assistant")
            ):
                continue
            echoed = len(turns[index]) > count and turns[index][count] == answers[parent]
            candidates.append(
                (count + echoed, not children[parent], calls[parent]["t_end"], parent)
            )
        shared, prefix = (max(candidates)[0], max(candidates)[3]) if candidates else (0, None)
        parents = {prefix} if prefix is not None else set()
        offset = histories[index][shared - 1][0] + 1 if shared else 0
        new = set().union(*[_words(m) for m in call["request"]["messages"][offset:]])
        content = {i for i in search if len(_words(calls[i]["answer"]) & new) >= 3}
        goals = {
            _text(m.get("content")).strip()
            for m in call["request"]["messages"]
            if m["role"] == "user"
        }
        delegated = {
            i: {key for key, goal in delegations[i].items() if goal in goals} for i in search
        }
        content.update(i for i, matches in delegated.items() if matches)
        branches.append(
            branches[prefix] if prefix is not None else set().union(*delegated.values())
        )
        known = set().union(*branches)
        returned = set()
        for message in call["request"]["messages"][offset:]:
            tool_id = message.get("tool_call_id")
            if message["role"] == "tool" and any(key[0] == tool_id for key in known):
                result, _ = json.JSONDecoder().raw_decode(_text(message.get("content")))
                returned.update(
                    (tool_id, row["task_index"])
                    for row in result.get("results", [])
                    if row["status"] == "completed" and row.get("api_calls", 0) > 0
                )
        content.update(i for i in ended if branches[i] & returned)
        parents |= content
        if ended and (prefix is None or children[prefix]):
            parents.add(max(ended, key=lambda i: calls[i]["t_end"]))
        kind = (
            "root"
            if not ended
            else "continuation"
            if prefix is not None
            else "start"
            if content
            else "time-only"
        )
        if kind == "time-only":
            if not all(_session_title(remaining) for remaining in calls[index:]):
                raise ValueError(
                    f"cannot link task request {index} of {len(calls)}; full trace retained"
                )
            return requests, {
                "nodes": extra,
                "cut": {"at": index, "of": len(calls)},
                "inferred_starts": sum(n["kind"] == "start" for n in extra),
            }
        parents = {p for p in parents if not any(p in ancestors[o] for o in parents if o != p)}
        ancestors.append(parents | set().union(*(ancestors[p] for p in parents)))
        if prefix is not None:
            children[prefix].append(index)
            prefixes[index] = prefix
            tips.discard(prefix)
        tips.add(index)
        start = max(calls[p]["t_end"] for p in parents) if parents else first
        requests.append(
            {
                "id": index,
                "parents": sorted(parents),
                "tool": round(call["t_send"] - start, 4),
                "out": call["usage"]["completion_tokens"],
                "messages": json.dumps(call["request"]["messages"], ensure_ascii=False),
                "tools": json.dumps(call["request"]["tools"], ensure_ascii=False)
                if call["request"].get("tools")
                else None,
            }
        )
        extra.append(
            {
                "kind": kind,
                "send": round(call["t_send"] - first, 4),
                "end": round(call["t_end"] - first, 4),
            }
        )
    return requests, {
        "nodes": extra,
        "cut": None,
        "inferred_starts": sum(n["kind"] == "start" for n in extra),
    }


@asynccontextmanager
async def _process(command, workspace, environment, log):
    """Own the whole process group, including tools and harness child processes."""
    with log.open("wb") as stream:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=workspace,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=stream,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            yield process
        finally:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), 3)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            # The parent may have exited while a tool process remains in its group.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)


async def live(
    items: list[dict],
    model: str,
    workload: str,
    proxy,
    output: Path,
    concurrency: int = 1,
) -> list[dict]:
    """Run n worker threads; each row gets a new harness history and proxy session.

    The caller owns the proxy's event loop. Threads own their subprocess and
    source environments; completed rows are written on the caller's loop.
    """
    if concurrency < 1:
        raise ValueError("live concurrency must be positive; run.py resolves -1")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = queue.Queue()
    recorded = output / "rows.jsonl"
    completed = (
        {r["id"]: r for r in (json.loads(line) for line in recorded.read_text().splitlines())}
        if recorded.exists()
        else {}
    )
    servers = list(proxy.servers)
    identities = {
        json.dumps(
            {
                **{key: server.get(key) for key in ("model", "engine", "checkpoint", "args")},
                "context_window": server.get("context_window", 131072),
                "env": {
                    key: value
                    for key, value in server.get("env", {}).items()
                    if key not in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
                },
            },
            sort_keys=True,
        )
        for server in proxy.servers.values()
    }
    for index, item in enumerate(items):
        if item["id"] in completed:
            continue
        server = item.get("server")
        if server is None:
            if len(identities) != 1:
                raise ValueError(
                    "rows must select a server when recording servers are not replicas"
                )
            server = servers[index % len(servers)]
        session = uuid4().hex
        proxy.register(session, server, workload)
        jobs.put((index, item, session, server))

    loop = asyncio.get_running_loop()
    stopped = threading.Event()
    running = []
    lock = threading.Lock()
    rows = [completed.get(item["id"]) for item in items]

    async def on_owner(awaitable):
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(awaitable, loop))

    with recorded.open("a") as destination:

        async def publish(index, row):
            rows[index] = row
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            destination.flush()

        async def capture(index, item, session, server_id):
            directory = output / session
            workspace, config = directory / "workspace", directory / "config"
            workspace.mkdir(parents=True)
            config.mkdir(mode=0o700)
            calls = []
            try:
                async with prepared(item, workspace) as item:
                    endpoint = (
                        (
                            await on_owner(proxy.container_url(item["container_gateway"]))
                            if item.get("container")
                            else proxy.url
                        )
                        + "/"
                        + session
                    )
                    version, exit_code = None, 0
                    if item["pool"] == "S":
                        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                            response = await client.post(
                                endpoint + "/v1/chat/completions",
                                json={"messages": item["messages"], "stream": False},
                            )
                            response.raise_for_status()
                    else:
                        command, environment = harness(
                            item,
                            workspace,
                            config,
                            endpoint,
                            context_window=proxy.servers[server_id].get("context_window", 131072),
                        )
                        prefix = []
                        if item.get("container"):
                            local_env = [
                                part
                                for key, value in environment.items()
                                if os.environ.get(key) != value
                                for part in ("--env", f"{key}={value}")
                            ]
                            prefix = [
                                "docker",
                                "exec",
                                "--user",
                                item["container_user"],
                                "--workdir",
                                item["container_workspace"],
                                *local_env,
                                item["container"],
                            ]
                        async with asyncio.timeout(item.get("timeout", 3600)):
                            if not item.get("command"):
                                version = (
                                    await source_command(
                                        [
                                            *prefix,
                                            *profile(item["harness"], item["pool"])["version_argv"],
                                        ],
                                        cwd=workspace,
                                        env=environment,
                                    )
                                ).strip()
                            async with _process(
                                [*prefix, *command],
                                workspace,
                                environment,
                                directory / "harness.log",
                            ) as process:
                                exit_code = await process.wait()
                                if exit_code:
                                    raise RuntimeError(
                                        f"{item['harness']} exited with status {exit_code}"
                                    )
                    calls = await on_owner(proxy.finish(session))
                    requests, extra = graph(calls)
                    if item["pool"] == "S":
                        requests[0]["out"] = OUTPUT_ALLOWANCE
                        children = 0
                    else:
                        children = sub_agents(calls, item["harness"], directory / "harness.log")
                    if item["pool"] == "A-MA" and children < 2:
                        raise ValueError("A-MA requires at least two recorded sub-agents")
                    server = proxy.servers[server_id]
                    row = {key: item[key] for key in ("id", "pool", "source", "item", "split")}
                    evaluation = item.get("evaluation") if item["pool"] == "S" else None
                    row.update(
                        model=get_model(server.get("model", model)).name,
                        requests=requests,
                        evaluation=json.loads(evaluation)
                        if isinstance(evaluation, str)
                        else evaluation,
                        harness=None
                        if item["pool"] == "S"
                        else json.dumps({"name": item["harness"], "version": version}),
                        extra=json.dumps(
                            {
                                **extra,
                                "session": session,
                                "server": server_id,
                                "engine": server["engine"],
                                "checkpoint": server.get("checkpoint"),
                                "context_window": server.get("context_window", 131072),
                                "sub_agents": children,
                                "exit_code": exit_code,
                                "abandoned": sum(call["abandoned"] for call in calls),
                            }
                        ),
                    )
                    await on_owner(publish(index, row))
            except Exception as error:
                (directory / "failure.json").write_text(
                    json.dumps({"id": item["id"], "error": str(error)}) + "\n"
                )
            finally:
                await on_owner(proxy.finish(session))

        async def worker():
            with lock:
                running.append((asyncio.get_running_loop(), asyncio.current_task()))
            while not stopped.is_set():
                try:
                    job = jobs.get_nowait()
                except queue.Empty:
                    return
                await capture(*job)

        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="live") as pool:
            ready = threading.Barrier(concurrency)

            def start_worker():
                ready.wait()
                asyncio.run(worker())

            futures = [loop.run_in_executor(pool, start_worker) for _ in range(concurrency)]
            completed = asyncio.gather(*futures, return_exceptions=True)
            try:
                await asyncio.shield(completed)
            except asyncio.CancelledError:
                stopped.set()
                with lock:
                    for owner, task in running:
                        if not task.done():
                            owner.call_soon_threadsafe(task.cancel)
                await completed
                raise
    successful = [row for row in rows if row is not None]
    if len(successful) != len(items):
        raise RuntimeError(
            f"{len(items) - len(successful)} recording sessions failed; "
            f"successful rows and diagnostics saved in {output}"
        )
    return successful
