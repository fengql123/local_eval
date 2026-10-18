"""Two capture/graph cases and two failure/cleanup cases."""

import asyncio
import json
import os
import runpy
import sqlite3
import sys
import threading
import tomllib
from pathlib import Path

import httpx
import pytest
from http_server import event, headers, server

from eval.bench import sources
from eval.bench.harnesses import harness, sub_agents
from eval.bench.live import graph, live
from eval.bench.protocols import Emitter, chat
from eval.bench.proxy import Proxy
from eval.bench.sources import raw_items
from eval.config import OUTPUT_ALLOWANCE
from eval.request_policy import request_fields


def item(name, command, server="a"):
    return dict(
        id=name,
        pool="A",
        source="test",
        item=name,
        split="public",
        harness="test",
        command=command,
        server=server,
    )


def servers(endpoint, second=None):
    return {
        name: dict(
            endpoint=url,
            policy=request_fields("Qwen3.6-35B-A3B", "vllm"),
            model="Qwen3.6-35B-A3B",
            engine="vllm",
        )
        for name, url in (("a", endpoint), ("b", second))
        if url is not None
    }


def test_concurrent_processes_capture_tool_sessions_without_forced_lengths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def source_images():
        task = tmp_path / "native-source"
        environment = task / "environment"
        environment.mkdir(parents=True)
        (environment / "Dockerfile").write_text("FROM original-source\n")
        commands = []
        cached = False

        async def command(argv):
            commands.append(argv)
            return "built" if cached or argv[1] == "build" else ""

        with monkeypatch.context() as patch:
            patch.setattr(sources, "source_command", command)
            patch.setattr(sources, "_BUILD_LOCK", tmp_path / "build.lock")
            (task / "task.toml").write_text(
                '[environment]\ndocker_image = "original/fix-git:tag"\n'
            )
            assert await sources._source_image(task, "tbench2", "fix-git") == "original/fix-git:tag"
            assert commands == []
            (task / "task.toml").write_text("[environment]\n")
            built = await sources._source_image(task, "tbench2", "fix-git")
            assert commands == [
                ["docker", "image", "ls", "--quiet", built],
                ["docker", "image", "ls", "--quiet", built],
                ["docker", "build", "--quiet", "--tag", built, str(environment)],
            ]
            cached = True
            commands.clear()
            assert await sources._source_image(task, "tbench2", "fix-git") == built
            assert commands == [["docker", "image", "ls", "--quiet", built]]
            (environment / "fixture.txt").write_text("changed source fixture")
            assert await sources._source_image(task, "tbench2", "fix-git") != built

    asyncio.run(source_images())
    raw = tmp_path / "raw"
    task = raw / "tasks" / "hbf" / "sample"
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("Implement the sample in /app.")
    for pool, source, name, selected_harness, data in (
        ("S", "aime2025", "single", None, {"messages": [{"role": "user", "content": "one"}]}),
        ("A-MA", "hbf", "multi", {"name": "codex"}, {"task": "tasks/hbf/sample"}),
    ):
        directory = raw / "sources" / pool / name
        directory.mkdir(parents=True)
        (directory / "row.json").write_text(
            json.dumps(
                dict(
                    id=name,
                    pool=pool,
                    source=source,
                    item=name,
                    split="private",
                    harness=selected_harness,
                    evaluation=None,
                    data=data,
                )
            )
        )
    assert raw_items("Qwen3.6-35B-A3B", "S-N", Path("raw"))[0]["messages"][0]["content"] == "one"
    selected = raw_items("Qwen3.6-35B-A3B", "A-MA", Path("raw"), harness="codex")
    assert selected[0]["id"] == "multi" and selected[0]["split"] == "private"
    assert "calibration" not in selected[0]
    assert selected[0]["task"] == task and callable(selected[0]["setup"])
    (tmp_path / "workspace").mkdir()
    (tmp_path / "config").mkdir()
    command, _ = harness(
        {**selected[0], "pool": "A", "container": "owned-task", "container_workspace": "/testbed"},
        tmp_path / "workspace",
        tmp_path / "config",
        "http://127.0.0.1:8000/session",
        context_window=65536,
    )
    assert command[command.index("-C") + 1] == "/testbed"
    assert "--dangerously-bypass-approvals-and-sandbox" in command
    assert "--sandbox" not in command
    host_command, _ = harness(
        {**selected[0], "pool": "A"},
        tmp_path / "workspace",
        tmp_path / "config",
        "http://127.0.0.1:8000/session",
        context_window=65536,
    )
    assert host_command[0] == "bwrap"
    assert host_command.index("--dangerously-bypass-approvals-and-sandbox") > host_command.index(
        "--"
    )
    assert "--sandbox" not in host_command
    config = tomllib.loads((tmp_path / "config" / "config.toml").read_text())
    assert config["features"]["view_image"] is False
    assert config["model_context_window"] == 65536
    assert config["model_auto_compact_token_limit"] == 32768
    harness(
        {**selected[0], "harness": "openclaw"},
        tmp_path / "workspace",
        tmp_path / "openclaw-config",
        "http://127.0.0.1:8000/session",
        context_window=65536,
    )
    openclaw = json.loads((tmp_path / "openclaw-config" / "openclaw.json").read_text())
    model = openclaw["models"]["providers"]["record"]["models"][0]
    assert model["contextTokens"] + model["maxTokens"] == model["contextWindow"] == 65536
    assert openclaw["agents"]["defaults"]["compaction"]["midTurnPrecheck"]["enabled"] is True
    assert openclaw["agents"]["defaults"]["workspace"] == str(tmp_path / "openclaw-config")
    assert openclaw["agents"]["defaults"]["cwd"] == str(tmp_path / "workspace")
    bootstrap = runpy.run_path(str(tmp_path / "openclaw-config" / "openclaw-multi.py"))
    succeeded, pending = bootstrap["succeeded"], bootstrap["pending"]
    execution = {"status": "terminal", "startedAt": 1, "endedAt": 2, "outcome": {"status": "ok"}}
    delivered = {
        "runId": "delivered",
        "childSessionKey": "child-1",
        "execution": execution,
        "delivery": {"status": "delivered"},
        "cleanupCompletedAt": 3,
    }
    suspended = {
        "runId": "suspended",
        "taskRunId": "task-2",
        "childSessionKey": "child-2",
        "execution": execution,
        "delivery": {
            "status": "suspended",
            "disposition": "permanent_failure",
            "suspendedAt": 3,
            "attemptCount": 10,
            "lastError": "sink unavailable",
        },
    }
    state = tmp_path / "openclaw-config" / "state"
    state.mkdir()
    with sqlite3.connect(state / "openclaw.sqlite") as connection:
        connection.execute("CREATE TABLE subagent_runs (run_id TEXT, payload_json TEXT)")
        connection.execute("CREATE TABLE task_runs (run_id TEXT, runtime TEXT, status TEXT)")
        connection.executemany(
            "INSERT INTO subagent_runs VALUES (?, ?)",
            [(r["runId"], json.dumps(r)) for r in (delivered, suspended)],
        )
        connection.executemany(
            "INSERT INTO task_runs VALUES (?, 'subagent', 'succeeded')",
            [("delivered",), ("task-2",)],
        )
    entries = bootstrap["children"]()
    assert len(entries) == 2 and all(succeeded(entry) and not pending([entry]) for entry in entries)
    assert entries[1]["delivery"] == suspended["delivery"]
    assert "cleanupCompletedAt" not in entries[1] and not pending(entries)
    for status in ("pending", "in_progress"):
        for entry in entries:
            assert pending([{**entry, "requesterSettleWake": {"status": status}}])
    for status in ("running", "interrupted"):
        unfinished = {**entries[1], "execution": {**execution, "status": status}}
        assert pending([unfinished]) and not succeeded(unfinished)
    for unfinished in (
        {**entries[1], "execution": {**execution, "endedAt": None}},
        {**entries[1], "execution": {**execution, "outcome": {"status": "error"}}},
        {**entries[1], "task_status": "failed"},
    ):
        assert pending([unfinished]) and not succeeded(unfinished)
    original = tmp_path / "workspace" / "AGENTS.md"
    existing = b"# Source instructions\nKeep this file intact.\n"
    original.write_bytes(existing)
    bootstrap["prepare_instructions"]()
    configured = (tmp_path / "openclaw-config" / "AGENTS.md").read_bytes()
    assert original.read_bytes() == existing and configured.startswith(existing)
    assert b"main session only" in configured and b"two distinct sub-agents" in configured
    original.unlink()
    bootstrap["prepare_instructions"]()
    assert not original.exists()
    assert (
        "Spawned sub-agents complete their assigned subtask"
        in (tmp_path / "openclaw-config" / "AGENTS.md").read_text()
    )
    for pool, required in (
        ("A", "without creating sub-agents"),
        ("A-MA", "two distinct sub-agents"),
    ):
        claude, _ = harness(
            {**selected[0], "harness": "claude-code", "pool": pool, "container": "owned-task"},
            tmp_path / "workspace",
            tmp_path / "claude-config",
            "http://127.0.0.1:8000/session",
        )
        assert required in claude[claude.index("--append-system-prompt") + 1]
        assert claude[-1].startswith(selected[0]["prompt"])
        codex, _ = harness(
            {**selected[0], "harness": "codex", "pool": pool, "container": "owned-task"},
            tmp_path / "workspace",
            tmp_path / "codex-config",
            "http://127.0.0.1:8000/session",
        )
        config = tomllib.loads((tmp_path / "codex-config" / "config.toml").read_text())
        assert required in config["developer_instructions"]
        assert ("one local model only" in config["developer_instructions"]) is (pool == "A-MA")
        assert config["features"]["multi_agent"] is (pool == "A-MA")
        assert codex[-1] == "-"
        assert selected[0]["prompt"] not in codex
        assert (tmp_path / "codex-config" / "prompt.txt").read_text() == (
            selected[0]["prompt"] + config["developer_instructions"]
        )
    program = tmp_path / "harness.py"
    program.write_text(
        """import json, os, sys, time, urllib.request
url = os.environ['RECORD_ENDPOINT'] + '/chat/completions'
messages = [{'role': 'user', 'content': json.dumps({'workspace': os.getcwd(), 'seed': open('seed').read()})}]
tools = [{'type': 'function', 'function': {'name': 'read', 'parameters': {}}}]
for turn in range(2):
    request = urllib.request.Request(url, json.dumps({'messages': messages, 'tools': tools}).encode(), {'Content-Type': 'application/json'})
    reply = json.load(urllib.request.urlopen(request))['choices'][0]['message']
    messages.append(reply)
    if reply['tool_calls']:
        messages.append({'role': 'tool', 'tool_call_id': reply['tool_calls'][0]['id'], 'content': 'read successfully'})
time.sleep(float(sys.argv[1]))
"""
    )

    async def run():
        active, peak = 0, 0
        arrivals, paired = 0, asyncio.Event()
        threads = set()

        def prepared(name):
            async def setup(workspace):
                threads.add(threading.get_ident())
                (workspace / "seed").write_text(name)
                return {}

            return setup

        async def respond(reader, writer, body):
            nonlocal active, peak, arrivals
            active += 1
            peak = max(peak, active)
            arrivals += 1
            if arrivals <= 2:
                if arrivals == 2:
                    paired.set()
                await asyncio.wait_for(paired.wait(), 5)
            await headers(writer)
            if not body.get("tools"):
                writer.write(event({"content": "1"}) + event(finish="stop"))
            elif len(body["messages"]) == 1:
                writer.write(event({"reasoning_content": "inspect"}))
                writer.write(
                    event(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "function": {"name": "re", "arguments": '{"file":'},
                                }
                            ]
                        }
                    )
                )
                writer.write(
                    event(
                        {
                            "tool_calls": [
                                {"index": 0, "function": {"name": "ad", "arguments": '"a.txt"}'}}
                            ]
                        }
                    )
                )
                writer.write(event(finish="tool_calls"))
            else:
                writer.write(event({"content": "finished"}) + event(finish="stop"))
            writer.write(event(count=3) + b"data: [DONE]\n\n")
            await writer.drain()
            active -= 1

        async with (
            server(respond) as (endpoint, bodies),
            server(respond) as (second, second_bodies),
            Proxy(servers(endpoint, second), tmp_path / "record") as proxy,
        ):
            recording = asyncio.create_task(
                live(
                    [
                        {
                            **item("one", [sys.executable, str(program), "0"]),
                            "setup": prepared("one"),
                        },
                        {
                            **item("two", [sys.executable, str(program), "0.3"], "b"),
                            "setup": prepared("two"),
                        },
                        {
                            **item("three", [sys.executable, str(program), "0"]),
                            "setup": prepared("three"),
                        },
                    ],
                    "Qwen3.6-35B-A3B",
                    "A-1",
                    proxy,
                    tmp_path / "record",
                    concurrency=2,
                )
            )
            destination = tmp_path / "record" / "rows.jsonl"
            while not recording.done() and (
                not destination.exists() or not destination.read_text()
            ):
                await asyncio.sleep(0.01)
            assert not recording.done(), "successful rows must be flushed before the batch finishes"
            rows = await recording
            async with Proxy(servers(endpoint), tmp_path / "single") as single_proxy:
                single = await live(
                    [
                        dict(
                            id="single",
                            pool="S",
                            source="test",
                            item="single",
                            split="public",
                            harness=None,
                            server="a",
                            evaluation={"method": "math_verify", "answer": "1"},
                            messages=[{"role": "user", "content": "one"}],
                        )
                    ],
                    "Qwen3.6-35B-A3B",
                    "S-1",
                    single_proxy,
                    Path("single"),
                )
        assert peak == 2
        assert len(threads) == 2
        assert len(rows) == 3
        assert all(len(row["requests"]) == 2 for row in rows)
        assert all(row["requests"][1]["parents"] == [0] for row in rows)
        assert all(request["out"] == 3 for row in rows for request in row["requests"])
        assert len({json.loads(row["requests"][0]["messages"])[0]["content"] for row in rows}) == 3
        assert {
            json.loads(body["messages"][0]["content"])["seed"]
            for body in bodies
            if body.get("tools")
        } == {"one", "three"}
        assert {json.loads(body["messages"][0]["content"])["seed"] for body in second_bodies} == {
            "two"
        }
        assert all(
            "ignore_eos" not in body and body["max_tokens"] == OUTPUT_ALLOWANCE
            for body in bodies + second_bodies
        )
        assert all("accuracy" not in row and "metrics" not in row for row in rows)
        assert len(list((tmp_path / "record").glob("*/trace.jsonl"))) == 3
        assert (tmp_path / "record" / "rows.jsonl").exists()
        assert single[0]["requests"][0]["out"] == OUTPUT_ALLOWANCE
        trace = json.loads(next((tmp_path / "single").glob("*/trace.jsonl")).read_text())
        assert trace["usage"]["completion_tokens"] == 3
        assert trace["answer"]["content"] == "1"

    asyncio.run(run())


def test_protocol_streams_and_fork_join_graph(tmp_path, monkeypatch):
    message = {"role": "user", "content": "task"}
    custom_tools = [
        {
            "type": "namespace",
            "name": "functions",
            "tools": [{"type": "custom", "name": "exec"}],
        }
    ]
    for arguments in ('{"cmd":"read file"}', '{"input":'):
        tool_call = {
            "id": "call_bad",
            "type": "function",
            "function": {"name": "exec", "arguments": arguments},
        }
        answer = {"role": "assistant", "content": None, "tool_calls": [tool_call]}
        emitter = Emitter("responses", {"tools": custom_tools})
        result = emitter.reply(answer, {"completion_tokens": 1}, "tool_calls")
        item = result["output"][0]
        assert item["type"] == "function_call" and item["namespace"] == "functions"
        assert item["arguments"] == arguments and "input" not in item
        following = chat(
            {
                "tools": custom_tools,
                "input": [
                    message,
                    item,
                    {"type": "function_call_output", "call_id": "call_bad", "output": "aborted"},
                ],
            },
            "responses",
        )
        assert following["messages"][1]["tool_calls"] == answer["tool_calls"] == [tool_call]
        assert following["messages"][2] == {
            "role": "tool",
            "tool_call_id": "call_bad",
            "content": "aborted",
        }
    function_tools = [{"type": "function", "name": "read", "parameters": {}}]
    assert chat(
        {"messages": [message], "tools": function_tools, "parallel_tool_calls": True}, "chat"
    )["parallel_tool_calls"]
    assert chat(
        {
            "messages": [message],
            "tools": [{"name": "read", "input_schema": {}}],
            "tool_choice": {"type": "any", "disable_parallel_tool_use": False},
        },
        "messages",
    )["parallel_tool_calls"]
    for api, body in (
        ("responses", {"input": [message], "tools": []}),
        ("chat", {"messages": [message]}),
    ):
        raw = {**body, "tool_choice": "auto", "parallel_tool_calls": False}
        original = json.dumps(raw)
        assert chat(raw, api) == {"messages": [message]}
        assert json.dumps(raw) == original
    with pytest.raises(ValueError, match="unsupported tool choice: allowed_tools"):
        chat(
            {"input": "task", "tools": function_tools, "tool_choice": {"type": "allowed_tools"}},
            "responses",
        )
    assert chat(
        {"input": [{"type": "additional_tools", "role": "developer", "tools": []}, message]},
        "responses",
    ) == {"messages": [message]}
    with pytest.raises(ValueError, match="unique tool names"):
        chat(
            {
                "tools": [{"type": "function", "name": "read"}],
                "input": [
                    {
                        "type": "additional_tools",
                        "role": "developer",
                        "tools": [
                            {
                                "type": "namespace",
                                "name": "other",
                                "tools": [{"type": "function", "name": "read"}],
                            }
                        ],
                    },
                    message,
                ],
            },
            "responses",
        )

    async def run():
        async def respond(reader, writer, body):
            await headers(writer)
            if body["messages"][-1]["content"] == "no tools":
                assert not {"tools", "tool_choice", "parallel_tool_calls"} & body.keys()
                writer.write(event({"content": "done"}) + event(finish="stop") + event(count=1))
                writer.write(b"data: [DONE]\n\n")
                await writer.drain()
                return
            function = body["tools"][0]["function"]
            name = "write" if body["messages"][-1]["content"] == "unknown" else function["name"]
            arguments = '{"input":"plain input"}' if function["name"] == "patch" else "{}"
            middle = len(arguments) // 2
            writer.write(event({"reasoning_content": "consider"}) + event({"content": "hello"}))
            writer.write(
                event(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_",
                                "function": {
                                    "name": name[:1],
                                    "arguments": arguments[:middle],
                                },
                            }
                        ]
                    }
                )
            )
            writer.write(
                event(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "a",
                                "function": {"name": name[1:], "arguments": arguments[middle:]},
                            }
                        ]
                    }
                )
            )
            writer.write(event(finish="tool_calls") + event(count=5) + b"data: [DONE]\n\n")
            await writer.drain()

        async with server(respond) as (endpoint, bodies):
            async with Proxy(servers(endpoint), tmp_path) as proxy:
                for session in ("claude", "codex", "patch", "unknown", "no-tools"):
                    proxy.register(session, "a", "A-MA")
                bridges = await asyncio.gather(
                    proxy.container_url("127.0.0.2"), proxy.container_url("127.0.0.2")
                )
                assert bridges[0] == bridges[1]
                emit_finish = Emitter.finish

                def completed(emitter, *args):
                    # A consumer can start its next request as soon as this frame arrives.
                    assert all(
                        call["t_end"] is not None
                        for calls in proxy.calls.values()
                        for call in calls
                    )
                    return emit_finish(emitter, *args)

                monkeypatch.setattr(Emitter, "finish", completed)
                async with httpx.AsyncClient() as client:
                    anthropic = await client.post(
                        proxy.url + "/claude/v1/messages",
                        json={
                            "system": [{"type": "text", "text": "system"}],
                            "messages": [{"role": "user", "content": "task"}],
                            "tools": [
                                {"name": "read", "input_schema": {}},
                                {"name": "inspect", "input_schema": {}},
                            ],
                            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
                            "stream": True,
                        },
                    )
                    responses = await client.post(
                        proxy.url + "/codex/v1/responses",
                        json={
                            "instructions": "system",
                            "input": [
                                {
                                    "type": "additional_tools",
                                    "role": "developer",
                                    "tools": [
                                        {"type": "function", "name": "inspect", "parameters": {}}
                                    ],
                                },
                                message,
                            ],
                            "tool_choice": "auto",
                            "parallel_tool_calls": False,
                            "tools": [
                                {
                                    "type": "namespace",
                                    "name": "functions",
                                    "tools": [
                                        {"type": "function", "name": "read", "parameters": {}}
                                    ],
                                }
                            ],
                            "stream": True,
                        },
                    )
                    custom = await client.post(
                        proxy.url + "/patch/v1/responses",
                        json={
                            "input": [
                                {
                                    "type": "additional_tools",
                                    "id": "at_child",
                                    "role": "developer",
                                    "tools": [
                                        {
                                            "type": "namespace",
                                            "name": "functions",
                                            "tools": [{"type": "custom", "name": "patch"}],
                                        }
                                    ],
                                },
                                message,
                            ],
                            "stream": True,
                        },
                    )
                    unknown = await client.post(
                        proxy.url + "/unknown/v1/responses",
                        json={
                            "input": "unknown",
                            "tools": [{"type": "function", "name": "read", "parameters": {}}],
                            "stream": True,
                        },
                    )
                    raw_empty = {
                        "input": "no tools",
                        "tools": [],
                        "tool_choice": "auto",
                        "parallel_tool_calls": False,
                        "stream": True,
                    }
                    empty = await client.post(proxy.url + "/no-tools/v1/responses", json=raw_empty)
                    assert empty.status_code == 200 and "response.completed" in empty.text
                    assert proxy.calls["no-tools"][0]["raw_request"] == raw_empty
                    assert proxy.calls["no-tools"][0]["error"] is None
                    for original in (anthropic, responses, custom):
                        completed_reply = await client.post(
                            original.request.url,
                            json={**json.loads(original.request.content), "stream": False},
                        )
                        payload = completed_reply.json()
                        blocks = payload["content"] if "content" in payload else payload["output"]
                        assert json.dumps(blocks).count("hello") == 1
                        assert json.dumps(blocks).count("consider") == 1
                        assert ("patch" if original is custom else "read") in json.dumps(blocks)
                assert '"type": "custom_tool_call"' in custom.text
                assert '"input": "plain input"' in custom.text
                assert '"namespace": "functions"' in custom.text
                assert "thinking_delta" in anthropic.text
                assert "tool_use" in anthropic.text
                assert '"output_tokens": 5' in anthropic.text
                assert "response.completed" in responses.text
                assert "response.completed" in unknown.text
                assert '"type": "function_call"' in unknown.text
                assert '"name": "write"' in unknown.text
                assert '"arguments": "{}"' in unknown.text
                assert proxy.calls["unknown"][0]["error"] is None
                assert '"namespace": "functions"' in responses.text
                assert proxy.calls["claude"][0]["request"] == proxy.calls["codex"][0]["request"]
                assert proxy.calls["claude"][0]["answer"]["tool_calls"] == [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ]
                assert all(body["messages"][0]["role"] == "system" for body in bodies[:2])
                assert all(body["parallel_tool_calls"] is False for body in bodies[:2])
                assert [tool["function"]["name"] for tool in bodies[1]["tools"]] == [
                    "read",
                    "inspect",
                ]
                assert bodies[2]["messages"] == [message]
                assert "tools" not in proxy.calls["patch"][0]["raw_request"]
                assert (
                    proxy.calls["patch"][0]["raw_request"]["input"][0]["type"] == "additional_tools"
                )

    asyncio.run(run())
    phrases = [
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo",
        "one two three four five six seven eight nine ten eleven",
        "red blue green yellow purple orange black white silver golden bronze",
    ]

    def call(messages, answer, sent, ended):
        return {
            "request": {"messages": messages},
            "answer": {"role": "assistant", "content": answer},
            "t_send": sent,
            "t_end": ended,
            "usage": {"completion_tokens": 5},
            "error": None,
            "abandoned": False,
        }

    calls = [
        call([{"role": "user", "content": "main task"}], phrases[0], 0, 1),
        call([{"role": "user", "content": "child: " + phrases[0]}], phrases[1], 1.2, 2),
        call([{"role": "user", "content": "other child: " + phrases[0]}], phrases[2], 1.3, 2.1),
        call(
            [{"role": "user", "content": "join " + phrases[1] + " " + phrases[2]}], "done", 2.5, 3
        ),
    ]
    requests, extra = graph(calls)
    assert requests[1]["parents"] == requests[2]["parents"] == [0]
    assert requests[3]["parents"] == [1, 2]
    assert [request["tool"] for request in requests] == [0, 0.2, 0.3, 0.4]
    assert extra["inferred_starts"] == 3
    assert sub_agents(calls, "codex") == 0
    history = [{"role": "user", "content": "inspect task"}]
    first = call([{"role": "system", "content": "budget 100"}, *history], "inspect", 0, 1)
    second = call(
        [
            {"role": "system", "content": "budget 90"},
            *history,
            first["answer"],
            {"role": "developer", "content": "budget 80"},
            {"role": "user", "content": "continue"},
        ],
        "done",
        2,
        3,
    )
    continued, details = graph([first, second])
    assert len(continued) == 2 and details["cut"] is None
    assert continued[1]["parents"] == [0]
    assert json.loads(continued[1]["messages"]) == second["request"]["messages"]
    unrelated = call(
        [{"role": "system", "content": "budget 90"}, {"role": "user", "content": "new task"}],
        "other",
        2,
        3,
    )
    with pytest.raises(ValueError, match="cannot link task request 1 of 2"):
        graph([first, unrelated])
    title_messages = [
        {"role": "system", "content": "You name chat sessions. Give this session a title."},
        {"role": "user", "content": "Opening message."},
    ]
    title = call(title_messages, "Session title", 4, 5)
    title["raw_request"] = {
        "messages": title_messages,
        "response_format": {"type": "json_schema", "json_schema": {"name": "session_title"}},
    }
    kept, details = graph([first, title])
    assert len(kept) == 1 and details["cut"] == {"at": 1, "of": 2}
    resumed = call([{"role": "user", "content": "Continue after compaction."}], "done", 6, 7)
    with pytest.raises(ValueError, match="cannot link task request 1 of 3"):
        graph([first, title, resumed])
    parent = call(history, "", 0, 1)
    parent["answer"]["tool_calls"] = [
        {
            "id": "delegation",
            "function": {
                "name": "delegate_task",
                "arguments": json.dumps(
                    {"tasks": [{"goal": "Return ok."}, {"goal": "Return ok2."}]}
                ),
            },
        }
    ]
    parent["answer"]["tool_calls"].append(
        {
            "id": "malformed",
            "function": {"name": "delegate_task", "arguments": '{"tasks":"invalid"}'},
        }
    )
    short_children = [
        call([{"role": "user", "content": goal}], answer, sent, ended)
        for goal, answer, sent, ended in [
            ("Return ok.", "ok", 1.2, 2),
            ("Return ok2.", "ok2", 1.3, 2.1),
        ]
    ]
    joined = call(
        [
            *history,
            parent["answer"],
            {
                "role": "tool",
                "tool_call_id": "delegation",
                "content": json.dumps(
                    {
                        "results": [
                            {
                                "task_index": i,
                                "status": "completed",
                                "api_calls": 1,
                                "summary": answer,
                            }
                            for i, answer in enumerate(("ok", "ok2"))
                        ]
                    }
                ),
            },
        ],
        "done",
        2.5,
        3,
    )
    short_graph, details = graph([parent, *short_children, joined])
    assert details["cut"] is None
    assert [row["parents"] for row in short_graph] == [[], [0], [0], [1, 2]]
    native = [
        {
            "abandoned": False,
            "request": {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "a", "function": {"name": "spawn_agent"}},
                            {"id": "b", "function": {"name": "spawn_agent"}},
                            {"id": "c", "function": {"name": "spawn_agent"}},
                        ],
                    },
                    {"role": "tool", "tool_call_id": "a", "content": '{"agent_id":"one"}'},
                    {"role": "tool", "tool_call_id": "b", "content": '{"agent_id":"two"}'},
                    {"role": "tool", "tool_call_id": "c", "content": '{"error":"limit reached"}'},
                ]
            },
        }
    ]
    assert sub_agents(native * 2, "codex") == 0
    assert sub_agents(native, "claude-code") == 0
    log = tmp_path / "claude.log"
    log.write_text(
        "\n".join(
            json.dumps(
                {
                    "tool_use_result": {"status": status, "agentId": child},
                    "note": "native\u0085output",
                },
                ensure_ascii=False,
            )
            for status, child in (
                ("completed", "one"),
                ("completed", "two"),
                ("completed", "one"),
                ("failed", "three"),
            )
        )
    )
    assert sub_agents(calls, "claude-code", log) == 2
    delegates = native[0]["request"]["messages"]
    for message in delegates[0]["tool_calls"]:
        message["function"]["name"] = "delegate_task"
    for message in delegates[1:3]:
        message["content"] = '{"results":[{"task_index":0,"status":"completed","api_calls":2}]}'
    delegates[3]["content"] = '{"error":"invalid tasks"}\n\n[Tool loop warning: count=3]'
    assert sub_agents(native * 2, "hermes") == 2


def test_missing_usage_preserves_trace_without_exporting_a_success(tmp_path):
    program = tmp_path / "harness.py"
    program.write_text(
        "import os, urllib.request; urllib.request.urlopen(urllib.request.Request(os.environ['RECORD_ENDPOINT']+'/chat/completions',b'{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}',{'Content-Type':'application/json'})).read(); import sys; sys.exit(int(sys.argv[1]))"
    )

    async def run():
        scenario = "missing"

        async def respond(reader, writer, body):
            await headers(writer)
            if scenario in ("partial", "render"):
                writer.write(
                    event(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "function": {
                                        "name": "patch" if scenario == "render" else "read",
                                        "arguments": "not JSON"
                                        if scenario == "render"
                                        else '{"file":',
                                    },
                                }
                            ]
                        }
                    )
                    + event(finish="tool_calls")
                    + event(count=3)
                    + (b"data: [DONE]\n\n" if scenario == "render" else b"")
                )
            else:
                writer.write(
                    event({"content": "visible text is not a token count"})
                    + event(finish="stop")
                    + (event(count=3) if scenario == "valid" else b"")
                    + b"data: [DONE]\n\n"
                )
            await writer.drain()

        async with server(respond) as (endpoint, _):
            for scenario, folder, exit_code in (
                ("missing", "record", "0"),
                ("partial", "partial", "0"),
                ("valid", "exit", "7"),
            ):
                async with Proxy(servers(endpoint), tmp_path / folder) as proxy:
                    with pytest.raises(RuntimeError, match="1 recording sessions failed"):
                        await live(
                            [item(scenario, [sys.executable, str(program), exit_code])],
                            "Qwen3.6-35B-A3B",
                            "A-1",
                            proxy,
                            tmp_path / folder,
                        )
            scenario = "render"
            async with Proxy(servers(endpoint), tmp_path / "render") as proxy:
                proxy.register("bad", "a", "A-1")
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        proxy.url + "/bad/v1/responses",
                        json={"input": "task", "tools": [{"type": "custom", "name": "patch"}]},
                    )
                assert response.status_code == 200
                native = response.json()["output"][0]
                assert native["type"] == "function_call"
                assert native["name"] == "patch" and native["arguments"] == "not JSON"
                await proxy.finish("bad")
                render = json.loads((tmp_path / "render" / "bad" / "trace.jsonl").read_text())
                assert render["error"] is None
                assert render["usage"]["completion_tokens"] == 3
                assert render["finish"] == "tool_calls"
                assert render["answer"]["tool_calls"][0]["function"] == {
                    "name": "patch",
                    "arguments": "not JSON",
                }
                requests, details = graph(proxy.calls["bad"])
                assert requests[0]["out"] == 3 and details["cut"] is None
        failed = json.loads(next((tmp_path / "exit").glob("*/failure.json")).read_text())
        assert "exited with status 7" in failed["error"]
        assert (tmp_path / "exit" / "rows.jsonl").read_text() == ""
        trace = json.loads(next((tmp_path / "record").glob("*/trace.jsonl")).read_text())
        assert trace["usage"] is None
        assert "missing server completion-token usage" in trace["error"]
        assert len(list((tmp_path / "record").glob("*/failure.json"))) == 1
        assert (tmp_path / "record" / "rows.jsonl").read_text() == ""
        partial = json.loads(next((tmp_path / "partial").glob("*/trace.jsonl")).read_text())
        assert "missing [DONE] stream terminator" in partial["error"]
        assert partial["usage"]["completion_tokens"] == 3
        assert partial["finish"] == "tool_calls"
        assert partial["answer"]["tool_calls"] == [
            {
                "id": "call_a",
                "type": "function",
                "function": {"name": "read", "arguments": '{"file":'},
            }
        ]
        assert len(list((tmp_path / "partial").glob("*/failure.json"))) == 1
        assert (tmp_path / "partial" / "rows.jsonl").read_text() == ""

    asyncio.run(run())


def test_cancelled_recording_stops_the_harness_process_group(tmp_path):
    program = tmp_path / "harness.py"
    child = "import time\nfrom pathlib import Path\nwhile True: Path('heartbeat').write_text(str(time.monotonic())); time.sleep(0.02)"
    program.write_text(
        "import os,subprocess,sys,time\nfrom pathlib import Path\n"
        + f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        + "Path('namespace').write_text(os.readlink('/proc/self/ns/pid'))\ntime.sleep(60)\n"
    )

    async def run():
        workspace = None

        async def cleanup():
            (tmp_path / "source-cleaned").write_text("done")

        async def setup(directory):
            nonlocal workspace
            workspace = directory
            return {"cleanup": cleanup}

        async def respond(reader, writer, body):
            await headers(writer)

        async with (
            server(respond) as (endpoint, _),
            Proxy(servers(endpoint), tmp_path / "record") as proxy,
        ):
            task = asyncio.create_task(
                live(
                    [{**item("cancel", [sys.executable, str(program)]), "setup": setup}],
                    "Qwen3.6-35B-A3B",
                    "A-1",
                    proxy,
                    tmp_path / "record",
                )
            )
            async with asyncio.timeout(10):
                while workspace is None or not (workspace / "heartbeat").is_file():
                    await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert (tmp_path / "source-cleaned").read_text() == "done"
        heartbeat = (workspace / "heartbeat").read_text()
        await asyncio.sleep(0.1)
        assert (workspace / "heartbeat").read_text() == heartbeat
        namespace = (workspace / "namespace").read_text()
        for path in Path("/proc").glob("[0-9]*/ns/pid"):
            try:
                if (path.parent.parent / "cmdline").read_bytes():
                    assert os.readlink(path) != namespace
            except (FileNotFoundError, PermissionError):
                pass  # Other host processes may exit while this test inspects namespaces.

    asyncio.run(run())


@pytest.mark.parametrize(
    "tool,status,error,expected",
    [
        ("wait", "completed", None, 2),
        ("notification", "completed", None, 2),
        ("spawn_agent", "running", None, 0),
        ("wait", "completed", "unsupported Responses input item: additional_tools", 0),
    ],
)
def test_codex_children_need_completion_and_recorded_requests(
    tmp_path, tool, status, error, expected
):
    log = tmp_path / "harness.log"
    item = {
        "type": "collab_tool_call",
        "tool": tool,
        "agents_states": {child: {"status": status} for child in ("one", "two")},
    }
    log.write_text((json.dumps({"item": item}) + "\n") * 2)
    calls = [
        {
            "raw_request": {"client_metadata": {"thread_id": child}},
            "abandoned": False,
            "error": error,
        }
        for child in ("one", "two")
    ]
    if tool == "notification":
        item["agents_states"].pop("two")
        log.write_text(json.dumps({"item": item}) + "\n")
        notice = {"agent_path": "two", "status": {"completed": "Reviewed the implementation"}}
        calls[0]["request"] = {
            "messages": [
                {
                    "role": "user",
                    "content": "<subagent_notification>\n"
                    + json.dumps(notice)
                    + "\n</subagent_notification>",
                }
            ]
        }
    assert sub_agents(calls, "codex", log) == expected
