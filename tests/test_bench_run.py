"""Two real-recording measurement cases and two failed-measurement cases."""

import asyncio
import json
import os
import shlex
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

import pyarrow.parquet as pq
import pytest
from http_server import event, headers, server

from eval.baselines import mapping
from eval.bench import run
from eval.bench.metrics import summarize
from eval.bench.replay import ReplayError
from eval.bench.serve import Server
from eval.request_policy import request_fields


def rows(pool):
    source = Path(
        os.environ.get("LOCAL_EVAL_DATA", Path(__file__).resolve().parents[1] / "data/dataset")
    )
    rows = pq.read_table(source / "workloads/Qwen3.6-35B-A3B.parquet").to_pylist()
    return [
        {
            **row,
            "requests": [{**request, "tool": 0} for request in row["requests"]],
            "source": "fixture",
            "item": row["id"],
            "evaluation": ('{"method":"math_verify","answer":"1"}' if pool == "S" else None),
        }
        for row in rows
        if any(
            w in row["public"]
            for w in {"S": ("S-1", "S-N"), "A": ("A-1", "A-N"), "A-MA": ("A-MA",)}[pool]
        )
    ]


async def reply(writer, *, count, text="scripted response"):
    await headers(writer)
    writer.write(event({"content": text}))
    await writer.drain()
    await asyncio.sleep(0.002)
    writer.write(event(finish="length") + event(count=count, cached=2) + b"data: [DONE]\n\n")
    await writer.drain()


def test_supplied_rows_reach_http_unchanged_for_every_workload():
    @asynccontextmanager
    async def serving(endpoint):
        async with Server(None, endpoint, health_timeout=2) as running:
            yield running

    async def scenario():
        for workload, pool in (
            ("S-1", "S"),
            ("S-N", "S"),
            ("A-1", "A"),
            ("A-N", "A"),
            ("A-MA", "A-MA"),
        ):
            selected = (
                rows(pool)[:2]
                if pool == "S"
                else [min(rows(pool), key=lambda x: len(x["requests"]))]
            )
            if workload in ("S-N", "A-N"):
                selected = [dict(selected[i % len(selected)], id=f"sample-{i}") for i in range(8)]
            policy = request_fields("Qwen3.6-35B-A3B")

            async def respond(reader, writer, body, workload=workload):
                await reply(writer, count=3 if workload.startswith("S") else body["max_tokens"])

            async with server(respond) as (endpoint, received):
                opened = (
                    serving(endpoint)
                    if workload in ("S-N", "A-1")
                    else Server(None, endpoint, health_timeout=2)
                )
                report = await run.measure(selected, workload, policy, server=opened)
            recorded = [request for row in selected for request in row["requests"]]
            assert report["valid"] and report["requests"] == len(recorded)
            assert ("ttft_mean" in report) == (workload not in ("S-N", "A-N"))
            assert not any(key.startswith("completion_") for key in report)
            assert len(received) == len(recorded) + 1
            assert received[0]["messages"] == run.WARMUP["messages"]
            assert received[0][policy.max_tokens] == 16
            expected = [
                policy.request(
                    json.loads(request["messages"]),
                    workload=workload,
                    output_limit=request["out"],
                    tools=json.loads(request["tools"]) if request["tools"] is not None else None,
                )
                for request in recorded
            ]
            assert sorted(
                map(
                    json.dumps,
                    received[1:],
                )
            ) == sorted(map(json.dumps, expected))
            assert report["output_tokens"] == sum(
                record["output_tokens"] for record in report["records"]
            )
            assert report["warmup"]["output_tokens"] <= 16
            assert {
                key: report[key] for key in summarize(workload, report["records"])
            } == summarize(workload, report["records"])

    asyncio.run(scenario())


@pytest.mark.parametrize("workload,n,count", [("S-1", 1, 3), ("S-N", 8, 16)])
def test_resume_opens_once_and_restarts_only_the_unfinished_unit(tmp_path, workload, n, count):
    async def scenario():
        selected = [
            {
                **rows("S")[0],
                "id": str(i),
                "requests": [
                    {
                        **rows("S")[0]["requests"][0],
                        "messages": json.dumps([{"role": "user", "content": str(i)}]),
                    }
                ],
            }
            for i in range(count)
        ]
        calls, lifecycle = [], []
        blocked, release = asyncio.Event(), asyncio.Event()

        async def respond(_reader, writer, body):
            if body["messages"] == run.WARMUP["messages"]:
                calls.append("warmup")
                return await reply(writer, count=3)
            key = int(body["messages"][-1]["content"])
            calls.append(key)
            if n <= key < 2 * n and not release.is_set():
                if sum(isinstance(key, int) and n <= key < 2 * n for key in calls) == n:
                    blocked.set()
                await release.wait()
                return
            await reply(writer, count=3)

        async with server(respond) as (endpoint, _):

            @asynccontextmanager
            async def owned():
                lifecycle.append("start")
                try:
                    async with Server(None, endpoint, health_timeout=2) as running:
                        yield running
                finally:
                    lifecycle.append("stop")

            async def measure():
                return await run.measure(
                    selected,
                    workload,
                    request_fields("Qwen3.6-35B-A3B"),
                    server=owned(),
                    concurrency=n,
                    progress=tmp_path,
                )

            pending = asyncio.create_task(measure())
            await asyncio.wait_for(blocked.wait(), 5)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert (tmp_path / "units/000000.json").exists()
            assert not (tmp_path / "units/000001.json").exists()
            release.set()
            result = await measure()
            assert result["requests"] == count
            assert lifecycle == ["start", "stop", "start", "stop"]
            assert calls.count("warmup") == 2
            assert calls.count(0) == 1
            assert all(calls.count(i) == 1 for i in range(1, n))
            assert all(calls.count(i) == 2 for i in range(n, 2 * n))
            assert all(calls.count(i) == 1 for i in range(2 * n, count))
            before = list(calls)
            assert await measure() == result
            assert calls == before and len(lifecycle) == 4

    asyncio.run(scenario())


def test_cli_launches_mapped_baseline_and_reports_public_rows(tmp_path):
    # A saved subset is already scoped: agents may change its scheduling mode.
    selected = [{**row, "public": ["S-N"]} for row in rows("S")[:2]]
    request = dict(selected[0]["requests"][0])
    messages = json.loads(request["messages"])
    messages[0]["content"] += "\u0085"
    request["messages"] = json.dumps(messages, ensure_ascii=False)
    selected[0] = {**selected[0], "requests": [request]}
    from datasets import Dataset

    source = tmp_path / "data" / "S"
    Dataset.from_list(selected).save_to_disk(str(source))
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"args": ["--workers", "2"], "env": {"TEST_PARALLEL": "2"}}))
    image_root = tmp_path / "images"
    baseline = image_root / "nvidia-cuda-x86" / "baselines" / "ollama"
    baseline.mkdir(parents=True)
    arguments, requests, pid_file = (
        tmp_path / name for name in ("launch-args.json", "requests.jsonl", "server.pid")
    )
    program = baseline / "server.py"
    program.write_text(f"""
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
Path({str(arguments)!r}).write_text(json.dumps({{
    "args": sys.argv[1:],
    "checkpoint": os.environ.get("CHECKPOINT"),
    "port": "8000",
    "alias": os.environ["FIXED_ALIAS"],
    "parallel": os.environ["TEST_PARALLEL"],
}}))
Path({str(pid_file)!r}).write_text(str(os.getpid()))
with Path({str(pid_file.with_suffix(".starts"))!r}).open("a") as started:
    started.write("start\\n")
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{{"data": [{{"id": "served"}}]}}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.calls = getattr(self.server, "calls", 0) + 1
        with Path({str(requests)!r}).open("a") as handle:
            handle.write(json.dumps(body) + "\\n")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write({event({"content": "scripted response"})!r})
        self.wfile.flush()
        time.sleep(0.002)
        self.wfile.write(
            {event(finish="stop") + b"data: [DONE]\n\n"!r}
            if self.server.calls == int(os.environ.get("FAIL_ON", "0"))
            else {event(finish="length") + event(count=3, cached=2) + b"data: [DONE]\n\n"!r}
        )
        self.wfile.flush()
    def log_message(self, *_args):
        pass
HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
""")
    launch = baseline / "launch.sh"
    launch.write_text(
        "export FIXED_ALIAS=served\n"
        f"exec {shlex.quote(sys.executable)} {shlex.quote(str(program))} "
        '--model "$CHECKPOINT" --served-model-name "$FIXED_ALIAS" '
        '--host 127.0.0.1 --port 8000 "$@"\n'
    )

    async def scenario():
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "eval.bench.run",
            "Qwen3.6-35B-A3B",
            "S-1",
            str(launch),
            "--params",
            str(params),
            "--data",
            str(tmp_path / "data/S"),
            cwd=tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
        assert process.returncode == 0, stderr.decode()
        message = stdout.decode().strip().splitlines()[-1]
        assert message.startswith("Measurement report: ")
        output = Path(message.removeprefix("Measurement report: "))
        assert output.name == "report.json" and output.parent.parent == tmp_path / "runs"
        UUID(output.parent.name)
        assert (output.parent / "server.log").is_file()
        report = json.loads(output.read_text())
        assert report["valid"] and report["requests"] == 2 and report["output_tokens"] == 6
        assert [record["row_id"] for record in report["records"]] == [row["id"] for row in selected]
        # A new Python hash seed must not change a baseline launch's resume identity.
        size = requests.stat().st_size
        resumed = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "eval.bench.run",
            "--resume",
            str(output.parent),
            cwd=tmp_path,
            env={**os.environ, "PYTHONHASHSEED": "1"},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, resumed_errors = await asyncio.wait_for(resumed.communicate(), 15)
        assert resumed.returncode == 0, resumed_errors.decode()
        assert requests.stat().st_size == size
        return report

    report = asyncio.run(scenario())
    item = mapping("nvidia-cuda-x86", "Qwen3.6-35B-A3B", "ollama")
    observed = json.loads(arguments.read_text())
    port = int(observed["port"])
    assert port == 8000
    assert report["model"] == item.model
    assert report["baseline"] == "ollama"
    assert report["checkpoint"] == item.checkpoint
    assert report["launch"] == str(launch)
    assert report["parameters"] == json.loads(params.read_text())
    assert observed == {
        "args": [
            "--model",
            item.checkpoint,
            "--served-model-name",
            "served",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            "2",
        ],
        "checkpoint": item.checkpoint,
        "port": str(port),
        "alias": "served",
        "parallel": "2",
    }
    received = [json.loads(line) for line in requests.read_text().splitlines()]
    policy = request_fields("Qwen3.6-35B-A3B", "ollama")
    expected = [
        policy.request(
            json.loads(row["requests"][0]["messages"]),
            workload="S-1",
            output_limit=row["requests"][0]["out"],
        )
        for row in selected
    ]
    assert received == [
        policy.request(run.WARMUP["messages"], workload="S-1", output_limit=16),
        *expected,
    ]
    assert pid_file.with_suffix(".starts").read_text().splitlines() == ["start"]
    assert not Path(f"/proc/{pid_file.read_text()}").exists()

    raw = tmp_path / "raw" / "sources" / "S" / "selected"
    raw.mkdir(parents=True)
    raw.joinpath("row.json").write_text(
        json.dumps(
            {
                **{key: selected[0][key] for key in ("id", "source", "item", "evaluation")},
                "source": "aime2025",
                "pool": "S",
                "split": "public",
                "harness": None,
                "data": {"messages": json.loads(selected[0]["requests"][0]["messages"])},
            }
        )
    )

    async def record_cli():
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "eval.bench.run",
            "Qwen3.6-35B-A3B",
            "S-N",
            str(candidate),
            "--mode",
            "record",
            "--data",
            str(tmp_path / "raw"),
            "--concurrency",
            "2",
            "--params",
            str(params),
            cwd=tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 5)
        assert process.returncode == 0, stderr.decode()
        output = Path(stdout.decode().strip().splitlines()[-1].removeprefix("Recorded rows: "))
        assert output.parent.parent == tmp_path / "runs"
        UUID(output.parent.name)
        assert not (output.parent / "report.json").exists()
        with output.open() as handle:
            captured = [json.loads(line) for line in handle]
        assert len(captured) == 1 and captured[0]["id"] == selected[0]["id"]
        assert len(list(output.parent.glob("*/trace.jsonl"))) == 1
        assert not Path(f"/proc/{pid_file.read_text()}").exists()

    async def quality_cli(launcher):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "eval.bench.run",
            "Qwen3.6-35B-A3B",
            "A-N",
            str(launcher),
            "--mode",
            "quality",
            "--data",
            str(source),
            "--concurrency",
            "-1",
            "--params",
            str(params),
            cwd=tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
        assert process.returncode == 0, stderr.decode()
        output = Path(stdout.decode().strip().splitlines()[-1].removeprefix("Accuracy report: "))
        result = json.loads(output.read_text())
        # The CPU fixture deliberately answers with non-gold text.
        assert result == {"accuracy": 0.0}
        return output

    asyncio.run(quality_cli(launch))
    # A completed candidate supplies its own optimized weights and parameters.
    candidate = tmp_path / "launch.sh"
    candidate.write_text(
        "export FIXED_ALIAS=served\n"
        f"exec {shlex.quote(sys.executable)} {shlex.quote(str(program))} "
        "--model /workspace/weights/optimized --host 127.0.0.1 --port 8000 --workers 4\n"
    )
    asyncio.run(quality_cli(candidate))
    candidate_input = json.loads(arguments.read_text())
    assert candidate_input["checkpoint"] is None
    assert candidate_input["args"] == [
        "--model",
        "/workspace/weights/optimized",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--workers",
        "4",
    ]
    assert not Path(f"/proc/{pid_file.read_text()}").exists()
    asyncio.run(record_cli())

    async def recording_server_map():
        async def respond(reader, writer, body):
            await reply(writer, count=3)

        # Exercise the actual --servers CLI against two independent HTTP peers.
        second_raw = raw.parent / "second"
        second_raw.mkdir()
        document = json.loads(raw.joinpath("row.json").read_text())
        document["id"] = "second"
        second_raw.joinpath("row.json").write_text(json.dumps(document))
        async with server(respond) as (a, bodies_a), server(respond) as (b, bodies_b):
            servers_file = tmp_path / "servers.yaml"
            servers_file.write_text(
                json.dumps(
                    {
                        "servers": {
                            "a": {
                                "endpoint": a,
                                "env": {"CUDA_VISIBLE_DEVICES": "0"},
                                "context_window": 65536,
                            },
                            "b": {
                                "endpoint": b,
                                "env": {"CUDA_VISIBLE_DEVICES": "1"},
                                "context_window": 65536,
                            },
                        }
                    }
                )
            )
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "eval.bench.run",
                "Qwen3.6-35B-A3B",
                "S-N",
                "--mode",
                "record",
                "--servers",
                str(servers_file),
                "--data",
                str(tmp_path / "raw"),
                "--concurrency",
                "-1",
                cwd=tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), 5)
            assert process.returncode == 0, stderr.decode()
            output = Path(stdout.decode().strip().splitlines()[-1].removeprefix("Recorded rows: "))
            captured = [json.loads(line) for line in output.read_text().split("\n") if line]
            assert len(bodies_a) == len(bodies_b) == 1
            assert {json.loads(row["extra"])["server"] for row in captured} == {"a", "b"}
            assert {json.loads(row["extra"])["context_window"] for row in captured} == {65536}
            for trace in output.parent.glob("*/trace.jsonl"):
                assert (
                    json.loads(trace.read_text().split("\n")[0])["server"]["context_window"]
                    == 65536
                )
            assert not (output.parent / "report.json").exists()
        # Attached servers remain owned by the caller through the recording run.

    asyncio.run(recording_server_map())

    async def failed_cli(fail_on):
        # Fail warmup directly, then fail measured replay inside nested TaskGroups.
        params.write_text(json.dumps({"env": {"TEST_PARALLEL": "2", "FAIL_ON": str(fail_on)}}))
        previous = set((tmp_path / "runs").iterdir())
        sent_before = len(requests.read_text().splitlines())
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "eval.bench.run",
            "Qwen3.6-35B-A3B",
            "S-1",
            str(launch),
            "--params",
            str(params),
            "--data",
            str(source),
            cwd=tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
        assert process.returncode != 0
        assert stdout.decode().startswith("Operation folder: ")
        (folder,) = set((tmp_path / "runs").iterdir()) - previous
        failure = json.loads((folder / "failure.json").read_text())
        assert failure["ok"] is False and failure["mode"] == "timing"
        (record,) = failure["records"]
        assert record["request_id"] == ("warmup" if fail_on == 1 else 0)
        assert record["text"] == "scripted response"
        assert record["errors"] == ["missing completion token usage"]
        assert "missing completion token usage" in stderr.decode()
        assert str(folder / "failure.json") in stderr.decode()
        assert not (folder / "report.json").exists()
        assert not Path(f"/proc/{pid_file.read_text()}").exists()
        assert len(requests.read_text().splitlines()) - sent_before == fail_on

    for fail_on in (1, 2):
        asyncio.run(failed_cli(fail_on))


def test_failed_warmup_keeps_context_and_does_not_start_timing():
    selected = rows("S")[:2]

    async def scenario():
        async def respond(reader, writer, body):
            await headers(writer)
            writer.write(
                event({"content": "incomplete"}) + event(finish="stop") + b"data: [DONE]\n\n"
            )
            await writer.drain()

        async with server(respond) as (endpoint, received):
            with pytest.raises(ReplayError) as caught:
                await run.measure(
                    selected,
                    "S-1",
                    request_fields("Qwen3.6-35B-A3B"),
                    server=Server(None, endpoint, health_timeout=2),
                )
            assert len(received) == 1
            assert caught.value.record["request_id"] == "warmup"
            assert "missing completion token usage" in caught.value.record["errors"]
            assert "missing completion token usage" in str(caught.value)

    asyncio.run(scenario())


def test_timed_stream_failure_cancels_siblings_and_returns_no_partial_score():
    selected = rows("S")[:8]

    async def scenario():
        started, disconnected = asyncio.Event(), asyncio.Event()
        calls = 0

        async def respond(reader, writer, body):
            nonlocal calls
            calls += 1
            if calls == 1:
                await reply(writer, count=3, text="warmup only")
            elif body["messages"] == json.loads(selected[0]["requests"][0]["messages"]):
                await started.wait()
                await headers(writer)
                writer.write(b'data: {"error":{"message":"timed failure"}}\n\ndata: [DONE]\n\n')
                await writer.drain()
            else:
                await headers(writer)
                writer.write(event({"content": "blocked sibling"}))
                await writer.drain()
                started.set()
                await reader.read()
                disconnected.set()

        async with server(respond) as (endpoint, received):
            with pytest.raises(ExceptionGroup):
                await run.measure(
                    selected,
                    "S-N",
                    request_fields("Qwen3.6-35B-A3B"),
                    server=Server(None, endpoint, health_timeout=2),
                )
            await asyncio.wait_for(disconnected.wait(), 2)
            assert len(received) == 9

    asyncio.run(scenario())


def test_quality_grades_only_rows_with_an_answer_key():
    graded, timing_only = (
        {**rows("S")[0], "evaluation": '{"method":"math_verify","answer":"1"}'},
        {**rows("S")[1], "evaluation": None},
    )
    seen = []

    async def scenario():
        async def respond(_reader, writer, body):
            seen.append(body["messages"][-1]["content"])
            await reply(writer, count=2, text="\\boxed{1}")

        async with server(respond) as (endpoint, _):
            result = await run.quality(
                [graded, timing_only],
                request_fields("Qwen3.6-35B-A3B"),
                server=Server(None, endpoint),
            )
        return result

    result = asyncio.run(scenario())
    assert set(result["items"]) == {graded["id"]} and len(seen) == 1
    with pytest.raises(ValueError, match="answer key"):
        asyncio.run(
            run.quality(
                [timing_only], request_fields("Qwen3.6-35B-A3B"), server=Server(None, "http://x")
            )
        )


def test_cold_warmup_timeout_identifies_startup_and_never_runs_measured_rows(monkeypatch):
    from eval.bench.serve import StartupError

    monkeypatch.setitem(run.WARMUP, "timeout_seconds", 0.05)

    async def scenario():
        async def respond(reader, writer, body):
            await reader.read()

        async with server(respond) as (endpoint, received):
            with pytest.raises(StartupError, match="warmup did not finish within 0.05s"):
                await run.measure(
                    rows("S")[:2],
                    "S-1",
                    request_fields("Qwen3.6-35B-A3B"),
                    server=Server(None, endpoint, health_timeout=2),
                )
            assert len(received) == 1
            assert received[0]["messages"] == run.WARMUP["messages"]

    asyncio.run(scenario())
