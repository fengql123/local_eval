"""Two real-recording measurement cases and two failed-measurement cases."""

import asyncio
import json
import os
import shlex
import sys
from pathlib import Path
from uuid import UUID

import pyarrow.parquet as pq
import pytest
from http_server import event, headers, server

from eval.baselines import mapping
from eval.bench import run
from eval.bench.metrics import summarize
from eval.bench.replay import ReplayError, replay
from eval.bench.serve import Server
from eval.request_policy import request_fields


def rows(pool):
    default = Path(__file__).resolve().parents[2] / "edge-harness-v2" / "local_eval_data"
    source = Path(os.environ.get("LOCAL_EVAL_DATA", default))
    return [
        row
        for row in pq.read_table(source / pool / "public-00000-of-00001.parquet").to_pylist()
        if row["model"] == "C2"
    ]


async def reply(writer, *, count, text="scripted response"):
    await headers(writer)
    writer.write(event({"content": text}))
    await writer.drain()
    await asyncio.sleep(0.002)
    writer.write(event(finish="length") + event(count=count, cached=2) + b"data: [DONE]\n\n")
    await writer.drain()


def test_supplied_rows_reach_http_unchanged_for_every_workload(monkeypatch):
    # Scheduling tests check real delays separately. Avoid minutes of recorded
    # idle time here while retaining the actual graph and HTTP request bodies.
    async def no_idle(_seconds):
        await asyncio.sleep(0)

    async def quick_replay(selected, workload, client, *, concurrency=None):
        return await replay(selected, workload, client, concurrency=concurrency, sleep=no_idle)

    monkeypatch.setattr(run, "replay", quick_replay)

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
            policy = request_fields("C2")

            async def respond(reader, writer, body, workload=workload):
                await reply(writer, count=3 if workload.startswith("S") else body["max_tokens"])

            async with server(respond) as (endpoint, received):
                report = await run.measure(
                    selected, workload, policy, server=Server(None, endpoint, health_timeout=2)
                )
            recorded = [request for row in selected for request in row["requests"]]
            assert report["valid"] and report["requests"] == len(recorded)
            assert len(received) == len(recorded) + 1
            assert received[0]["messages"] == json.loads(recorded[0]["messages"])
            expected = [
                policy.request(
                    json.loads(request["messages"]),
                    workload=workload,
                    output_limit=request["out"],
                    tools=json.loads(request["tools"]) if request["tools"] is not None else None,
                )
                for request in recorded
            ]
            assert sorted(map(json.dumps, received[1:])) == sorted(map(json.dumps, expected))
            assert report["output_tokens"] == sum(
                record["output_tokens"] for record in report["records"]
            )
            assert report["warmup"]["row_id"] == selected[0]["id"]
            assert {
                key: report[key] for key in summarize(workload, report["records"])
            } == summarize(workload, report["records"])

    asyncio.run(scenario())


def test_cli_launches_mapped_baseline_and_reports_public_rows(tmp_path):
    selected = rows("S")[:2]
    source = tmp_path / "data" / "S" / "rows.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("".join(json.dumps(row) + "\n" for row in selected))
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
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{{"data": [{{"id": "served"}}]}}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with Path({str(requests)!r}).open("a") as handle:
            handle.write(json.dumps(body) + "\\n")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write({event({"content": "scripted response"})!r})
        self.wfile.flush()
        time.sleep(0.002)
        self.wfile.write({event(finish="length") + event(count=3, cached=2) + b"data: [DONE]\n\n"!r})
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
            "C2",
            "S-1",
            str(launch),
            "--params",
            str(params),
            cwd=tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 5)
        assert process.returncode == 0, stderr.decode()
        message = stdout.decode().strip()
        assert message.startswith("Measurement report: ")
        output = Path(message.removeprefix("Measurement report: "))
        assert output.name == "report.json" and output.parent.parent == tmp_path / "bench"
        UUID(output.parent.name)
        assert (output.parent / "server.log").is_file()
        report = json.loads(output.read_text())
        assert report["valid"] and report["requests"] == 2 and report["output_tokens"] == 6
        assert [record["row_id"] for record in report["records"]] == [row["id"] for row in selected]
        return report

    report = asyncio.run(scenario())
    item = mapping("nvidia-cuda-x86", "C2", "ollama")
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
    policy = request_fields("C2", "ollama")
    expected = [
        policy.request(
            json.loads(row["requests"][0]["messages"]),
            workload="S-1",
            output_limit=row["requests"][0]["out"],
        )
        for row in selected
    ]
    assert received == [expected[0], *expected]
    assert not Path(f"/proc/{pid_file.read_text()}").exists()

    async def quality_cli(launcher):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "eval.bench.run",
            "C2",
            "A-N",
            str(launcher),
            "--mode",
            "quality",
            "--concurrency",
            "-1",
            "--params",
            str(params),
            cwd=tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 5)
        assert process.returncode == 0, stderr.decode()
        output = Path(stdout.decode().strip().removeprefix("Accuracy report: "))
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
                    request_fields("C2"),
                    server=Server(None, endpoint, health_timeout=2),
                )
            assert len(received) == 1
            assert caught.value.record["row_id"] == selected[0]["id"]
            assert "missing completion token usage" in caught.value.record["errors"]

    asyncio.run(scenario())


def test_timed_stream_failure_cancels_siblings_and_returns_no_partial_score():
    selected = rows("S")[:2]

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
                    request_fields("C2"),
                    server=Server(None, endpoint, health_timeout=2),
                )
            await asyncio.wait_for(disconnected.wait(), 2)
            assert len(received) == 3

    asyncio.run(scenario())
