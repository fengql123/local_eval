"""Server ownership: two common cases and two lifecycle failure cases."""

import asyncio
import json
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from eval.baselines import mapping
from eval.bench.serve import Server
from eval.launch import Launch

HTTP_SERVER = """
import json, os, subprocess, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

if os.environ.get("CHILD_PID"):
    child = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
    Path(os.environ["CHILD_PID"]).write_text(str(child.pid))
if os.environ.get("LAUNCH_ARGS"):
    Path(os.environ["LAUNCH_ARGS"]).write_text(json.dumps({
        "args": sys.argv[1:],
        "checkpoint": os.environ["CHECKPOINT"],
        "port": os.environ["PORT"],
        "alias": os.environ["FIXED_ALIAS"],
    }))
print(os.environ.get("SERVER_MARKER"), os.getcwd(), flush=True)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/v1/models" else 404)
        self.end_headers()
        self.wfile.write(b'{"data": []}')
    def log_message(self, *_args):
        pass
HTTPServer(("127.0.0.1", int(os.environ["PORT"])), Handler).serve_forever()
"""


def launch_script(directory, code):
    program, launch = directory / "server.py", directory / "launch.sh"
    program.write_text(code)
    launch.write_text(
        "export FIXED_ALIAS=served\n"
        f"exec {shlex.quote(sys.executable)} {shlex.quote(str(program))} "
        '--model "$CHECKPOINT" --served-model-name "$FIXED_ALIAS" '
        '--host 127.0.0.1 --port "$PORT" "$@"\n'
    )
    return launch


def endpoint():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}", port


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


async def stopped(pid):
    async with asyncio.timeout(1):
        while alive(pid):
            await asyncio.sleep(0.01)


def test_owned_server_is_ready_and_cleans_up_its_whole_group(tmp_path):
    address, port = endpoint()
    child_file, log = tmp_path / "child.pid", tmp_path / "server.log"
    arguments = tmp_path / "launch-args.json"
    item = mapping("nvidia-cuda-x86", "C2", "freetoken")
    launch = launch_script(tmp_path, HTTP_SERVER)

    async def scenario():
        async with Server(
            Launch(
                launch,
                item.checkpoint,
                port,
                args=("--workers", "2"),
                env={
                    "SERVER_MARKER": "started",
                    "CHILD_PID": str(child_file),
                    "LAUNCH_ARGS": str(arguments),
                    "PYTHONUNBUFFERED": "1",
                },
            ),
            address,
            cwd=tmp_path,
            log=log,
            health_timeout=2,
        ) as server:
            assert server.endpoint == address
            assert server.served_model == "served"
            async with httpx.AsyncClient(trust_env=False) as client:
                assert (await client.get(f"{server.endpoint}/v1/models")).json() == {"data": []}
            child_pid = int(child_file.read_text())
            assert alive(child_pid)
            assert json.loads(arguments.read_text()) == {
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
            }
        await stopped(child_pid)

    asyncio.run(scenario())
    assert f"started {tmp_path}" in log.read_text()


def test_attaching_does_not_stop_the_existing_server():
    address, port = endpoint()
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", HTTP_SERVER],
        env={**os.environ, "CHECKPOINT": "unused", "PORT": str(port), "FIXED_ALIAS": "served"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    async def scenario():
        async with Server(None, address, health_timeout=2):
            pass
        assert process.poll() is None
        async with httpx.AsyncClient(trust_env=False) as client:
            assert (await client.get(f"{address}/v1/models")).is_success

    try:
        asyncio.run(scenario())
    finally:
        process.terminate()
        process.wait(timeout=2)


def test_failed_startup_and_timeout_clean_up_descendants(tmp_path):
    address, _port = endpoint()
    child_file = tmp_path / "child.pid"
    command = """
import os, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
Path(os.environ["CHILD_PID"]).write_text(str(child.pid))
if os.environ["TEST_MODE"] == "fail":
    sys.exit(7)
time.sleep(60)
"""
    launch = launch_script(tmp_path, command)

    async def scenario():
        for mode, error in (("fail", RuntimeError), ("timeout", TimeoutError)):
            with pytest.raises(error):
                async with Server(
                    Launch(
                        launch,
                        "unused",
                        _port,
                        env={"CHILD_PID": str(child_file), "TEST_MODE": mode},
                    ),
                    address,
                    health_timeout=0.3,
                ):
                    pytest.fail("a server with no HTTP listener must not become ready")
            await stopped(int(child_file.read_text()))

    asyncio.run(scenario())


def test_cancellation_during_readiness_stops_the_group(tmp_path):
    address, _port = endpoint()
    parent_file = tmp_path / "parent.pid"
    command = """
import os, signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(os.environ["PARENT_PID"]).write_text(str(os.getpid()))
time.sleep(60)
"""
    launch = launch_script(tmp_path, command)

    async def scenario():
        async def start():
            async with Server(
                Launch(launch, "unused", _port, env={"PARENT_PID": str(parent_file)}),
                address,
                health_timeout=5,
            ):
                pytest.fail("cancelled startup must not enter the context")

        task = asyncio.create_task(start())
        async with asyncio.timeout(2):
            while not parent_file.exists():
                await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await stopped(int(parent_file.read_text()))

    asyncio.run(scenario())
