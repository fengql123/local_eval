"""Server ownership: two common cases and two lifecycle failure cases."""

import asyncio
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from eval.baselines import mapping
from eval.bench import serve
from eval.bench.serve import Server, ShutdownError, StartupError
from eval.launch import Launch

HTTP_SERVER = """
import json, os, subprocess, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

if os.environ.get("CHILD_PID"):
    child_file = Path(os.environ["CHILD_PID"])
    child_file.unlink(missing_ok=True)
    code = ("import ctypes,os,signal,threading,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            "Path(os.environ['CHILD_PID']).write_text(str(os.getpid())); ")
    code += ("threading.Thread(target=time.sleep,args=(60,)).start(); ctypes.CDLL(None).pthread_exit(None)"
             if os.environ.get("THREAD_AFTER_LEADER") else "time.sleep(60)")
    child = subprocess.Popen([sys.executable, "-c", code])
    while not child_file.exists():
        time.sleep(0.001)
if os.environ.get("LAUNCH_ARGS"):
    Path(os.environ["LAUNCH_ARGS"]).write_text(json.dumps({
        "args": sys.argv[1:],
        "checkpoint": os.environ.get("CHECKPOINT"),
        "uid": os.getuid(),
    }))
print(os.environ.get("SERVER_MARKER"), os.getcwd(), flush=True)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            time.sleep(float(os.environ.get("MODEL_DELAY", "0")))
        health = os.environ.get("HEALTH_FILE")
        if self.path == "/health" and health:
            payload = json.loads(Path(health).read_text())
            self.send_response(payload.pop("http_status", 200))
            seen = Path(os.environ["HEALTH_SEEN"])
            pending = seen.with_suffix(".tmp")
            pending.write_text(json.dumps(payload))
            pending.replace(seen)
        else:
            self.send_response(200 if self.path == "/v1/models" else 404)
            payload = {"data": [{"id": os.environ.get("TEST_ALIAS", "served")}]}
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())
    def log_message(self, *_args):
        pass
HTTPServer(("127.0.0.1", int(os.environ["TEST_PORT"])), Handler).serve_forever()
"""


def launch_script(directory, code, port, *, template=True):
    program, launch = directory / "server.py", directory / "launch.sh"
    program.write_text(code)
    launch.write_text(
        f"export TEST_PORT={port}\n"
        + ('export CHECKPOINT="/weights/optimized"\n' if not template else "")
        + f"exec {shlex.quote(sys.executable)} {shlex.quote(str(program))} "
        + '--model "$CHECKPOINT" --served-model-name served '
        + f'--host 127.0.0.1 --port {port} "$@"\n'
    )
    return launch


def endpoint():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}", port


def alive(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] not in ("Z", "X") or int(fields[17]) > 1
    except (FileNotFoundError, ProcessLookupError):
        return False


async def stopped(pid):
    async with asyncio.timeout(1):
        while alive(pid):
            await asyncio.sleep(0.01)


def test_owned_template_and_completed_server_clean_up_the_whole_group(tmp_path, monkeypatch):
    address, port = endpoint()
    child_file, log = tmp_path / "child.pid", tmp_path / "server.log"
    arguments = tmp_path / "launch-args.json"
    item = mapping("nvidia-cuda-x86", "Qwen3.6-35B-A3B", "freetoken")
    killpg = os.killpg

    def deliver_kill(pgid):
        try:
            killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def delayed_worker_exit(pgid, sig):
        # Killing a GPU worker can finish after its API parent has already exited.
        # Delay delivery to a real child so the context must wait for that child.
        if sig == signal.SIGKILL:
            asyncio.get_running_loop().call_later(0.15, deliver_kill, pgid)
        else:
            killpg(pgid, sig)

    monkeypatch.setattr(os, "killpg", delayed_worker_exit)

    async def scenario():
        for template in (True, False):
            script = launch_script(tmp_path, HTTP_SERVER, port, template=template)
            launch = Launch(
                script,
                item.checkpoint,
                args=("--workers", "2"),
                env={
                    "SERVER_MARKER": "started",
                    "CHILD_PID": str(child_file),
                    "LAUNCH_ARGS": str(arguments),
                    "PYTHONUNBUFFERED": "1",
                    "THREAD_AFTER_LEADER": "" if template else "1",
                },
            )
            async with Server(launch, address, cwd=tmp_path, log=log, health_timeout=2) as server:
                assert server.endpoint == address
                assert server.served_model == "served"
                async with httpx.AsyncClient(trust_env=False) as client:
                    assert (await client.get(f"{address}/v1/models")).json() == {
                        "data": [{"id": "served"}]
                    }
                child_pid = int(child_file.read_text())
                assert alive(child_pid)
                if not template:
                    async with asyncio.timeout(1):
                        while Path(f"/proc/{child_pid}/stat").read_text().split()[2] != "Z":
                            await asyncio.sleep(0.01)
                    assert alive(child_pid)  # Its other thread still owns resources.
                checkpoint = item.checkpoint if template else "/weights/optimized"
                assert json.loads(arguments.read_text()) == {
                    "args": [
                        "--model",
                        checkpoint,
                        "--served-model-name",
                        "served",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                        "--workers",
                        "2",
                    ],
                    "checkpoint": checkpoint,
                    "uid": os.getuid(),
                }
            try:
                assert not alive(child_pid), "server exit returned while its worker was alive"
            finally:
                deliver_kill(server._process.pid)
                await stopped(child_pid)

    asyncio.run(scenario())
    assert f"started {tmp_path}" in log.read_text()


@pytest.mark.parametrize("health", ("unsupported", "loading", "unavailable", "error"))
def test_attaching_does_not_stop_the_existing_server(tmp_path, health):
    address, port = endpoint()
    health_file, seen = tmp_path / "health.json", tmp_path / "health-seen.json"

    def set_health(value):
        pending = health_file.with_suffix(".tmp")
        pending.write_text(json.dumps(value))
        pending.replace(health_file)

    initial = {
        "loading": {"status": "loading"},
        "unavailable": {"http_status": 503},
        "error": {"status": "error", "message": "backend worker failed"},
    }
    environment = {**os.environ, "TEST_PORT": str(port)}
    if health == "unsupported":
        environment["MODEL_DELAY"] = "1.2"
    else:
        set_health(initial[health])
        environment.update(HEALTH_FILE=str(health_file), HEALTH_SEEN=str(seen))
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", HTTP_SERVER],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    async def scenario():
        async def attach():
            async with Server(None, address, health_timeout=3):
                pass

        started = asyncio.create_task(attach())
        if health != "unsupported":
            async with asyncio.timeout(2):
                while not seen.exists():
                    await asyncio.sleep(0.01)
            if health == "error":
                with pytest.raises(StartupError, match="backend worker failed"):
                    await started
            else:
                assert not started.done()  # /v1/models already advertises served.
                if health == "loading":
                    set_health({"status": "ok", "maintenance": "rebuilding"})
                    async with asyncio.timeout(2):
                        while json.loads(seen.read_text()).get("maintenance") != "rebuilding":
                            await asyncio.sleep(0.01)
                    assert not started.done()
                set_health(
                    {"status": "ok", "maintenance": "serving"}
                    if health == "loading"
                    else {"status": "ok"}
                )
                await started
        else:
            await started
        assert process.poll() is None
        async with httpx.AsyncClient(trust_env=False) as client:
            assert (await client.get(f"{address}/v1/models")).is_success

    try:
        asyncio.run(scenario())
    finally:
        process.terminate()
        process.wait(timeout=2)


def test_wrong_port_alias_exit_and_occupied_endpoint_fail_before_replay(tmp_path):
    address, port = endpoint()
    child_file, log = tmp_path / "child.pid", tmp_path / "server.log"
    command = """
import os, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
Path(os.environ["CHILD_PID"]).write_text(str(child.pid))
sys.exit(7)
"""

    async def scenario():
        for mode in ("wrong-port", "wrong-alias", "exit"):
            actual_port = endpoint()[1] if mode == "wrong-port" else port
            script = launch_script(
                tmp_path, command if mode == "exit" else HTTP_SERVER, actual_port
            )
            with pytest.raises(StartupError) as failure:
                async with Server(
                    Launch(
                        script,
                        "unused",
                        env={
                            "CHILD_PID": str(child_file),
                            "TEST_ALIAS": "other" if mode == "wrong-alias" else "served",
                        },
                    ),
                    address,
                    log=log,
                    health_timeout=0.3,
                ):
                    pytest.fail("a wrong endpoint, alias or dead process must not become ready")
            assert failure.value.endpoint == address
            assert failure.value.log == log
            await stopped(int(child_file.read_text()))

        occupied_script = launch_script(tmp_path, HTTP_SERVER, port)
        async with Server(Launch(occupied_script, "unused"), address, health_timeout=2):
            blocked_log = tmp_path / "blocked.log"
            with pytest.raises(StartupError, match="already occupied"):
                async with Server(Launch(occupied_script, "unused"), address, log=blocked_log):
                    pytest.fail("an existing listener must not count as the new server")
            assert not blocked_log.exists()

    asyncio.run(scenario())


def test_cancelled_startup_and_stalled_worker_shutdown(tmp_path, monkeypatch):
    address, port = endpoint()
    parent_file = tmp_path / "parent.pid"
    command = """
import os, signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(os.environ["PARENT_PID"]).write_text(str(os.getpid()))
time.sleep(60)
"""
    script = launch_script(tmp_path, command, port)

    async def scenario():
        async def start():
            async with Server(
                Launch(script, "unused", env={"PARENT_PID": str(parent_file)}),
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
        assert not alive(int(parent_file.read_text()))

        # A stuck worker must fail cleanup, not permit a new server on its GPU.
        child_file = tmp_path / "child.pid"
        stuck_script = launch_script(tmp_path, HTTP_SERVER, port)
        server = Server(
            Launch(stuck_script, "unused", env={"CHILD_PID": str(child_file)}),
            address,
            health_timeout=2,
        )
        killpg = os.killpg

        def stuck_worker(pgid, sig):
            if sig != signal.SIGKILL:
                killpg(pgid, sig)

        monkeypatch.setattr(os, "killpg", stuck_worker)
        monkeypatch.setattr(serve, "_STOP_TIMEOUT", 0.05)
        try:
            with pytest.raises(ShutdownError, match="live workers") as failure:
                async with server:
                    child_pid = int(child_file.read_text())
            assert str(child_pid) in str(failure.value)
            assert alive(child_pid)
            assert server._process.returncode is not None
        finally:
            killpg(server._process.pid, signal.SIGKILL)
            await stopped(int(child_file.read_text()))

    asyncio.run(scenario())
