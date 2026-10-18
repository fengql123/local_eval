"""Host filesystem isolation and a narrowly scoped per-world tool capability."""

import asyncio
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from eval.bench.harnesses import _host_sandbox
from eval.bench.sources import _TOOL_CLIENT, _TOOL_SCRIPT, _tool_broker


def session(root):
    workspace, config = root / "workspace", root / "config"
    workspace.mkdir(parents=True)
    config.mkdir()
    return workspace, config


def test_host_filesystem_only_exposes_own_writable_workspace(tmp_path):
    workspace, config = session(tmp_path / "one")
    sibling, _ = session(tmp_path / "two")
    (sibling / "answer.txt").write_text("another session's answer")
    (tmp_path / "raw-answers.json").write_text("private source rows")
    (workspace / "fixture.txt").write_text("this session")
    executable = tmp_path / "private-install" / "bin" / "custom-agent"
    executable.parent.mkdir(parents=True)
    (executable.parent.parent / "secret").write_text("unrelated installation state")
    hidden = [
        str(executable.parent.parent / "secret"),
        str(sibling / "answer.txt"),
        str(tmp_path / "raw-answers.json"),
        str(Path(__file__).parents[1] / "pyproject.toml"),
        str(Path(__file__).parents[1] / ".env"),
        str(Path.home() / ".claude"),
        str(Path.home() / ".codex/auth.json"),
        "/run/docker.sock",
        "/var/run/docker.sock",
    ]
    code = r"""
import json, os, pathlib, sys
for name in json.loads(sys.argv[1]):
    assert not os.access(name, os.R_OK), name
    assert not os.access('/proc/1/root' + name, os.R_OK), name
assert pathlib.Path('fixture.txt').read_text() == 'this session'
pathlib.Path('answer.txt').write_text('own answer')
pathlib.Path(os.environ['HOME'], 'private.txt').write_text('own state')
pathlib.Path('/tmp/own-temp').write_text('private temp')
assert not os.access(sys.executable, os.W_OK)
print('isolated')
"""
    executable.write_text(f"#!{sys.executable}\n" + code)
    executable.chmod(0o755)
    command, environment = _host_sandbox(
        [str(executable), json.dumps(hidden)],
        {"PATH": os.environ["PATH"]},
        workspace,
        config,
    )
    output = config / "probe.log"
    with output.open("wb") as log:
        result = subprocess.run(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=20
        )
    assert result.returncode == 0, output.read_text()
    assert output.read_text().strip() == "isolated"
    assert (workspace / "answer.txt").read_text() == "own answer"
    assert (config / "home/private.txt").read_text() == "own state"
    assert not (Path("/tmp") / "own-temp").exists()
    assert (sibling / "answer.txt").read_text() == "another session's answer"


async def request(socket, value):
    reader, writer = await asyncio.open_unix_connection(socket)
    writer.write(json.dumps(value).encode() + b"\n")
    await writer.drain()
    response = json.loads(await reader.readline())
    writer.close()
    await writer.wait_closed()
    return response


def test_each_tool_client_only_reaches_its_assigned_world(tmp_path):
    async def run():
        sockets = []
        for world in ("first", "second"):
            workspace, config = session(tmp_path / world)
            (workspace / "tools").write_text(_TOOL_CLIENT)
            (workspace / "tools").chmod(0o755)
            native = config / "native"
            (native / "task").mkdir(parents=True)
            (native / "task/config.json").write_text(
                json.dumps({"seed": 7, "allowed_services": []})
            )
            (native / "task/initial_state.json").write_text(json.dumps({"name": world, "count": 0}))
            (native / "tool_defs.json").write_text('[{"name":"fetch"},{"name":"fail"}]')
            program = (
                r"""
import json, os, random, sys, types
class WorldState(types.SimpleNamespace):
    meta = types.SimpleNamespace()
def fetch(name, world):
    world.count += 1
    print('tool note', file=sys.stderr)
    return json.dumps([world.name, world.count, os.getpid(), random.random(), name])
def fail(world):
    world.count += 1
    raise ValueError('native tool failed after mutation')
for module, fields in [
    ('automationbench.schema.world', {'WorldState': WorldState}),
    ('automationbench.tools.api', {'API_TOOLS': [fetch, fail]}),
]:
    value = types.ModuleType(module)
    value.__dict__.update(fields)
    sys.modules[module] = value
sys.argv = ['tool.py', 'automationbench']
"""
                + "\nexec("
                + repr(_TOOL_SCRIPT.replace("/opt/automationbench", str(native)))
                + ")"
            )
            async with _tool_broker([sys.executable, "-u", "-c", program]) as socket:
                sockets.append(socket)
                assert socket.stat().st_mode & 0o777 == 0o600
                command, environment = _host_sandbox(
                    [str(workspace / "tools"), "fetch", '{"name": "account"}'],
                    {"PATH": os.environ["PATH"]},
                    workspace,
                    config,
                    tool_socket=socket,
                )
                stdout, stderr = config / "stdout", config / "stderr"
                with stdout.open("wb") as out, stderr.open("wb") as err:
                    process = await asyncio.create_subprocess_exec(
                        *command,
                        env=environment,
                        stdout=out,
                        stderr=err,
                    )
                    await asyncio.wait_for(process.wait(), 20)
                assert process.returncode == 0, stderr.read_text()
                first = json.loads(stdout.read_text())
                expected_random = random.Random(7)
                assert first[:2] == [world, 1]
                assert first[3:] == [expected_random.random(), "account"]
                assert stderr.read_text() == "tool note\n"
                replies = await asyncio.gather(
                    request(socket, ["fetch", '{"name":"child-one"}']),
                    request(socket, ["fetch", '{"name":"child-two"}']),
                )
                values = sorted(
                    (json.loads(reply["stdout"]) for reply in replies), key=lambda row: row[1]
                )
                assert [row[1] for row in values] == [2, 3]
                assert all(row[2] == first[2] for row in values)
                assert [row[3] for row in values] == [
                    expected_random.random(),
                    expected_random.random(),
                ]
                failed = await request(socket, ["fail"])
                assert (
                    failed["status"] == 1
                    and "native tool failed after mutation" in failed["stderr"]
                )
                last = await request(socket, ["fetch", '{"name":"after-error"}'])
                assert json.loads(last["stdout"])[1:3] == [5, first[2]]
            assert not socket.exists()
        assert sockets[0] != sockets[1]

    asyncio.run(run())


def test_tool_broker_rejects_command_injection_and_preserves_exit_status(tmp_path):
    async def run():
        program = r"""
import json, sys
print('{"ready": true}', flush=True)
for line in sys.stdin:
    print(json.dumps({'stdout': json.dumps([sys.argv[1], *json.loads(line)]),
        'stderr': '', 'status': 7}), flush=True)
"""
        async with _tool_broker([sys.executable, "-c", program, "own-world"]) as socket:
            for invalid in (
                {"container": "other", "command": "sh"},
                [],
                ["docker", "exec", "other"],
                [4],
            ):
                response = await request(socket, invalid)
                assert response["status"] == 2 and response["stdout"] == ""
            # Even a shell-looking tool name remains a single argument to this world.
            response = await request(socket, ["fetch; docker exec other sh", "{}"])
            assert response["status"] == 7
            assert json.loads(response["stdout"]) == [
                "own-world",
                "fetch; docker exec other sh",
                "{}",
            ]

    asyncio.run(run())


def test_canceling_broker_stops_only_its_inflight_tool(tmp_path):
    pidfile = tmp_path / "tool.pid"
    program = r"""
import os, sys, time, pathlib
print('{"ready": true}', flush=True)
for line in sys.stdin:
    pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
    time.sleep(120)
"""

    async def run():
        opened = asyncio.Future()

        async def owner():
            async with _tool_broker([sys.executable, "-c", program, str(pidfile)]) as socket:
                opened.set_result(socket)
                await asyncio.Event().wait()

        task = asyncio.create_task(owner())
        socket = await opened
        reader, writer = await asyncio.open_unix_connection(socket)
        writer.write(b'["wait"]\n')
        await writer.drain()
        async with asyncio.timeout(10):
            while not pidfile.exists():
                await asyncio.sleep(0.01)
        pid = int(pidfile.read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await reader.read() == b""
        writer.close()
        await writer.wait_closed()
        assert not socket.exists()
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

    asyncio.run(run())
