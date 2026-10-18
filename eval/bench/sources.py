"""Map trusted raw rows to independent sessions and preserve their source runtimes."""

import asyncio
import csv
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import tarfile
import tempfile
import tomllib
from contextlib import AsyncExitStack, asynccontextmanager
from functools import partial
from pathlib import Path

from ..config import TABLES
from ..request_policy import pool_of
from .harnesses import registry


@asynccontextmanager
async def prepared(item, workspace):
    if item.get("setup"):
        item = {**item, **(await item["setup"](workspace) or {})}
    try:
        if (
            item.get("pool") != "S"
            and not item.get("container")
            and not (workspace / ".git").exists()
        ):
            await source_command(["git", "init", "--quiet"], cwd=workspace)
        yield item
    finally:
        if item.get("cleanup"):
            await item["cleanup"]()


_TOOL_SCRIPT = r"""import ast
import inspect
import io
import json
import random
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

source = sys.argv[1]
protocol = sys.stdout
sys.stdout = sys.stderr
if source == 'automationbench':
    from automationbench.schema.world import WorldState
    from automationbench.tools.api import API_TOOLS
    root = Path('/opt/automationbench')
    config = json.loads((root / 'task/config.json').read_text())
    definitions = json.loads((root / 'tool_defs.json').read_text())
    world = WorldState(**json.loads((root / 'task/initial_state.json').read_text()))
    world.meta.allowed_services = config['allowed_services']
    random.seed(config['seed'])
    functions = {fn.__name__: fn for fn in API_TOOLS}

    def invoke(name, args):
        fn = functions[name]
        args = {k: v for k, v in args.items() if not (isinstance(v, dict) and not v)}
        if 'world' in inspect.signature(fn).parameters:
            args['world'] = world
        return fn(**args)
else:
    from agentdojo.agent_pipeline.tool_execution import is_string_list, tool_result_to_str
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suite
    root = Path('/opt/agentdojo')
    config = json.loads((root / 'task/config.json').read_text())
    suite = get_suite(config['benchmark_version'], config['suite'])
    task = suite.get_user_task_by_id(config['user_task'])
    runtime = FunctionsRuntime(suite.tools)
    world = task.init_environment(suite.load_and_inject_default_environment({}))
    definitions = [{'name': f.name, 'description': f.description,
        'parameters': f.parameters.model_json_schema()}
        for f in runtime.functions.values()]

    def invoke(name, args):
        args = {k: ast.literal_eval(v) if isinstance(v, str) and is_string_list(v) else v
            for k, v in args.items()}
        result, error = runtime.run_function(world, name, args)
        return error if error else tool_result_to_str(result)

print(json.dumps({'ready': True}), file=protocol, flush=True)
for line in sys.stdin:
    stdout, stderr = io.StringIO(), io.StringIO()
    status = 0
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            arguments = json.loads(line)
            name = arguments[0]
            args = json.loads(arguments[1]) if len(arguments) > 1 else {}
            print(json.dumps(definitions) if name == '--list' else invoke(name, args))
        except Exception:
            # Match a failed native CLI call while retaining its world mutations.
            traceback.print_exc()
            status = 1
    print(json.dumps({'stdout': stdout.getvalue(), 'stderr': stderr.getvalue(),
        'status': status}), file=protocol, flush=True)
"""

_TOOL_CLIENT = """#!/usr/bin/env python3
import json
import socket
import sys

with socket.socket(socket.AF_UNIX) as connection:
    connection.connect('/run/record-tools.sock')
    connection.sendall(json.dumps(sys.argv[1:]).encode() + b'\\n')
    with connection.makefile('rb') as response:
        result = json.loads(response.readline())
sys.stdout.write(result['stdout'])
sys.stderr.write(result['stderr'])
sys.exit(result['status'])
"""


@asynccontextmanager
async def _tool_broker(prefix):
    """Keep one native world alive behind its session's private tool socket."""
    handlers = set()
    exchange = asyncio.Lock()
    with (
        tempfile.TemporaryDirectory(prefix="local-eval-tools-") as directory,
        tempfile.TemporaryFile() as errors,
    ):
        process = await asyncio.create_subprocess_exec(
            *prefix,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=errors,
            limit=1024 * 1024,
            start_new_session=True,
        )

        async def response():
            line = await process.stdout.readline()
            if line:
                return json.loads(line)
            await process.wait()
            errors.seek(0)
            return dict(
                stdout="",
                stderr=errors.read().decode() or "The session's tool process exited.\n",
                status=process.returncode or 1,
            )

        async def handle(reader, writer):
            task = asyncio.current_task()
            handlers.add(task)
            try:
                try:
                    arguments = json.loads(await reader.readline())
                    if not (
                        isinstance(arguments, list)
                        and 1 <= len(arguments) <= 2
                        and all(isinstance(argument, str) for argument in arguments)
                    ):
                        raise ValueError("expected a tool name and optional JSON arguments")
                except ValueError as error:
                    result = dict(stdout="", stderr=f"Invalid tool call: {error}\n", status=2)
                else:
                    async with exchange:
                        if process.returncode is None:
                            try:
                                process.stdin.write(json.dumps(arguments).encode() + b"\n")
                                await process.stdin.drain()
                            except (BrokenPipeError, ConnectionResetError):
                                pass  # EOF below retains the native process's real failure.
                        result = await response()
                writer.write(json.dumps(result).encode() + b"\n")
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                handlers.remove(task)

        try:
            async with asyncio.timeout(1800):
                ready = await response()
            if ready != {"ready": True}:
                raise RuntimeError(ready["stderr"])
            socket = Path(directory) / "tools.sock"
            server = await asyncio.start_unix_server(handle, path=socket, limit=1024 * 1024)
            socket.chmod(0o600)
            try:
                yield socket
            finally:
                server.close()
                pending = list(handlers)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                await server.wait_closed()
        finally:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()


async def source_command(
    argv: list[str], *, cwd: Path | None = None, env: dict | None = None
) -> str:
    """Own a preparation process and preserve its real exit status."""
    process = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd, env=env, stdout=asyncio.subprocess.PIPE, start_new_session=True
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=1800)
    finally:
        if process.returncode is None:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, argv, output=stdout.decode())
    return stdout.decode()


_BUILD_LOCK = Path(tempfile.gettempdir()) / "local-eval-record-images.lock"


@asynccontextmanager
async def _image_build_lock():
    """Keep cold Docker preparations from competing across recording processes."""
    with _BUILD_LOCK.open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.1)
        yield


async def _build_image(context: Path, source: str, item: str) -> str:
    digest = hashlib.sha256()
    for path in sorted(context.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(context)).encode() + b"\0")
            digest.update(path.read_bytes())
    tag = f"local-eval/record-{source}:{item.replace('/', '-')}-{digest.hexdigest()[:12]}"
    if not (await source_command(["docker", "image", "ls", "--quiet", tag])).strip():
        async with _image_build_lock():
            if not (await source_command(["docker", "image", "ls", "--quiet", tag])).strip():
                await source_command(["docker", "build", "--quiet", "--tag", tag, str(context)])
    return tag


async def _native_image(image: str) -> tuple[str, str]:
    """Cache the native harnesses and recording user on an untouched source image."""
    if not (await source_command(["docker", "image", "ls", "--quiet", image])).strip():
        async with _image_build_lock():
            if not (await source_command(["docker", "image", "ls", "--quiet", image])).strip():
                await source_command(["docker", "pull", image])
    image_id, cwd = json.loads(
        await source_command(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "[{{json .Id}},{{json .Config.WorkingDir}}]",
                image,
            ]
        )
    )
    cwd = cwd or "/"
    packages = sorted(
        {entry["npm"] for entry in registry()["harnesses"].values() if "npm" in entry}
    )
    uid, gid = os.getuid(), os.getgid()
    with tempfile.TemporaryDirectory(prefix="local-eval-record-native-") as directory:
        build = Path(directory)
        (build / "Dockerfile").write_text(
            f"FROM {image}\n# Resolved source: {image_id}\nUSER root\n"
            "RUN export DEBIAN_FRONTEND=noninteractive; "
            "apt-get update -qq && apt-get install -y -qq --no-install-recommends "
            "curl ca-certificates git sudo "
            "&& curl -fsSL https://deb.nodesource.com/setup_22.x -o /tmp/record-node.sh "
            "&& bash /tmp/record-node.sh && apt-get install -y -qq nodejs "
            "&& npm install --global --cache /tmp/record-npm "
            + shlex.join(packages)
            + " && rm -rf /var/lib/apt/lists/* /tmp/record-node.sh /tmp/record-npm\n"
            + f"RUN groupadd --non-unique --gid {gid} record "
            + f"&& useradd --no-log-init --non-unique --uid {uid} --gid {gid} "
            + "--create-home --shell /bin/bash record && printf '%s\\n' "
            + shlex.quote(f"#{uid} ALL=(ALL) NOPASSWD: ALL")
            + " > /etc/sudoers.d/record && chmod 0440 /etc/sudoers.d/record"
            + (f" && chown -R {uid}:{gid} " + shlex.quote(cwd) if cwd != "/" else "")
            + "\n"
        )
        return await _build_image(build, "native", "runtime"), cwd


async def _source_image(task: Path, source: str, item: str, tools: bool = False) -> str:
    if not tools:
        image = tomllib.loads((task / "task.toml").read_text())["environment"].get("docker_image")
        if image:
            return image
    context = task / "environment" / "tools" if tools else task / "environment"
    if not tools:
        return await _build_image(context, source, item)
    # Only upstream libraries and initial state enter the tools image. Do not
    # copy the previous Harbor exporter, MCP server, oracle or graders.
    with tempfile.TemporaryDirectory(prefix="local-eval-record-tools-") as directory:
        build = Path(directory)
        shutil.copy2(context / "requirements.txt", build / "requirements.txt")
        (build / "task").mkdir()
        config = json.loads((context / "task/config.json").read_text())
        config.pop("grade_token", None)
        (build / "task/config.json").write_text(json.dumps(config))
        if source == "automationbench":
            shutil.copytree(context / "src", build / "src")
            shutil.copy2(context / "task/initial_state.json", build / "task/initial_state.json")
            shutil.copy2(context / "tool_defs.json", build / "tool_defs.json")
        base = "python:3.13-slim" if source == "automationbench" else "python:3.12.14-slim-bookworm"
        (build / "Dockerfile").write_text(
            f"FROM {base}\n"
            "ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1\n"
            f"COPY requirements.txt /opt/{source}/requirements.txt\n"
            f"RUN pip install -r /opt/{source}/requirements.txt\n"
            f"COPY task/ /opt/{source}/task/\n"
            + (
                "COPY src/ /opt/automationbench/src/\n"
                "COPY tool_defs.json /opt/automationbench/tool_defs.json\n"
                "ENV PYTHONPATH=/opt/automationbench/src\n"
                if source == "automationbench"
                else ""
            )
        )
        return await _build_image(build, source, item)


async def _source_cleanup(container: str) -> None:
    await source_command(["docker", "rm", "--force", container])


async def _source_prepare(task: Path, source: str, item: str, workspace: Path) -> dict:
    """Prepare fixtures or the real upstream runtime without a Harbor job."""
    if source == "hbf":
        with tarfile.open(task / "environment/setup.tar") as archive:
            archive.extractall(workspace, filter="data")
        instructions = "".join(
            f"\nSource task instructions ({name}):\n" + (workspace / name).read_text()
            for name in ("AGENTS.md", "CLAUDE.md")
            if (workspace / name).is_file()
        )
        skills = sorted(workspace.glob(".agents/skills/*/SKILL.md"))
        if skills:
            instructions += "\nSource task skills (read the corresponding SKILL.md when needed):\n"
            instructions += "\n".join(str(path.relative_to(workspace)) for path in skills)
        return {"prompt_append": instructions}
    if source == "clawbench":
        environment = task / "environment/task/environment"
        await source_command(["bash", str(environment / "setup.sh"), str(workspace / "workspace")])
        if (environment / "data").is_dir():
            shutil.copytree(environment / "data", workspace / "workspace", dirs_exist_ok=True)
        return {}
    tools = source in ("automationbench", "agentdojo")
    image = await _source_image(task, source, item, tools)
    if not tools:
        image, cwd = await _native_image(image)
    container = "local-eval-record-" + hashlib.sha256(str(workspace).encode()).hexdigest()[:16]
    async with AsyncExitStack() as resources:
        creation = asyncio.create_task(
            source_command(
                [
                    "docker",
                    "run",
                    "--detach",
                    "--name",
                    container,
                    *([] if tools else ["--volume", f"{workspace.parent}:{workspace.parent}"]),
                    image,
                    "sleep",
                    "infinity",
                ]
            )
        )

        async def cleanup():
            # A canceled caller still owns any container Docker creates later.
            await asyncio.gather(creation, return_exceptions=True)
            existing = await source_command(
                ["docker", "ps", "--all", "--quiet", "--filter", f"name=^/{container}$"]
            )
            if existing.strip():
                await _source_cleanup(container)

        resources.push_async_callback(cleanup)
        await asyncio.shield(creation)
        if tools:
            script = workspace / "tool.py"
            script.write_text(_TOOL_SCRIPT)
            await source_command(["docker", "cp", str(script), f"{container}:/opt/record-tool.py"])
            prefix = [
                "docker",
                "exec",
                "--interactive",
                container,
                "python3",
                "-B",
                "-u",
                "/opt/record-tool.py",
                source,
            ]
            socket = await resources.enter_async_context(_tool_broker(prefix))
            (workspace / "tools").write_text(_TOOL_CLIENT)
            (workspace / "tools").chmod(0o755)
            out = {
                "tool_socket": socket,
                "prompt_append": (
                    "\nThe apps and accounts are simulated. The source task's MCP functions are "
                    "provided through ./tools in this session; use it to interact with them. "
                    "The source exporter's /app/answer.txt path maps to ./answer.txt in this "
                    "session's workspace. Write the final answer there, without accessing /app. "
                    "Run ./tools --list to read the available function names and JSON argument schemas. "
                    'Call a function with ./tools FUNCTION_NAME \'{"argument": "value"}\'. '
                    "The tool state belongs to this session and is shared with its sub-agents.\n"
                ),
            }
        else:
            # Preserve the task's filesystem and working directory. Native
            # harness CLIs execute here rather than in an empty host copy.
            instructions = await source_command(
                [
                    "docker",
                    "exec",
                    "--workdir",
                    cwd,
                    container,
                    "bash",
                    "-c",
                    'for name in AGENTS.md CLAUDE.md; do if [ -f "$name" ]; then '
                    'printf "\nSource task instructions (%s):\n" "$name"; cat "$name"; fi; done',
                ]
            )
            out = {
                "prompt_append": instructions
                + "\nUse sudo for administrative commands when the source task requires them.",
                "container": container,
                "container_workspace": cwd,
                "container_user": f"{os.getuid()}:{os.getgid()}",
                "container_gateway": (
                    await source_command(
                        [
                            "docker",
                            "inspect",
                            "--format",
                            "{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}",
                            container,
                        ]
                    )
                ).strip(),
                "env": {"HOME": "/home/record"},
            }
        out["cleanup"] = resources.pop_all().aclose
        return out


def raw_items(
    model: str,
    workload: str,
    data_root: Path | str | None = None,
    limit: int | None = None,
    harness: str | None = None,
) -> list[dict]:
    """Read trusted selected raw items and prepare each live session independently.

    Raw roots contain sources/<pool>/<row-id>/row.json and tasks/<source>/<item>.
    Their selected IDs and splits are already established before harness expansion.
    S-1/S-N share S; A-1/A-N share A. No prepared replay rows are read here.
    """
    root = (Path(data_root) if data_root is not None else Path("data/raw")).resolve()
    pool = pool_of(workload)
    with (TABLES / "data.csv").open(newline="") as handle:
        allocations = {row["source"]: row for row in csv.DictReader(handle) if row["pool"] == pool}
    items = []
    for directory in sorted((root / "sources" / pool).iterdir()):
        raw = json.loads((directory / "row.json").read_text())
        allocation = allocations[raw["source"]]
        selected_harness = raw["harness"]["name"] if raw["harness"] else None
        if selected_harness and selected_harness not in allocation["harnesses"].split(","):
            continue
        if harness is not None and selected_harness != harness:
            continue
        entry = {key: raw[key] for key in ("id", "pool", "source", "item", "split", "evaluation")}
        entry.update({"model": model, "harness": selected_harness})
        if pool == "S":
            entry["messages"] = raw["data"]["messages"]
        else:
            task = root / raw["data"]["task"]
            prompt = (task / "instruction.md").read_text()
            # The source export's /app note describes its former container;
            # local fixture sources now use the session's own working directory.
            if raw["source"] in ("hbf", "clawbench"):
                prompt = prompt.replace("/app/workspace", "./workspace").replace("/app", ".")
            entry.update(
                {
                    "prompt": prompt,
                    "task": task,
                    "setup": partial(_source_prepare, task, raw["source"], raw["item"]),
                }
            )
        items.append(entry)
    return items[:limit] if limit is not None else items
