"""Cached native runtimes: reuse, session isolation, invalidation and process contention."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from eval.bench import sources
from eval.bench.harnesses import harness, registry


@pytest.fixture
def docker(tmp_path, monkeypatch):
    state = dict(
        images=set(),
        containers=set(),
        recipes={},
        calls=[],
        source_id="sha256:first",
        cwd="/testbed",
    )

    async def command(argv):
        state["calls"].append(argv)
        if argv[1:3] == ["image", "ls"]:
            return "cached" if argv[-1] in state["images"] else ""
        if argv[1:3] == ["image", "inspect"]:
            return json.dumps([state["source_id"], state["cwd"]])
        if argv[1] == "pull":
            state["images"].add(argv[-1])
        elif argv[1] == "build":
            state["recipes"][argv[4]] = (Path(argv[-1]) / "Dockerfile").read_text()
            state["images"].add(argv[4])
        elif argv[1] == "run":
            state["containers"].add(argv[argv.index("--name") + 1])
        elif argv[1] == "ps":
            name = argv[-1].removeprefix("name=^/").removesuffix("$")
            return name if name in state["containers"] else ""
        elif argv[1] == "rm":
            state["containers"].remove(argv[-1])
        elif argv[1] == "exec":
            return "Original source instructions.\n"
        elif argv[1] == "inspect":
            return "172.17.0.1\n"
        return ""

    monkeypatch.setattr(sources, "source_command", command)
    monkeypatch.setattr(sources, "_BUILD_LOCK", tmp_path / "build.lock")
    return state


def test_native_runtime_reuses_all_harnesses_without_warm_lock(docker, monkeypatch):
    image, cwd = asyncio.run(sources._native_image("original/source:tag"))
    recipe = docker["recipes"][image]
    assert recipe.startswith("FROM original/source:tag\n# Resolved source: sha256:first\n")
    assert cwd == "/testbed" and "COPY " not in recipe
    for entry in registry()["harnesses"].values():
        if "npm" in entry:
            assert entry["npm"] in recipe
    assert f"chown -R {os.getuid()}:{os.getgid()} /testbed" in recipe
    assert "sudoers.d/record" in recipe and "setup_22.x" in recipe
    assert "useradd --no-log-init" in recipe

    def unexpected_lock(*args):
        pytest.fail("a cached runtime must not wait for another image's build")

    monkeypatch.setattr(sources.fcntl, "flock", unexpected_lock)
    assert asyncio.run(sources._native_image("original/source:tag")) == (image, cwd)
    assert len(docker["recipes"]) == 1
    assert sum(call[1] == "pull" for call in docker["calls"]) == 1


def test_rows_get_fresh_containers_and_private_configs(docker, tmp_path, monkeypatch):
    task = tmp_path / "source"
    task.mkdir()
    (task / "task.toml").write_text('[environment]\ndocker_image = "original/source:tag"\n')
    (task / "fixture.txt").write_text("untouched fixture")
    sessions = []

    async def run():
        for index in range(2):
            workspace = tmp_path / str(index) / "workspace"
            workspace.mkdir(parents=True)
            item = await sources._source_prepare(task, "swebenchv", "sample", workspace)
            config = workspace.parent / "config"
            _, environment = harness(
                {**item, "harness": "claude-code", "pool": "A", "prompt": "Solve the task"},
                workspace,
                config,
                f"http://proxy/session-{index}",
            )
            assert environment["CLAUDE_CONFIG_DIR"] == str(config)
            assert "GIT_CEILING_DIRECTORIES" not in environment
            assert item["container_workspace"] == "/testbed"
            assert item["container_user"] == f"{os.getuid()}:{os.getgid()}"
            sessions.append(item)
        for item in sessions:
            await item["cleanup"]()

    asyncio.run(run())
    runs = [call for call in docker["calls"] if call[1] == "run"]
    assert len(runs) == 2 and len(docker["recipes"]) == 1
    assert runs[0][runs[0].index("--name") + 1] != runs[1][runs[1].index("--name") + 1]
    assert runs[0][runs[0].index("--volume") + 1] != runs[1][runs[1].index("--volume") + 1]
    assert sum(call[1] == "rm" for call in docker["calls"]) == 2
    assert (task / "fixture.txt").read_text() == "untouched fixture"
    assert all("apt-get" not in call[-1] and "chown" not in call[-1] for call in docker["calls"])

    command = sources.source_command

    async def cancellation():
        started, release = asyncio.Event(), asyncio.Event()

        async def delayed_create(argv):
            if argv[1] == "run":
                started.set()
                await release.wait()
            return await command(argv)

        monkeypatch.setattr(sources, "source_command", delayed_create)
        workspace = tmp_path / "canceled" / "workspace"
        workspace.mkdir(parents=True)
        preparation = asyncio.create_task(
            sources._source_prepare(task, "swebenchv", "sample", workspace)
        )
        await started.wait()
        preparation.cancel()
        await asyncio.sleep(0)
        assert not preparation.done()  # Cleanup waits for Docker's in-flight create.
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await preparation
        assert not docker["containers"]

    asyncio.run(cancellation())
    failure = subprocess.CalledProcessError(125, ["docker", "run"])

    async def failed_create(argv):
        if argv[1] == "run":
            raise failure
        return await command(argv)

    monkeypatch.setattr(sources, "source_command", failed_create)
    with pytest.raises(subprocess.CalledProcessError) as raised:
        asyncio.run(sources._source_prepare(task, "swebenchv", "sample", tmp_path / "failed"))
    assert raised.value is failure
    assert not docker["containers"]
    assert sum(call[1] == "rm" for call in docker["calls"]) == 3


def test_source_recipe_and_identity_changes_invalidate_cache(docker, monkeypatch):
    def prepare():
        return asyncio.run(sources._native_image("original/source:tag"))[0]

    images = [prepare()]
    docker["source_id"] = "sha256:changed"
    images.append(prepare())
    table = registry()
    table["harnesses"]["pi"]["npm"] += "@next"
    monkeypatch.setattr(sources, "registry", lambda: table)
    images.append(prepare())
    monkeypatch.setattr(sources.os, "getuid", lambda: 42420)
    images.append(prepare())
    monkeypatch.setattr(sources.os, "getgid", lambda: 42421)
    images.append(prepare())
    assert len(set(images)) == len(images) == len(docker["recipes"])
    docker["cwd"] = ""
    root_image, cwd = asyncio.run(sources._native_image("original/source:tag"))
    assert cwd == "/" and "chown -R" not in docker["recipes"][root_image]


def test_cold_builds_serialize_across_processes_and_cancel_cleanly(tmp_path, monkeypatch):
    contexts = [tmp_path / "a", tmp_path / "b"]
    for context in contexts:
        context.mkdir()
        (context / "Dockerfile").write_text(f"FROM {context.name}\n")
    program = r"""
import asyncio, os, sys
from pathlib import Path
from eval.bench import sources
root, context = map(Path, sys.argv[1:])
sources._BUILD_LOCK = root / 'build.lock'
async def command(argv):
    if argv[1:3] == ['image', 'ls']:
        return 'cached' if (root / argv[-1].replace('/', '_')).exists() else ''
    tag = argv[4]
    active = root / 'active'
    active.mkdir()  # Fails if different source images build concurrently.
    with (root / 'builds').open('a') as handle:
        handle.write(tag + '\n')
    await asyncio.sleep(0.2)
    (root / tag.replace('/', '_')).touch()
    active.rmdir()
    return 'built'
sources.source_command = command
async def run():
    (root / f'ready-{os.getpid()}').touch()
    while len(list(root.glob('ready-*'))) != 3:
        await asyncio.sleep(0.01)
    await sources._build_image(context, 'test', 'runtime')
asyncio.run(run())
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", program, str(tmp_path), str(context)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for context in (contexts[0], contexts[0], contexts[1])
    ]
    try:
        for process in processes:
            _, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    builds = (tmp_path / "builds").read_text().splitlines()
    assert len(builds) == len(set(builds)) == 2
    monkeypatch.setattr(sources, "_BUILD_LOCK", tmp_path / "build.lock")

    async def cancellation():
        async def waiter():
            async with sources._image_build_lock():
                pytest.fail("the first caller still owns the lock")

        async with sources._image_build_lock():
            task = asyncio.create_task(waiter())
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        async with sources._image_build_lock():
            pass

    asyncio.run(cancellation())


def test_prepared_host_rows_keep_git_operations_inside_their_workspaces(tmp_path):
    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run(["git", "init", "--quiet", str(outer)], check=True)

    async def run():
        for name in ("one", "two"):
            workspace = outer / "attempt" / name / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "fixture.txt").write_text(name)
            async with sources.prepared({"pool": "A"}, workspace):
                _, environment = harness(
                    {"pool": "A", "harness": "claude-code", "prompt": "Solve the task"},
                    workspace,
                    workspace.parent / "config",
                    "http://proxy/session",
                )
                assert environment["GIT_CEILING_DIRECTORIES"] == str(outer / "attempt")
                root = await sources.source_command(
                    ["git", "rev-parse", "--show-toplevel"], cwd=workspace, env=environment
                )
                assert Path(root.strip()) == workspace
                with pytest.raises(subprocess.CalledProcessError):
                    await sources.source_command(
                        ["git", "rev-parse", "--show-toplevel"],
                        cwd=workspace.parent,
                        env=environment,
                    )
                await sources.source_command(
                    ["git", "add", "fixture.txt"], cwd=workspace, env=environment
                )
                await sources.source_command(
                    [
                        "git",
                        "-c",
                        "user.name=Recorder test",
                        "-c",
                        "user.email=test@example.invalid",
                        "-c",
                        "core.hooksPath=/dev/null",
                        "commit",
                        "--no-gpg-sign",
                        "--quiet",
                        "-m",
                        name,
                    ],
                    cwd=workspace,
                    env=environment,
                )
                assert (workspace / "fixture.txt").read_text() == name
        for name in ("one", "two"):
            subject = await sources.source_command(
                ["git", "log", "-1", "--format=%s"], cwd=outer / "attempt" / name / "workspace"
            )
            assert subject.strip() == name

    asyncio.run(run())
    assert not list((outer / ".git/refs/heads").iterdir())


def test_prepared_preserves_source_git_and_skips_container_and_s(tmp_path, monkeypatch):
    async def unexpected(*args, **kwargs):
        pytest.fail("existing repositories, containers and S must not be initialized")

    monkeypatch.setattr(sources, "source_command", unexpected)

    async def run():
        for name in ("git-directory", "git-file", "container", "S"):
            workspace = tmp_path / name
            workspace.mkdir()
            marker = workspace / ".git"
            if name == "git-directory":
                marker.mkdir()
                (marker / "HEAD").write_text("ref: refs/heads/source\n")
            elif name == "git-file":
                marker.write_text("gitdir: /source/worktrees/task\n")
            item = {"pool": "S" if name == "S" else "A"}
            if name == "container":
                item["container"] = "source-container"
            async with sources.prepared(item, workspace):
                pass
            if name == "git-directory":
                assert (marker / "HEAD").read_text() == "ref: refs/heads/source\n"
            elif name == "git-file":
                assert marker.read_text() == "gitdir: /source/worktrees/task\n"
            else:
                assert not marker.exists()

    asyncio.run(run())


def test_prepared_cleans_up_if_git_initialization_fails(tmp_path, monkeypatch):
    failure = subprocess.CalledProcessError(128, ["git", "init"])
    cleaned = []

    async def command(*args, **kwargs):
        raise failure

    async def cleanup():
        cleaned.append(True)

    async def setup(workspace):
        return {"cleanup": cleanup}

    monkeypatch.setattr(sources, "source_command", command)

    async def run():
        with pytest.raises(subprocess.CalledProcessError) as raised:
            async with sources.prepared({"pool": "A", "setup": setup}, tmp_path):
                pytest.fail("initialization failed before the harness could start")
        assert raised.value is failure

    asyncio.run(run())
    assert cleaned == [True]
