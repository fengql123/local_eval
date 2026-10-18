"""Native provider staging and resume: two common cases and two failure boundaries."""

import asyncio
import json
import os
import shlex
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.environments.daytona import environment as native_daytona
from harbor.environments.docker.docker import DockerEnvironment as NativeDockerEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from eval.providers import checkpoint, transfer
from eval.providers import daytona as cloud_provider
from eval.providers import docker as local_provider
from eval.providers.daytona import DaytonaEnvironment
from eval.verify.machine import check_cuda


def source_plan(tmp_path):
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (context / "runtime.json").write_text(
        json.dumps({"phase": "verification", "image": {"identity": "fixture"}})
    )
    images = {}
    for phase in ("search", "coding", "verification"):
        (context / f"{phase}.Dockerfile").write_text("FROM ubuntu:24.04\n")
        images[phase] = {
            "identity": phase,
            "dockerfile": f"{phase}.Dockerfile",
            "tag": f"image-{phase}",
            "ready": False,
        }
    return {"image": {"identity": "source-identity", "context": str(context), "images": images}}


def test_shared_setup_copies_scoped_inputs_without_changing_host_permissions(tmp_path, monkeypatch):
    view = tmp_path / "prepared"
    view.mkdir(mode=0o700)
    rows = view / "state.json"
    rows.write_text("{}")
    rows.chmod(0o600)
    (tmp_path / "setup.sh").write_text("#!/bin/bash\n")
    payload = tmp_path / "payload"
    payload.mkdir()
    runtime = {
        "phase": "coding",
        "payload": {"source": str(payload)},
        "files": [
            {"source": str(view), "target": "/inputs/public", "read_only": True},
            {"source": str(rows), "target": "/var/lib/local-eval/task.json", "read_only": True},
        ],
    }
    uploads, commands = [], []

    class Environment:
        environment_dir = tmp_path
        trial_paths = SimpleNamespace(trial_dir=tmp_path)
        task_env_config = SimpleNamespace(build_timeout_sec=600)

        async def ensure_dirs(self, dirs):
            pass

        async def exec(self, command, **kwargs):
            commands.append(command)
            return SimpleNamespace(
                return_code=0, stdout='{"cpu": "fixture", "fingerprint": "device-a"}'
            )

        async def upload_dir(self, source, target):
            uploads.append((str(source), target))

        upload_file = upload_dir

        async def download_file(self, source, target):
            target.write_text(
                '{"cpu":"fixture","fingerprint":"device-a"}'
                if target.name == "hardware.json"
                else "{}"
            )

    asyncio.run(transfer.stage(Environment(), runtime))
    assert (str(payload), "/opt/eval") in uploads
    assert (str(tmp_path / "setup.sh"), "/root/local-eval-setup.sh") in uploads
    assert (str(view), "/inputs/public") in uploads
    assert (str(rows), "/var/lib/local-eval/task.json") in uploads
    assert commands[-1] == "bash /root/local-eval-setup.sh /root/local-eval-runtime.json"
    assert not any("baseline" in command or "mount" in command for command in commands)
    assert view.stat().st_mode & 0o777 == 0o700
    assert rows.stat().st_mode & 0o777 == 0o600
    assert not (tmp_path / "assets").exists()
    assert json.loads((tmp_path / "hardware.json").read_text()) == {
        "cpu": "fixture",
        "fingerprint": "device-a",
    }

    # Images already contain an empty submission directory. Only the readiness
    # marker makes it a previously staged workspace that should replace a handoff.
    snapshot = tmp_path / "snapshot.json"
    runtime["checkpoint"] = str(snapshot)
    runtime["files"].append({"source": str(view), "target": "/workspace"})

    async def restore(*args):
        pass

    monkeypatch.setattr(local_provider.checkpoint, "restore", restore)
    for ready in (False, True):
        files = {"/workspace": {"kind": "dir"}}
        if ready:
            files["/workspace/.harbor/inputs-ready.json"] = {"kind": "file"}
        snapshot.write_text(json.dumps({"files": files}))
        uploads.clear()
        asyncio.run(transfer.stage(Environment(), runtime))
        assert (str(view), "/workspace") in uploads


def test_daytona_source_builds_native_resources_and_scoped_users_without_volumes(
    tmp_path, monkeypatch
):
    plan = source_plan(tmp_path)
    client = object()  # No volume API is used.
    environment = DaytonaEnvironment(
        environment_dir=Path(plan["image"]["context"]),
        environment_name="verification",
        session_id="fixture",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=EnvironmentConfig(gpus=1, gpu_types=["RTX-4090"]),
        auto_labels=False,
    )
    assert environment.startup_ready is False
    params = SimpleNamespace(volumes=None)

    async def native_resolve(instance, received, resources, *, force_build):
        assert received is client
        return params

    monkeypatch.setattr(
        native_daytona.DaytonaEnvironment, "_resolve_start_sandbox_params", native_resolve
    )
    result = asyncio.run(
        environment._resolve_start_sandbox_params(client, "native-resources", force_build=False)
    )
    assert result is params and result.spot is True and result.os_user == "root"
    assert result.volumes is None
    environment.task_env_config.gpu_types = ["B300", "H200", "B300"]
    assert [gpu.value for gpu in environment._resolve_daytona_gpu_types()] == ["B300", "H200"]
    executed = []

    async def native_exec(instance, command, **kwargs):
        executed.append((command, kwargs["user"]))
        return SimpleNamespace(return_code=0)

    monkeypatch.setattr(native_daytona.DaytonaEnvironment, "exec", native_exec)
    with environment.with_default_user("judge"):
        asyncio.run(environment.exec("set -euo pipefail; printf bash-ready"))
    assert executed[0][1] == "judge" and executed[0][0].startswith("setpriv --no-new-privs ")
    assert (
        subprocess.run(
            shlex.split(executed[0][0]), check=True, capture_output=True, text=True
        ).stdout
        == "bash-ready"
    )

    scripts = tmp_path / "scripts"
    scripts.mkdir(mode=0o700)
    (scripts / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    (scripts / "test.sh").chmod(0o755)
    uploaded = []

    async def upload_file(source, target):
        with tarfile.open(source) as archive:
            assert archive.getmember(".").mode == 0o700
            assert archive.getmember("./test.sh").mode == 0o755

    async def sandbox_exec(command, **kwargs):
        uploaded.append(command)
        return SimpleNamespace(return_code=0)

    environment._sdk_upload_file = upload_file
    environment._sandbox_exec = sandbox_exec
    for user in ("judge", "root"):
        with environment.with_default_user(user):
            asyncio.run(environment._sdk_upload_dir(scripts, "/tests"))
        transfer, cleanup = uploaded[-2:]
        assert "tar --no-same-owner" in transfer and cleanup.startswith("rm -f ")
        assert ("chown -hR -- judge /tests" in transfer) == (user == "judge")


@pytest.mark.parametrize(
    "cpu_name,accepted",
    [
        ("AMD EPYC 9455 48-Core Processor", True),
        ("AMD EPYC 9B45", True),
        ("Intel(R) Xeon(R) 6776P", False),
        ("AMD EPYC 945", False),
    ],
)
def test_cloud_cpu_admission_precedes_downloads(tmp_path, monkeypatch, cpu_name, accepted):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    manifest = {
        "image": {"snapshot": "source-built-fixture"},
        "allocation_check": {"cpu_prefixes": ["AMD EPYC 9455", "AMD EPYC 9B45"]},
    }
    (environment_dir / "runtime.json").write_text(json.dumps(manifest))
    trial = tmp_path / "trial"
    trial.mkdir()
    environment = DaytonaEnvironment(
        environment_dir=environment_dir,
        environment_name="agent",
        session_id="fixture",
        trial_paths=TrialPaths(trial_dir=trial),
        task_env_config=EnvironmentConfig(gpus=1, gpu_types=["RTX-PRO-6000"]),
        auto_labels=False,
    )
    staged = []

    async def start(instance, force_build):
        instance._sandbox = SimpleNamespace(id="fixture-sandbox")

    async def exec(command, **kwargs):
        assert kwargs["user"] == "root"
        if "/proc/cpuinfo" in command:
            return SimpleNamespace(return_code=0, stdout=cpu_name)
        if command.startswith("nvidia-smi"):
            return SimpleNamespace(return_code=0, stdout="NVIDIA RTX PRO 6000, 97887 MiB\n")
        return SimpleNamespace(return_code=0, stdout='{"return_code":0,"devices":1}')

    async def runtime(instance, manifest):
        staged.append("runtime")

    async def stage(instance, manifest):
        staged.append("inputs")

    monkeypatch.setattr(native_daytona.DaytonaEnvironment, "start", start)
    monkeypatch.setattr(cloud_provider, "stage", stage)
    environment.exec = exec
    if accepted:
        asyncio.run(environment.start(False))
        assert staged == ["inputs"]
    else:
        with pytest.raises(RuntimeError, match="CPU allocation rejected before downloads"):
            asyncio.run(environment.start(False))
        assert staged == []
    assert json.loads((trial / "cpu-admission.json").read_text())["accepted"] is accepted


@pytest.mark.parametrize(
    "requested,cuda_code,cuda_count,declared,unusable",
    [
        (1, 0, 1, True, False),
        (2, 0, 2, False, False),
        (1, 100, 0, False, True),
        (2, 0, 1, True, True),
    ],
)
def test_cloud_gpu_is_cuda_usable_before_runtime_or_downloads(
    tmp_path, monkeypatch, requested, cuda_code, cuda_count, declared, unusable
):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    allocation = {"gpus": requested, "gpu": "RTX PRO 6000"} if declared else {"gpus": requested}
    manifest = {"image": {"snapshot": "source-built-fixture"}}
    if declared:
        manifest["allocation_check"] = allocation
    (environment_dir / "runtime.json").write_text(json.dumps(manifest))
    trial = tmp_path / "trial"
    trial.mkdir()
    environment = DaytonaEnvironment(
        environment_dir=environment_dir,
        environment_name="search",
        session_id="fixture",
        trial_paths=TrialPaths(trial_dir=trial),
        task_env_config=EnvironmentConfig(gpus=requested, gpu_types=["RTX-PRO-6000"]),
        auto_labels=False,
    )
    staged = []

    async def start(instance, force_build):
        instance._sandbox = SimpleNamespace(id="fixture-sandbox")

    async def exec(command, **kwargs):
        assert kwargs["user"] == "root"
        if command.startswith("nvidia-smi"):
            return SimpleNamespace(
                return_code=0, stdout="NVIDIA RTX PRO 6000, 97887 MiB\n" * requested
            )
        # A driver reset still reports the correct GPU through nvidia-smi;
        # CUDA initialization must succeed before the agent can use it.
        assert "cuInit" in command and "cuDeviceGetCount" in command
        return SimpleNamespace(
            return_code=bool(cuda_code),
            stdout=json.dumps({"return_code": cuda_code, "devices": cuda_count}),
        )

    async def runtime(instance, manifest):
        staged.append("runtime")

    async def stage(instance, manifest):
        staged.append("inputs")

    monkeypatch.setattr(native_daytona.DaytonaEnvironment, "start", start)
    monkeypatch.setattr(cloud_provider, "stage", stage)
    environment.exec = exec
    if unusable:
        with pytest.raises(RuntimeError, match="CUDA allocation unusable before downloads"):
            asyncio.run(environment.start(False))
        assert staged == []
    else:
        asyncio.run(environment.start(False))
        assert staged == ["inputs"]
    evidence = json.loads((trial / "allocation.json").read_text())
    assert evidence["requested"] == allocation
    assert evidence["cuda"]
    import ctypes

    class Driver:
        def cuInit(self, flags):
            return cuda_code

        def cuDeviceGetCount(self, pointer):
            pointer._obj.value = cuda_count
            return 0

    monkeypatch.setattr(ctypes, "CDLL", lambda name: Driver())
    manifest["category"] = "nvidia-cuda-x86"
    if unusable:
        with pytest.raises(RuntimeError, match="CUDA allocation unusable before downloads"):
            check_cuda(manifest)
    else:
        check_cuda(manifest)


@pytest.mark.parametrize("fail", [False, True])
def test_task_build_precedes_setup_and_propagates_failure(tmp_path, monkeypatch, fail):
    plan = source_plan(tmp_path)
    context = Path(plan["image"]["context"])
    runtime = {"phase": "coding", "image": {"identity": "fixture"}, "gpu_ids": [], "files": []}
    (context / "runtime.json").write_text(json.dumps(runtime))
    trial = tmp_path / "trial"
    trial.mkdir()
    environment = local_provider.DockerEnvironment(
        environment_dir=context,
        environment_name="coding",
        session_id="fixture",
        trial_paths=TrialPaths(trial_dir=trial),
        task_env_config=EnvironmentConfig(memory_mb=4096),
    )
    events = []

    async def build(instance, force_build):
        assert instance._dockerfile_path == context / "Dockerfile"
        assert instance.task_env_config.docker_image is None
        events.append("build")
        if fail:
            raise RuntimeError("source build failed")

    async def compose(*args, **kwargs):
        return SimpleNamespace(stdout="new-container")

    async def stage(*args, **kwargs):
        events.append("setup")

    async def keep(*args):
        pass

    monkeypatch.setattr(NativeDockerEnvironment, "start", build)
    monkeypatch.setattr(local_provider, "stage", stage)
    monkeypatch.setattr(local_provider.checkpoint, "keep", keep)
    monkeypatch.setattr(environment, "_run_docker_compose_command", compose)
    if fail:
        with pytest.raises(RuntimeError, match="source build failed"):
            asyncio.run(environment.start(False))
        assert not environment.retained_record.exists()
        assert events == ["build"]
    else:
        asyncio.run(environment.start(False))
        assert json.loads(environment.retained_record.read_text())["startup_ready"] is True
        assert events == ["build", "setup"]


def test_setup_failure_propagates_without_publishing_readiness(tmp_path):
    (tmp_path / "setup.sh").write_text("#!/bin/bash\nexit 1\n")
    downloaded = []

    class Environment:
        environment_dir = tmp_path
        task_env_config = SimpleNamespace(build_timeout_sec=600)
        trial_paths = SimpleNamespace(trial_dir=tmp_path)

        async def ensure_dirs(self, dirs):
            pass

        async def upload_dir(self, *args):
            pass

        upload_file = upload_dir

        async def exec(self, command, **kwargs):
            assert kwargs["user"] == "root"
            return SimpleNamespace(return_code=1, stderr="setup failed", stdout="")

        async def download_file(self, *args):
            downloaded.append(args)

    with pytest.raises(RuntimeError, match="setup failed"):
        asyncio.run(transfer.stage(Environment(), {"payload": {"source": "fixture"}, "files": []}))
    assert downloaded == []


@pytest.mark.parametrize("ready,fail", [(True, False), (False, False), (False, True)])
def test_retained_local_resume_preserves_filesystem_and_retries_unfinished_startup(
    tmp_path, monkeypatch, ready, fail
):
    async def checkpoint_idle(*args):
        await asyncio.sleep(100)

    monkeypatch.setattr(local_provider.checkpoint, "keep", checkpoint_idle)
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    previous = tmp_path / "retained.json"
    previous.write_text(
        json.dumps(
            {"container_id": "same-container", "session_id": "original", "startup_ready": ready}
        )
    )
    runtime = {
        "phase": "verification",
        "category": "nvidia-cuda-x86",
        "image": {"tag": "already-built"},
        "payload": {"source": str(tmp_path / "payload")},
        "gpu_ids": [],
        "retained_environment": str(previous),
        "files": [],
    }
    (environment_dir / "runtime.json").write_text(json.dumps(runtime))
    trial = tmp_path / "trial"
    trial.mkdir()
    environment = local_provider.DockerEnvironment(
        environment_dir=environment_dir,
        environment_name="test",
        session_id="new-attempt",
        trial_paths=TrialPaths(trial_dir=trial),
        task_env_config=EnvironmentConfig(docker_image="already-built", memory_mb=4096),
    )
    events = []

    async def docker(*args, **kwargs):
        assert args[1] == "same-container"
        if args[0] == "exec":
            assert args[-1] == (
                "if command -v dpkg >/dev/null; then "
                "dpkg --configure -a || apt-get -f install -y; fi"
            )
        events.append(args[0])
        return "exists" if args[0] == "inspect" else ""

    async def unexpected_start(*args, **kwargs):
        pytest.fail("native fresh start would discard the retained filesystem")

    async def interpreter(*args, **kwargs):
        events.append("interpreter")

    async def capture(*args, **kwargs):
        events.append("capture")

    async def stage(*args, **kwargs):
        events.append("stage")
        if fail:
            raise RuntimeError("startup interrupted")

    async def ensure_dirs(*args, **kwargs):
        pass

    monkeypatch.setattr(local_provider, "_docker", docker)
    monkeypatch.setattr(NativeDockerEnvironment, "start", unexpected_start)
    monkeypatch.setattr(local_provider.checkpoint, "capture", capture)
    monkeypatch.setattr(local_provider, "stage", stage)
    monkeypatch.setattr(environment, "ensure_dirs", ensure_dirs)
    if fail:
        with pytest.raises(RuntimeError, match="startup interrupted"):
            asyncio.run(environment.start(False))
        assert not environment.retained_record.exists()
    else:
        asyncio.run(environment.start(False))
        assert json.loads(environment.retained_record.read_text())["startup_ready"] is True
    assert events == ["inspect", "stop", "start", "exec"] + (
        ["capture", "stage"] if ready else ["stage"]
    )


@pytest.mark.parametrize(
    "visible_kib,accepted",
    [
        (1511 * 1024**2, True),
        (141 * 1024**2, True),
        (88 * 1024**2, False),
        (127506842, True),  # First whole KiB at 95% of 128 GiB.
    ],
)
def test_cloud_ram_admission_precedes_evaluator_and_checkpoint_downloads(
    tmp_path, monkeypatch, visible_kib, accepted
):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    manifest = {"image": {"snapshot": "source-built-fixture"}}
    (environment_dir / "runtime.json").write_text(json.dumps(manifest))
    trial = tmp_path / "trial"
    trial.mkdir()
    environment = DaytonaEnvironment(
        environment_dir=environment_dir,
        environment_name="agent",
        session_id="fixture",
        trial_paths=TrialPaths(trial_dir=trial),
        task_env_config=EnvironmentConfig(gpus=1, gpu_types=["RTX-PRO-6000"], memory_mb=128 * 1024),
        auto_labels=False,
    )
    staged = []

    async def start(instance, force_build):
        instance._sandbox = SimpleNamespace(id="fixture-sandbox")

    async def exec(command, **kwargs):
        assert kwargs["user"] == "root"
        if "/proc/meminfo" in command:
            return SimpleNamespace(return_code=0, stdout=str(visible_kib))
        if command.startswith("nvidia-smi"):
            return SimpleNamespace(return_code=0, stdout="NVIDIA RTX PRO 6000, 97887 MiB\n")
        assert "cuInit" in command
        return SimpleNamespace(return_code=0, stdout='{"return_code":0,"devices":1}')

    async def runtime(instance, manifest):
        staged.append("runtime")

    async def stage(instance, manifest):
        staged.append("inputs")

    monkeypatch.setattr(native_daytona.DaytonaEnvironment, "start", start)
    monkeypatch.setattr(cloud_provider, "stage", stage)
    environment.exec = exec
    if accepted:
        asyncio.run(environment.start(False))
        assert staged == ["inputs"]
    else:
        with pytest.raises(RuntimeError, match="RAM allocation rejected before downloads"):
            asyncio.run(environment.start(False))
        assert staged == []
    assert json.loads((trial / "memory-admission.json").read_text()) == {
        "requested_memory_mb": 128 * 1024,
        "host_memory_kib": visible_kib,
        "accepted": accepted,
    }


def test_native_claude_resume_preserves_harbors_existing_app_project(tmp_path, monkeypatch):
    sessions = tmp_path / "logs/agent/sessions"
    native = sessions / "projects/-app"
    native.mkdir(parents=True)
    (native / "session.jsonl").write_text('{"sessionId":"native"}\n')
    stream = sessions.parent / "claude-code.txt"
    stream.write_text("saved output\n")
    stream.chmod(0o444)
    unrelated = sessions.parent / "protected.log"
    unrelated.write_text("unrelated output")
    unrelated.chmod(0o400)
    monkeypatch.setattr(checkpoint, "SESSIONS", sessions)
    monkeypatch.setitem(checkpoint.ACCOUNTS, "agent", (os.getuid(), os.getgid()))
    checkpoint.restore_session_paths({"agent": "claude-code", "project": "-workspace"}, "agent")
    assert native.is_dir() and not native.is_symlink()
    assert (native / "session.jsonl").read_text() == '{"sessionId":"native"}\n'
    assert (sessions / "projects/-workspace").is_dir()
    assert stream.stat().st_mode & 0o200
    with stream.open("a") as output:
        output.write("resumed output\n")
    assert stream.read_text() == "saved output\nresumed output\n"
    assert unrelated.stat().st_mode & 0o777 == 0o400
