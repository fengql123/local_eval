"""Native Daytona source builds and scoped sandbox-local input staging."""

from __future__ import annotations

import asyncio
import json
import shlex
import tempfile
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from daytona.common.errors import (
    DaytonaBadGatewayError,
    DaytonaBadRequestError,
    DaytonaConnectionError,
    DaytonaInternalServerError,
    DaytonaNotFoundError,
    DaytonaRateLimitError,
    DaytonaServiceUnavailableError,
    DaytonaTimeoutError,
)
from harbor.environments.daytona.environment import (
    DAYTONA_GPU_TYPE_MAP,
    DaytonaClientManager,
)
from harbor.environments.daytona.environment import (
    DaytonaEnvironment as NativeDaytonaEnvironment,
)
from harbor.environments.daytona.snapshots import DaytonaSnapshotService
from harbor.environments.tar_transfer import pack_dir_to_file
from tenacity import retry_any, retry_if_exception, retry_if_exception_type, stop_after_delay

from ..bench.progress import publish
from . import checkpoint
from .checkpoint import capture_review_history
from .transfer import VerifierLogs, stage

_COMMAND_READ_RETRY = {
    "stop": stop_after_delay(60),
    "retry": retry_if_exception_type(
        (
            DaytonaBadGatewayError,
            DaytonaConnectionError,
            DaytonaInternalServerError,
            DaytonaRateLimitError,
            DaytonaServiceUnavailableError,
            DaytonaTimeoutError,
        )
    )
    | retry_if_exception(
        lambda error: (
            isinstance(error, DaytonaBadRequestError)
            and "failed to resolve container IP" in str(error)
        )
    ),
}


_SPOT_CAPACITY_RETRY = retry_if_exception(
    lambda error: (
        isinstance(error, DaytonaBadRequestError)
        and "no available gpu spot runners" in str(error).lower()
    )
)
_NATIVE_CREATE_RETRY = NativeDaytonaEnvironment._create_sandbox.retry


class SourceSnapshotService(DaytonaSnapshotService):
    async def _create_with_retry(self, daytona, snapshot_name, resources):
        """Let the SDK finish one source build and preserve its failure evidence."""
        from daytona import CreateSnapshotParams, Image

        client, should_close = self._build_client_for_snapshot(daytona)
        try:
            await client.snapshot.create(
                CreateSnapshotParams(
                    name=snapshot_name,
                    image=Image.from_dockerfile(str(self._dockerfile_path)),
                    resources=resources,
                ),
                timeout=14400,
            )
        finally:
            if should_close:
                await client.close()
        return snapshot_name

    async def _get_with_retry(self, daytona, snapshot_name):
        """Map current SDK build states to Harbor's pending/error lifecycle."""
        from daytona._async.snapshot import SnapshotState

        snapshot = await super()._get_with_retry(daytona, snapshot_name)
        snapshot.state = {
            SnapshotState.BUILDING: SnapshotState.PENDING,
            SnapshotState.PULLING: SnapshotState.PENDING,
            SnapshotState.SNAPSHOTTING: SnapshotState.PENDING,
            SnapshotState.BUILD_FAILED: SnapshotState.ERROR,
        }.get(snapshot.state, snapshot.state)
        return snapshot

    async def _wait_for_active(self, daytona, snapshot_name, timeout=14400):
        return await super()._wait_for_active(daytona, snapshot_name, timeout)


async def release(record: Path) -> None:
    """Stop only the sandbox recorded by this interrupted phase before replacing it."""
    value = json.loads(record.read_text())
    if value.get("stopped"):
        return
    manager = await DaytonaClientManager.get_instance()
    client = await manager.get_client()
    try:
        sandbox = await client.get(value["sandbox_id"])
        if str(sandbox.state).lower().split(".")[-1] not in {
            "stopped",
            "stopping",
            "destroying",
            "destroyed",
        }:
            await sandbox.stop()
    except DaytonaNotFoundError:
        pass
    publish(record, {**value, "stopped": True})


class DaytonaEnvironment(VerifierLogs, NativeDaytonaEnvironment):
    # Queue individual trials when matching spot runners are occupied. Harbor's
    # build timeout still bounds waiting; native cancellation/cleanup is preserved.
    _create_sandbox = NativeDaytonaEnvironment._create_sandbox.retry_with(
        retry=retry_any(_SPOT_CAPACITY_RETRY, _NATIVE_CREATE_RETRY.retry),
        wait=lambda state: 30 if _SPOT_CAPACITY_RETRY(state) else _NATIVE_CREATE_RETRY.wait(state),
    )

    # Give status/log polling one minute to recover using the original command IDs.
    _get_session_command_with_retry = (
        NativeDaytonaEnvironment._get_session_command_with_retry.retry_with(**_COMMAND_READ_RETRY)
    )
    _get_session_command_logs_with_retry = (
        NativeDaytonaEnvironment._get_session_command_logs_with_retry.retry_with(
            **_COMMAND_READ_RETRY
        )
    )

    def _resolve_daytona_gpu_types(self):
        """Keep native aliases and include GPU types supported by the current SDK."""
        from daytona import GpuType

        types = self.task_env_config.gpu_types
        if not types:
            return None
        names = {**DAYTONA_GPU_TYPE_MAP, **{gpu.value.lower(): gpu.value for gpu in GpuType}}
        return list(
            dict.fromkeys(
                GpuType(names[name.lower().strip()])
                for name in types
                if name.lower().strip() in names
            )
        )

    def __init__(self, *, environment_dir: Path, **kwargs):
        self.startup_ready = False
        self.manifest = json.loads((environment_dir / "runtime.json").read_text())
        super().__init__(environment_dir=environment_dir, **kwargs)

    def _snapshots(self):
        if self._snapshot_service is None:
            from hashlib import sha256

            profile = sha256(
                json.dumps(asdict(self._sandbox_resources()), sort_keys=True, default=str).encode()
            ).hexdigest()[:12]
            self._snapshot_service = SourceSnapshotService(
                logger=self.logger,
                env_hash=(
                    self.manifest["image"].get("identity", self.environment_id)[:20] + profile
                ),
                dockerfile_path=self._dockerfile_path,
            )
        return self._snapshot_service

    async def _upload_environment_dir_after_start(self) -> None:
        # Small private inputs are uploaded explicitly by stage(), not into an agent's workdir.
        pass

    async def is_dir(self, path: str, *, user: str | None = None) -> bool:
        """Native filesystem stat raises on missing paths; shell tests return false."""
        result = await self.exec(f"test -d {shlex.quote(str(path))}", user=user)
        return result.return_code == 0

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        """Keep native user scoping; submitted binaries cannot change that identity."""
        user = self._resolve_user(user)
        if user not in (None, "root", "0", 0):
            command = shlex.join(["setpriv", "--no-new-privs", "/bin/bash", "-c", command])
        return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)

    async def _sdk_upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Preserve modes and links while honoring Harbor's scoped upload user."""
        remote = f"/tmp/local-eval-upload-{uuid4().hex}.tar.gz"
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "upload.tar.gz"
            await asyncio.to_thread(pack_dir_to_file, source_dir, archive)
            await self._sdk_upload_file(archive, remote)
        try:
            command = (
                f"mkdir -p {shlex.quote(target_dir)} && "
                f"tar --no-same-owner -xzf {shlex.quote(remote)} -C {shlex.quote(target_dir)}"
            )
            user = self._resolve_user(None)
            if user not in (None, "root", "0", 0):
                command += f" && chown -hR -- {shlex.quote(str(user))} {shlex.quote(target_dir)}"
            result = await self._sandbox_exec(
                command,
                shell="sh -c",
                timeout_sec=600,
            )
            if result.return_code:
                raise RuntimeError(result.stderr or result.stdout)
        finally:
            await self._sandbox_exec(f"rm -f {shlex.quote(remote)}", shell="sh -c", timeout_sec=10)

    async def _resolve_start_sandbox_params(self, daytona, resources, *, force_build: bool):
        params = await super()._resolve_start_sandbox_params(
            daytona, resources, force_build=force_build
        )
        if snapshot_name := getattr(params, "snapshot", None):
            snapshot = await daytona.snapshot.get(snapshot_name)
            actual = (snapshot.cpu, snapshot.mem, snapshot.disk, snapshot.gpu)
            expected = (resources.cpu, resources.memory, resources.disk, resources.gpu)
            gpu_types = resources.gpu_type or []
            gpu_types = gpu_types if isinstance(gpu_types, list) else [gpu_types]
            if actual != expected or (gpu_types and snapshot.gpu_type not in gpu_types):
                raise RuntimeError(
                    f"snapshot {snapshot_name} has resources {actual}, GPU {snapshot.gpu_type}; "
                    f"requested {expected}, GPUs {gpu_types}; build a matching source snapshot"
                )
        params.os_user = "root"
        if self._effective_gpus > 0:
            params.spot = self._kwargs.get("spot", True)
        return params

    async def start(self, force_build: bool) -> None:
        if self.manifest.get("previous_environment"):
            await release(Path(self.manifest["previous_environment"]))
        await super().start(force_build)
        publish(
            self.trial_paths.trial_dir / "environment.json",
            {"provider": "daytona", "sandbox_id": self._sandbox.id, "stopped": False},
        )
        if self._effective_gpus > 0:
            allocation = {"gpus": self._effective_gpus, **self.manifest.get("allocation_check", {})}
            if prefixes := allocation.get("cpu_prefixes"):
                cpu = await self.exec(
                    "python3 -c \"from pathlib import Path; print(next(line.split(':', 1)[1].strip() "
                    "for line in Path('/proc/cpuinfo').read_text().splitlines() "
                    "if line.startswith('model name')))\"",
                    user="root",
                )
                name = cpu.stdout.strip()
                accepted = cpu.return_code == 0 and any(
                    name == prefix or name.startswith(prefix + " ") for prefix in prefixes
                )
                publish(
                    self.trial_paths.trial_dir / "cpu-admission.json",
                    {
                        "cpu_name": name,
                        "accepted_prefixes": prefixes,
                        "accepted": accepted,
                    },
                )
                if not accepted:
                    raise RuntimeError(f"CPU allocation rejected before downloads: {name}")
            result = await self.exec(
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", user="root"
            )
            devices = result.stdout.strip().splitlines()
            cuda = await self.exec(
                shlex.join(
                    [
                        "python3",
                        "-c",
                        "import ctypes,json; driver=ctypes.CDLL('libcuda.so.1'); "
                        "count=ctypes.c_int(); code=driver.cuInit(0); "
                        "code=code or driver.cuDeviceGetCount(ctypes.byref(count)); "
                        "print(json.dumps({'return_code':code,'devices':count.value})); "
                        "raise SystemExit(bool(code))",
                    ]
                ),
                user="root",
            )
            evidence = {
                "requested": allocation,
                "devices": devices,
                "cuda": json.loads(cuda.stdout) if not cuda.return_code else cuda.stdout.strip(),
            }
            publish(self.trial_paths.trial_dir / "allocation.json", evidence)
            if cuda.return_code or evidence["cuda"]["devices"] != allocation["gpus"]:
                raise RuntimeError(f"CUDA allocation unusable before downloads: {evidence}")
            if (
                result.return_code
                or len(devices) != allocation["gpus"]
                or (
                    allocation.get("gpu")
                    and any(allocation["gpu"] not in device for device in devices)
                )
            ):
                raise RuntimeError(f"actual GPU allocation differs before downloads: {evidence}")
            if self.task_env_config.memory_mb is not None:
                memory = await self.exec(
                    'python3 -c "from pathlib import Path; print(next(line.split()[1] '
                    "for line in Path('/proc/meminfo').read_text().splitlines() "
                    "if line.startswith('MemTotal:')))\"",
                    user="root",
                )
                if memory.return_code:
                    raise RuntimeError("RAM allocation probe failed before downloads")
                visible_kib = int(memory.stdout.strip())
                requested_kib = self.task_env_config.memory_mb * 1024
                accepted = visible_kib >= requested_kib * 0.95
                evidence = {
                    "requested_memory_mb": self.task_env_config.memory_mb,
                    "host_memory_kib": visible_kib,
                    "accepted": accepted,
                }
                publish(self.trial_paths.trial_dir / "memory-admission.json", evidence)
                if not accepted:
                    raise RuntimeError(f"RAM allocation rejected before downloads: {evidence}")
        await stage(self, self.manifest)
        self._workspace_checkpoint = asyncio.create_task(checkpoint.keep(self))

    async def stop(self, delete: bool) -> None:
        try:
            if self._sandbox is not None:
                await checkpoint.stop(self)
                await capture_review_history(self, self.manifest)
        finally:
            await super().stop(delete)
            record = self.trial_paths.trial_dir / "environment.json"
            if record.exists():
                publish(record, {**json.loads(record.read_text()), "stopped": True})

    async def _stop_sandbox(self) -> None:
        # GPU sandboxes are ephemeral: stopping triggers provider deletion.
        if self._effective_gpus > 0:
            await self._sandbox.stop()
        else:
            await super()._stop_sandbox()
