"""Native local Harbor sandboxes with scoped copied inputs and retained resume."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from harbor.environments.docker.docker import DockerEnvironment as NativeDockerEnvironment

from ..bench.progress import publish
from . import checkpoint
from .checkpoint import capture_review_history
from .transfer import VerifierLogs, stage


async def _docker(*args: str, check: bool = True) -> str:
    process = await asyncio.create_subprocess_exec(
        "docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if check and process.returncode:
        raise RuntimeError(stderr.decode().strip())
    return stdout.decode().strip()


async def release_retained(environment_dir: Path, retained_record: Path) -> None:
    """Release a completed task only after Harbor has saved its final evidence."""
    if not retained_record.exists():
        return
    record = json.loads(retained_record.read_text())
    await _docker("rm", "-f", record["container_id"])
    retained_record.unlink()


class ConversationDockerEnvironment(NativeDockerEnvironment):
    """Preserve reviewer conversations before native Docker cleanup."""

    async def stop(self, delete: bool) -> None:
        try:
            await capture_review_history(self, {"phase": "verification"})
        finally:
            await super().stop(delete)


class DockerEnvironment(VerifierLogs, ConversationDockerEnvironment):
    def __init__(self, *, environment_dir: Path, trial_paths, **kwargs):
        self.manifest = json.loads((environment_dir / "runtime.json").read_text())
        self.retained_record = trial_paths.trial_dir / "retained.json"
        self.startup_ready = False
        retained = self.manifest.get("retained_environment")
        self.retained = json.loads(Path(retained).read_text()) if retained else None
        if self.retained:
            kwargs["session_id"] = self.retained["session_id"]
        self.overlay = trial_paths.trial_dir / "docker-runtime.json"
        main = {
            "user": "root",
            "init": True,
            "security_opt": ["no-new-privileges:true"],
            "memswap_limit": kwargs["task_env_config"].memory_mb * 1048576,
        }
        ids = self.manifest["gpu_ids"]
        if ids:
            main["deploy"] = {
                "resources": {
                    "reservations": {
                        "devices": [
                            {"driver": "nvidia", "device_ids": ids, "capabilities": ["gpu"]}
                        ]
                    }
                }
            }
        else:
            main["runtime"] = "runc"
        self.overlay.write_text(json.dumps({"services": {"main": main}}, indent=2))
        compose = [*kwargs.pop("extra_docker_compose", []), self.overlay]
        super().__init__(
            environment_dir=environment_dir,
            trial_paths=trial_paths,
            extra_docker_compose=compose,
            **kwargs,
        )
        main["memswap_limit"] = self.task_env_config.memory_mb * 1048576
        self.overlay.write_text(json.dumps({"services": {"main": main}}, indent=2))
        # Mutable logs stay in the retained container; trial downloads are immutable snapshots.
        self._mounts = [m for m in self._mounts if not m["target"].startswith("/logs/")]
        self._keep_containers = True

    @property
    def capabilities(self):
        return super().capabilities.model_copy(
            update={
                "gpus": bool(self.manifest["gpu_ids"]),
                "mounted": False,
            }
        )

    async def _upload_environment_dir_after_start(self) -> None:
        pass  # Inputs are supplied at their declared paths by trusted startup.

    async def start(self, force_build: bool) -> None:
        if self.retained:
            # Reconnect without native start(), which runs Compose down and destroys the filesystem.
            exists = await _docker("inspect", self.retained["container_id"], check=False)
            if not exists:
                self.retained = None
                self.session_id = self.trial_paths.trial_dir.name
                return await self.start(force_build)
            await _docker("stop", self.retained["container_id"])
            self._use_prebuilt = True
            self._mounts_compose_path = self._write_mounts_compose_file()
            self._resources_compose_path = self._write_resources_compose_file()
            self._env_compose_path = self._write_env_compose_file()
            self._write_egress_control_services_compose_file()
            await _docker("start", self.retained["container_id"])
            # An interrupted native agent setup can leave unpacked system
            # packages behind. Finish that transaction before its installer reruns.
            await _docker(
                "exec",
                self.retained["container_id"],
                "sh",
                "-c",
                "if command -v dpkg >/dev/null; then "
                "dpkg --configure -a || apt-get -f install -y; fi",
            )
            await self.ensure_dirs(["/logs/agent", "/logs/verifier", "/logs/artifacts"])
            if self.retained.get("startup_ready", True):
                self.startup_ready = True
                await checkpoint.capture(self)
                self.manifest["checkpoint"] = str(
                    self.trial_paths.trial_dir / "checkpoint/latest.json"
                )
            await stage(self, self.manifest)
            self.startup_ready = True
            publish(self.retained_record, {**self.retained, "startup_ready": True})
            self._workspace_checkpoint = asyncio.create_task(checkpoint.keep(self))
            return
        await super().start(force_build)
        container = await self._run_docker_compose_command(["ps", "--all", "--quiet", "main"])
        publish(
            self.retained_record,
            {
                "provider": "docker",
                "container_id": container.stdout.strip(),
                "session_id": self.session_id,
                "startup_ready": False,
            },
        )
        await stage(self, self.manifest)
        self.startup_ready = True
        publish(
            self.retained_record,
            {**json.loads(self.retained_record.read_text()), "startup_ready": True},
        )
        self._workspace_checkpoint = asyncio.create_task(checkpoint.keep(self))

    async def stop(self, delete: bool) -> None:
        await checkpoint.stop(self)
        container = await self._run_docker_compose_command(
            ["ps", "--all", "--quiet", "main"], check=False
        )
        container_id = container.stdout.strip()
        await super().stop(delete)
        if container_id:
            publish(
                self.retained_record,
                {
                    "provider": "docker",
                    "container_id": container_id,
                    "session_id": self.session_id,
                    "startup_ready": self.startup_ready,
                },
            )
