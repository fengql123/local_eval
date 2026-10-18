"""Supply trusted files and run a task's setup script before its agent starts."""

from __future__ import annotations

import json
import shlex
import tempfile
from pathlib import Path

from ..timings import record
from . import checkpoint


class VerifierLogs:
    """Keep native agent and verifier logs inside their protected workspace state."""

    async def _verifier_logs(self, operation, dirs, chmod):
        phase = self.manifest["phase"]
        logger = {"search": "search", "coding": "agent", "verification": "judge"}[phase]
        policies = {
            "/logs/agent": f"{logger}:{logger} 0700",
            "/logs/artifacts": "agent:agent 0700" if phase == "coding" else "root:root 0700",
            "/logs/verifier": "root:judge 0770" if phase == "verification" else "root:root 0700",
        }
        ordinary = [path for path in dirs if str(path) not in policies]
        if ordinary:
            await operation(ordinary, chmod=chmod)
        result = None
        for path in dirs:
            if str(path) in policies:
                # Harbor removes symlink roots when resetting a directory. Reset
                # the physical workspace directory and preserve the native alias.
                target = "/workspace/.harbor" + str(path)
                result = await operation([target], chmod=False)
                owner, mode = policies[str(path)].split()
                await execute(self, f"chown {owner} {target} && chmod {mode} {target}")
        return result

    async def ensure_dirs(self, dirs, *, chmod=True):
        return await self._verifier_logs(super().ensure_dirs, dirs, chmod)

    async def empty_dirs(self, dirs, *, chmod=True):
        return await self._verifier_logs(super().empty_dirs, dirs, chmod)


async def execute(environment, command: str, *, timeout: int = 600, env: dict | None = None):
    result = await environment.exec(command, user="root", env=env, timeout_sec=timeout)
    if result.return_code:
        raise RuntimeError(result.stderr or result.stdout)
    return result


async def upload(environment, source: Path, target: str) -> None:
    await execute(environment, f"mkdir -p {shlex.quote(str(Path(target).parent))}")
    if source.is_dir():
        await environment.upload_dir(source, target)
    else:
        await environment.upload_file(source, target)


async def stage(environment, runtime: dict) -> None:
    """Restore work, upload declared inputs, then execute the trusted task setup."""
    environment.startup_ready = False
    await environment.ensure_dirs(["/logs/agent", "/logs/verifier", "/logs/artifacts"])
    await execute(
        environment,
        "python3 -c "
        + shlex.quote(
            "from pathlib import Path; import shutil; "
            "root=Path('/workspace'); "
            "[(shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()) "
            "for p in root.iterdir() if p.name not in ('output','.harbor')] "
            "if root.exists() else None"
        ),
    )
    with record(environment.trial_paths.trial_dir, "setup.payload_upload"):
        await environment.upload_dir(runtime["payload"]["source"], "/opt/eval")
    await environment.upload_file(
        environment.environment_dir / "setup.sh", "/root/local-eval-setup.sh"
    )
    if runtime.get("checkpoint"):
        await checkpoint.restore(environment, runtime["checkpoint"])
    with record(environment.trial_paths.trial_dir, "setup.input_upload"):
        for item in runtime["files"]:
            await upload(environment, Path(item["source"]), item["target"])
    with tempfile.TemporaryDirectory() as folder:
        manifest = Path(folder) / "runtime.json"
        manifest.write_text(json.dumps(runtime))
        await environment.upload_file(manifest, "/root/local-eval-runtime.json")
    with record(environment.trial_paths.trial_dir, "setup.script"):
        await execute(
            environment,
            "bash /root/local-eval-setup.sh /root/local-eval-runtime.json",
            timeout=environment.task_env_config.build_timeout_sec,
            env={
                "PATH": "/opt/python/bin:/usr/local/go/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            },
        )
    for source, name in (
        ("/root/local-eval-hardware.json", "hardware.json"),
        (
            "/workspace/.harbor/inputs-ready.json"
            if runtime["phase"] != "search"
            else "/workspace/inputs-ready.json",
            "inputs-ready.json",
        ),
    ):
        await environment.download_file(source, environment.trial_paths.trial_dir / name)
    environment.startup_ready = True
