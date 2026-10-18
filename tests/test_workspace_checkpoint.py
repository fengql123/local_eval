"""Incremental workspace capture: two recovery paths and two unsafe/partial boundaries."""

import asyncio
import io
import json
import os
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.providers import checkpoint


class LocalTransport:
    def __init__(self, root):
        self.trial_paths = SimpleNamespace(trial_dir=root / "trial")
        self.trial_paths.trial_dir.mkdir()
        self.startup_ready = True
        self.fail_download = False
        self.before_archive = None
        self.downloads = []

    async def exec(self, command, **kwargs):
        if self.before_archive and "removed.json" in command:
            callback, self.before_archive = self.before_archive, None
            callback()
        child = await asyncio.create_subprocess_exec(
            "bash", "-c", command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await child.communicate()
        return SimpleNamespace(
            return_code=child.returncode, stdout=stdout.decode(), stderr=stderr.decode()
        )

    async def download_file(self, source, target):
        self.downloads.append(Path(target))
        if self.fail_download and str(source).endswith("changed.tar"):
            raise ConnectionError("transport lost")
        shutil.copyfile(source, target)

    async def upload_file(self, source, target):
        shutil.copyfile(source, target)


def fixture(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace/output"
    workspace.mkdir(parents=True)
    runtime = tmp_path / "workspace/runtime"
    runtime.mkdir()
    (runtime / "installed-library.so").write_bytes(b"generated")
    monkeypatch.setattr(checkpoint, "ROOTS", (str(workspace),))
    env = LocalTransport(tmp_path)
    return workspace, env, env.trial_paths.trial_dir / "checkpoint/latest.json"


def test_incremental_capture_restores_files_links_permissions_and_removals(tmp_path, monkeypatch):
    workspace, env, latest = fixture(tmp_path, monkeypatch)
    (workspace / "submission.py").write_text("original")
    (workspace / "submission.py").chmod(0o640)
    (workspace / "current").symlink_to("submission.py")
    temporary = workspace / ".codex/tmp/session"
    temporary.mkdir(parents=True)
    (temporary / ".lock").write_text("transient")
    cache_file = workspace / "kernel.o.d"
    cache_file.write_text("compiler dependency cache")
    env.before_archive = cache_file.unlink
    asyncio.run(checkpoint.capture(env))
    before = json.loads(latest.read_text())
    assert str(cache_file) not in before["files"]
    assert not any("/runtime/" in name for name in before["files"])
    assert not any(str(temporary.parent) in path for path in before["files"])
    first = before["files"][str(workspace / "submission.py")]["object"]
    assert all(not path.is_relative_to(latest.parent) for path in env.downloads)
    original = latest.parent / "objects" / first
    inode = original.stat().st_ino
    (workspace / "submission.py").touch()
    asyncio.run(checkpoint.capture(env))
    assert original.stat().st_ino == inode
    (workspace / "completed.json").write_text('{"requests":8}')
    asyncio.run(checkpoint.capture(env))
    assert (
        json.loads(latest.read_text())["files"][str(workspace / "submission.py")]["object"] == first
    )
    (workspace / "submission.py").write_text("changed")
    (workspace / "completed.json").unlink()
    stamp = 1_790_123_456_123_456_789
    os.utime(workspace / "submission.py", ns=(stamp, stamp))
    asyncio.run(checkpoint.capture(env))
    (workspace / "submission.py").unlink()
    (workspace / "current").unlink()
    asyncio.run(checkpoint.restore(env, str(latest)))
    assert (workspace / "submission.py").read_text() == "changed"
    assert (workspace / "submission.py").stat().st_mode & 0o777 == 0o640
    assert (workspace / "current").readlink() == Path("submission.py")
    assert not (workspace / "completed.json").exists()
    assert (workspace / "submission.py").stat().st_mtime_ns == stamp
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    resumed = LocalTransport(replacement)
    asyncio.run(checkpoint.restore(resumed, str(latest)))
    restored = resumed.trial_paths.trial_dir / "checkpoint/latest.json"
    data = json.loads(restored.read_text())
    identity = data["files"][str(workspace / "submission.py")]["object"]
    assert (restored.parent / "objects" / identity).stat().st_ino == (
        latest.parent / "objects" / identity
    ).stat().st_ino
    # No unchanged dependency should need another archive download after restore.
    resumed.fail_download = True
    asyncio.run(checkpoint.capture(resumed))


@pytest.mark.parametrize("declared", [False, None])
def test_capture_before_setup_cannot_replace_a_ready_restore_seed(tmp_path, monkeypatch, declared):
    workspace, environment, latest = fixture(tmp_path, monkeypatch)
    (workspace / "submission.py").write_text("unfinished work")
    if declared is None:
        del environment.startup_ready
    else:
        environment.startup_ready = declared
    asyncio.run(checkpoint.capture(environment))
    assert json.loads(latest.read_text())["ready"] is False
    environment.startup_ready = True
    asyncio.run(checkpoint.capture(environment))
    assert json.loads(latest.read_text())["ready"] is True


@pytest.mark.parametrize("logs", ["/logs/agent", "/workspace/.harbor/logs/agent"])
def test_native_conversations_are_available_without_materializing_model_files(tmp_path, logs):
    objects = tmp_path / "objects"
    objects.mkdir()
    (objects / "history").write_text('{"type":"session_meta"}\n')
    (objects / "weights").write_text("not a conversation")
    latest = tmp_path / "latest.json"
    latest.write_text(
        json.dumps(
            {
                "files": {
                    f"{logs}/sessions/2026/10/16/rollout-chat.jsonl": {
                        "kind": "file",
                        "object": "history",
                        "mtime": 1_790_123_456_123_456_789,
                    },
                    "/workspace/weights/model.bin": {"kind": "file", "object": "weights"},
                }
            }
        )
    )
    logs = checkpoint.conversations(latest, tmp_path / "agent")
    session = logs / "sessions/2026/10/16/rollout-chat.jsonl"
    assert session.exists()
    assert session.stat().st_mtime_ns == 1_790_123_456_123_456_789
    assert not (logs / "weights").exists()


@pytest.mark.parametrize("failure", ["download", "permission"])
def test_failed_capture_keeps_the_previous_committed_checkpoint(tmp_path, monkeypatch, failure):
    if failure == "permission" and os.geteuid() == 0:
        pytest.skip("root can read files without permission bits")
    workspace, env, latest = fixture(tmp_path, monkeypatch)
    (workspace / "work.py").write_text("saved")
    asyncio.run(checkpoint.capture(env))
    before = latest.read_bytes()
    (workspace / "work.py").write_text("unfinished change")
    if failure == "download":
        env.fail_download = True
        expected, message = ConnectionError, "transport lost"
    else:
        env.before_archive = lambda: (workspace / "work.py").chmod(0o000)
        expected, message = RuntimeError, "PermissionError"
    try:
        with pytest.raises(expected, match=message):
            asyncio.run(checkpoint.capture(env))
    finally:
        (workspace / "work.py").chmod(0o600)
    assert latest.read_bytes() == before
    objects = latest.parent / "objects"
    assert all(
        (objects / meta["object"]).exists()
        for meta in json.loads(before)["files"].values()
        if meta["kind"] == "file"
    )


def test_unexpected_or_missing_archive_members_cannot_advance_a_checkpoint(tmp_path):
    archive = tmp_path / "archive.tar"
    with tarfile.open(archive, "w") as tar:
        entry = tarfile.TarInfo("etc/forbidden")
        entry.size = 1
        tar.addfile(entry, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unexpected"):
        checkpoint.ingest(archive, tmp_path / "objects", {}, set())
    with tarfile.open(archive, "w"):
        pass
    with pytest.raises(ValueError, match="incomplete"):
        checkpoint.ingest(
            archive, tmp_path / "objects", {"/workspace/missing": {}}, {"/workspace/missing"}
        )
