"""Incremental workspace and conversation checkpoints for every Harbor provider.

The provider transports one manifest and one archive of changed files. The host
keeps only the current committed file objects. No provider volume is required.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import os
import shlex
import shutil
import tarfile
import tempfile
from contextlib import suppress
from decimal import Decimal
from pathlib import Path, PurePosixPath
from uuid import uuid4

from ..bench.progress import publish
from ..timings import record
from ..verify.access import ACCOUNTS, directory, writable

ROOTS = (
    "/workspace/output",
    "/workspace/.harbor/logs/agent",
    "/workspace/.harbor/logs/verifier",
    "/workspace/.harbor/codex-home/sessions",
    "/workspace/.harbor/reviewer-history",
    "/var/lib/local-eval/reference",
    "/var/lib/local-eval/verification",
)
INDEX = r"""
import json, os, stat
from pathlib import Path
roots, output = ROOTS, OUTPUT
files = {}
for root in roots:
    base = Path(root)
    if not base.exists(): continue
    if base.resolve() != base: raise ValueError(f'checkpoint root is a symbolic link: {base}')
    for parent, dirs, names in os.walk(base, followlinks=False):
        if Path(parent).name == '.codex':
            dirs[:] = [name for name in dirs if name != 'tmp']
        for name in ['', *dirs, *names]:
            p = Path(parent) / name
            if p.name in {'.credentials.json', 'auth.json', 'credentials.json'}:
                continue
            if str(p.parent) == '/var/lib/local-eval/verification' and p.name in {'prepare.json','install.json'}:
                continue
            if p.is_symlink():
                resolved = p.resolve()
                if not any(resolved.is_relative_to(r) for r in roots):
                    raise ValueError(f'checkpoint link leaves saved outputs: {p}')
            try: s = p.lstat()
            except FileNotFoundError: continue
            kind = 'link' if stat.S_ISLNK(s.st_mode) else 'dir' if stat.S_ISDIR(s.st_mode) else 'file' if stat.S_ISREG(s.st_mode) else None
            if kind:
                files[str(p)] = {'kind':kind, 'size':s.st_size, 'mtime':s.st_mtime_ns,
                    'mode':stat.S_IMODE(s.st_mode), 'uid':s.st_uid, 'gid':s.st_gid,
                    **({'link':os.readlink(p)} if kind=='link' else {})}
Path(output).write_text(json.dumps(files))
"""
ARCHIVE = r"""
import json, os, stat, tarfile
from pathlib import Path
base = Path(REMOTE)
removed = []
def open_file(path):
    directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in Path(path).parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        return os.open(Path(path).name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
with tarfile.open(base / 'changed.tar', 'w:gz', compresslevel=1, dereference=True, copybufsize=8 * 1024 * 1024) as bundle:
    for name in (base / 'files').read_bytes().split(b'\0'):
        if not name: continue
        path = '/' + name.decode()
        try:
            descriptor = open_file(path)
        except FileNotFoundError:
            removed.append(path)
            continue
        with os.fdopen(descriptor, 'rb') as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ValueError(f'checkpoint file changed type: {path}')
            info = bundle.gettarinfo(fileobj=source, arcname=path.lstrip('/'))
            bundle.addfile(info, source)
(base / 'removed.json').write_text(json.dumps(removed))
"""


def allowed(path: str) -> bool:
    p = PurePosixPath(path)
    return ".." not in p.parts and any(p.is_relative_to(root) for root in ROOTS)


def ingest(archive: Path, objects: Path, entries: dict, changed: set[str]) -> None:
    objects.mkdir(exist_ok=True)
    with tarfile.open(archive) as bundle:
        for member in bundle:
            name = "/" + member.name.lstrip("/")
            if name not in changed or not allowed(name) or not member.isfile():
                raise ValueError(f"unexpected checkpoint member {member.name}")
            temporary = objects / (uuid4().hex + ".partial")
            digest = hashlib.sha256()
            with bundle.extractfile(member) as source, temporary.open("wb") as target:
                while chunk := source.read(8 * 1024 * 1024):
                    digest.update(chunk)
                    target.write(chunk)
            identity = digest.hexdigest()
            target = objects / identity
            if target.exists():
                temporary.unlink()
            else:
                temporary.replace(target)
            entries[name]["object"] = identity
    missing = [name for name in changed if not entries[name].get("object")]
    if missing:
        raise ValueError(f"incomplete checkpoint archive: {missing[:3]}")


async def capture(environment) -> None:
    with record(environment.trial_paths.trial_dir, "checkpoint.capture"):
        await _capture(environment)


async def _capture(environment) -> None:
    folder = environment.trial_paths.trial_dir / "checkpoint"
    folder.mkdir(mode=0o700, exist_ok=True)
    objects = folder / "objects"
    latest = folder / "latest.json"
    previous = json.loads(latest.read_text())["files"] if latest.exists() else {}
    remote = "/tmp/local-eval-checkpoint-" + uuid4().hex
    result = await environment.exec(f"mkdir -m 700 {remote}", user="root")
    if result.return_code:
        raise RuntimeError(
            result.stderr or result.stdout or f"remote command exited {result.return_code}"
        )
    try:
        script = INDEX.replace("ROOTS", repr(ROOTS)).replace("OUTPUT", repr(remote + "/index.json"))
        with record(environment.trial_paths.trial_dir, "checkpoint.index"):
            result = await environment.exec(shlex.join(["python3", "-c", script]), user="root")
            if result.return_code:
                raise RuntimeError(
                    result.stderr or result.stdout or f"remote command exited {result.return_code}"
                )
        with tempfile.TemporaryDirectory(prefix="local-eval-capture-") as work:
            work = Path(work)
            with record(environment.trial_paths.trial_dir, "checkpoint.index_download"):
                await environment.download_file(remote + "/index.json", work / "index.json")
            entries = json.loads((work / "index.json").read_text())
            if any(not allowed(path) for path in entries):
                raise ValueError("checkpoint path leaves its declared roots")
            changed = set()
            for path, meta in entries.items():
                if meta["kind"] != "file":
                    continue
                old = previous.get(path, {})
                if all(old.get(k) == meta[k] for k in ("size", "mtime")) and old.get("object"):
                    meta["object"] = old["object"]
                else:
                    changed.add(path)
            if changed:
                (work / "files").write_bytes(
                    b"\0".join(p.lstrip("/").encode() for p in sorted(changed)) + b"\0"
                )
                await environment.upload_file(work / "files", remote + "/files")
                script = ARCHIVE.replace("REMOTE", repr(remote))
                with record(
                    environment.trial_paths.trial_dir, "checkpoint.pack", files=len(changed)
                ):
                    result = await environment.exec(
                        shlex.join(["nice", "-n", "19", "python3", "-c", script]),
                        user="root",
                        timeout_sec=3600,
                    )
                    if result.return_code:
                        raise RuntimeError(
                            result.stderr
                            or result.stdout
                            or f"remote command exited {result.return_code}"
                        )
                await environment.download_file(remote + "/removed.json", work / "removed.json")
                removed = set(json.loads((work / "removed.json").read_text()))
                if not removed <= changed:
                    raise ValueError("unexpected removed checkpoint paths")
                for path in removed:
                    del entries[path]
                changed -= removed
                with record(environment.trial_paths.trial_dir, "checkpoint.download"):
                    await environment.download_file(remote + "/changed.tar", work / "changed.tar")
                with record(environment.trial_paths.trial_dir, "checkpoint.ingest"):
                    await asyncio.to_thread(ingest, work / "changed.tar", objects, entries, changed)
            import time

            publish(
                latest,
                {
                    "at": time.time(),
                    "files": entries,
                    "ready": getattr(environment, "startup_ready", False),
                },
            )
            referenced = {m["object"] for m in entries.values() if m.get("object")}
            if objects.exists():
                for path in objects.iterdir():
                    if path.name not in referenced:
                        path.unlink()
    finally:
        # Lost transports must not keep a cancelled task alive for remote cleanup.
        with suppress(Exception):
            async with asyncio.timeout(5):
                await environment.exec(f"rm -rf -- {remote}", user="root", timeout_sec=5)


async def keep(environment) -> None:
    while True:
        try:
            await capture(environment)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # A lost transport must not destroy the last committed checkpoint.
            environment.logger.warning("Checkpoint capture failed: %s", error)
        await asyncio.sleep(15)


async def stop(environment) -> None:
    task = getattr(environment, "_workspace_checkpoint", None)
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    try:
        async with asyncio.timeout(120):
            await capture(environment)
    except Exception as error:
        environment.logger.warning("Final checkpoint unavailable: %s", error)


def archive_checkpoint(source: Path, target: Path) -> None:
    """Rebuild only declared files, preserving identity and modes for trusted startup."""
    entries = json.loads(source.read_text())["files"]
    for name in entries:
        if not allowed(name):
            raise ValueError(f"invalid restore path {name}")
        if entries[name]["kind"] == "link":
            link_target = str(PurePosixPath(name).parent / entries[name]["link"])
            if not allowed(os.path.normpath(link_target)):
                raise ValueError(f"checkpoint link leaves saved outputs: {name}")
        if any(
            entries.get(str(parent), {}).get("kind") == "link"
            for parent in PurePosixPath(name).parents
        ):
            raise ValueError("checkpoint files traverse a symbolic link")
    with tarfile.open(target, "w:gz", compresslevel=1) as bundle:
        for name, meta in sorted(
            entries.items(), key=lambda item: (len(PurePosixPath(item[0]).parts), item[0])
        ):
            item = tarfile.TarInfo(name.lstrip("/"))
            item.mode, item.uid, item.gid = meta["mode"] & 0o777, meta["uid"], meta["gid"]
            item.pax_headers["mtime"] = str(Decimal(meta["mtime"]) / 1_000_000_000)
            if meta["kind"] == "dir":
                item.type = tarfile.DIRTYPE
                bundle.addfile(item)
            elif meta["kind"] == "link":
                item.type, item.linkname = tarfile.SYMTYPE, meta["link"]
                bundle.addfile(item)
            else:
                path = source.parent / "objects" / meta["object"]
                item.size = path.stat().st_size
                with path.open("rb") as handle:
                    bundle.addfile(item, handle)


def _seed_checkpoint(source: Path, folder: Path) -> None:
    """Reuse restored file objects before the next incremental capture starts."""
    latest = folder / "latest.json"
    if latest.exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    objects = folder / "objects"
    objects.mkdir(exist_ok=True)
    saved = json.loads(source.read_text())
    for meta in saved["files"].values():
        if meta["kind"] != "file":
            continue
        original = source.parent / "objects" / meta["object"]
        meta["size"] = original.stat().st_size
        target = objects / meta["object"]
        if not target.exists():
            try:
                os.link(original, target)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copyfile(original, target)
    if not latest.exists():
        publish(latest, saved)


async def restore(environment, source: str) -> None:
    with tempfile.TemporaryDirectory() as work:
        archive = Path(work) / "checkpoint.tar"
        with record(environment.trial_paths.trial_dir, "restore.pack"):
            await asyncio.to_thread(archive_checkpoint, Path(source), archive)
        remote = "/tmp/local-eval-restore-" + uuid4().hex + ".tar"
        with record(
            environment.trial_paths.trial_dir, "restore.upload", bytes=archive.stat().st_size
        ):
            await environment.upload_file(archive, remote)
        try:
            with record(environment.trial_paths.trial_dir, "restore.extract"):
                result = await environment.exec(
                    f"tar -C / --same-owner -xf {remote}", user="root", timeout_sec=3600
                )
                if result.return_code:
                    raise RuntimeError(
                        result.stderr
                        or result.stdout
                        or f"remote command exited {result.return_code}"
                    )
            with record(environment.trial_paths.trial_dir, "restore.seed"):
                await asyncio.to_thread(
                    _seed_checkpoint, Path(source), environment.trial_paths.trial_dir / "checkpoint"
                )
        finally:
            await environment.exec(f"rm -f -- {remote}", user="root")


def conversations(source: Path, destination: Path) -> Path:
    """Materialize only native logs, leaving large workspace objects in the checkpoint."""
    import shutil

    files = json.loads(source.read_text())["files"]
    for root in (
        "/logs/agent",
        "/workspace/.harbor/logs/agent",
        "/tmp/codex-home/sessions",
        "/workspace/.harbor/codex-home/sessions",
    ):
        for name, meta in files.items():
            if meta["kind"] != "file" or not PurePosixPath(name).is_relative_to(root):
                continue
            target = (
                destination
                / ("sessions" if "codex-home" in root else "")
                / PurePosixPath(name).relative_to(root)
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source.parent / "objects" / meta["object"], target)
            if "mtime" in meta:
                os.utime(target, ns=(meta["mtime"], meta["mtime"]))
    return destination


SESSIONS = Path("/logs/agent/sessions")
REVIEW_HISTORY = "/workspace/.harbor/reviewer-history"


def restore_session_paths(resume: dict, account: str) -> None:
    """Restore native CLI sessions in their original project directory."""
    if resume.get("agent") not in {"claude-code", "codex"}:
        return
    if SESSIONS.is_symlink():
        raise ValueError("native resume sessions must be a directory")
    directory(SESSIONS, account, 0o755)
    if resume["agent"] == "claude-code":
        projects = SESSIONS / "projects"
        project = projects / resume.get("project", "-workspace")
        if projects.is_symlink() or project.is_symlink():
            raise ValueError("native resume project must be a directory")
        directory(projects, account, 0o755)
        directory(project, account, 0o755)
    writable(SESSIONS, account)
    stream = SESSIONS.parent / "claude-code.txt"
    if stream.is_file() and not stream.is_symlink():
        os.chown(stream, *ACCOUNTS[account])
        stream.chmod(0o644)


async def capture_review_history(environment, runtime: dict) -> None:
    """Export native reviewer conversations before the disposable sandbox stops."""
    if runtime["phase"] != "verification":
        return
    if not await environment.is_dir(REVIEW_HISTORY, user="root"):
        return
    target = environment.trial_paths.trial_dir / "steps/review/verifier/reviewer-history"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as folder:
        saved = Path(folder) / "reviewer-history"
        await environment.download_dir(REVIEW_HISTORY, saved)
        if target.exists():
            shutil.rmtree(target)
        saved.replace(target)


async def restore_conversation(environment, runtime: dict) -> None:
    """Restore native chat inputs without changing retained candidate outputs."""
    from .transfer import execute, upload

    for item in runtime["files"]:
        if item["target"].startswith((str(SESSIONS), REVIEW_HISTORY)):
            await upload(environment, Path(item["source"]), item["target"])
    if runtime.get("resume") or any(
        item["target"].startswith(REVIEW_HISTORY) for item in runtime["files"]
    ):
        script = (
            "import json; from pathlib import Path; "
            "from eval.providers.checkpoint import restore_session_paths; "
            "from eval.verify.access import writable; "
            "runtime=json.loads(Path('/root/local-eval-runtime.json').read_text()); "
            "restore_session_paths(runtime['resume'], "
            "{'search':'search','coding':'agent','verification':'verifier'}[runtime['phase']]) "
            "if runtime.get('resume') else None; "
            "p=Path('/workspace/.harbor/reviewer-history'); "
            "writable(p,'judge') if p.exists() else None"
        )
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "runtime.json"
            manifest.write_text(json.dumps(runtime))
            await environment.upload_file(manifest, "/root/local-eval-runtime.json")
        await execute(environment, shlex.join(["/opt/python/bin/python", "-I", "-c", script]))
