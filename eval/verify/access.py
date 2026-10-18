"""Phase-owned outputs, generated runtimes and protected verification inputs."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ACCOUNTS = {
    "root": (0, 0),
    "agent": (1001, 1001),
    "search": (1002, 1002),
    "judge": (1003, 1003),
    "verifier": (1004, 1004),
    "reference": (1005, 1005),
}


def runtime_root(account: str) -> Path:
    name = "candidate" if account in ("agent", "verifier") else account
    return Path("/workspace/runtime") / name


def directory(path: Path, account: str, mode: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chown(path, *ACCOUNTS[account])
    path.chmod(mode)


def readonly(path: Path, *, private: bool = False, account: str = "root") -> None:
    """Protect copied files without following links into another workspace."""
    paths = (path, *path.rglob("*")) if path.is_dir() and not path.is_symlink() else (path,)
    for item in paths:
        if item.is_symlink():
            os.chown(item, 0, 0, follow_symlinks=False)
            continue
        os.chown(item, 0, ACCOUNTS[account][1] if private else 0)
        item.chmod(
            (0o750 if private else 0o555)
            if item.is_dir()
            else (0o440 if private else 0o444) | (item.stat().st_mode & 0o111)
        )


def writable(path: Path, account: str) -> None:
    for item in (path, *path.rglob("*")):
        if item.is_symlink():
            os.chown(item, *ACCOUNTS[account], follow_symlinks=False)
            continue
        os.chown(item, *ACCOUNTS[account])
        item.chmod(0o755 if item.is_dir() else 0o644 | (item.stat().st_mode & 0o111))


def project_writable(project: Path, account: str) -> None:
    """Imported source cannot take ownership of framework or generated runtime roots."""
    for child in project.iterdir():
        if child.name not in (".harbor", "runtime", "output"):
            writable(child, account)
    os.chown(project, 0, ACCOUNTS[account][1])
    project.chmod(0o1775)


def prepare_workspace(phase: str) -> None:
    project = Path("/workspace")
    internal = project / ".harbor"
    output = project / "output"
    actor = {"search": "search", "coding": "agent", "verification": "verifier"}[phase]
    for path in (internal, internal / "logs", project / "runtime"):
        directory(path, "root", 0o755)
    directory(output, "root" if phase == "verification" else actor, 0o755)
    if phase == "coding":
        directory(output / "submission", actor, 0o755)
    if phase != "verification":
        directory(output / "experiments", actor, 0o755)
    accounts = (actor,) if phase != "verification" else ("verifier", "reference", "judge")
    for account in accounts:
        home = (
            Path("/home/search")
            if account == "search"
            else internal / "agent"
            if account == "agent"
            else internal / "scratch" / account
        )
        directory(home, account, 0o700)
        for path in (runtime_root(account), *(runtime_root(account) / n for n in ("cache", "tmp"))):
            directory(path, account, 0o700)
    os.chown(project, 0, ACCOUNTS[actor][1])
    project.chmod(0o1775)

    def alias(source: str, target: Path, owner: str, mode: int = 0o700):
        directory(target, owner, mode)
        path = Path(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            if path.resolve() != target.resolve():
                raise ValueError(f"framework path points outside its workspace: {path}")
            return
        if path.exists():
            shutil.copytree(path, target, dirs_exist_ok=True, symlinks=True)
            shutil.rmtree(path)
        path.symlink_to(target, target_is_directory=True)

    logger = actor if phase != "verification" else "judge"
    alias("/logs/agent", internal / "logs/agent", logger)
    alias("/logs/artifacts", internal / "logs/artifacts", actor if phase == "coding" else "root")
    alias("/logs/verifier", internal / "logs/verifier", "root")
    if phase != "verification":
        for name in ("codex-home", "codex-secrets", "claude-code-settings"):
            alias(f"/tmp/{name}", internal / name, actor)
    else:
        directory(output / "review", "judge", 0o2750)
        directory(internal / "reviewer-history", "judge", 0o700)
        directory(internal / "review-rubric", "root", 0o755)
    for path in ("/tmp", "/var/tmp", "/run/lock"):
        directory(Path(path), "root", 0o755)


def reference_workspace() -> Path:
    root = runtime_root("reference")
    for path in (root, *(root / name for name in ("baselines", "weights", "cache", "tmp"))):
        directory(path, "reference", 0o700)
    return root


def seal_submission(submission: Path) -> None:
    """Freeze authored and installed code; leave only designated scratch writable."""
    if submission.is_symlink():
        raise ValueError(f"review input must be a directory, not a link: {submission}")
    for path in submission.iterdir():
        if path.name not in (".harbor", "output", "runtime"):
            readonly(path)
    runtime = runtime_root("verifier")
    if runtime.is_dir():
        for path in runtime.iterdir():
            if path.name not in ("cache", "tmp"):
                readonly(path)
        os.chown(runtime, 0, 0)
        runtime.chmod(0o555)
    os.chown(submission, 0, 0)
    submission.chmod(0o555)


def reset_runtime(account: str) -> Path:
    """Clear transient state without deleting generated installations or native histories."""
    uid = ACCOUNTS[account][0]
    for root in map(Path, ("/tmp", "/var/tmp", "/dev/shm", "/run/lock", f"/home/{account}")):
        for item in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if item.lstat().st_uid == uid:
                if item.is_dir() and not item.is_symlink():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    home = Path("/workspace/.harbor/scratch") / account
    for path in (home, *(runtime_root(account) / name for name in ("cache", "tmp"))):
        if path.exists():
            shutil.rmtree(path)
        directory(path, account, 0o700)
    return home
