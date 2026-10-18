"""Keep native reviewer conversations while Rewardkit handles execution and scoring."""

from __future__ import annotations

import shutil
import sys
import tomllib
from pathlib import Path
from uuid import UUID

from rewardkit.__main__ import main as rewardkit_main
from rewardkit.agents import ClaudeCodeBackend, CodexBackend, register_agent

from ..history import _events

HISTORY = Path("/workspace/.harbor/reviewer-history")
SUPPORTED = {"claude-code", "codex"}


def conversation(history: Path, agent: str) -> str | None:
    """Resume this backend's native chat; another backend keeps its own history."""
    if agent not in SUPPORTED:
        if history.exists() and any(history.iterdir()):
            raise ValueError(f"reviewer {agent!r} does not support conversation resume")
        return None
    native = history / agent / ("projects" if agent == "claude-code" else "sessions")
    pattern = "*/*.jsonl" if agent == "claude-code" else "**/*.jsonl"
    transcripts = list(native.glob(pattern))
    if not transcripts:
        if (history / ".resume-required").exists() or (
            (history / agent).exists()
            and any(path.is_file() for path in (history / agent).rglob("*"))
        ):
            raise ValueError(f"missing native {agent} conversation for reviewer resume")
        return None
    path = max(transcripts, key=lambda item: item.stat().st_mtime_ns)
    events = _events(path)
    if agent == "claude-code":
        session = path.stem
        valid = any(event.get("sessionId") == session for event in events)
    else:
        header = events[0] if events else {}
        session = header.get("payload", {}).get("id")
        valid = header.get("type") == "session_meta" and isinstance(session, str)
    if not valid:
        raise ValueError(f"invalid native {agent} conversation: {path}")
    UUID(session)
    return session


def keep_history(native: Path, persistent: Path) -> None:
    """Only conversation files persist; native backend credentials/config stay transient."""
    persistent.mkdir(parents=True, exist_ok=True)
    native.parent.mkdir(parents=True, exist_ok=True)
    if native.is_symlink():
        if native.resolve() != persistent.resolve():
            raise ValueError(f"native reviewer history points elsewhere: {native}")
        return
    if native.exists():
        shutil.copytree(native, persistent, dirs_exist_ok=True)
        shutil.rmtree(native)
    native.symlink_to(persistent, target_is_directory=True)


class ClaudeReview(ClaudeCodeBackend):
    """Use the native Claude session rather than starting a replacement conversation."""

    async def __aenter__(self):
        conversation(HISTORY, self.name)
        await super().__aenter__()
        config = Path(self.env.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude"))
        keep_history(config / "projects", HISTORY / self.name / "projects")
        return self

    def _build_command(self, prompt, schema):
        command = super()._build_command(prompt, schema)
        session = conversation(HISTORY, self.name)
        if session:
            command.extend(["--resume", session])
        return command


class CodexReview(CodexBackend):
    """Retain only native Codex sessions beyond Rewardkit's temporary Codex home."""

    reasoning_effort: str | None = None

    async def __aenter__(self):
        conversation(HISTORY, self.name)
        await super().__aenter__()
        keep_history(self.codex_home / "sessions", HISTORY / self.name / "sessions")
        return self

    def _build_command(self, schema_path, output_path):
        command = super()._build_command(schema_path, output_path)
        if self.reasoning_effort:
            command[-1:-1] = ["-c", f'model_reasoning_effort="{self.reasoning_effort}"']
        session = conversation(HISTORY, self.name)
        if session:
            command[-1:] = ["resume", session, "-"]
        return command


def main() -> None:
    """Extend Rewardkit's public backend registry without changing its CLI or rubric."""
    rubric = tomllib.loads((Path(sys.argv[1]) / "rules.toml").read_text())
    conversation(HISTORY, rubric["judge"]["judge"])
    CodexReview.reasoning_effort = rubric["judge"]["reasoning_effort"]
    register_agent(ClaudeReview)
    register_agent(CodexReview)
    rewardkit_main()


if __name__ == "__main__":
    main()
