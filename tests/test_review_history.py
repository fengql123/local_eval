"""Two native reviewer continuations and two cases that must not start a fresh chat."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import pytest
from rewardkit.agents import ClaudeCodeBackend, CodexBackend, get_agent
from rewardkit.models import AgentJudge

from eval.verify import review

CLAUDE = "062a7a75-21be-4b0d-aedc-e0e1ddfc4176"
CODEX = "01935c98-3f27-7f6c-a249-d801a2e8d979"


def transcript(history, agent, session):
    if agent == "claude-code":
        path = history / agent / "projects/-workspace-submission" / f"{session}.jsonl"
        events = [
            {"type": "user", "sessionId": session, "message": {"content": "Inspect code"}},
            {
                "type": "assistant",
                "sessionId": session,
                "message": {"content": [{"type": "tool_use", "name": "Read", "id": "read1"}]},
            },
            {
                "type": "user",
                "sessionId": session,
                "message": {"content": [{"type": "tool_result", "tool_use_id": "read1"}]},
            },
        ]
    else:
        path = history / agent / "sessions/2026/10/06" / f"rollout-{session}.jsonl"
        events = [
            {"type": "session_meta", "payload": {"id": session, "cwd": "/workspace/submission"}},
            {"type": "response_item", "payload": {"type": "function_call", "call_id": "read1"}},
            {
                "type": "response_item",
                "payload": {"type": "function_call_output", "call_id": "read1", "output": "code"},
            },
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    return path


def test_claude_keeps_live_native_history_and_resumes_same_conversation(tmp_path, monkeypatch):
    history, home = tmp_path / "history", tmp_path / "home"
    monkeypatch.setattr(review, "HISTORY", history)
    monkeypatch.setattr(Path, "home", lambda: home)

    async def native_enter(self):
        self.binary = Path("/native/claude")
        self.env = {"CLAUDE_CONFIG_DIR": str(home / "official-claude-config")}
        return self

    monkeypatch.setattr(ClaudeCodeBackend, "__aenter__", native_enter)
    judge = AgentJudge(agent="claude-code", model="claude-review-model")

    async def exercise():
        async with review.ClaudeReview(judge, "/workspace/submission") as first:
            initial = first._build_command("inspect", {"type": "object"})
            assert "--resume" not in initial
            assert first.env == {"CLAUDE_CONFIG_DIR": str(home / "official-claude-config")}
            native = home / "official-claude-config/projects"
            assert native.is_symlink()
            path = transcript(history, "claude-code", CLAUDE)
            raw = path.read_bytes()
            assert (native / path.relative_to(history / "claude-code/projects")).read_bytes() == raw
        (history / ".resume-required").touch()
        async with review.ClaudeReview(judge, "/workspace/submission") as resumed:
            command = resumed._build_command("inspect", {"type": "object"})
            assert command[-2:] == ["--resume", CLAUDE]
            assert command[command.index("--model") + 1] == "claude-review-model"
            assert command[command.index("--json-schema") + 1] == '{"type": "object"}'
            assert path.read_bytes() == raw

    asyncio.run(exercise())


def test_codex_keeps_sessions_through_native_home_cleanup_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(review.CodexReview, "reasoning_effort", "xhigh")
    history = tmp_path / "history"
    monkeypatch.setattr(review, "HISTORY", history)

    async def native_enter(self):
        self._temp_dir = tempfile.TemporaryDirectory(dir=tmp_path)
        self.codex_home = Path(self._temp_dir.name)
        self.binary = Path("/native/codex")
        self.env = {"CODEX_HOME": str(self.codex_home)}
        (self.codex_home / "config.toml").write_text("# native configuration\n")
        return self

    monkeypatch.setattr(CodexBackend, "__aenter__", native_enter)
    judge = AgentJudge(agent="codex", model="openai/review-model")

    async def exercise():
        async with review.CodexReview(judge, "/workspace/submission") as first:
            schema, output = tmp_path / "schema.json", tmp_path / "answer.json"
            command = first._build_command(schema, output)
            assert "resume" not in command
            assert command[command.index("-c") + 1] == 'model_reasoning_effort="xhigh"'
            native_home = first.codex_home
            assert (native_home / "sessions").is_symlink()
            path = transcript(history, "codex", CODEX)
            raw = path.read_bytes()
        assert not native_home.exists() and path.read_bytes() == raw
        assert not list(history.rglob("config.toml"))
        (history / ".resume-required").touch()
        async with review.CodexReview(judge, "/workspace/submission") as resumed:
            command = resumed._build_command(schema, output)
            assert command[-3:] == ["resume", CODEX, "-"]
            assert command[command.index("--output-schema") + 1] == str(schema)
            assert command[command.index("--output-last-message") + 1] == str(output)
            assert command[command.index("-m") + 1] == "review-model"
            assert command[command.index("-c") + 1] == 'model_reasoning_effort="xhigh"'
            assert path.read_bytes() == raw

    asyncio.run(exercise())


@pytest.mark.parametrize("agent", ["claude-code", "codex"])
def test_reviewer_launcher_receives_the_required_configured_effort(tmp_path, monkeypatch, agent):
    (tmp_path / "rules.toml").write_text(
        f'[judge]\njudge = "{agent}"\nreasoning_effort = "xhigh"\n'
    )
    monkeypatch.setattr(sys, "argv", ["review", str(tmp_path)])
    monkeypatch.setattr(review, "HISTORY", tmp_path / "history")
    monkeypatch.setattr(review.CodexReview, "reasoning_effort", None)
    calls = []
    monkeypatch.setattr(review, "rewardkit_main", lambda: calls.append("started"))
    review.main()
    assert calls == ["started"]
    assert review.CodexReview.reasoning_effort == "xhigh"


def test_required_missing_and_corrupt_history_cannot_start_new_chat(tmp_path):
    (tmp_path / ".resume-required").touch()
    with pytest.raises(ValueError, match="missing native claude-code conversation"):
        review.conversation(tmp_path, "claude-code")
    path = transcript(tmp_path, "claude-code", CLAUDE)
    path.write_text("not native JSON\n")
    with pytest.raises(json.JSONDecodeError):
        review.conversation(tmp_path, "claude-code")
    path.unlink()
    codex = transcript(tmp_path, "codex", CODEX)
    codex.write_text(json.dumps({"type": "response_item", "payload": {"id": CODEX}}))
    with pytest.raises(ValueError, match="invalid native codex conversation"):
        review.conversation(tmp_path, "codex")


def test_unsupported_agent_resume_fails_and_registration_keeps_native_names(tmp_path):
    assert review.conversation(tmp_path, "custom-reviewer") is None
    transcript(tmp_path, "claude-code", CLAUDE)
    with pytest.raises(ValueError, match="does not support conversation resume"):
        review.conversation(tmp_path, "custom-reviewer")
    # Explicitly selecting Codex starts its own session, preserving the earlier
    # Claude review rather than fabricating a cross-backend conversation.
    assert review.conversation(tmp_path, "codex") is None
    assert review.conversation(tmp_path, "claude-code") == CLAUDE
    own = tmp_path / "codex/sessions/state.txt"
    own.parent.mkdir(parents=True)
    own.write_text("existing Codex history without a native transcript")
    with pytest.raises(ValueError, match="missing native codex conversation"):
        review.conversation(tmp_path, "codex")
    review.register_agent(review.ClaudeReview)
    review.register_agent(review.CodexReview)
    assert isinstance(get_agent(AgentJudge(agent="claude-code")), review.ClaudeReview)
    assert isinstance(get_agent(AgentJudge(agent="codex")), review.CodexReview)


def test_native_command_progress_recovers_only_from_an_exact_complete_record(tmp_path):
    from eval.history import _events

    path = transcript(tmp_path, "codex", CODEX)
    original = path.read_bytes()
    event = {
        "timestamp": "later",
        "type": "event_msg",
        "payload": {
            "type": "item_completed",
            "item": {
                "type": "CommandExecution",
                "id": "exec-1",
                "command": "python -c 'print(1)'",
                "aggregated_output": "1\n",
                "exit_code": 0,
            },
        },
    }
    complete = json.dumps(event).encode()
    partial = complete[:-12].replace(b'"later"', b'"earlier"')
    path.write_bytes(original + partial + b"\n" + complete + b"\n")
    raw = path.read_bytes()
    events = _events(path)
    assert events[-2] == {**event, "timestamp": "earlier"}
    assert events[-1] == event
    assert path.read_bytes() == raw
    path.write_bytes(original + b'{"timestamp":"earlier",\n' + complete + b"\n")
    with pytest.raises(json.JSONDecodeError):
        _events(path)


def test_native_command_progress_without_matching_completion_is_not_discarded(tmp_path):
    from eval.history import _events

    path = transcript(tmp_path, "codex", CODEX)
    with path.open("ab") as target:
        target.write(b'{"type":"event_msg","payload":{"type":"item_completed"\n')
        target.write(b'{"type":"response_item","payload":{"type":"message"}}\n')
    with pytest.raises(json.JSONDecodeError):
        _events(path)
