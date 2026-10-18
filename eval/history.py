"""Preserve native agent conversations when a stopped Harbor trial is retried."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harbor.models.trial.config import AgentConfig


@dataclass(frozen=True)
class History:
    trajectory: Path
    files: tuple[dict, ...] = ()
    runtime: dict | None = None


def _zero(value):
    if isinstance(value, dict):
        return {key: _zero(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_zero(item) for item in value]
    return 0 if isinstance(value, (int, float)) and not isinstance(value, bool) else value


def _events(path: Path) -> list[dict]:
    data = path.read_bytes()
    # Only an unterminated last line can be a partially written event. Earlier
    # malformed events still raise instead of silently erasing conversation.
    lines = data.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        try:
            json.loads(lines[-1])
        except json.JSONDecodeError:
            lines.pop()
    events, truncated = {}, {}
    for index, line in enumerate(lines):
        if line.strip():
            try:
                events[index] = json.loads(line)
            except json.JSONDecodeError as error:
                truncated[index] = error

    # Codex can leave a partial command-progress record before writing the same
    # complete record. Recover only exact prefixes of later command telemetry;
    # never discard or reconstruct conversation messages from a guess.
    def without_timestamp(line):
        return re.sub(rb'("timestamp"\s*:\s*")[^"]*(")', rb"\1\2", line.rstrip(b"\n"), count=1)

    for index, error in truncated.items():
        prefix = without_timestamp(lines[index])
        for later, event in events.items():
            payload = event.get("payload") or {}
            item = payload.get("item") or {}
            if (
                later > index
                and event.get("type") == "event_msg"
                and payload.get("type") == "item_completed"
                and item.get("type") == "CommandExecution"
                and item.get("id")
                and re.search(rb'"id"\s*:\s*' + re.escape(json.dumps(item["id"]).encode()), prefix)
                and without_timestamp(lines[later]).startswith(prefix)
            ):
                suffix = without_timestamp(lines[later])[len(prefix) :]
                events[index] = json.loads(lines[index].rstrip(b"\n") + suffix)
                break
        else:
            raise error
    return [events[index] for index in sorted(events)]


def rate_limit_reset(logs: Path) -> float | None:
    """Use Claude's native rejected-quota event; successful retries clear it."""
    path = logs / "claude-code.txt"
    if not path.is_file():
        return None
    reset = None
    for event in _events(path):
        if event.get("type") == "rate_limit_event":
            info = event["rate_limit_info"]
            reset = info.get("resetsAt") if info["status"] == "rejected" else None
        elif event.get("type") == "result" and not event.get("is_error"):
            reset = None
    return reset


def _reset_billing(path: Path) -> None:
    """Change counters in the upload copy; keep every conversation event intact."""
    if path.suffix == ".jsonl":
        events = _events(path)
        for event in events:
            message = event.get("message") or {}
            if "usage" in message:
                message["usage"] = _zero(message["usage"])
            if event.get("type") == "cost-state":
                event["totalCostUSD"] = 0
                event["modelUsage"] = _zero(event.get("modelUsage", {}))
            payload = event.get("payload") or {}
            if event.get("type") == "event_msg" and payload.get("type") == "token_count":
                info = payload.get("info") or {}
                for key in ("total_token_usage", "last_token_usage", "total_cost", "cost_usd"):
                    if key in info:
                        info[key] = _zero(info[key])
        path.write_text("".join(json.dumps(event) + "\n" for event in events))
    else:
        trajectory = json.loads(path.read_text())
        for step in trajectory.get("steps", ()):
            metrics = step.get("metrics") or {}
            for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "cost_usd"):
                if key in metrics:
                    metrics[key] = 0
        metrics = trajectory.get("final_metrics") or {}
        for key in (
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_cached_tokens",
            "total_cost_usd",
        ):
            if key in metrics:
                metrics[key] = 0
        path.write_text(json.dumps(trajectory) + "\n")


def prepare_history(agent: AgentConfig, logs: Path, destination: Path, *, coding: bool) -> History:
    """Prefer lossless Claude/Codex sessions; otherwise use Harbor's ATIF capability.

    The old logs remain immutable. A coding upload has old billing counters reset,
    because Harbor's parser includes loaded historical messages in the next trial.
    """
    from harbor.agents.factory import AgentFactory
    from harbor.models.agent.name import AgentName

    implementation = AgentFactory.get_agent_class_from_config(agent)
    capabilities = implementation.capabilities
    destination.mkdir(parents=True)
    owner = "agent" if coding else "search"
    if capabilities.load_native_trajectory and issubclass(
        implementation, AgentFactory.get_agent_class(AgentName.CLAUDE_CODE)
    ):
        sessions = tuple((logs / "sessions/projects").glob("*/*.jsonl"))
        if not sessions and agent.load_trajectory:
            sessions = (Path(agent.load_trajectory),)
        if not sessions:
            raise ValueError(f"Claude conversation is missing from {logs}")
        source = max(sessions, key=lambda path: path.stat().st_mtime).resolve()
        project_name = source.parent.name if source.parent.name.startswith("-") else "-workspace"
        project = destination / "projects" / project_name
        project.mkdir(parents=True)
        trajectory = project / source.name
        shutil.copyfile(source, trajectory)
        tools = source.with_suffix("")
        if tools.is_dir():
            shutil.copytree(tools, trajectory.with_suffix(""), symlinks=True)
        memory = source.parent / "memory"
        if memory.is_dir():
            shutil.copytree(memory, project / "memory", symlinks=True)
        files = ({"source": str(project.parent), "target": "/logs/agent/sessions/projects"},)
        runtime = {"agent": "claude-code", "project": project_name}
    elif capabilities.load_native_trajectory and issubclass(
        implementation, AgentFactory.get_agent_class(AgentName.CODEX)
    ):
        sessions = tuple((logs / "sessions").rglob("rollout-*.jsonl"))
        if not sessions and agent.load_trajectory:
            sessions = (Path(agent.load_trajectory),)
        if not sessions:
            raise ValueError(f"Codex conversation is missing from {logs}")
        source = max(sessions, key=lambda path: path.stat().st_mtime)
        dated = Path(*source.name.removeprefix("rollout-")[:10].split("-")) / source.name
        trajectory = destination / "sessions" / dated
        trajectory.parent.mkdir(parents=True)
        shutil.copyfile(source, trajectory)
        files = ({"source": str(destination / "sessions"), "target": "/logs/agent/sessions"},)
        runtime = {"agent": "codex"}
    elif capabilities.load_atif_trajectory and (logs / "trajectory.json").is_file():
        trajectory = destination / "trajectory.json"
        shutil.copyfile(logs / "trajectory.json", trajectory)
        files, runtime = (), None
    else:
        raise ValueError(f"Agent {agent.name!r} cannot restore its recorded conversation at {logs}")
    for path in destination.rglob("*.jsonl"):
        path.write_text("".join(json.dumps(event) + "\n" for event in _events(path)))
    if coding:
        for path in destination.rglob("*.jsonl"):
            _reset_billing(path)
        if trajectory.suffix == ".json":
            _reset_billing(trajectory)
    if runtime is not None:
        runtime.update(coding=coding, session_id=trajectory.stem)
    return History(
        trajectory,
        tuple({**item, "owner": owner, "read_only": False} for item in files),
        runtime,
    )


def partial_trajectory(agent: AgentConfig, logs: Path, destination: Path) -> Path | None:
    """Use Harbor's own parser on preserved logs when interruption left no result."""
    from harbor.agents.factory import AgentFactory
    from harbor.models.agent.context import AgentContext

    saved = logs / "trajectory.json"
    if saved.exists():
        return saved
    native = tuple((logs / "sessions").rglob("*.jsonl"))
    implementation = AgentFactory.get_agent_class_from_config(agent)
    if (
        agent.name not in ("claude-code", "codex")
        or not native
        or not implementation.capabilities.atif
    ):
        return None
    # A parser may leave only scratch files when the chat has no billable steps.
    # Rebuild that derived copy on retry; the preserved native logs stay untouched.
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    if agent.name == "claude-code":
        shutil.copytree(
            logs / "sessions/projects", destination / "sessions/projects", symlinks=True
        )
        stream = logs / "claude-code.txt"
        if stream.exists():
            shutil.copyfile(stream, destination / stream.name)
    else:
        shutil.copytree(logs / "sessions", destination / "sessions")
    parser = implementation(logs_dir=destination, model_name=agent.model_name, **agent.kwargs)
    parser.populate_context_post_run(AgentContext())
    path = destination / "trajectory.json"
    return path if path.exists() else None
