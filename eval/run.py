"""Run native Harbor search, coding and verification jobs in the foreground."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import uuid
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import yaml
from harbor.job import Job
from harbor.models.job.config import JobConfig
from harbor.models.trial.config import AgentConfig as HarborAgentConfig
from harbor.models.trial.config import EnvironmentConfig, TaskConfig

from .baselines import mappings
from .config import AgentConfig, RunConfig, Task, load_config, select_tasks
from .costs import (
    Attempt,
    Usage,
    from_harbor_result,
    read_ledger,
    summarize,
    update_ledger,
    with_claude_cost,
)
from .device import check_environment
from .history import partial_trajectory, prepare_history, rate_limit_reset
from .providers import job_environment
from .setup import prepare, task_inputs
from .timings import collect as collect_timings


class _AgentLimit(Exception):
    def __init__(self, agent: str, phase: str, reset_at: float):
        self.agent, self.phase, self.reset_at = agent, phase, reset_at
        super().__init__(
            f"{agent} usage limit until {datetime.fromtimestamp(reset_at, UTC).isoformat()}"
        )


async def _wait_agent_limit(folder: Path, state: dict, task_id: str, limit: _AgentLimit) -> None:
    """Wait without a sandbox or a resource lease; preserve the native conversation."""
    entry = state["tasks"][task_id]
    entry.update(
        status="waiting",
        phase=limit.phase,
        cycle=state["cycle"],
        resume_after=datetime.fromtimestamp(limit.reset_at, UTC).isoformat(),
        error=str(limit),
    )
    _write_json(folder / "state.json", state)
    print(f"{folder.name}/{task_id}: {limit}; waiting without a sandbox", flush=True)
    while (remaining := limit.reset_at - datetime.now(UTC).timestamp()) > 0:
        await asyncio.sleep(min(remaining, 60))
    entry.pop("resume_after", None)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}-{uuid.uuid4()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _handoff(selected: dict) -> dict:
    """Keep only a baseline selection, or an unsupported verdict that makes SLO."""
    if selected.get("supported") is False:
        return {"supported": False}
    return {key: selected[key] for key in ("baseline", "args", "env")}


def _agent_settings(agent: AgentConfig) -> dict:
    return {
        "name": agent.name,
        "model": agent.model,
        "kwargs": agent.kwargs,
        "budget": agent.budget_seconds,
    }


def _freeze_host(folder: Path) -> None:
    """Resume with the coordinator that created the run, even after source changes."""
    checkout = Path(__file__).resolve().parents[1]
    destination = folder / "host.partial"
    for name in ("eval", "images", "tables", "templates"):
        shutil.copytree(
            checkout / name,
            destination / name,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".venv", ".git", ".cache", "node_modules", "build", "dist"
            ),
            symlinks=True,
        )
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(checkout / name, destination / name)
    destination.rename(folder / "host")


def _save_config(config: RunConfig, folder: Path) -> None:
    """Freeze defaults and paths; native adapters inherit the host environment."""
    supplied = None
    if config.search.reference is not None:
        supplied = folder / "provided-reference"
        supplied.mkdir()
        for task in select_tasks(config.tasks):
            if not mappings(config.environment.device_category, task.model):
                # Without a mapped baseline the pair is SLO and selects nothing.
                continue
            source = config.search.reference
            if source.is_dir():
                source = source / f"{task.task_id}.json"
            _write_json(supplied / f"{task.task_id}.json", _handoff(json.loads(source.read_text())))
    value = {
        "dataset": config.dataset,
        "concurrency": config.concurrency,
        "max_retries": config.max_retries,
        "environment": asdict(config.environment),
        "tasks": {
            "models": list(config.tasks.models) if config.tasks.models is not None else "all",
            "workloads": list(config.tasks.workloads)
            if config.tasks.workloads is not None
            else "all",
        },
        **(
            {"agents": {key: _agent_settings(value) for key, value in config.agents.items()}}
            if config.agents
            else {"agent": _agent_settings(config.agent)}
        ),
        "search": {
            "agent": _agent_settings(config.search.agent),
            **({"reference": str(supplied)} if supplied is not None else {}),
        },
        "evaluation": {
            "concurrency": config.evaluation.concurrency,
            "install_timeout": config.evaluation.install_timeout_seconds,
            "health_timeout": config.evaluation.health_timeout_seconds,
            "quality_timeout": config.evaluation.quality_timeout_seconds,
            "slo_budget": config.evaluation.slo_budget_seconds,
            "slo_request_cap": config.evaluation.slo_request_cap_seconds,
            "slo_repeats": config.evaluation.slo_repeats,
            "agent": _agent_settings(config.evaluation.agent),
        },
        "out": str(config.out.resolve()),
    }
    (folder / "run.yaml").write_text(yaml.safe_dump(value, sort_keys=False))


def _resume_config(folder: Path) -> RunConfig:
    """Resume the frozen run configuration without asking for another YAML."""
    return load_config(folder / "run.yaml")


def _concurrency(config: RunConfig, paths: tuple[Path, ...]) -> int:
    """Cloud uses native concurrency; local GPU tasks need disjoint allocations."""
    count = min(config.concurrency, len(paths)) if paths else 1
    if config.environment.provider != "docker" or count == 1:
        return count
    allocated: set[str] = set()
    for task in paths[:count]:
        manifest = task / "environment/runtime.json"
        devices = (
            set(json.loads(manifest.read_text()).get("gpu_ids", ())) if manifest.exists() else set()
        )
        if not devices or allocated & devices:
            return 1
        allocated.update(devices)
    return count


def _harbor_agent(agent: AgentConfig) -> dict:
    """Describe a native Harbor agent without resolving credentials or its class."""
    # Harbor resolves built-in names and module.path:ClassName itself. Other
    # adapters own authentication; an omitted model stays unset in their config.
    model_options = {} if agent.model is None else {"model_name": agent.model}
    return HarborAgentConfig(
        name=agent.name,
        kwargs=agent.kwargs,
        override_timeout_sec=agent.budget_seconds,
        **model_options,
    ).model_dump(mode="json", exclude_unset=True, context={"redact_sensitive_env": False})


def _job_config(
    config: RunConfig,
    folder: Path,
    paths: tuple[Path, ...],
    *,
    phase: str = "coding",
    job_name: str | None = None,
) -> JobConfig:
    job_name = job_name or phase
    saved = folder / "jobs" / job_name / "config.json"
    if saved.exists():
        return JobConfig.model_validate_json(saved.read_text())
    if phase == "verify":
        agent = HarborAgentConfig(name="nop")
    elif config.agent.name == "oracle":
        # Oracle is a complete pipeline sanity check, including reference search.
        agent = HarborAgentConfig(name="oracle")
    else:
        agent = HarborAgentConfig.model_validate(
            _harbor_agent(config.search.agent if phase == "search" else config.agent)
        )
    if phase != "verify":
        role = "search" if phase == "search" else "agent"
        home = f"/workspace/.harbor/{role}" if phase == "coding" else f"/home/{role}"
        agent.env = {**agent.env, "HOME": home, "USER": role, "LOGNAME": role}
        runtime = (
            "/workspace/runtime/search" if phase == "search" else "/workspace/runtime/candidate"
        )
        agent.env.update(TMPDIR=f"{runtime}/tmp", XDG_CACHE_HOME=f"{runtime}/cache")
    return JobConfig(
        job_name=job_name,
        jobs_dir=folder / "jobs",
        n_concurrent_trials=_concurrency(config, paths),
        environment=job_environment(config)
        if not paths or (paths[0] / "environment/runtime.json").exists()
        else EnvironmentConfig(import_path="eval.providers.docker:ConversationDockerEnvironment"),
        agents=[agent],
        tasks=[TaskConfig(path=path) for path in paths],
    )


def _trial_directories(folder: Path, phase: str):
    """Original jobs and immutable one-task continuation jobs share a logical phase."""
    state_path = folder / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    records = state.get("jobs")
    jobs = (
        [folder / "jobs" / record["name"] for record in records if record["phase"] == phase]
        if records is not None
        else [
            job
            for job in sorted((folder / "jobs").glob(f"{phase}*"))
            if job.name == phase or job.name.startswith(f"{phase}-resume-")
        ]
    )
    for job in jobs:
        if not job.exists():
            continue
        for path in sorted(job.iterdir()):
            if path.is_dir() and (
                (path / "result.json").exists() or (path / "config.json").exists()
            ):
                yield path


def _stage_task(folder: Path, phase: str, task_id: str) -> Path:
    """Saved runs retain their original layout; fresh runs group tasks by stage."""
    manifest = json.loads((folder / "prepared.json").read_text())
    if manifest.get("schema", 1) < 2:
        return folder / phase / task_id
    names = {"search": "search", "coding": "agent", "verification": "evaluation"}
    return folder / "tasks" / names[phase] / task_id


def _trial_results(folder: Path, job: str) -> dict[str, tuple[Path, dict]]:
    """Select the explicitly current attempt, including failed or unfinished retries."""
    state_path = folder / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    current = state.get("attempts", {}).get(job, {})
    results = {}
    for trial in _trial_directories(folder, job):
        path = trial / "result.json"
        if path.exists():
            result = json.loads(path.read_text())
        else:
            result = {"config": json.loads((trial / "config.json").read_text())}
        task_id = Path(result["config"]["task"]["path"]).name
        if task_id in current and trial.parent.name != current[task_id]["job"]:
            continue
        if _outcome(result)[0] == "complete" and not _handoff_ready(path, job, result):
            result["_missing_handoff"] = True
        previous = results.get(task_id)
        if previous is None or path.parent.stat().st_mtime >= previous[0].parent.stat().st_mtime:
            results[task_id] = path, result
    return results


def _outcome(result: dict) -> tuple[str, dict | None]:
    """Keep infrastructure errors separate from native verifier and budget verdicts."""
    error = result.get("exception_info")
    if (
        not result.get("finished_at")
        or (error and error.get("exception_type") != "AgentTimeoutError")
        or result.get("_missing_handoff")
        or any(step.get("exception_info") for step in result.get("step_results") or ())
    ):
        return "infrastructure_error", None
    rewards = (result.get("verifier_result") or {}).get("rewards")
    if not rewards:
        return "infrastructure_error", None
    return ("candidate_failed" if rewards.get("reward") == 0 else "complete"), rewards


def _artifact(result_path: Path, source: str) -> Path | None:
    """Resolve only a declared directory from Harbor's single-task manifest."""
    artifacts = result_path.parent / "artifacts"
    manifest = artifacts / "manifest.json"
    if not manifest.exists():
        return None
    entries = json.loads(manifest.read_text())
    entry = next(
        (
            entry
            for entry in entries
            if entry["source"] == source
            and entry["status"] == "ok"
            and entry["type"] == "directory"
            and entry.get("service") in (None, "main")
        ),
        None,
    )
    if entry is None:
        return None
    # Harbor's manifest destination includes "artifacts/", relative to the trial.
    return artifacts.parent / entry["destination"]


def _handoff_ready(result_path: Path, phase: str, result: dict) -> bool:
    """A native reward alone does not complete an unavailable phase handoff."""
    if phase == "search":
        task = Path(result["config"]["task"]["path"])
        portable = (task / "environment/runtime.json").exists()
        source = _artifact(
            result_path, "/var/lib/local-eval/reference" if portable else "/root/eval/reference"
        )
        return source is not None and (source / "reference.json").is_file()
    if phase == "coding":
        source = _artifact(result_path, "/var/lib/local-eval/submission")
        return source is not None and source.is_dir()
    return True


def _planned(folder: Path) -> dict[str, dict]:
    """Selected catalog tasks; classification follows reference resolution."""
    manifest = json.loads((folder / "prepared.json").read_text())
    return {task["task_id"]: task for task in manifest.get("tasks", ())}


def _no_baselines(folder: Path, task_id: str) -> bool:
    manifest = json.loads((folder / "prepared.json").read_text())
    task = _planned(folder)[task_id]
    return not manifest["assets"][task["model"]]["baselines"]


def _selection(folder: Path, task_id: str, reference: Path | None, results: dict) -> dict | None:
    """The supplied or searched handoff, once available."""
    if reference is not None:
        path = reference / f"{task_id}.json"
        return json.loads(path.read_text()) if path.is_file() else None
    recorded = results.get(task_id)
    if recorded is None or _outcome(recorded[1])[0] != "complete":
        return None
    source = _artifact(recorded[0], "/var/lib/local-eval/reference")
    if source is None or not (source / "reference.json").is_file():
        return None
    return json.loads((source / "reference.json").read_text())


def _render_classified(folder: Path, config: RunConfig | None, task_id: str, kind: str) -> None:
    """Render coding and verification only once their reference outcome is known."""
    from .tasks import ImageSpec, render_task

    coding = _stage_task(folder, "coding", task_id)
    verification = _stage_task(folder, "verification", task_id)
    metadata = coding / "environment/task.json"
    if (
        metadata.is_file()
        and json.loads(metadata.read_text()).get("type") == kind
        and all((path / "task.toml").is_file() for path in (coding, verification))
    ):
        return
    config = config or _resume_config(folder)
    manifest = json.loads((folder / "prepared.json").read_text())
    task = Task(**{**_planned(folder)[task_id], "type": kind})
    description = manifest["image"]
    image = ImageSpec(
        description["category"], Path(description["context"]), description["identity"], description
    )
    inputs = {
        **task_inputs(manifest["inputs"], task),
        "quality": manifest["quality"],
    }
    render_task(
        task,
        config,
        image,
        folder,
        inputs,
        manifest["assets"][task.model],
        include_search=False,
    )


def _coding_tasks(
    folder: Path,
    paths: tuple[Path, ...],
    *,
    reference: Path | None = None,
    config: RunConfig | None = None,
) -> tuple[Path, ...]:
    """Classify each pair, then transfer SOTA reference parameters.

    Candidates keep their canonical model checkpoint. Search supplies only the
    reference parameters; the reference retains its own mapped checkpoint.
    """
    results = _trial_results(folder, "search")
    eligible = []
    for path in paths:
        task = _stage_task(folder, "coding", path.name)
        selection = (
            {"supported": False}
            if _no_baselines(folder, path.name)
            else _selection(folder, path.name, reference, results)
        )
        if selection is None:
            continue
        kind = "SOTA" if selection and selection.get("supported") is not False else "SLO"
        _render_classified(folder, config, path.name, kind)
        if kind == "SLO":
            eligible.append(task)
            continue
        description = json.loads((task / "environment/task.json").read_text())
        target_checkpoint = {"source": description["assets"]["checkpoint"]}
        target = task / "reference/metadata"
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "reference.json", _handoff(selection))
        for phase in ("coding", "verification"):
            runtime_path = _stage_task(folder, phase, path.name) / "environment/runtime.json"
            runtime = json.loads(runtime_path.read_text())
            task_path = _stage_task(folder, phase, path.name) / "environment/task.json"
            metadata = json.loads(task_path.read_text())
            if metadata.get("schema", 1) >= 5:
                runtime["target"] = target_checkpoint
                metadata["target"] = target_checkpoint
                _write_json(task_path, metadata)
            runtime["files"] = [
                item
                for item in runtime["files"]
                if item["target"] not in ("/var/lib/local-eval/reference", "/inputs/reference.json")
            ]
            runtime["files"].append(
                {
                    "source": str(target),
                    "target": "/var/lib/local-eval/reference",
                    "read_only": True,
                }
            )
            if phase == "coding":
                runtime["files"].append(
                    {
                        "source": str(target / "reference.json"),
                        "target": "/inputs/reference.json",
                        "read_only": True,
                    }
                )
            _write_json(runtime_path, runtime)
        eligible.append(task)
    return tuple(eligible)


def _verification_tasks(folder: Path, paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Seed fresh verification with the successful coding attempt's submission only."""
    results = _trial_results(folder, "coding")
    eligible = []
    for path in paths:
        recorded = results.get(path.name)
        if recorded is None or _outcome(recorded[1])[0] != "complete":
            continue
        result_path, result = recorded
        submission = _artifact(result_path, "/var/lib/local-eval/submission")
        if submission is None or not submission.is_dir():
            continue
        task = _stage_task(folder, "verification", path.name)
        target = task / "environment/submission"
        handoff = task / "environment/submission-attempt.json"
        if not handoff.exists() or json.loads(handoff.read_text())["attempt"] != result["id"]:
            staging = task / (".submission-" + str(uuid.uuid4()))
            shutil.copytree(submission, staging, symlinks=True)
            if target.exists():
                shutil.rmtree(target)
            staging.rename(target)
            _write_json(handoff, {"attempt": result["id"]})
        runtime_path = task / "environment/runtime.json"
        runtime = json.loads(runtime_path.read_text())
        runtime["files"] = [item for item in runtime["files"] if item["target"] != "/workspace"]
        runtime["files"].append(
            {
                "source": str(target),
                "target": "/workspace",
                "read_only": False,
            }
        )
        _write_json(runtime_path, runtime)
        eligible.append(task)
    return tuple(eligible)


def _slo_summary(report: dict) -> dict:
    """The placement and per-tier evidence a results table needs."""
    scored = report["slo"]
    return {
        key: scored[key]
        for key in ("kind", "tier", "tier_met", "score", "coverage", "throughput")
        if key in scored
    } | {
        "tiers": [
            {
                key: tier[key]
                for key in ("id", "met", "score", "attainment", "interval", "uncertain", "x")
                if key in tier
            }
            for tier in scored["tiers"]
        ]
    }


def _final_results(folder: Path, paths: tuple[Path, ...]) -> dict:
    results = {phase: _trial_results(folder, phase) for phase in ("search", "coding", "verify")}
    timings = {
        phase: collect_timings(_trial_directories(folder, phase))
        for phase in ("search", "coding", "verify")
    }
    manifest = json.loads((folder / "prepared.json").read_text())
    tasks = {}
    for path in paths:
        recorded = {phase: trials.get(path.name) for phase, trials in results.items()}
        supplied = folder / "provided-reference" / f"{path.name}.json"
        static = _no_baselines(folder, path.name)
        selection = _selection(
            folder, path.name, supplied.parent if supplied.is_file() else None, results["search"]
        )
        kind = (
            "SLO"
            if static or (selection or {}).get("supported") is False
            else "SOTA"
            if selection
            else None
        )
        status, rewards, phase = "infrastructure_error", None, "classification"
        for phase in ("coding", "verify") if supplied.is_file() or static else recorded:
            trial = recorded[phase]
            status, rewards = _outcome(trial[1]) if trial else ("infrastructure_error", None)
            if status != "complete":
                break
        checked = recorded["verify"]
        report = None
        if status == "complete":
            step = "slo" if kind == "SLO" else "measure"
            path_report = checked[0].parent / "steps" / step / "verifier" / "report.json"
            report = json.loads(path_report.read_text()) if path_report.exists() else None
            if report is not None and report["outcome"] == "candidate_failure":
                status = "candidate_failed"
        if status == "candidate_failed":
            rewards = {"score": 0.0}
        score = (
            rewards.get("score") if rewards and status in ("complete", "candidate_failed") else None
        )
        tasks[path.name] = {
            "status": status,
            "stage": phase,
            **({"type": kind} if kind is not None else {}),
            "rewards": rewards,
            "score": score,
            "timings": {
                "search": timings["search"].get(path.name)
                or (
                    {"reused": True}
                    if supplied.is_file()
                    else {"skipped": True}
                    if static
                    else None
                ),
                "coding": timings["coding"].get(path.name),
                "verification": timings["verify"].get(path.name),
            },
            "evaluation_protocol": (checked[1].get("evaluation_protocol") if checked else None)
            or {
                "dataset": manifest.get("dataset_identity", "legacy"),
            },
            "search_result": str(recorded["search"][0]) if recorded["search"] else None,
            "coding_result": str(recorded["coding"][0]) if recorded["coding"] else None,
            "verification_result": str(checked[0]) if checked else None,
            **({"provided_reference": str(supplied)} if supplied.is_file() else {}),
            **(
                {
                    key: report[key]
                    for key in ("candidate_checkpoint", "reference_checkpoint")
                    if key in report
                }
                if report
                else {}
            ),
            **({"gains": report["gains"]} if report and "gains" in report else {}),
            **({"machine": report["machine"]} if report and "machine" in report else {}),
            **({"slo": _slo_summary(report)} if report and "slo" in report else {}),
        }
    _write_json(folder / "result.json", {"tasks": tasks})
    return tasks


def _collect_costs(folder: Path) -> None:
    """Save finalized coding evidence; search never creates a cost attempt."""
    attempts = []
    known = {attempt.attempt_id for attempt in read_ledger(folder / "costs.json")}
    for trial in _trial_directories(folder, "coding"):
        result_path = trial / "result.json"
        if not result_path.exists():
            identity = f"interrupted:{trial.parent.name}/{trial.name}"
            if identity in known:
                continue
            settings = json.loads((trial / "config.json").read_text())
            agent = HarborAgentConfig.model_validate(settings["agent"])
            if agent.name in ("oracle", "nop"):
                continue
            trajectory = partial_trajectory(
                agent, _saved_inputs(trial) / "agent", folder / "interrupted-usage" / trial.name
            )
            if trajectory is None and not (trial / "agent-start.json").exists():
                continue
            recorded = json.loads(trajectory.read_text()) if trajectory is not None else {}
            metrics = recorded.get("final_metrics") or {}
            cost = metrics.get("total_cost_usd")
            attempts.append(
                Attempt(
                    identity,
                    Path(settings["task"]["path"]).name,
                    trial.name,
                    "implementation",
                    agent.name,
                    (recorded.get("agent") or {}).get("version"),
                    agent.model_name or (recorded.get("agent") or {}).get("model_name"),
                    None,
                    Usage(
                        metrics.get("total_prompt_tokens"),
                        metrics.get("total_cached_tokens"),
                        metrics.get("total_completion_tokens"),
                        cost,
                        "reported" if cost is not None else "unknown",
                    ),
                    "interrupted.trajectory.final_metrics",
                    tuple(
                        str(path)
                        for path in (trajectory, trial / "config.json")
                        if path is not None
                    ),
                )
            )
            attempts[-1] = with_claude_cost(
                attempts[-1], _saved_inputs(trial) / "agent/claude-code.txt"
            )
            continue
        result = json.loads(result_path.read_text())
        cancelled = (result.get("exception_info") or {}).get("exception_type") == "CancelledError"
        if not result.get("finished_at") and not cancelled:
            continue
        if str(result["id"]) in known:
            continue
        agent_config = (result.get("config") or {}).get("agent") or {}
        if agent_config.get("name") in ("oracle", "nop"):
            # Native solutions execute scripts, without coding-model usage.
            continue
        if (
            result.get("agent_result") is None
            and not result.get("agent_execution")
            and not (trial / "agent-start.json").exists()
        ):
            continue  # Allocation/setup never started a coding agent.
        trajectory = result_path.parent / "agent" / "trajectory.json"
        if not trajectory.exists():
            preserved = folder / "interrupted-usage" / trial.name / "trajectory.json"
            trajectory = (
                preserved
                if preserved.is_file()
                else partial_trajectory(
                    HarborAgentConfig.model_validate(agent_config),
                    _saved_inputs(trial) / "agent",
                    preserved.parent,
                )
            )
        evidence = [str(result_path)]
        trajectory_data = None
        if trajectory is not None and trajectory.exists():
            evidence.append(str(trajectory))
            trajectory_data = json.loads(trajectory.read_text())
        attempts.append(
            from_harbor_result(
                result,
                trajectory=trajectory_data,
                evidence=evidence,
            )
        )
        stream = trial / "agent/claude-code.txt"
        if not stream.is_file():
            stream = _saved_inputs(trial) / "agent/claude-code.txt"
        attempts[-1] = with_claude_cost(attempts[-1], stream)
    ledger = update_ledger(folder / "costs.json", attempts)
    _write_json(folder / "cost-summary.json", summarize(ledger))


def _saved_inputs(trial: Path) -> Path:
    """Use the last complete live snapshot when a lost sandbox could not return its logs."""
    current = trial / "checkpoint/latest.json"
    if current.exists() and not tuple((trial / "agent/sessions").rglob("*.jsonl")):
        from .providers.checkpoint import conversations

        destination = trial / "checkpoint/logs"
        conversations(current, destination / "agent")
        return destination
    latest = trial / "live-checkpoints/latest.json"
    if latest.is_file() and not tuple((trial / "agent/sessions").rglob("*.jsonl")):
        return Path(json.loads(latest.read_text())["path"])
    return trial


def _resume_task(
    folder: Path, phase: str, task: Path, trial: Path, agent
) -> tuple[Path, Path | None]:
    """Restore this phase's workspace and native chat, without changing old attempts."""
    from .providers.checkpoint import conversations

    root = folder / "resume" / phase / str(uuid.uuid4())
    target = root / task.name
    shutil.copytree(task, target, symlinks=True)
    runtime_path = target / "environment/runtime.json"
    runtime = json.loads(runtime_path.read_text())
    result_path = trial / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    settings = result.get("config") or json.loads((trial / "config.json").read_text())
    previous = Path(settings["task"]["path"]) / "environment/runtime.json"
    inherited = json.loads(previous.read_text())
    hardware = trial / "hardware.json"
    expected = inherited.get("expected_hardware") or (
        json.loads(hardware.read_text()).get("fingerprint")
        if hardware.exists() and (trial / "inputs-ready.json").exists()
        else None
    )
    if expected:
        runtime["expected_hardware"] = expected
    snapshot = trial / "checkpoint/latest.json"
    staged = (trial / "inputs-ready.json").exists() or (trial / "agent-start.json").exists()
    saved_ready = (
        staged and snapshot.exists() and json.loads(snapshot.read_text()).get("ready", True)
    )
    if inherited.get("checkpoint") and not saved_ready:
        # A failed allocation can capture the empty image before restoring inputs.
        # It must not replace the workspace that this attempt was meant to resume.
        snapshot = Path(inherited["checkpoint"])
    if snapshot.exists():
        runtime["checkpoint"] = str(snapshot)
    environment = trial / "environment.json"
    if environment.exists():
        runtime["previous_environment"] = str(environment)
    elif inherited.get("previous_environment"):
        runtime["previous_environment"] = inherited["previous_environment"]
    retained = trial / "retained.json"
    if retained.exists():
        runtime["retained_environment"] = str(retained)
    progress_path = trial / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    complete = progress.get("agent_complete") or inherited.get("agent_complete", False)
    if phase != "verify" and complete:
        finished = progress.get("agent_finished_at")
        fresh = snapshot.exists() and (
            not finished
            or json.loads(snapshot.read_text()).get("at", 0)
            >= datetime.fromisoformat(finished).timestamp()
        )
        if fresh or retained.exists():
            runtime["agent_complete"] = True
        elif result_path.exists():
            source = _artifact(
                result_path,
                "/var/lib/local-eval/reference"
                if phase == "search"
                else "/var/lib/local-eval/submission",
            )
            if source and source.is_dir():
                if phase == "search":
                    selected = json.loads((source / "reference.json").read_text())
                    if selected.get("supported") is False:
                        for attempt in selected["attempts"]:
                            if attempt.get("log"):
                                attempt["log"] = str(
                                    Path("/workspace/output/reference") / attempt["log"]
                                )
                            attempt.pop("log_error", None)
                        selected = {key: selected[key] for key in ("supported", "attempts")}
                    else:
                        selected = _handoff(selected)
                    selection = root / "selection.json"
                    _write_json(selection, selected)
                    runtime["files"].extend(
                        [
                            {
                                "source": str(source),
                                "target": "/workspace/output/reference",
                                "owner": "search",
                                "read_only": False,
                            },
                            {
                                "source": str(selection),
                                "target": "/workspace/output/selection.json",
                                "owner": "search",
                                "read_only": False,
                            },
                        ]
                    )
                else:
                    runtime["files"].append(
                        {
                            "source": str(source),
                            "target": "/workspace/output/submission",
                            "owner": "agent",
                            "read_only": False,
                        }
                    )
                runtime["agent_complete"] = True
    trajectory = None
    if phase == "verify":
        # Fresh setup recreates dependencies and weights. Native step markers
        # preserve completed evidence; preparation always runs in a new environment.
        if retained.exists() or snapshot.exists():
            runtime["completed_steps"] = progress.get("completed_steps", [])
        if retained.exists():
            _remaining_steps(target, [s for s in runtime["completed_steps"] if s != "prepare"])
    elif not runtime.get("agent_complete") and agent.name not in ("oracle", "nop"):
        agent = agent.model_copy(
            update={
                "load_trajectory": agent.load_trajectory
                or (settings.get("agent") or {}).get("load_trajectory")
            }
        )
        logs = trial / "agent"
        started = (trial / "agent-start.json").exists() or bool(result.get("agent_execution"))
        exported = tuple((logs / "sessions").rglob("*.jsonl"))
        if snapshot.exists() and not exported and (started or not agent.load_trajectory):
            # Final native exports include events after the last periodic capture.
            # A setup-only retry must keep the trajectory it was about to load.
            logs = conversations(snapshot, root / "native")
        # A checkpoint lost before the first event has no recoverable chat. Its
        # failed attempt stays recorded; retry the original task instruction.
        if (
            tuple((logs / "sessions").rglob("*.jsonl"))
            or (logs / "trajectory.json").exists()
            or agent.load_trajectory
        ):
            history = prepare_history(agent, logs, root / "history", coding=phase == "coding")
            trajectory = history.trajectory
            runtime["files"].extend(history.files)
            if history.runtime is not None:
                runtime["resume"] = history.runtime
        elif started:
            _write_json(
                root / "history-unavailable.json",
                {"trial": str(trial), "action": "restart original prompt"},
            )
    _write_json(runtime_path, runtime)
    return target, trajectory


def _remaining_steps(task: Path, completed: list[str]) -> None:
    """Keep native Harbor steps from the first missing trusted completion record."""
    definition = task / "task.toml"
    if not definition.exists():
        return
    text = definition.read_text()
    blocks = re.split(r"(?m)^\[\[steps\]\]\s*$", text)
    pending = []
    unfinished = False
    for block in blocks[1:]:
        step = tomllib.loads("[[steps]]\n" + block)["steps"][0]
        unfinished = unfinished or step["name"] not in completed
        if unfinished:
            pending.append("[[steps]]\n" + block)
    definition.write_text(blocks[0] + "".join(pending))


async def _checkpoint(event) -> None:
    """Record successful native step outputs before Harbor starts the next operation."""
    trial = event.config.trials_dir / event.trial_name
    path = trial / "progress.json"
    value = json.loads(path.read_text()) if path.exists() else {}
    runtime_path = event.config.task.path / "environment/runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    completed = list(runtime.get("completed_steps", ()))
    for step in event.result.step_results or ():
        if step.exception_info is not None or step.verifier_result is None:
            break
        if not step.verifier_result.rewards or step.verifier_result.rewards.get("reward") != 1:
            break
        if step.step_name not in completed:
            completed.append(step.step_name)
    value["completed_steps"] = completed
    if not event.result.step_results and event.result.agent_execution is not None:
        value["agent_complete"] = (
            event.result.agent_execution.finished_at is not None
            and event.result.exception_info is None
        )
        if value["agent_complete"]:
            value["agent_finished_at"] = event.result.agent_execution.finished_at.isoformat()
    _write_json(path, value)


async def _agent_started(event) -> None:
    """A host-side native hook makes hard interruptions distinguishable from setup."""
    steps = event.result.step_results or ()
    _write_json(
        event.config.trials_dir / event.trial_name / "agent-start.json",
        {"started_at": event.timestamp.isoformat(), "step": steps[-1].step_name if steps else None},
    )
    await _checkpoint(event)


async def _verification_started(event) -> None:
    await _checkpoint(event)


async def _trial_ended(event) -> None:
    """Publish final completion only after native results and artifacts are saved."""
    result = event.result.model_dump(mode="json")
    status, rewards = _outcome(result)
    if status == "infrastructure_error":
        return
    trial = event.config.trials_dir / event.trial_name
    runtime_path = event.config.task.path / "environment/runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    phase = runtime.get("phase", event.config.task.path.parent.name)
    if status == "complete" and not _handoff_ready(trial / "result.json", phase, result):
        return
    _write_json(trial / "completed.json", {"status": status, "rewards": rewards})
    record = trial / "retained.json"
    if record.exists():
        from .providers.docker import release_retained

        await release_retained(event.config.task.path / "environment", record)


async def _execute_job(config: JobConfig) -> None:
    job = await Job.create(config)
    job.on_agent_started(_agent_started)
    job.on_verification_started(_verification_started)
    job.on_trial_ended(_trial_ended)
    await job.run()


async def _quality_ready(folder: Path, paths: tuple[Path, ...], state: dict) -> None:
    """Resolve the frozen per-model quality bounds before allocating verification."""
    from datasets import load_from_disk

    from .quality import model_range

    manifest = json.loads((folder / "prepared.json").read_text())
    rows = load_from_disk(str(Path(manifest["quality"]["source"]) / "questions")).to_list()
    for path in paths:
        task = json.loads((path / "environment/task.json").read_text())
        model_range(task["model"], rows)


def _agent_seconds(trial: Path) -> float:
    started = trial / "agent-start.json"
    if not started.exists():
        return 0
    start = datetime.fromisoformat(json.loads(started.read_text())["started_at"]).timestamp()
    result = trial / "result.json"
    finished = (
        (json.loads(result.read_text()).get("agent_execution") or {}).get("finished_at")
        if result.exists()
        else None
    )
    if finished:
        return max(0, datetime.fromisoformat(finished).timestamp() - start)
    checkpoint = trial / "checkpoint/latest.json"
    return (
        max(0, json.loads(checkpoint.read_text()).get("at", start) - start)
        if checkpoint.exists()
        else 0
    )


async def _run_phase(
    config: RunConfig,
    folder: Path,
    paths: tuple[Path, ...],
    phase: str,
    *,
    job_name: str | None = None,
    state: dict | None = None,
) -> None:
    """Native retries load each task's own conversation; completed trials stay immutable."""
    job_config = _job_config(config, folder, paths, phase=phase, job_name=job_name)
    trials = tuple(_trial_directories(folder, phase))
    outcomes = _trial_results(folder, phase)
    pending = tuple(
        task
        for task in paths
        if task.name not in outcomes
        or _outcome(outcomes[task.name][1])[0] == "infrastructure_error"
    )
    if not pending:
        return
    agent_name = job_config.agents[0].name
    reset_at = (state or {}).get("agent_limits", {}).get(agent_name, 0)
    if reset_at > datetime.now(UTC).timestamp():
        raise _AgentLimit(agent_name, phase, reset_at)
    histories = {}
    for trial in trials:
        path = trial / "result.json"
        settings = (
            json.loads(path.read_text())["config"]
            if path.exists()
            else json.loads((trial / "config.json").read_text())
        )
        histories.setdefault(Path(settings["task"]["path"]).name, []).append(trial)

    async def execute(job: JobConfig) -> None:
        if state is not None and "jobs" in state:
            if not any(record["name"] == job.job_name for record in state["jobs"]):
                state["jobs"].append(
                    {
                        "name": job.job_name,
                        "phase": phase,
                        "task_ids": [task.path.name for task in job.tasks],
                    }
                )
            attempts = state.setdefault("attempts", {}).setdefault(phase, {})
            for task in job.tasks:
                attempts[task.path.name] = {"job": job.job_name}
            _write_json(folder / "state.json", state)
        await _execute_job(job)
        if state is not None and job.agents[0].name == "claude-code":
            for trial in (job.jobs_dir / job.job_name).iterdir():
                if (
                    not (trial / "config.json").is_file()
                    or not (trial / "agent-start.json").is_file()
                ):
                    # Setup may restore an earlier account's rejected-quota log.
                    # Only this attempt's agent execution can establish a limit.
                    continue
                reset_at = rate_limit_reset(_saved_inputs(trial) / "agent")
                if reset_at is not None and reset_at > datetime.now(UTC).timestamp():
                    state.setdefault("agent_limits", {})["claude-code"] = reset_at
                    _write_json(folder / "state.json", state)
                    raise _AgentLimit("claude-code", phase, reset_at)

    if (
        not any(task.name in histories for task in pending)
        and not (folder / "jobs" / job_config.job_name).exists()
    ):
        job_config.tasks = [TaskConfig(path=task) for task in pending]
        await execute(job_config)
        return
    for task in pending:
        matches = histories.get(task.name, [])
        retry = job_config.model_copy(deep=True)
        retry.job_name = f"{job_config.job_name}-resume-{uuid.uuid4()}"
        retry.n_concurrent_trials = 1
        if phase != "verify" and retry.agents[0].name not in ("oracle", "nop"):
            budget = (
                config.search.agent.budget_seconds
                if phase == "search"
                else config.agent.budget_seconds
            )
            remaining = budget - sum(_agent_seconds(t) for t in matches)
            retry.agents[0].override_timeout_sec = max(1, int(remaining))
        if matches:
            previous = (
                outcomes[task.name][0].parent
                if task.name in outcomes
                else max(matches, key=lambda path: path.stat().st_mtime)
            )
            latest_environment = previous / "environment.json"
            if (
                not (previous / "inputs-ready.json").exists()
                and not (previous / "agent-start.json").exists()
            ):
                # Admission/setup failures have no newer workspace or conversation.
                # Keep the latest attempt for verdicts, but recover the last ready one.
                ready = [
                    trial
                    for trial in matches
                    if (trial / "inputs-ready.json").exists()
                    or (trial / "agent-start.json").exists()
                ]
                if ready:
                    previous = max(ready, key=lambda path: path.stat().st_mtime)
            resumed, history = _resume_task(folder, phase, task, previous, retry.agents[0])
            if latest_environment.exists():
                runtime = resumed / "environment/runtime.json"
                metadata = json.loads(runtime.read_text())
                metadata["previous_environment"] = str(latest_environment)
                _write_json(runtime, metadata)
            retry.tasks = [TaskConfig(path=resumed)]
            retry.agents[0].load_trajectory = str(history) if history is not None else None
            progress = previous / "progress.json"
            runtime_path = resumed / "environment/runtime.json"
            completed_agent = runtime_path.exists() and json.loads(runtime_path.read_text()).get(
                "agent_complete", False
            )
            if phase != "verify" and (
                completed_agent
                or (
                    (previous / "retained.json").is_file()
                    and progress.is_file()
                    and json.loads(progress.read_text()).get("agent_complete")
                )
            ):
                retry.agents[0] = HarborAgentConfig(name="nop")
        else:
            retry.tasks = [TaskConfig(path=task)]
            retry.agents[0].load_trajectory = None
        await execute(retry)


def _resources(config: RunConfig) -> asyncio.Queue:
    resources = asyncio.Queue()
    ids = config.environment.kwargs.get("gpu_ids", [])
    if config.environment.provider == "docker" and ids:
        groups = ids if isinstance(ids[0], list) else [ids]
        flat = [str(gpu) for group in groups for gpu in group]
        if len(flat) != len(set(flat)) or any(not group for group in groups):
            raise ValueError("local GPU groups must be nonempty and must not overlap")
        for group in groups[: config.concurrency]:
            resources.put_nowait([str(gpu) for gpu in group])
    else:
        for _ in range(config.concurrency):
            resources.put_nowait([])
    return resources


def _assign(folder: Path, task_id: str, group: list[str]) -> None:
    for name in ("search", "coding", "verification"):
        runtime_path = _stage_task(folder, name, task_id) / "environment/runtime.json"
        if runtime_path.exists():
            runtime = json.loads(runtime_path.read_text())
            runtime["gpu_ids"] = group
            _write_json(runtime_path, runtime)


def _state(folder: Path) -> dict:
    path = folder / "state.json"
    state = (
        json.loads(path.read_text())
        if path.exists()
        else {"run_id": folder.name, "jobs": [], "tasks": {}}
    )
    state["cycle"] = state.get("cycle", 0) + 1
    state["status"] = "running"
    _write_json(path, state)
    return state


async def run(config_path: Path | None = None, *, resume: Path | None = None) -> Path:
    """One resource pool, one run folder, independent pipelines and bounded recovery."""
    fresh = resume is None
    config = load_config(config_path) if fresh else _resume_config(resume.resolve())
    if config.agent is None:
        raise ValueError("provide agent or agents")
    folder = config.out.resolve() / str(uuid.uuid4()) if fresh else resume.resolve()
    if fresh:
        folder.mkdir(parents=True)
    lock = None
    try:
        if fresh:
            _save_config(config, folder)
            _freeze_host(folder)
            config = _resume_config(folder)
        print(f"Run folder: {folder}", flush=True)
        lock = (folder / "run.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = _state(folder)
    except BaseException:
        if lock is not None:
            lock.close()
        if fresh:
            shutil.rmtree(folder)
        raise
    contexts = [(config, folder, state)]

    def save(directory, status):
        _write_json(directory / "state.json", status)

    try:
        if config.environment.provider == "docker":
            check_environment()
        if not (folder / "prepared.json").exists():
            await prepare(config, folder)
        manifest = json.loads((folder / "prepared.json").read_text())
        paths = tuple(Path(path) for path in manifest["task_paths"])
        resources = _resources(config)
        if config.agents:
            from .tasks import ImageSpec, render_task

            contexts = []
            description = manifest["image"]
            image = ImageSpec(
                description["category"],
                Path(description["context"]),
                description["identity"],
                description,
            )
            for key, agent in config.agents.items():
                branch = folder / "agents" / key
                branch.mkdir(parents=True, exist_ok=True)
                settings = replace(
                    config, agent=agent, agents={}, search=replace(config.search, reference=None)
                )
                if not (branch / "prepared.json").exists():
                    _save_config(settings, branch)
                    branch_paths = []
                    for value in manifest["tasks"]:
                        task = Task(**value)
                        inputs = {
                            **task_inputs(manifest["inputs"], task),
                            "quality": manifest["quality"],
                        }
                        branch_paths.append(
                            render_task(
                                task,
                                settings,
                                image,
                                branch,
                                inputs,
                                manifest["assets"][task.model],
                                include_search=False,
                            )
                        )
                    _write_json(
                        branch / "prepared.json",
                        {**manifest, "task_paths": list(map(str, branch_paths))},
                    )
                contexts.append((settings, branch, _state(branch)))

        async def phase(settings, directory, status, task_id, path, name):
            entry = status["tasks"].setdefault(task_id, {})
            for attempt in range(config.max_retries + 1):
                entry.update(status="running", phase=name, retry=attempt, cycle=status["cycle"])
                save(directory, status)
                try:
                    await _run_phase(
                        settings,
                        directory,
                        (path,),
                        name,
                        job_name=f"{name}-{task_id}",
                        state=status,
                    )
                    result = _trial_results(directory, name).get(task_id)
                    outcome = _outcome(result[1])[0] if result else "infrastructure_error"
                    if outcome != "infrastructure_error":
                        entry["status"] = outcome
                        save(directory, status)
                        return outcome
                    entry["error"] = "Harbor phase incomplete; restoring its saved checkpoint"
                except _AgentLimit:
                    raise
                except Exception as error:
                    entry["error"] = f"{type(error).__name__}: {error}"
                _collect_costs(directory)
                save(directory, status)
                if attempt < config.max_retries:
                    delay = min(10 * 2**attempt, 60)
                    print(
                        f"{directory.name}/{task_id}: resuming {name} in {delay}s "
                        f"({attempt + 1}/{config.max_retries})",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
            entry["status"] = "incomplete"
            save(directory, status)
            return "infrastructure_error"

        leases = {}

        async def search(path):
            task_id = path.name
            reference = folder / "references" / f"{task_id}.json"
            if reference.exists():
                return True
            if _no_baselines(folder, task_id):
                selected = {"supported": False}
            elif config.search.reference:
                selected = _selection(folder, task_id, config.search.reference, {})
            else:
                group = await resources.get()
                try:
                    _assign(folder, task_id, group)
                    outcome = await phase(
                        config,
                        folder,
                        state,
                        task_id,
                        _stage_task(folder, "search", task_id),
                        "search",
                    )
                    selected = (
                        _selection(folder, task_id, None, _trial_results(folder, "search"))
                        if outcome == "complete"
                        else None
                    )
                except _AgentLimit as limit:
                    resources.put_nowait(group)
                    await _wait_agent_limit(folder, state, task_id, limit)
                    return await search(path)
                except BaseException:
                    resources.put_nowait(group)
                    raise
                if outcome != "complete":
                    resources.put_nowait(group)
                    return False
                leases[task_id] = group
            reference.parent.mkdir(exist_ok=True)
            _write_json(reference, _handoff(selected))
            return True

        from .report import build

        async def pipeline(settings, directory, status, path, searched):
            task_id = path.name
            entry = status["tasks"].setdefault(task_id, {})
            if entry.get("status") in ("candidate_failed", "excluded") or (
                entry.get("status") == "complete" and entry.get("phase") == "verify"
            ):
                return
            if not await searched:
                entry["status"] = (
                    "candidate_failed"
                    if state["tasks"].get(task_id, {}).get("status") == "candidate_failed"
                    else "incomplete"
                )
                save(directory, status)
                return
            group = leases.pop(task_id) if task_id in leases else await resources.get()
            limited = None
            try:
                _assign(directory, task_id, group)
                supplied = directory / "provided-reference"
                source = folder / "references" / f"{task_id}.json"
                if source.exists():
                    supplied.mkdir(exist_ok=True)
                    shutil.copyfile(source, supplied / source.name)
                coding = _coding_tasks(directory, (path,), reference=supplied, config=settings)
                if not coding:
                    entry["status"] = "excluded"
                    return
                _assign(directory, task_id, group)
                outcome = await phase(settings, directory, status, task_id, coding[0], "coding")
                if outcome != "complete":
                    entry["status"] = "incomplete" if outcome == "infrastructure_error" else outcome
                    return
                verification = _verification_tasks(directory, coding)
                if not verification:
                    entry.update(
                        status="incomplete", error="completed coding submission is unavailable"
                    )
                    return
                await _quality_ready(directory, verification, status)
                outcome = await phase(
                    settings, directory, status, task_id, verification[0], "verify"
                )
                entry["status"] = "incomplete" if outcome == "infrastructure_error" else outcome
            except _AgentLimit as limit:
                limited = limit
            except Exception as error:
                entry.update(status="incomplete", error=f"{type(error).__name__}: {error}")
            finally:
                _final_results(directory, paths)
                _collect_costs(directory)
                save(directory, status)
                build([context for _, context, _ in contexts], folder / "results")
                resources.put_nowait(group)
            if limited is not None:
                await _wait_agent_limit(directory, status, task_id, limited)
                return await pipeline(settings, directory, status, path, searched)

        async with asyncio.TaskGroup() as workers:
            searches = {path.name: workers.create_task(search(path)) for path in paths}
            for settings, directory, status in contexts:
                for path in paths:
                    workers.create_task(
                        pipeline(settings, directory, status, path, searches[path.name])
                    )
        state["status"] = (
            "incomplete"
            if any(
                v.get("status") == "incomplete"
                for _, _, status in contexts
                for v in status["tasks"].values()
            )
            else "complete"
        )
        build([directory for _, directory, _ in contexts], folder / "results")
        if config.agents:
            ledger = update_ledger(
                folder / "costs.json",
                [
                    attempt
                    for _, directory, _ in contexts
                    for attempt in read_ledger(directory / "costs.json")
                ],
            )
            _write_json(folder / "cost-summary.json", summarize(ledger))
            _write_json(
                folder / "result.json",
                {
                    "agents": {
                        directory.name: str(directory / "result.json")
                        for _, directory, _ in contexts
                    }
                },
            )
    finally:
        error = sys.exception()
        if error is not None:
            state["status"] = (
                "interrupted"
                if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
                else "incomplete"
            )
            state["error"] = f"{type(error).__name__}: {error}"
        for _, directory, status in contexts:
            _collect_costs(directory)
            save(directory, status)
        save(folder, state)
        lock.close()
    return folder


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, help="configuration for a new run")
    parser.add_argument("--resume", type=Path, metavar="FOLDER", help="existing UUID run folder")
    args = parser.parse_args(argv)
    if (args.config is None) == (args.resume is None):
        parser.error("provide CONFIG or --resume FOLDER, exclusively")
    if args.resume:
        host = args.resume.resolve() / "host"
        frozen = host / "eval/run.py"
        if frozen.is_file() and frozen.resolve() != Path(__file__).resolve():
            result = subprocess.run(
                [sys.executable, "-m", "eval.run", "--resume", str(args.resume.resolve())],
                cwd=host,
                env={**os.environ, "PYTHONPATH": str(host)},
            )
            raise SystemExit(result.returncode)
    asyncio.run(run(args.config, resume=args.resume))


if __name__ == "__main__":
    main()
