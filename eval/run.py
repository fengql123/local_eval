"""Run native Harbor search, coding and verification jobs in the foreground."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import uuid
from pathlib import Path

import yaml
from harbor.job import Job
from harbor.models.job.config import JobConfig
from harbor.models.trial.config import AgentConfig as HarborAgentConfig
from harbor.models.trial.config import EnvironmentConfig, TaskConfig

from .config import AgentConfig, RunConfig, load_config
from .config import model as get_model
from .costs import from_harbor_result, summarize, update_ledger
from .device import check_environment
from .setup import prepare


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _agent_settings(agent: AgentConfig) -> dict:
    return {
        "name": agent.name,
        "model": agent.model,
        "kwargs": agent.kwargs,
        "auth": agent.auth,
        "env": list(agent.env),
        "budget": agent.budget_seconds,
    }


def _save_config(config: RunConfig, folder: Path) -> None:
    """Freeze all defaults and the absolute output root; store env names only."""
    value = {
        "dataset": config.dataset,
        "gpus": list(config.gpus) if config.gpus is not None else "all",
        "tasks": {
            "models": list(config.tasks.models) if config.tasks.models is not None else "all",
            "workloads": list(config.tasks.workloads)
            if config.tasks.workloads is not None
            else "all",
            "types": list(config.tasks.types) if config.tasks.types is not None else "all",
        },
        "agent": _agent_settings(config.agent),
        "search": {
            "mode": config.search.mode,
            "agent": _agent_settings(config.search.agent),
        },
        "evaluation": {
            "install_timeout": config.evaluation.install_timeout_seconds,
            "health_timeout": config.evaluation.health_timeout_seconds,
            "agent": _agent_settings(config.evaluation.agent),
        },
        "out": str(config.out.resolve()),
    }
    (folder / "run.yaml").write_text(yaml.safe_dump(value, sort_keys=False))


def _harbor_agent(agent: AgentConfig) -> dict:
    """Describe a native Harbor agent without resolving credentials or its class."""
    env = {name: "${" + name + "}" for name in agent.env}
    if agent.name == "codex" and agent.auth == "login":
        # Harbor 0.23's Codex adapter supplies and uploads the host login file.
        # Honor an explicit documented auth path supplied through agent.env.
        if "CODEX_AUTH_JSON_PATH" not in env:
            env["CODEX_FORCE_AUTH_JSON"] = "true"
    elif agent.name == "codex":
        # Explicit API-key mode must not inherit Harbor's host login opt-ins.
        env.update(CODEX_FORCE_AUTH_JSON="false", CODEX_AUTH_JSON_PATH="")
    # Harbor resolves built-in names and module.path:ClassName itself. Other
    # adapters own authentication; an omitted model stays unset in their config.
    model_options = {} if agent.model is None else {"model_name": agent.model}
    return HarborAgentConfig(
        name=agent.name,
        kwargs=agent.kwargs,
        env=env,
        override_timeout_sec=agent.budget_seconds,
        **model_options,
    ).model_dump(mode="json", exclude_unset=True, context={"redact_sensitive_env": False})


def _job_config(
    config: RunConfig, folder: Path, paths: tuple[Path, ...], *, phase: str = "coding"
) -> JobConfig:
    saved = folder / "jobs" / phase / "config.json"
    if saved.exists():
        job = JobConfig.model_validate_json(saved.read_text())
        if job.agents[0].name == "codex":
            # Harbor redacts literal AUTH controls; recover them from frozen YAML.
            role = config.search.agent if phase == "search" else config.agent
            controls = _harbor_agent(role)["env"]
            for name in ("CODEX_FORCE_AUTH_JSON", "CODEX_AUTH_JSON_PATH"):
                if name in controls:
                    job.agents[0].env[name] = controls[name]
        return job
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
        agent.env = {**agent.env, "HOME": f"/home/{role}", "USER": role, "LOGNAME": role}
    return JobConfig(
        job_name=phase,
        jobs_dir=folder / "jobs",
        n_concurrent_trials=1,
        environment=EnvironmentConfig(type="docker"),
        agents=[agent],
        tasks=[TaskConfig(path=path) for path in paths],
    )


def _trial_results(folder: Path, job: str) -> dict[str, tuple[Path, dict]]:
    """Use Harbor's recorded task path to pair stage trials for each task."""
    results = []
    for path in (folder / "jobs" / job).glob("*/result.json"):
        result = json.loads(path.read_text())
        results.append((path, result))
    results.sort(key=lambda item: (item[1].get("finished_at") or "", str(item[0])))
    return {Path(result["config"]["task"]["path"]).name: (path, result) for path, result in results}


def _outcome(result: dict) -> tuple[str, dict | None]:
    """Keep native infrastructure errors separate from a valid zero reward."""
    if (
        not result.get("finished_at")
        or result.get("exception_info")
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


def _coding_tasks(folder: Path, paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Hand off selected parameters and trusted read-only weight mounts after search."""
    results = _trial_results(folder, "search")
    assets = json.loads((folder / "prepared.json").read_text())["assets"]
    eligible = []
    for path in paths:
        recorded = results.get(path.name)
        if recorded is None or _outcome(recorded[1])[0] != "complete":
            continue
        source = _artifact(recorded[0], "/root/eval/reference")
        if source is None or not all(
            (source / name).is_file() for name in ("reference.json", "public.json")
        ):
            continue
        task = folder / "coding" / path.name
        metadata = json.loads((task / "environment" / "task.json").read_text())
        selection = json.loads((source / "reference.json").read_text())
        asset = assets[get_model(metadata["model"]).key]["baselines"][selection["baseline"]]
        target = task / "reference" / "metadata"
        weights = task / "reference" / "weights"
        if not (target / "reference.json").exists():
            if not weights.exists():
                if asset["path"] is None:
                    weights.mkdir()
                else:
                    weights.symlink_to(Path(asset["path"]).resolve(), target_is_directory=True)
            # Commit the protected reference last; it marks a complete handoff.
            for name in ("public.json", "reference.json"):
                temporary = target / ("." + name + "-" + str(uuid.uuid4()))
                shutil.copyfile(source / name, temporary)
                temporary.chmod(0o644 if name == "public.json" else 0o600)
                temporary.replace(target / name)
        eligible.append(task)
    return tuple(eligible)


def _verification_tasks(folder: Path, paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Seed native prepare workdirs once from successful coding artifacts."""
    results = _trial_results(folder, "coding")
    eligible = []
    for path in paths:
        recorded = results.get(path.name)
        if recorded is None:
            continue
        result_path, result = recorded
        if _outcome(result)[0] != "complete":
            continue
        submission = _artifact(result_path, "/workspace/submission")
        if submission is None or not submission.is_dir():
            continue
        task = folder / "verification" / path.name
        target = task / "steps" / "prepare" / "workdir" / "submission"
        if not target.exists():
            # Harbor uploads the entire workdir: interrupted copies stay outside it.
            staging = task / (".submission-" + str(uuid.uuid4()))
            shutil.copytree(submission, staging, symlinks=True)
            staging.rename(target)
        eligible.append(task)
    return tuple(eligible)


def _final_results(folder: Path, paths: tuple[Path, ...]) -> None:
    results = {phase: _trial_results(folder, phase) for phase in ("search", "coding", "verify")}
    tasks = {}
    for path in paths:
        recorded = {phase: trials.get(path.name) for phase, trials in results.items()}
        for phase in recorded:
            trial = recorded[phase]
            status, rewards = _outcome(trial[1]) if trial else ("infrastructure_error", None)
            if status != "complete":
                break
        checked = recorded["verify"]
        if status == "complete":
            report = checked[0].parent / "steps" / "measure" / "verifier" / "report.json"
            if report.exists() and json.loads(report.read_text())["outcome"] == "candidate_failure":
                status = "candidate_failed"
        tasks[path.name] = {
            "status": status,
            "stage": phase,
            "rewards": rewards,
            "search_result": str(recorded["search"][0]) if recorded["search"] else None,
            "coding_result": str(recorded["coding"][0]) if recorded["coding"] else None,
            "verification_result": str(checked[0]) if checked else None,
        }
    _write_json(folder / "result.json", {"tasks": tasks})


def _collect_costs(folder: Path) -> None:
    """Save finalized coding evidence; search never creates a cost attempt."""
    attempts = []
    for result_path in sorted((folder / "jobs" / "coding").glob("*/result.json")):
        result = json.loads(result_path.read_text())
        cancelled = (result.get("exception_info") or {}).get("exception_type") == "CancelledError"
        if not result.get("finished_at") and not cancelled:
            continue
        agent_config = (result.get("config") or {}).get("agent") or {}
        if agent_config.get("name") == "oracle":
            # Native solutions execute scripts, without coding-model usage.
            continue
        trajectory = result_path.parent / "agent" / "trajectory.json"
        evidence = [str(result_path)]
        trajectory_data = None
        if trajectory.exists():
            evidence.append(str(trajectory))
            trajectory_data = json.loads(trajectory.read_text())
        attempts.append(
            from_harbor_result(
                result,
                trajectory=trajectory_data,
                evidence=evidence,
            )
        )
    ledger = update_ledger(folder / "costs.json", attempts)
    _write_json(folder / "cost-summary.json", summarize(ledger))


def _remove_cancelled(folder: Path) -> None:
    """Remove cancelled trials only after _collect_costs persisted their usage."""
    for job in ("search", "coding", "verify"):
        for result_path in (folder / "jobs" / job).glob("*/result.json"):
            result = json.loads(result_path.read_text())
            if (result.get("exception_info") or {}).get("exception_type") == "CancelledError":
                shutil.rmtree(result_path.parent)


async def run(config_path: Path | None = None, *, resume: Path | None = None) -> Path:
    """Prepare once, then await Harbor in this process. Resume never takes a YAML."""
    if resume is not None:
        folder = resume.resolve()
        config = load_config(folder / "run.yaml")
    else:
        config = load_config(config_path)
        folder = config.out.resolve() / str(uuid.uuid4())
        folder.mkdir(parents=True)
        _save_config(config, folder)
    print(f"Run folder: {folder}", flush=True)
    state = {"run_id": folder.name, "status": "preparing"}
    _write_json(folder / "run.json", state)
    try:
        # Job.create removes incomplete trials. Preserve finalized usage first.
        _collect_costs(folder)
        if resume is not None:
            _remove_cancelled(folder)
        check_environment()
        saved = folder / "prepared.json"
        if saved.exists():
            paths = tuple(Path(path) for path in json.loads(saved.read_text())["task_paths"])
        else:
            paths = prepare(config, folder).task_paths
        search = tuple(folder / "search" / path.name for path in paths)
        job_config = _job_config(config, folder, search, phase="search")
        state["status"] = "searching"
        _write_json(folder / "run.json", state)
        job = await Job.create(job_config)
        await job.run()
        coding = _coding_tasks(folder, paths)
        if coding:
            state["status"] = "coding"
            _write_json(folder / "run.json", state)
            job = await Job.create(_job_config(config, folder, coding))
            await job.run()
        verification = _verification_tasks(folder, coding)
        if verification:
            state["status"] = "verifying"
            _write_json(folder / "run.json", state)
            job = await Job.create(_job_config(config, folder, verification, phase="verify"))
            await job.run()
        _final_results(folder, paths)
    finally:
        # Finalization is part of the run lifecycle, including interruption.
        # No exception is caught or converted into fallback evidence.
        error = sys.exception()
        state["status"] = (
            "interrupted"
            if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
            else "failed"
            if error
            else "finalizing"
        )
        if error is not None:
            state["error_type"] = type(error).__name__
        state["cost_collection_complete"] = False
        _write_json(folder / "run.json", state)
        _collect_costs(folder)
        state["cost_collection_complete"] = True
        if error is None:
            state["status"] = "complete"
        _write_json(folder / "run.json", state)
    return folder


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, help="configuration for a new run")
    parser.add_argument("--resume", type=Path, metavar="FOLDER", help="existing UUID run folder")
    args = parser.parse_args(argv)
    if (args.config is None) == (args.resume is None):
        parser.error("provide CONFIG or --resume FOLDER, exclusively")
    asyncio.run(run(args.config, resume=args.resume))


if __name__ == "__main__":
    main()
