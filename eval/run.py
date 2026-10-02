"""Prepare one foreground Harbor job or resume its saved configuration.

The job targets eval.agents:PhasedAgent, the Phase 5 coordinator contract. That
runtime adapter is deliberately not implemented in Phase 2, like the verifier.
"""

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
        },
        "out": str(config.out.resolve()),
    }
    (folder / "run.yaml").write_text(yaml.safe_dump(value, sort_keys=False))


def _harbor_agent(agent: AgentConfig) -> dict:
    """Describe a Harbor delegate without resolving credentials or loading its class."""
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


def _job_config(config: RunConfig, folder: Path, paths: tuple[Path, ...]) -> JobConfig:
    saved = folder / "jobs" / "eval" / "config.json"
    if saved.exists():
        return JobConfig.model_validate_json(saved.read_text())
    return JobConfig(
        job_name="eval",
        jobs_dir=folder / "jobs",
        n_concurrent_trials=1,
        environment=EnvironmentConfig(type="docker"),
        agents=[
            HarborAgentConfig(
                import_path="eval.agents:PhasedAgent",
                kwargs={
                    "search": _harbor_agent(config.search.agent),
                    "coding": _harbor_agent(config.agent),
                    "search_mode": config.search.mode,
                },
            )
        ],
        tasks=[TaskConfig(path=path) for path in paths],
    )


def _collect_costs(folder: Path) -> None:
    """Save finalized coding evidence; search never creates a cost attempt."""
    attempts = []
    for result_path in sorted((folder / "jobs" / "eval").glob("*/result.json")):
        result = json.loads(result_path.read_text())
        cancelled = (result.get("exception_info") or {}).get("exception_type") == "CancelledError"
        if not result.get("finished_at") and not cancelled:
            continue
        agent_config = (result.get("config") or {}).get("agent") or {}
        # Harbor leaves step_results unset until native multi-step execution starts.
        if (
            agent_config.get("import_path") == "eval.agents:PhasedAgent"
            or result.get("step_results") is not None
        ):
            coding = next(
                (
                    step
                    for step in result.get("step_results") or ()
                    if step["step_name"] == "coding"
                ),
                None,
            )
            if coding is None:
                continue
            trajectory = result_path.parent / "steps" / "coding" / "agent" / "trajectory.json"
            delegate = result["config"]["agent"]["kwargs"]["coding"]
            result = {
                **result,
                "agent_result": None,
                "step_results": [coding],
                "agent_info": {
                    "name": delegate.get("name") or delegate.get("import_path"),
                    "model_info": {"name": delegate.get("model_name")},
                },
            }
        else:
            trajectory = result_path.parent / "agent" / "trajectory.json"
        evidence = [str(result_path)]
        trajectory_data = None
        if trajectory.exists():
            evidence.append(str(trajectory))
            trajectory_data = json.loads(trajectory.read_text())
            if result.get("step_results") is not None:
                result["agent_info"]["model_info"]["provider"] = (
                    trajectory_data.get("agent") or {}
                ).get("provider")
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
    for result_path in (folder / "jobs" / "eval").glob("*/result.json"):
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
        job_config = _job_config(config, folder, paths)
        state["status"] = "running"
        _write_json(folder / "run.json", state)
        job = await Job.create(job_config)
        await job.run()
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
