"""Two common runner cases and two edges; no paid calls or real container starts."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
import yaml
from harbor.agents.factory import AgentFactory
from harbor.models.trial.config import AgentConfig as HarborAgentConfig
from harbor.models.trial.result import StepResult

from eval import run as runner
from eval.costs import read_ledger

FIXTURE = Path(__file__).parent / "fixtures" / "harbor_trial" / "result.json"


def write_config(path, *, agent=None):
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": "owner/recordings",
                "tasks": {"models": ["C2"], "workloads": ["S-1"]},
                "agent": agent
                if agent is not None
                else {
                    "name": "codex",
                    "model": "test-agent",
                    "env": ["CUSTOM_CREDENTIAL"],
                    "budget": "1h",
                    "kwargs": {"reasoning_effort": "high"},
                },
                "out": "runs",
            }
        )
    )


def write_trial(folder, *, number=1, cancelled=False, coding=True, coding_agent=None):
    folder.mkdir(parents=True)
    result = json.loads(FIXTURE.read_text())
    result["id"] = str(UUID(int=number))
    result["finished_at"] = "2026-10-01T00:00:00+00:00"
    result["verifier_result"] = {"rewards": {"reward": 0}}
    usage = result.pop("agent_result")
    result["agent_info"] = {"name": "phased", "version": "test-only-coordinator"}
    result["step_results"] = [
        {
            "step_name": "search",
            "agent_result": {"n_input_tokens": 1000000, "n_output_tokens": 2000000, "cost_usd": 99},
        }
    ]
    delegate = coding_agent or {"name": "codex", "model_name": "fixture-coding-model"}
    result["config"] = {
        "agent": {
            "import_path": "eval.agents:PhasedAgent",
            "kwargs": {"coding": delegate},
        }
    }
    if coding:
        result["step_results"].append({"step_name": "coding", "agent_result": usage})
        trajectory = folder / "steps" / "coding" / "agent" / "trajectory.json"
        trajectory.parent.mkdir(parents=True)
        trajectory.write_text(
            json.dumps(
                {
                    "agent": {
                        "name": delegate.get("name"),
                        "model_name": delegate.get("model_name"),
                        "version": "test-coding-version",
                    },
                    "final_metrics": {
                        "total_prompt_tokens": 100,
                        "total_cached_tokens": 40,
                        "total_completion_tokens": 10,
                    },
                }
            )
        )
    for step in result["step_results"]:
        StepResult.model_validate(step)
    if cancelled:
        result["exception_info"] = {"exception_type": "CancelledError"}
    (folder / "result.json").write_text(json.dumps(result))


def test_new_run_saves_uuid_config_and_awaits_harbor(tmp_path, monkeypatch):
    source = tmp_path / "config.yaml"
    write_config(source)
    monkeypatch.setenv("CUSTOM_CREDENTIAL", "never-save-this-value")
    monkeypatch.setattr(runner, "check_environment", lambda: None)
    calls = []

    def prepare(config, folder):
        paths = (folder / "tasks" / "one", folder / "tasks" / "two")
        (folder / "prepared.json").write_text(json.dumps({"task_paths": list(map(str, paths))}))
        assert (folder / "run.yaml").is_file()
        return SimpleNamespace(task_paths=paths)

    class FakeJob:
        @classmethod
        async def create(cls, config):
            calls.append(config)
            return cls()

        async def run(self):
            calls.append("awaited")

    monkeypatch.setattr(runner, "prepare", prepare)
    monkeypatch.setattr(runner, "Job", FakeJob)
    folder = asyncio.run(runner.run(source))
    assert UUID(folder.name).version == 4
    saved = yaml.safe_load((folder / "run.yaml").read_text())
    assert saved["out"] == str(tmp_path / "runs")
    assert saved["agent"]["budget"] == 3600
    assert saved["agent"]["env"] == ["CUSTOM_CREDENTIAL"]
    coordinator = calls[0].agents[0]
    coding = HarborAgentConfig.model_validate(coordinator.kwargs["coding"])
    search = HarborAgentConfig.model_validate(coordinator.kwargs["search"])
    assert coordinator.import_path == "eval.agents:PhasedAgent"
    assert not {"model_name", "env", "override_timeout_sec"} & coordinator.model_fields_set
    assert coordinator.kwargs["search_mode"] == "combined"
    assert saved["search"]["agent"]["name"] == "claude-code"
    assert coding.env == {
        "CUSTOM_CREDENTIAL": "${CUSTOM_CREDENTIAL}",
        "CODEX_FORCE_AUTH_JSON": "true",
    }
    assert "never-save-this-value" not in calls[0].model_dump_json()
    assert coding.kwargs == {"reasoning_effort": "high"}
    assert coding.name == "codex"
    assert coding.model_name == "test-agent"
    assert search.name == "claude-code" and search.model_name is None
    assert "model_name" not in search.model_fields_set
    assert search.env == {} and search.override_timeout_sec == 7200
    assert AgentFactory.get_agent_class_from_config(coding).name() == "codex"
    assert AgentFactory.get_agent_class_from_config(search).name() == "claude-code"
    assert calls[0].n_concurrent_trials == 1 and len(calls[0].tasks) == 2
    assert calls[1] == "awaited"
    assert json.loads((folder / "run.json").read_text())["status"] == "complete"
    monkeypatch.setenv("CODEX_FORCE_AUTH_JSON", "true")
    monkeypatch.setenv("CODEX_AUTH_JSON_PATH", "/inherited/login.json")
    config = runner.load_config(folder / "run.yaml")
    api_config = replace(config, agent=replace(config.agent, auth="api_key", model=None))
    api_job = runner._job_config(
        api_config, folder / "api-key", tuple(task.path for task in calls[0].tasks)
    )
    api_coding = HarborAgentConfig.model_validate(api_job.agents[0].kwargs["coding"])
    assert api_coding.env["CODEX_FORCE_AUTH_JSON"] == "false"
    assert api_coding.env["CODEX_AUTH_JSON_PATH"] == ""
    assert api_coding.model_name is None
    assert "model_name" not in api_coding.model_fields_set
    assert not {"model_name", "env", "override_timeout_sec"} & api_job.agents[0].model_fields_set
    runner._save_config(api_config, folder)
    assert runner.load_config(folder / "run.yaml").agent.model is None


def test_folder_resume_preserves_costs_before_cleanup_and_keeps_zero_reward(tmp_path, monkeypatch):
    folder = tmp_path / "existing-run"
    folder.mkdir()
    write_config(
        folder / "run.yaml",
        agent={
            "name": "harbor.agents.nop:NopAgent",
            "env": ["CUSTOM_CREDENTIAL"],
            "kwargs": {},
        },
    )
    paths = [str(folder / "tasks" / "one")]
    (folder / "prepared.json").write_text(json.dumps({"task_paths": paths}))
    trials = folder / "jobs" / "eval"
    completed, cancelled = trials / "completed", trials / "cancelled"
    write_trial(completed, number=1)
    write_trial(cancelled, number=2, cancelled=True)
    write_trial(trials / "search-only", number=4, coding=False)
    search_cancelled = trials / "search-cancelled"
    write_trial(search_cancelled, number=5, cancelled=True, coding=False)
    before_steps = trials / "before-steps"
    write_trial(before_steps, number=6, cancelled=True, coding=False)
    result_path = before_steps / "result.json"
    partial = json.loads(result_path.read_text())
    partial.update(step_results=None, agent_result=None, finished_at=None)
    result_path.write_text(json.dumps(partial))
    saved_job = runner._job_config(
        runner.load_config(folder / "run.yaml"), folder, tuple(map(Path, paths))
    )
    (trials / "config.json").write_text(saved_job.model_dump_json(exclude_unset=True))
    monkeypatch.setattr(runner, "check_environment", lambda: None)

    def no_prepare(*args):
        pytest.fail("resume should use prepared inputs and tasks")

    class FakeJob:
        @classmethod
        async def create(cls, config):
            assert completed.is_dir() and not cancelled.exists()
            assert not search_cancelled.exists() and not before_steps.exists()
            assert len(read_ledger(folder / "costs.json")) == 2
            assert config.tasks[0].path == Path(paths[0])
            assert config == saved_job
            coordinator = config.agents[0]
            assert coordinator.import_path == "eval.agents:PhasedAgent"
            agent = HarborAgentConfig.model_validate(coordinator.kwargs["coding"])
            assert agent.name == "harbor.agents.nop:NopAgent"
            assert agent.model_name is None and "model_name" not in agent.model_fields_set
            assert agent.env == {"CUSTOM_CREDENTIAL": "${CUSTOM_CREDENTIAL}"}
            assert AgentFactory.get_agent_class_from_config(agent).name() == "nop"
            restored = type(agent).model_validate_json(agent.model_dump_json(exclude_unset=True))
            assert "model_name" not in restored.model_fields_set
            return cls()

        async def run(self):
            write_trial(trials / "retry", number=3)

    monkeypatch.setattr(runner, "prepare", no_prepare)
    monkeypatch.setattr(runner, "Job", FakeJob)
    assert asyncio.run(runner.run(resume=folder)) == folder
    assert len(read_ledger(folder / "costs.json")) == 3
    assert completed.is_dir()
    assert not (folder / "runs").exists()
    summary = json.loads((folder / "cost-summary.json").read_text())
    assert summary["total_tokens"] == 330
    assert summary["cost_usd"] is None
    attempts = read_ledger(folder / "costs.json")
    assert all(attempt.usage_source == "step_results" for attempt in attempts)
    assert all(attempt.agent_name == "codex" for attempt in attempts)
    assert all(attempt.agent_version == "test-coding-version" for attempt in attempts)
    assert all(
        any("steps/coding/agent/trajectory.json" in path for path in attempt.evidence)
        for attempt in attempts
    )


def test_interruption_records_finalized_usage_and_resumable_status(tmp_path, monkeypatch):
    source = tmp_path / "config.yaml"
    write_config(source)
    monkeypatch.setattr(runner, "check_environment", lambda: None)
    monkeypatch.setattr(
        runner, "prepare", lambda config, folder: SimpleNamespace(task_paths=(folder / "task",))
    )

    class FakeJob:
        @classmethod
        async def create(cls, config):
            instance = cls()
            instance.folder = config.jobs_dir.parent
            return instance

        async def run(self):
            trial = self.folder / "jobs" / "eval" / "interrupted"
            write_trial(trial, cancelled=True)
            result_path = trial / "result.json"
            partial = json.loads(result_path.read_text())
            partial["step_results"][-1]["agent_result"] = None
            result_path.write_text(json.dumps(partial))
            raise asyncio.CancelledError

    monkeypatch.setattr(runner, "Job", FakeJob)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner.run(source))
    folder = next((tmp_path / "runs").iterdir())
    state = json.loads((folder / "run.json").read_text())
    assert state["status"] == "interrupted" and state["error_type"] == "CancelledError"
    assert state["cost_collection_complete"]
    attempts = read_ledger(folder / "costs.json")
    assert len(attempts) == 1
    assert attempts[0].usage_source == "step_results+trajectory.final_metrics"
    assert attempts[0].usage.total_tokens == 110
    assert (folder / "run.yaml").is_file()
    malformed = folder / "jobs" / "eval" / "malformed"
    malformed.mkdir()
    (malformed / "result.json").write_text("{")
    with pytest.raises(json.JSONDecodeError):
        runner._collect_costs(folder)


def test_cli_rejects_config_with_resume_and_removed_foreground_flag(capsys):
    with pytest.raises(SystemExit) as error:
        runner.main(["run.yaml", "--resume", "runs/one"])
    assert error.value.code == 2
    assert "exclusively" in capsys.readouterr().err
    with pytest.raises(SystemExit) as error:
        runner.main(["run.yaml", "--foreground"])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
