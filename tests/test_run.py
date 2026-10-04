"""Two complete/resumed native pipelines and two interruption/failure edges."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
import yaml
from harbor.agents.factory import AgentFactory

from eval import run as runner
from eval.costs import read_ledger

FIXTURE = Path(__file__).parent / "fixtures" / "harbor_trial" / "result.json"


def write_config(path, *, oracle=False):
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": "owner/recordings",
                "tasks": {"models": ["C2"], "workloads": ["S-1"]},
                "agent": {
                    "name": "oracle" if oracle else "codex",
                    "model": None,
                    "env": ["CUSTOM_CREDENTIAL"],
                    "budget": "1h",
                    "kwargs": {"reasoning_effort": "high"},
                },
                "search": {"agent": {"name": "claude-code", "budget": "30m"}},
                "out": "runs",
            }
        )
    )


def prepared_tasks(folder, names=("one", "two")):
    source = folder / "models" / "C2" / "source"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    physical = {"source": "owner/model", "path": str(source), "checkpoint": str(source)}
    assets = {
        "C2": {
            "root": str(source.parent),
            "source": physical,
            "baselines": {
                "freetoken": physical,
                "ollama": {"source": "model:tag", "path": None, "checkpoint": "model:tag"},
            },
        }
    }
    paths = tuple(folder / "coding" / name for name in names)
    for path in paths:
        (path / "environment").mkdir(parents=True)
        (path / "environment" / "task.json").write_text('{"model": "C2"}')
        (path / "reference" / "metadata").mkdir(parents=True)
        (path / "reference" / "metadata" / "public.json").write_text("{}")
        (folder / "search" / path.name).mkdir(parents=True)
        (folder / "verification" / path.name / "steps" / "prepare" / "workdir").mkdir(parents=True)
    (folder / "prepared.json").write_text(
        json.dumps(
            {
                "task_paths": list(map(str, paths)),
                "assets": assets,
            }
        )
    )
    return paths


def write_trial(
    folder,
    task,
    phase,
    *,
    number=1,
    rewards=None,
    cancelled=False,
    infrastructure=False,
    publish=True,
    baseline="freetoken",
):
    folder.mkdir(parents=True)
    result = json.loads(FIXTURE.read_text())
    result.update(
        id=str(UUID(int=number)),
        finished_at="2026-10-01T00:00:00+00:00",
        verifier_result={"rewards": rewards if rewards is not None else {"reward": 1}},
        config={"task": {"path": str(task)}, "agent": {"name": "codex"}},
    )
    if phase == "search":
        result["config"]["agent"]["name"] = "claude-code"
        result["agent_result"] = {
            "n_input_tokens": 1000000,
            "n_output_tokens": 2000000,
            "cost_usd": 99,
        }
    if phase == "verify":
        result["config"]["agent"]["name"] = "nop"
        result["agent_result"] = None
        result["step_results"] = [
            {"step_name": name, "verifier_result": {"rewards": {"reward": 1}}}
            for name in ("prepare", "review", "measure")
        ]
        result["step_results"][-1]["verifier_result"] = result["verifier_result"]
    if cancelled or infrastructure:
        result["exception_info"] = {
            "exception_type": "CancelledError" if cancelled else "RuntimeError"
        }
    (folder / "result.json").write_text(json.dumps(result))
    if phase == "coding":
        trajectory = folder / "agent" / "trajectory.json"
        trajectory.parent.mkdir()
        trajectory.write_text(
            json.dumps(
                {
                    "agent": {"name": "codex", "version": "test-coding-version"},
                    "final_metrics": {
                        "total_prompt_tokens": 100,
                        "total_cached_tokens": 40,
                        "total_completion_tokens": 10,
                    },
                }
            )
        )
    if not publish or phase == "verify":
        return
    artifacts = folder / "artifacts"
    if phase == "search":
        source, destination = "/root/eval/reference", "artifacts/root/eval/reference"
        reference = artifacts / "root" / "eval" / "reference"
        reference.mkdir(parents=True)
        selection = {
            "baseline": baseline,
            "args": ["--batch-size", "8"],
            "env": {},
            "checkpoint": "/models/reference" if baseline == "freetoken" else "model:tag",
        }
        for name in ("reference.json", "public.json"):
            (reference / name).write_text(json.dumps(selection))
    else:
        source, destination = "/workspace/submission", "artifacts/workspace/submission"
        submission = artifacts / "workspace" / "submission"
        submission.mkdir(parents=True)
        for name in ("install.sh", "launch.sh", "server.py"):
            (submission / name).write_text("submitted code\n")
    (artifacts / "runtime").mkdir()
    (artifacts / "runtime" / "do-not-transfer").write_text("old installation")
    (artifacts / "manifest.json").write_text(
        json.dumps(
            [
                {"source": source, "destination": destination, "status": "ok", "type": "directory"},
                {
                    "source": "/workspace/runtime",
                    "destination": "artifacts/runtime",
                    "status": "ok",
                    "type": "directory",
                },
            ]
        )
    )


def test_native_three_jobs_preserve_agents_handoff_and_actual_scores(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    write_config(config_path)
    monkeypatch.setenv("CUSTOM_CREDENTIAL", "never-save-this-value")
    monkeypatch.setattr(runner, "check_environment", lambda: None)
    monkeypatch.setattr(
        runner, "prepare", lambda config, folder: SimpleNamespace(task_paths=prepared_tasks(folder))
    )
    calls = []

    class FakeJob:
        @classmethod
        async def create(cls, config):
            calls.append(config)
            job_folder = config.jobs_dir / config.job_name
            job_folder.mkdir(parents=True, exist_ok=True)
            (job_folder / "config.json").write_text(config.model_dump_json(exclude_unset=True))
            instance = cls()
            instance.config = config
            return instance

        async def run(self):
            config = self.config
            for number, task in enumerate(config.tasks, 1):
                if config.job_name == "coding":
                    selected = json.loads(
                        (task.path / "reference" / "metadata" / "reference.json").read_text()
                    )
                    weights = task.path / "reference" / "weights"
                    assert selected["baseline"] == ("freetoken" if number == 1 else "ollama")
                    assert (
                        weights.is_symlink()
                        if number == 1
                        else weights.is_dir() and not weights.is_symlink()
                    )
                    assert not (task.path / "runtime").exists()
                elif config.job_name == "verify":
                    source = task.path / "steps" / "prepare" / "workdir" / "submission"
                    assert sorted(path.name for path in source.iterdir()) == [
                        "install.sh",
                        "launch.sh",
                        "server.py",
                    ]
                    assert not (source.parent / "runtime").exists()
                write_trial(
                    config.jobs_dir / config.job_name / task.path.name,
                    task.path,
                    config.job_name,
                    number=number,
                    baseline="freetoken" if number == 1 else "ollama",
                    rewards={
                        "throughput": 50 if number == 1 else 0,
                        "ttft": 50 if number == 1 else 0,
                    }
                    if config.job_name == "verify"
                    else None,
                )

    monkeypatch.setattr(runner, "Job", FakeJob)
    folder = asyncio.run(runner.run(config_path))
    assert UUID(folder.name).version == 4
    assert [config.job_name for config in calls] == ["search", "coding", "verify"]
    search, coding, verify = (config.agents[0] for config in calls)
    assert search.name == "claude-code" and search.override_timeout_sec == 1800
    assert coding.name == "codex" and coding.override_timeout_sec == 3600
    assert verify.name == "nop"
    for agent in (search, coding, verify):
        assert agent.import_path is None and agent.model_name is None
        assert "model_name" not in agent.model_fields_set
        assert AgentFactory.get_agent_class_from_config(agent).name() == agent.name
    assert coding.kwargs == {"reasoning_effort": "high"}
    assert coding.env == {
        "CUSTOM_CREDENTIAL": "${CUSTOM_CREDENTIAL}",
        "CODEX_FORCE_AUTH_JSON": "true",
        "HOME": "/home/agent",
        "USER": "agent",
        "LOGNAME": "agent",
    }
    assert search.env == {"HOME": "/home/search", "USER": "search", "LOGNAME": "search"}
    assert "never-save-this-value" not in "".join(config.model_dump_json() for config in calls)
    result = json.loads((folder / "result.json").read_text())["tasks"]
    assert result["one"]["rewards"] == {"throughput": 50, "ttft": 50}
    assert result["two"]["rewards"] == {"throughput": 0, "ttft": 0}
    assert all(
        task["status"] == "complete" and task["stage"] == "verify" for task in result.values()
    )
    measured = folder / "jobs" / "verify" / "two" / "steps" / "measure" / "verifier"
    measured.mkdir(parents=True)
    (measured / "report.json").write_text('{"outcome": "candidate_failure"}')
    runner._final_results(folder, tuple(task.path for task in calls[1].tasks))
    failed = json.loads((folder / "result.json").read_text())["tasks"]["two"]
    assert failed["status"] == "candidate_failed" and failed["rewards"] == {
        "throughput": 0,
        "ttft": 0,
    }
    assert len(read_ledger(folder / "costs.json")) == 2
    assert json.loads((folder / "cost-summary.json").read_text())["total_tokens"] == 220
    assert json.loads((folder / "run.json").read_text())["status"] == "complete"
    config = runner.load_config(folder / "run.yaml")
    api = replace(config, agent=replace(config.agent, auth="api_key"))
    api_agent = runner._job_config(api, folder / "api", (), phase="coding").agents[0]
    assert (
        api_agent.env["CODEX_FORCE_AUTH_JSON"] == "false"
        and api_agent.env["CODEX_AUTH_JSON_PATH"] == ""
    )
    oracle = replace(config, agent=replace(config.agent, name="oracle"))
    assert all(
        runner._job_config(oracle, folder / "oracle", (), phase=phase).agents[0].name == "oracle"
        for phase in ("search", "coding")
    )
    custom = replace(
        config, agent=replace(config.agent, name="harbor.agents.nop:NopAgent", kwargs={})
    )
    native = runner._job_config(custom, folder / "custom", ()).agents[0]
    assert AgentFactory.get_agent_class_from_config(native).name() == "nop"
    assert "model_name" not in native.model_fields_set


def test_resume_reuses_three_configs_and_preserves_costs_before_cancel_cleanup(
    tmp_path, monkeypatch
):
    folder = tmp_path / "existing"
    folder.mkdir()
    write_config(folder / "run.yaml")
    paths = prepared_tasks(folder, ("one",))
    config = runner.load_config(folder / "run.yaml")
    saved = {}
    for number, phase in enumerate(("search", "coding", "verify"), 1):
        task = folder / ("verification" if phase == "verify" else phase) / "one"
        completed = folder / "jobs" / phase / "completed"
        write_trial(
            completed,
            task,
            phase,
            number=number,
            rewards={"reward": 0} if phase == "verify" else None,
        )
        write_trial(
            folder / "jobs" / phase / "cancelled", task, phase, number=number + 10, cancelled=True
        )
        job_config = runner._job_config(config, folder, (task,), phase=phase)
        (completed.parent / "config.json").write_text(
            job_config.model_dump_json(exclude_unset=True)
        )
        saved[phase] = job_config
    assert runner._coding_tasks(folder, paths) == paths
    verification = runner._verification_tasks(folder, paths)
    copy = verification[0] / "steps" / "prepare" / "workdir" / "submission"
    (copy / "launch.sh").write_text("keep completed source handoff")
    monkeypatch.setattr(runner, "check_environment", lambda: None)
    monkeypatch.setattr(
        runner, "prepare", lambda *_: pytest.fail("prepared resume must not repeat setup")
    )
    calls = []

    class FakeJob:
        @classmethod
        async def create(cls, config):
            calls.append(config.job_name)
            assert config == saved[config.job_name]
            assert all(not (folder / "jobs" / phase / "cancelled").exists() for phase in saved)
            assert len(read_ledger(folder / "costs.json")) == 2
            return cls()

        async def run(self):
            return None  # Native Harbor retains completed trials, including zero reward.

    monkeypatch.setattr(runner, "Job", FakeJob)
    assert asyncio.run(runner.run(resume=folder)) == folder
    assert calls == ["search", "coding", "verify"]
    assert (copy / "launch.sh").read_text() == "keep completed source handoff"
    outcome = json.loads((folder / "result.json").read_text())["tasks"]["one"]
    assert outcome["status"] == "candidate_failed" and outcome["rewards"] == {"reward": 0}
    assert outcome["stage"] == "verify"
    assert len(read_ledger(folder / "costs.json")) == 2
    assert all(attempt.agent_name == "codex" for attempt in read_ledger(folder / "costs.json"))

    monkeypatch.delenv("CODEX_FORCE_AUTH_JSON", raising=False)
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    for phase, auth in (("coding", "login"), ("search", "api_key")):
        role = replace(config.agent, model="synthetic-model", auth=auth)
        frozen = (
            replace(config, agent=role)
            if phase == "coding"
            else replace(config, search=replace(config.search, agent=role))
        )
        run = tmp_path / f"native-{phase}-{auth}"
        original = runner._job_config(frozen, run, paths, phase=phase)
        native_config = run / "jobs" / phase / "config.json"
        native_config.parent.mkdir(parents=True)
        native_config.write_text(original.model_dump_json(indent=4, exclude_defaults=True))
        stored = json.loads(native_config.read_text())["agents"][0]
        assert stored["env"]["CODEX_FORCE_AUTH_JSON"] == "****"
        if auth == "api_key":
            assert stored["env"]["CODEX_AUTH_JSON_PATH"] == "****"
        resumed = runner._job_config(frozen, run, (), phase=phase)
        assert resumed.agents[0].env == original.agents[0].env
        assert resumed.agents[0].env["CUSTOM_CREDENTIAL"] == "${CUSTOM_CREDENTIAL}"
        assert resumed.agents[0].model_name == "synthetic-model"
        assert resumed.agents[0].kwargs == original.agents[0].kwargs
        assert resumed.tasks == original.tasks and resumed == original

    oracle = replace(
        config,
        agent=replace(config.agent, name="oracle"),
        search=replace(config.search, agent=config.agent),
    )
    run = tmp_path / "native-oracle"
    native = runner._job_config(oracle, run, paths, phase="search")
    native_config = run / "jobs" / "search" / "config.json"
    native_config.parent.mkdir(parents=True)
    native_config.write_text(native.model_dump_json(exclude_defaults=True))
    resumed = runner._job_config(oracle, run, (), phase="search").agents[0]
    assert resumed.name == "oracle" and "CODEX_FORCE_AUTH_JSON" not in resumed.env


def test_interruption_records_native_coding_trajectory_and_resumable_state(tmp_path, monkeypatch):
    source = tmp_path / "config.yaml"
    write_config(source)
    monkeypatch.setattr(runner, "check_environment", lambda: None)
    monkeypatch.setattr(
        runner,
        "prepare",
        lambda config, folder: SimpleNamespace(task_paths=prepared_tasks(folder, ("one",))),
    )

    class FakeJob:
        @classmethod
        async def create(cls, config):
            instance = cls()
            instance.config = config
            return instance

        async def run(self):
            config = self.config
            trial = config.jobs_dir / config.job_name / "attempt"
            write_trial(
                trial, config.tasks[0].path, config.job_name, cancelled=config.job_name == "coding"
            )
            if config.job_name == "coding":
                path = trial / "result.json"
                result = json.loads(path.read_text())
                result.update(agent_result=None, finished_at=None)
                path.write_text(json.dumps(result))
                raise asyncio.CancelledError

    monkeypatch.setattr(runner, "Job", FakeJob)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner.run(source))
    folder = next((tmp_path / "runs").iterdir())
    state = json.loads((folder / "run.json").read_text())
    assert state["status"] == "interrupted" and state["error_type"] == "CancelledError"
    assert state["cost_collection_complete"]
    attempts = read_ledger(folder / "costs.json")
    assert len(attempts) == 1 and attempts[0].usage.total_tokens == 110
    assert attempts[0].usage_source == "trajectory.final_metrics"
    assert not (folder / "jobs" / "verify").exists()
    assert (folder / "run.yaml").is_file()

    copy_run = tmp_path / "copy-interrupted"
    paths = prepared_tasks(copy_run, ("one",))
    write_trial(copy_run / "jobs" / "coding" / "complete", paths[0], "coding")
    workdir = copy_run / "verification" / "one" / "steps" / "prepare" / "workdir"
    copytree = runner.shutil.copytree

    def interrupt_copy(source, destination, **kwargs):
        destination.mkdir()
        (destination / "launch.sh").write_text("partial copy")
        raise KeyboardInterrupt

    monkeypatch.setattr(runner.shutil, "copytree", interrupt_copy)
    with pytest.raises(KeyboardInterrupt):
        runner._verification_tasks(copy_run, paths)
    assert list(workdir.iterdir()) == []
    interrupted = list(workdir.parent.parent.parent.glob(".submission-*"))
    assert len(interrupted) == 1
    assert (interrupted[0] / "launch.sh").read_text() == "partial copy"

    monkeypatch.setattr(runner.shutil, "copytree", copytree)
    assert runner._verification_tasks(copy_run, paths) == (workdir.parent.parent.parent,)
    assert list(workdir.iterdir()) == [workdir / "submission"]
    submission = workdir / "submission"
    assert sorted(path.name for path in submission.iterdir()) == [
        "install.sh",
        "launch.sh",
        "server.py",
    ]
    assert (submission / "launch.sh").read_text() == "submitted code\n"


def test_failed_search_and_missing_handoffs_do_not_fabricate_scores(tmp_path, monkeypatch, capsys):
    for arguments in (["run.yaml", "--resume", "runs/one"], ["run.yaml", "--foreground"]):
        with pytest.raises(SystemExit) as error:
            runner.main(arguments)
        assert error.value.code == 2
        capsys.readouterr()
    source = tmp_path / "config.yaml"
    write_config(source)
    names = (
        "search-rejected",
        "search-infra",
        "reference-missing",
        "coding-rejected",
        "submission-missing",
    )
    monkeypatch.setattr(runner, "check_environment", lambda: None)
    monkeypatch.setattr(
        runner,
        "prepare",
        lambda config, folder: SimpleNamespace(task_paths=prepared_tasks(folder, names)),
    )
    calls = []

    class FakeJob:
        @classmethod
        async def create(cls, config):
            calls.append(config.job_name)
            assert config.job_name != "verify", "no successful source handoff exists"
            if config.job_name == "coding":
                assert [task.path.name for task in config.tasks] == [
                    "coding-rejected",
                    "submission-missing",
                ]
            instance = cls()
            instance.config = config
            return instance

        async def run(self):
            config = self.config
            for number, task in enumerate(config.tasks, 1):
                name = task.path.name
                rejected = (
                    name == "search-rejected"
                    or name == "coding-rejected"
                    and config.job_name == "coding"
                )
                write_trial(
                    config.jobs_dir / config.job_name / name,
                    task.path,
                    config.job_name,
                    number=number,
                    rewards={"reward": 0} if rejected else None,
                    infrastructure=name == "search-infra",
                    publish=name not in ("reference-missing", "submission-missing")
                    or config.job_name == "search",
                )
                if name == "reference-missing":
                    (
                        config.jobs_dir
                        / config.job_name
                        / name
                        / "artifacts/root/eval/reference/reference.json"
                    ).unlink()

    monkeypatch.setattr(runner, "Job", FakeJob)
    folder = asyncio.run(runner.run(source))
    assert calls == ["search", "coding"]
    tasks = json.loads((folder / "result.json").read_text())["tasks"]
    for name in ("search-rejected", "coding-rejected"):
        assert tasks[name]["status"] == "candidate_failed" and tasks[name]["rewards"] == {
            "reward": 0
        }
    for name in ("search-infra", "reference-missing", "submission-missing"):
        assert tasks[name]["status"] == "infrastructure_error" and tasks[name]["rewards"] is None
    assert all(task["verification_result"] is None for task in tasks.values())
