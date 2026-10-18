"""Native task handoffs, concurrent coordination and checkpoint recovery."""

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
import yaml
from harbor.agents.factory import AgentFactory

from eval import run as runner
from eval.costs import read_ledger
from eval.history import partial_trajectory, prepare_history, rate_limit_reset

FIXTURE = Path(__file__).parent / "fixtures" / "harbor_trial" / "result.json"


def native_history(logs, *, cost=7):
    project = logs / "sessions/projects/-workspace"
    project.mkdir(parents=True)
    identity = str(UUID(int=82))
    events = [
        {
            "type": "user",
            "uuid": "user-1",
            "sessionId": identity,
            "message": {"role": "user", "content": "Keep working on the serving implementation."},
        },
        {
            "type": "assistant",
            "uuid": "assistant-1",
            "parentUuid": "user-1",
            "sessionId": identity,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "The previous benchmark needs checking.",
                        "signature": "native-signature",
                    },
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "Read",
                        "input": {"file_path": "/workspace/submission/server.py"},
                    },
                ],
                "usage": {"input_tokens": 12, "output_tokens": 4, "cache_read_input_tokens": 3},
            },
        },
        {
            "type": "cost-state",
            "sessionId": identity,
            "totalCostUSD": cost,
            "modelUsage": {"synthetic-model": {"costUSD": cost, "inputTokens": 12}},
            "totalDuration": 300,
        },
    ]
    path = project / f"{identity}.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    tools = project / identity / "tool-results"
    tools.mkdir(parents=True)
    (tools / "output.txt").write_text("Complete previous tool output")
    return path, events


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
            for name in ("reference", "prepare", "review", "measure")
        ]
        result["step_results"][-1]["verifier_result"] = result["verifier_result"]
    if cancelled or infrastructure:
        result["exception_info"] = {
            "exception_type": "CancelledError" if cancelled else "RuntimeError"
        }
    (folder / "result.json").write_text(json.dumps(result))
    if phase == "coding":
        sessions = folder / "agent/sessions/2026/10/16"
        sessions.mkdir(parents=True)
        (sessions / f"rollout-2026-10-16-{UUID(int=82)}.jsonl").write_text(
            json.dumps({"type": "session_meta", "payload": {"id": str(UUID(int=82))}}) + "\n"
        )
        trajectory = folder / "agent" / "trajectory.json"
        trajectory.parent.mkdir(exist_ok=True)
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
        portable = (task / "environment/runtime.json").exists()
        source = "/var/lib/local-eval/reference" if portable else "/root/eval/reference"
        destination = (
            "artifacts/var/lib/local-eval/reference"
            if portable
            else "artifacts/root/eval/reference"
        )
        reference = folder / destination
        reference.mkdir(parents=True)
        selection = {
            "baseline": baseline,
            "args": ["--batch-size", "8"],
            "env": {},
            "checkpoint": "/models/reference" if baseline == "freetoken" else "model:tag",
        }
        (reference / "reference.json").write_text(json.dumps(selection))
    else:
        source, destination = "/var/lib/local-eval/submission", "artifacts/workspace/submission"
        submission = artifacts / "workspace" / "submission"
        submission.mkdir(parents=True)
        for name in ("install.sh", "obtain_weight.sh", "launch.sh", "server.py"):
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


def test_missing_chat_fails_and_partial_usage_without_trial_result_stays_known(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    write_config(folder / "run.yaml")
    tasks = prepared_tasks(folder, ("one",), portable=True)
    trial = folder / "jobs/coding/interrupted"
    (trial / "agent").mkdir(parents=True)
    (trial / "config.json").write_text(
        json.dumps(
            {
                "agent": {"name": "claude-code", "model_name": "synthetic-model"},
                "task": {"path": str(tasks[0])},
            }
        )
    )
    (trial / "agent/trajectory.json").write_text(
        json.dumps(
            {
                "agent": {"version": "test-version"},
                "final_metrics": {
                    "total_prompt_tokens": 17,
                    "total_completion_tokens": 5,
                    "total_cost_usd": 2.5,
                },
            }
        )
    )
    runner._collect_costs(folder)
    attempt = read_ledger(folder / "costs.json")[0]
    assert attempt.attempt_id == "interrupted:coding/interrupted"
    assert attempt.usage.input_tokens == 17 and attempt.usage.cost_usd == 2.5
    assert attempt.usage.cached_tokens is None
    agent = runner.HarborAgentConfig(name="claude-code")
    with pytest.raises(ValueError, match="Claude conversation is missing"):
        runner._resume_task(folder, "coding", tasks[0], trial, agent)
    assert trial.exists() and not (trial / "result.json").exists()
    runner._collect_costs(folder)
    assert read_ledger(folder / "costs.json") == (attempt,)
    native = folder / "jobs/coding/native-only"
    native.mkdir()
    shutil_settings = json.loads((trial / "config.json").read_text())
    (native / "config.json").write_text(json.dumps(shutil_settings))
    native_history(native / "agent")
    runner._collect_costs(folder)
    usage = read_ledger(folder / "costs.json")[-1]
    assert usage.usage.input_tokens == 15 and usage.usage.output_tokens == 4
    assert usage.usage.cost_usd is None  # An unknown model has no invented price.
    assert (folder / "interrupted-usage/native-only/trajectory.json").exists()


def test_interrupted_usage_parser_can_retry_an_incomplete_scratch_directory(tmp_path):
    logs, scratch = tmp_path / "agent", tmp_path / "parsed"
    session, events = native_history(logs)
    session.write_text(json.dumps(events[-1]) + "\n")
    agent = runner.HarborAgentConfig(name="claude-code", model_name="synthetic-model")
    assert partial_trajectory(agent, logs, scratch) is None
    (scratch / "stale.txt").write_text("incomplete parser output")
    session.write_text("".join(json.dumps(event) + "\n" for event in events))
    original = session.read_bytes()
    trajectory = partial_trajectory(agent, logs, scratch)
    assert trajectory == scratch / "trajectory.json"
    assert json.loads(trajectory.read_text())["final_metrics"]["total_completion_tokens"] == 4
    assert not (scratch / "stale.txt").exists()
    assert session.read_bytes() == original


def test_bootstrap_failures_preserve_the_last_ready_workspace_and_native_chat(
    tmp_path, monkeypatch
):
    folder = tmp_path / "run"
    folder.mkdir()
    source = folder / "run.yaml"
    write_config(source)
    settings = yaml.safe_load(source.read_text())
    settings["agent"]["name"] = "claude-code"
    source.write_text(yaml.safe_dump(settings))
    task = prepared_tasks(folder, ("one",), portable=True)[0]
    original = folder / "jobs/coding/original"
    write_trial(original, task, "coding", infrastructure=True, publish=False)
    session, events = native_history(original / "agent")
    (original / "inputs-ready.json").write_text("{}")
    snapshot = original / "checkpoint/latest.json"
    objects = snapshot.parent / "objects"
    objects.mkdir(parents=True)
    (objects / "native-chat").write_bytes(session.read_bytes())
    snapshot.write_text(
        json.dumps(
            {
                "at": 1,
                "files": {
                    "/workspace/.harbor/inputs-ready.json": {"kind": "file"},
                    f"/logs/agent/sessions/projects/-workspace/{session.name}": {
                        "kind": "file",
                        "object": "native-chat",
                    },
                },
            }
        )
    )
    events.append(
        {
            "type": "user",
            "uuid": "after-checkpoint",
            "sessionId": session.stem,
            "message": {"role": "user", "content": "Preserve the final exported conversation."},
        }
    )
    session.write_text("".join(json.dumps(event) + "\n" for event in events))
    agent = runner.HarborAgentConfig(name="claude-code")
    resumed, history = runner._resume_task(folder, "coding", task, original, agent)
    assert json.loads(history.read_text().splitlines()[-1])["uuid"] == "after-checkpoint"
    bootstrap = folder / "jobs/coding-resume-first/bootstrap"
    write_trial(bootstrap, resumed, "coding", infrastructure=True, publish=False)
    result = json.loads((bootstrap / "result.json").read_text())
    result["config"]["agent"] = {"name": "claude-code", "load_trajectory": str(history)}
    (bootstrap / "result.json").write_text(json.dumps(result))
    empty_snapshot = bootstrap / "checkpoint/latest.json"
    empty_snapshot.parent.mkdir()
    # Older providers could incorrectly label a failed admission as ready.
    empty_snapshot.write_text(json.dumps({"ready": True, "files": {"/workspace": {"kind": "dir"}}}))
    recovered, chat = runner._resume_task(folder, "coding", task, bootstrap, agent)
    runtime = json.loads((recovered / "environment/runtime.json").read_text())
    assert runtime["checkpoint"] == str(snapshot)
    assert runtime["resume"]["session_id"] == session.stem
    assert (
        json.loads(chat.read_text().splitlines()[1])["message"]["content"]
        == events[1]["message"]["content"]
    )
    assert agent.load_trajectory is None
    assert json.loads(chat.read_text().splitlines()[-1])["uuid"] == "after-checkpoint"

    # Recover an existing chain already damaged by an older bootstrap retry.
    runtime["checkpoint"] = str(empty_snapshot)
    runtime.pop("resume")
    runtime["files"] = []
    (recovered / "environment/runtime.json").write_text(json.dumps(runtime))
    latest = folder / "jobs/coding-resume-latest/bootstrap"
    write_trial(latest, recovered, "coding", infrastructure=True, publish=False)
    allocation = latest / "environment.json"
    allocation.write_text('{"provider":"daytona","sandbox_id":"latest-allocation"}')
    launched = []

    async def execute(job):
        launched.append(job)

    monkeypatch.setattr(runner, "_execute_job", execute)
    asyncio.run(runner._run_phase(runner.load_config(source), folder, (task,), "coding"))
    assert len(launched) == 1
    recovered = launched[0].tasks[0].path
    runtime = json.loads((recovered / "environment/runtime.json").read_text())
    assert runtime["previous_environment"] == str(allocation)
    assert runtime["checkpoint"] == str(snapshot)
    assert runtime["resume"]["session_id"] == session.stem
    assert Path(launched[0].agents[0].load_trajectory).name == session.name
    assert session.read_text().splitlines() == [json.dumps(event) for event in events]


def test_bootstrap_retry_of_a_new_task_does_not_invent_conversation(tmp_path, monkeypatch):
    folder = tmp_path / "run"
    folder.mkdir()
    source = folder / "run.yaml"
    write_config(source)
    task = prepared_tasks(folder, ("one",), portable=True)[0]
    bootstrap = folder / "jobs/coding/bootstrap"
    bootstrap.mkdir(parents=True)
    (bootstrap / "config.json").write_text(
        json.dumps({"task": {"path": str(task)}, "agent": {"name": "codex"}})
    )
    launched = []

    async def execute(job):
        launched.append(job)

    monkeypatch.setattr(runner, "_execute_job", execute)
    asyncio.run(runner._run_phase(runner.load_config(source), folder, (task,), "coding"))
    assert len(launched) == 1
    assert launched[0].agents[0].load_trajectory is None
    runtime = json.loads((launched[0].tasks[0].path / "environment/runtime.json").read_text())
    assert "resume" not in runtime and "checkpoint" not in runtime
    runner._collect_costs(folder)
    assert read_ledger(folder / "costs.json") == ()

    failed = json.loads(FIXTURE.read_text())
    failed.update(
        id="admission-failure",
        finished_at="2026-10-01T00:00:00+00:00",
        agent_result=None,
        agent_execution=None,
        exception_info={"exception_type": "RuntimeError"},
        config={"task": {"path": str(task)}, "agent": {"name": "codex"}},
    )
    (bootstrap / "result.json").write_text(json.dumps(failed))
    runner._collect_costs(folder)
    assert read_ledger(folder / "costs.json") == ()


def test_agent_without_history_capability_cannot_silently_restart(tmp_path, monkeypatch):
    from harbor.agents.capabilities import AgentCapabilities

    monkeypatch.setattr(
        AgentFactory,
        "get_agent_class_from_config",
        lambda config: SimpleNamespace(capabilities=AgentCapabilities()),
    )
    agent = runner.HarborAgentConfig(name="custom.agent:Agent")
    with pytest.raises(ValueError, match="cannot restore its recorded conversation"):
        prepare_history(agent, tmp_path / "logs", tmp_path / "upload", coding=False)
    trial = tmp_path / "trials" / "custom-interrupted"
    trial.mkdir(parents=True)
    event = SimpleNamespace(
        config=SimpleNamespace(
            trials_dir=trial.parent, task=SimpleNamespace(path=tmp_path / "task")
        ),
        trial_name=trial.name,
        result=SimpleNamespace(step_results=[], agent_execution=None),
        timestamp=datetime(2026, 10, 6, tzinfo=UTC),
    )
    asyncio.run(runner._agent_started(event))
    assert json.loads((trial / "agent-start.json").read_text()) == {
        "started_at": "2026-10-06T00:00:00+00:00",
        "step": None,
    }


@pytest.mark.parametrize("current", ["pass", "reject", "error", "unfinished"])
def test_current_attempt_cannot_be_masked_by_an_old_result(tmp_path, current):
    paths = prepared_tasks(tmp_path, ("one",))
    task = tmp_path / "verification/one"
    previous = tmp_path / "jobs/verify/old"
    active = tmp_path / "jobs/verify-resume-current/current"
    if current == "unfinished":
        active.mkdir(parents=True)
        (active / "config.json").write_text(json.dumps({"task": {"path": str(task)}}))
    else:
        write_trial(
            active,
            task,
            "verify",
            number=2,
            rewards={"reward": 0 if current == "reject" else 1},
            infrastructure=current == "error",
        )
    # Even a more recently touched old file cannot replace the selected attempt.
    write_trial(previous, task, "verify", rewards={"reward": 0})
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {"phase": "verify", "name": name}
                    for name in ("verify", "verify-resume-current")
                ],
                "attempts": {"verify": {"one": {"job": "verify-resume-current"}}},
            }
        )
    )
    selected, result = runner._trial_results(tmp_path, "verify")[paths[0].name]
    assert selected.parent == active
    expected = {
        "pass": "complete",
        "reject": "candidate_failed",
        "error": "infrastructure_error",
        "unfinished": "infrastructure_error",
    }
    assert runner._outcome(result)[0] == expected[current]


@pytest.mark.parametrize(
    ("error_type", "reward", "expected"),
    [
        ("AgentTimeoutError", 0, "candidate_failed"),
        ("AgentTimeoutError", 1, "complete"),
        ("AgentSetupTimeoutError", 0, "infrastructure_error"),
        ("AgentTimeoutError", None, "infrastructure_error"),
    ],
)
def test_agent_budget_keeps_native_verdict_but_setup_failure_does_not(error_type, reward, expected):
    result = {
        "finished_at": "2026-10-14T16:00:00Z",
        "exception_info": {"exception_type": error_type},
        "verifier_result": {"rewards": None if reward is None else {"reward": reward}},
    }
    assert runner._outcome(result)[0] == expected


def coordinator(tmp_path, monkeypatch, *, provider="docker", agents=None):
    from datasets import Dataset
    from test_quality_suite import questions

    from eval import setup

    settings = {
        "dataset": "owner/prepared",
        "out": str(tmp_path / "runs"),
        "concurrency": 2,
        "max_retries": 1,
        "environment": {"provider": provider},
        "tasks": {"models": ["Qwen3.6-35B-A3B"], "workloads": ["S-1", "S-N", "A-1"]},
        "agent": {"name": "codex", "kwargs": {"reasoning_effort": "xhigh"}},
        "search": {"agent": {"name": "codex", "kwargs": {"reasoning_effort": "xhigh"}}},
        "evaluation": {
            "concurrency": 8,
            "agent": {"name": "codex", "kwargs": {"reasoning_effort": "xhigh"}},
        },
    }
    if agents:
        settings.pop("agent")
        settings["agents"] = {
            label: {**agent, "kwargs": {"reasoning_effort": "xhigh", **agent.get("kwargs", {})}}
            for label, agent in agents.items()
        }
    source = tmp_path / "run.yaml"
    source.write_text(yaml.safe_dump(settings))

    def data(*args, split, **kwargs):
        return Dataset.from_list(
            [
                {
                    "id": f"{pool}-{role}-{i}",
                    "model": "Qwen3.6-35B-A3B",
                    "public": (["S-1", "S-N"] if pool == "S" else ["A-1", "A-N"])
                    if role == "public"
                    else [],
                    "private": (["S-1", "S-N"] if pool == "S" else ["A-1", "A-N"])
                    if role == "private"
                    else [],
                    "requests": [
                        {
                            "id": 0,
                            "parents": [],
                            "out": 3,
                            "tools": None,
                            "messages": '[{"role":"user","content":"prompt"}]',
                        }
                    ],
                }
                for role in ("public", "private")
                for pool in ("S", "A")
                for i in range(8)
            ]
        )

    async def quality(*args):
        pass

    monkeypatch.setattr(setup, "load_dataset", data)
    monkeypatch.setattr("eval.quality.load_dataset", lambda *a, **k: Dataset.from_list(questions()))
    monkeypatch.setattr(runner, "check_environment", lambda: None)
    monkeypatch.setattr(runner, "_quality_ready", quality)
    calls, attempts = [], {}
    active = peak = 0
    blocked = set()

    async def execute(job):
        nonlocal active, peak
        task = job.tasks[0].path
        phase = json.loads((task / "environment/runtime.json").read_text())["phase"]
        phase = "verify" if phase == "verification" else phase
        key = (str(job.jobs_dir.parent), task.name, phase)
        attempts[key] = attempts.get(key, 0) + 1
        calls.append((job.jobs_dir.parent.name, task.name, phase, job.agents[0].name))
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        write_trial(
            job.jobs_dir / job.job_name / task.name,
            task,
            phase,
            number=len(calls),
            infrastructure=(task.name, phase) in blocked,
        )
        active -= 1

    monkeypatch.setattr(runner, "_execute_job", execute)
    return source, calls, attempts, blocked, lambda: peak


@pytest.mark.parametrize(
    ("stage", "error_type"), [("_freeze_host", KeyboardInterrupt), ("_state", OSError)]
)
def test_failed_initialization_is_removed_and_can_start_cleanly(
    tmp_path, monkeypatch, stage, error_type
):
    source, calls, _, _, _ = coordinator(tmp_path, monkeypatch)
    previous = tmp_path / "runs/existing-run"
    previous.mkdir(parents=True)
    saved = previous / "submission.py"
    saved.write_text("keep the previous run")
    original = getattr(runner, stage)
    failed = []

    def fail(folder):
        failed.append(folder)
        partial = folder / "host.partial"
        partial.mkdir()
        (partial / "unfinished.py").write_text("partial initialization")
        raise error_type("initialization interrupted")

    monkeypatch.setattr(runner, stage, fail)
    with pytest.raises(error_type, match="initialization interrupted"):
        asyncio.run(runner.run(source))
    assert not failed[0].exists()
    assert list(previous.parent.iterdir()) == [previous]
    assert saved.read_text() == "keep the previous run"
    assert calls == []

    monkeypatch.setattr(runner, stage, original)
    folder = asyncio.run(runner.run(source))
    assert folder != failed[0]
    assert (folder / "host/eval/run.py").is_file()
    assert not (folder / "host.partial").exists()
    assert json.loads((folder / "state.json").read_text())["status"] == "complete"
    assert saved.read_text() == "keep the previous run"


def test_resume_startup_failure_preserves_the_run_and_releases_its_lock(tmp_path, monkeypatch):
    source, calls, _, _, _ = coordinator(tmp_path, monkeypatch)
    folder = asyncio.run(runner.run(source))
    submission = next((folder / "jobs").rglob("submission/server.py"))
    saved = {
        path: path.read_bytes()
        for path in (
            folder / "run.yaml",
            folder / "state.json",
            folder / "host/eval/run.py",
            submission,
        )
    }
    count = len(calls)
    original = runner._state

    def fail(folder):
        raise OSError("cannot write startup state")

    monkeypatch.setattr(runner, "_state", fail)
    with pytest.raises(OSError, match="cannot write startup state"):
        asyncio.run(runner.run(resume=folder))
    assert all(path.read_bytes() == content for path, content in saved.items())
    assert len(calls) == count

    monkeypatch.setattr(runner, "_state", original)
    assert asyncio.run(runner.run(resume=folder)) == folder
    assert len(calls) == count


def test_failure_after_initialization_preserves_the_run_for_resume(tmp_path, monkeypatch):
    source, calls, _, _, _ = coordinator(tmp_path, monkeypatch)
    original = runner.prepare

    async def fail(config, folder):
        raise RuntimeError("preparation failed")

    monkeypatch.setattr(runner, "prepare", fail)
    with pytest.raises(RuntimeError, match="preparation failed"):
        asyncio.run(runner.run(source))
    folder = next((tmp_path / "runs").iterdir())
    assert (folder / "run.yaml").is_file()
    assert (folder / "host/eval/run.py").is_file()
    assert json.loads((folder / "state.json").read_text())["status"] == "incomplete"
    assert calls == []

    monkeypatch.setattr(runner, "prepare", original)
    assert asyncio.run(runner.run(resume=folder)) == folder
    assert json.loads((folder / "state.json").read_text())["status"] == "complete"


@pytest.mark.parametrize("provider", ["docker", "daytona"])
def test_one_command_runs_independent_native_pipelines_under_one_concurrency_limit(
    tmp_path, monkeypatch, provider
):
    source, calls, _, _, peak = coordinator(tmp_path, monkeypatch, provider=provider)
    folder = asyncio.run(runner.run(source))
    assert peak() == 2
    for task in {c[1] for c in calls}:
        assert [c[2] for c in calls if c[1] == task] == ["search", "coding", "verify"]
    assert calls[2][2] == "coding"  # No global search-phase barrier.
    assert all(agent == ("nop" if phase == "verify" else "codex") for _, _, phase, agent in calls)
    assert json.loads((folder / "state.json").read_text())["status"] == "complete"
    assert len(read_ledger(folder / "costs.json")) == 3
    assert (folder / "results/index.html").exists()
    assert len(tuple((folder / "references").glob("*.json"))) == 3


@pytest.mark.parametrize(
    ("retries", "delays"),
    [(0, []), (1, [10]), (2, [10, 20]), (5, [10, 20, 40, 60, 60])],
)
def test_run_resume_skips_finished_tasks_and_retries_only_the_failed_phase(
    tmp_path, monkeypatch, retries, delays
):
    source, calls, _, blocked, _ = coordinator(tmp_path, monkeypatch)
    settings = yaml.safe_load(source.read_text())
    settings["max_retries"] = retries
    source.write_text(yaml.safe_dump(settings))
    observed = []
    real_sleep = asyncio.sleep

    async def sleep(seconds):
        if seconds >= 10:
            observed.append(seconds)
            await real_sleep(0)
        else:
            await real_sleep(seconds)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    failed = ("qwen36-35b-a3b_A-1", "coding")
    blocked.add(failed)
    folder = asyncio.run(runner.run(source))
    assert sum(c[1:3] == failed for c in calls) == retries + 1
    assert observed == delays
    assert json.loads((folder / "state.json").read_text())["status"] == "incomplete"
    old = len(calls)
    blocked.clear()
    asyncio.run(runner.run(resume=folder))
    assert [c[2] for c in calls[old:]] == ["coding", "verify"]
    assert observed == delays
    assert json.loads((folder / "state.json").read_text())["status"] == "complete"


def test_multiple_agents_share_search_only_inside_the_same_run(tmp_path, monkeypatch):
    source, calls, _, _, peak = coordinator(
        tmp_path, monkeypatch, agents={"first": {"name": "codex"}, "second": {"name": "oracle"}}
    )
    folder = asyncio.run(runner.run(source))
    assert len([c for c in calls if c[2] == "search"]) == 3
    assert len([c for c in calls if c[2] == "coding"]) == 6
    assert peak() == 2
    assert (folder / "agents/first/result.json").exists()
    assert (folder / "agents/second/result.json").exists()
    assert len(json.loads((folder / "results/report.json").read_text())) == 6


def test_local_gpu_groups_are_disjoint_and_hold_a_whole_task_allocation():
    from eval.config import parse_config

    config = parse_config(
        {
            "concurrency": 8,
            "environment": {"kwargs": {"gpu_ids": [[0, 1], [2, 3]]}},
            "search": {"agent": {"name": "codex", "kwargs": {"reasoning_effort": "xhigh"}}},
            "evaluation": {"agent": {"name": "codex", "kwargs": {"reasoning_effort": "xhigh"}}},
        }
    )
    pool = runner._resources(config)
    assert pool.qsize() == 2
    assert pool.get_nowait() == ["0", "1"]
    config = replace(
        config, environment=replace(config.environment, kwargs={"gpu_ids": [[0, 1], [1, 2]]})
    )
    with pytest.raises(ValueError, match="overlap"):
        runner._resources(config)


def write_config(path, *, oracle=False):
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": "owner/recordings",
                "environment": {"cpus": 4, "memory_mb": 16384},
                "tasks": {"models": ["Qwen3.6-35B-A3B"], "workloads": ["S-1"]},
                "agent": {
                    "name": "oracle" if oracle else "codex",
                    "model": None,
                    "budget": "1h",
                    "kwargs": {"reasoning_effort": "high"},
                },
                "search": {
                    "agent": {
                        "name": "claude-code",
                        "budget": "30m",
                        "kwargs": {"reasoning_effort": "high"},
                    }
                },
                "evaluation": {
                    "agent": {
                        "name": "claude-code",
                        "kwargs": {"reasoning_effort": "high"},
                    }
                },
                "out": "runs",
            }
        )
    )


def prepared_tasks(folder, names=("one", "two"), *, portable=False):
    assets = {
        "source": "owner/model",
        "baselines": {"freetoken": "owner/model", "ollama": "model:tag"},
    }
    paths = tuple(folder / "coding" / name for name in names)
    for path in paths:
        (path / "reference/metadata").mkdir(parents=True)
        for phase in ("search", "coding", "verification"):
            env = folder / phase / path.name / "environment"
            env.mkdir(parents=True, exist_ok=True)
            (env / "task.json").write_text(
                json.dumps({"model": "Qwen3.6-35B-A3B", "assets": assets})
            )
            (env / "runtime.json").write_text(json.dumps({"phase": phase, "files": []}))
    (folder / "prepared.json").write_text(
        json.dumps({"task_paths": list(map(str, paths)), "assets": {"Qwen3.6-35B-A3B": assets}})
    )
    return paths


@pytest.mark.parametrize("completed_phase", ["search", "coding"])
def test_resume_after_phase_completion_still_finishes_the_task(
    tmp_path, monkeypatch, completed_phase
):
    source, calls, _, _, _ = coordinator(tmp_path, monkeypatch)
    folder = asyncio.run(runner.run(source))
    task_id = "qwen36-35b-a3b_A-1"
    state_path = folder / "state.json"
    state = json.loads(state_path.read_text())
    remaining = ("coding", "verify") if completed_phase == "search" else ("verify",)
    state["jobs"] = [
        job
        for job in state["jobs"]
        if task_id not in job["task_ids"] or job["phase"] not in remaining
    ]
    for phase in remaining:
        state["attempts"][phase].pop(task_id)
    state["tasks"][task_id].update(status="complete", phase=completed_phase)
    state["status"] = "interrupted"
    state_path.write_text(json.dumps(state))
    before = len(calls)

    asyncio.run(runner.run(resume=folder))

    assert [call[2] for call in calls[before:]] == list(remaining)
    entry = json.loads(state_path.read_text())["tasks"][task_id]
    assert entry["status"] == "complete" and entry["phase"] == "verify"


@pytest.mark.parametrize("window", ["five_hour", "seven_day", "allowed", "successful_retry"])
def test_native_quota_reset_uses_structured_events(tmp_path, window):
    assert rate_limit_reset(tmp_path) is None
    reset = 2_000_000_000
    events = [
        {
            "type": "rate_limit_event",
            "rate_limit_info": {
                "status": "rejected",
                "resetsAt": reset,
                "rateLimitType": window,
            },
        }
    ]
    if window == "allowed":
        events.append({"type": "rate_limit_event", "rate_limit_info": {"status": "allowed"}})
    elif window == "successful_retry":
        events.append({"type": "result", "is_error": False})
    else:
        events.append({"type": "result", "is_error": True, "subtype": "success"})
    (tmp_path / "claude-code.txt").write_text("".join(json.dumps(e) + "\n" for e in events))
    assert rate_limit_reset(tmp_path) == (
        None if window in ("allowed", "successful_retry") else reset
    )


@pytest.mark.parametrize("started", [True, False])
@pytest.mark.parametrize("rejected", [True, False])
def test_restored_quota_log_requires_current_agent_execution(
    tmp_path, monkeypatch, started, rejected
):
    source = tmp_path / "run.yaml"
    write_config(source)
    task = prepared_tasks(tmp_path, ("one",), portable=True)[0]
    config = runner.load_config(source)
    config = replace(config, agent=replace(config.agent, name="claude-code"))
    state = {"jobs": []}
    reset = 2_000_000_000

    async def execute(job):
        trial = job.jobs_dir / job.job_name / "trial"
        (trial / "agent").mkdir(parents=True)
        (trial / "config.json").write_text("{}")
        if started:
            (trial / "agent-start.json").write_text("{}")
        event = {
            "type": "rate_limit_event",
            "rate_limit_info": {
                "status": "rejected" if rejected else "allowed",
                "resetsAt": reset,
            },
        }
        (trial / "agent/claude-code.txt").write_text(json.dumps(event) + "\n")

    monkeypatch.setattr(runner, "_execute_job", execute)
    if started and rejected:
        with pytest.raises(runner._AgentLimit):
            asyncio.run(runner._run_phase(config, tmp_path, (task,), "coding", state=state))
        assert state["agent_limits"]["claude-code"] == reset
    else:
        asyncio.run(runner._run_phase(config, tmp_path, (task,), "coding", state=state))
        assert not state.get("agent_limits")


@pytest.mark.parametrize("provider", ["docker", "daytona"])
def test_quota_wait_releases_resource_for_other_agents(tmp_path, monkeypatch, provider):
    source, calls, _, _, _ = coordinator(
        tmp_path,
        monkeypatch,
        provider=provider,
        agents={"limited": {"name": "claude-code"}, "other": {"name": "codex"}},
    )
    settings = yaml.safe_load(source.read_text())
    settings["concurrency"] = 1
    source.write_text(yaml.safe_dump(settings))
    original_phase, original_wait = runner._run_phase, runner._wait_agent_limit
    reset_at = None
    waits = []

    async def phase(config, folder, paths, name, **kwargs):
        nonlocal reset_at
        if name == "coding" and config.agent.name == "claude-code":
            if reset_at is None:
                reset_at = datetime.now(UTC).timestamp() + 0.4
            if datetime.now(UTC).timestamp() < reset_at:
                raise runner._AgentLimit("claude-code", name, reset_at)
        await original_phase(config, folder, paths, name, **kwargs)

    async def wait(folder, state, task_id, limit):
        await original_wait(folder, state, task_id, limit)
        assert any(c[0] == "other" and c[2] == "verify" for c in calls)
        waits.append(task_id)

    monkeypatch.setattr(runner, "_run_phase", phase)
    monkeypatch.setattr(runner, "_wait_agent_limit", wait)
    folder = asyncio.run(runner.run(source))
    assert waits
    assert json.loads((folder / "state.json").read_text())["status"] == "complete"
    assert len([c for c in calls if c[0] == "limited" and c[2] == "coding"]) == 3


def test_finished_coding_does_not_wait_for_a_limited_agent(tmp_path, monkeypatch):
    source, calls, _, _, _ = coordinator(tmp_path, monkeypatch)
    folder = asyncio.run(runner.run(source))
    previous = len(calls)
    task = runner._stage_task(folder, "coding", "qwen36-35b-a3b_S-1")
    state = json.loads((folder / "state.json").read_text())
    state["agent_limits"] = {"codex": datetime.now(UTC).timestamp() + 3600}
    asyncio.run(
        runner._run_phase(runner.load_config(source), folder, (task,), "coding", state=state)
    )
    assert len(calls) == previous


def test_expired_agent_limit_allows_a_native_job(tmp_path, monkeypatch):
    folder = tmp_path / "run"
    folder.mkdir()
    source = folder / "run.yaml"
    write_config(source)
    task = prepared_tasks(folder, ("one",), portable=True)[0]
    launched = []

    async def execute(job):
        launched.append(job)

    monkeypatch.setattr(runner, "_execute_job", execute)
    state = {"agent_limits": {"codex": datetime.now(UTC).timestamp() - 1}}
    asyncio.run(
        runner._run_phase(runner.load_config(source), folder, (task,), "coding", state=state)
    )
    assert len(launched) == 1
