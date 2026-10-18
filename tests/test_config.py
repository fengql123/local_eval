"""Two normal config cases and two invalid-input boundaries."""

import pytest

from eval.config import (
    TaskSelection,
    load_config,
    model,
    parse_config,
    reviewer_env,
    select_tasks,
)


def configured(value, **kwargs):
    agent = {"name": "claude-code", "kwargs": {"reasoning_effort": "xhigh"}}
    return parse_config(
        {
            **value,
            "search": {"agent": agent, **value.get("search", {})},
            "evaluation": {"agent": agent, **value.get("evaluation", {})},
        },
        **kwargs,
    )


def test_local_config_uses_declared_category_and_native_agent(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "secret-test-value")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("REWARDKIT_FORCE_SUBSCRIPTION", raising=False)
    config = configured(
        {
            "tasks": {"models": ["Qwen3.6-35B-A3B"], "workloads": ["S-1"]},
            "environment": {
                "device_category": "nvidia-cuda-x86",
                "kwargs": {"gpu_ids": [0, "GPU-example"]},
            },
            "agent": {
                "name": "custom.module:Agent",
                "kwargs": {"reasoning_effort": "high", "option": "value"},
            },
        },
        base=tmp_path,
    )
    assert config.environment.kwargs == {"gpu_ids": [0, "GPU-example"]}
    assert not hasattr(config.environment, "storage")
    assert config.environment.device_category == "nvidia-cuda-x86"
    assert config.agent.name == "custom.module:Agent" and config.agent.model is None
    assert config.agent.kwargs == {"reasoning_effort": "high", "option": "value"}
    assert reviewer_env(config.evaluation.agent) == ("CLAUDE_CODE_OAUTH_TOKEN",)
    assert model("Qwen3.6-35B-A3B").name == "Qwen3.6-35B-A3B"
    assert config.evaluation.replay_concurrency("S-N") == 32
    assert config.evaluation.replay_concurrency("S-1") == 1
    assert select_tasks(config.tasks)[0].task_id == "qwen36-35b-a3b_S-1"
    assert not (tmp_path / "runs").exists()


def test_cloud_native_kwargs_and_shared_yaml_agents_are_preserved(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("""dataset: owner/prepared
concurrency: 2
environment:
  provider: daytona
  device_category: nvidia-cuda-x86
  kwargs: {gpus: 1, gpu_types: [RTX-4090], spot: true}
agent: &agent
  name: claude-code
  model: supplied-model
  kwargs: {reasoning_effort: xhigh}
  budget: 24h
search: {agent: *agent}
evaluation: {agent: *agent, concurrency: {S-N: 32, A-N: 8}}
out: results
""")
    config = load_config(path)
    assert config.environment.kwargs == {"gpus": 1, "gpu_types": ["RTX-4090"], "spot": True}
    assert all(
        agent.kwargs["reasoning_effort"] == "xhigh"
        for agent in (config.agent, config.search.agent, config.evaluation.agent)
    )
    assert config.out == tmp_path / "results"
    assert config.evaluation.replay_concurrency("S-N") == 32
    assert config.evaluation.replay_concurrency("A-N") == 8
    assert all(
        agent.budget_seconds == 86400
        for agent in (config.agent, config.search.agent, config.evaluation.agent)
    )


def test_removed_fields_and_invalid_resource_selection_are_rejected():
    for value in (
        {"gpus": [0]},
        {"storage": {}},
        {"environment": {"storage": {"persistent": True}}},
        {"environment": {"image": "unused"}},
        {"agent": {"name": "codex", "auth": "login"}},
    ):
        with pytest.raises(TypeError):
            configured(value)
    with pytest.raises(ValueError, match="local GPU IDs"):
        configured({"environment": {"provider": "daytona", "kwargs": {"gpu_ids": [0]}}})
    for value in (0, -1, "2", True):
        with pytest.raises(ValueError, match="concurrency"):
            configured({"concurrency": value})
    for value in (7, -1, {"S-N": 15}, {"A-1": 8}):
        with pytest.raises(ValueError, match="evaluation.concurrency"):
            configured({"evaluation": {"concurrency": value}})


def test_empty_task_intersection_and_unknown_reviewer_are_rejected():
    with pytest.raises(ValueError, match="select no tasks"):
        select_tasks(TaskSelection(models=("Qwen3.6-35B-A3B",), workloads=()))
    with pytest.raises(ValueError, match="unknown Rewardkit"):
        configured({"evaluation": {"agent": {"name": "unknown"}}})


def test_classification_is_not_a_config_filter_and_slo_settings_parse():
    assert len(select_tasks(TaskSelection())) == 35
    with pytest.raises(TypeError, match="types"):
        configured({"tasks": {"types": ["SLO"]}})
    assert configured({}).evaluation.concurrency == {"S-N": 32, "A-N": 8}
    evaluation = configured(
        {"evaluation": {"slo_budget": "2h", "slo_request_cap": "5m", "slo_repeats": 3}}
    ).evaluation
    assert (evaluation.slo_budget_seconds, evaluation.slo_request_cap_seconds) == (7200, 300)
    assert evaluation.slo_repeats == 3
    assert (
        configured({"evaluation": {"quality_timeout": "2h"}}).evaluation.quality_timeout_seconds
        == 7200
    )
    defaults = configured({}).evaluation
    assert (defaults.slo_budget_seconds, defaults.slo_request_cap_seconds) == (21600, 600)
    assert defaults.quality_timeout_seconds == 18000
    with pytest.raises(ValueError, match="slo_repeats"):
        configured({"evaluation": {"slo_repeats": 0}})


@pytest.mark.parametrize("location", ["agent", "agents.second", "search.agent", "evaluation.agent"])
def test_every_agent_requires_an_explicit_reasoning_effort(location):
    agent = {"name": "codex", "kwargs": {"reasoning_effort": "xhigh"}}
    value = {"agent": agent, "search": {"agent": agent}, "evaluation": {"agent": agent}}
    missing = {"name": "codex"}
    if location == "agent":
        value["agent"] = missing
    elif location == "agents.second":
        value.pop("agent")
        value["agents"] = {"first": agent, "second": missing}
    else:
        value[location.split(".")[0]] = {"agent": missing}
    with pytest.raises(ValueError, match=rf"{location}\.kwargs\.reasoning_effort"):
        parse_config(value)
    # Omitting the whole search/reviewer mapping cannot silently choose an effort.
    if location in ("search.agent", "evaluation.agent"):
        value.pop(location.split(".")[0])
        with pytest.raises(ValueError, match=rf"{location}\.kwargs\.reasoning_effort"):
            parse_config(value)


@pytest.mark.parametrize("effort", [None, "", "  ", 1])
def test_reasoning_effort_cannot_be_empty_or_implicit(effort):
    with pytest.raises(ValueError, match="evaluation.agent.kwargs.reasoning_effort"):
        configured(
            {
                "evaluation": {
                    "agent": {"name": "claude-code", "kwargs": {"reasoning_effort": effort}}
                }
            }
        )
