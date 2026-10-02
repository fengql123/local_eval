import pytest

from eval.config import QUALITY_SUBSET, TaskSelection, load_config, parse_config, select_tasks


def test_task_filters_select_their_intersection():
    config = parse_config({"tasks": {"models": ["C2"], "workloads": ["S-1"], "types": ["SOTA"]}})
    assert [(task.task_id, task.model) for task in select_tasks(config.tasks)] == [
        ("qwen36-35b-a3b_S-1", "Qwen3.6-35B-A3B")
    ]
    assert QUALITY_SUBSET == 24
    assert config.search.agent.name == "claude-code"
    assert config.search.agent.model is None


def test_load_config_resolves_all_authored_settings_without_side_effects(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        """dataset: owner/recordings
gpus: [0, GPU-example]
tasks:
  models: all
  workloads: [S-N, A-N]
  types: all
agent:
  name: codex
  model: example-model
  kwargs: {reasoning_effort: high}
  auth: api_key
  env: [CUSTOM_API_KEY]
  budget: 30m
search:
  mode: single
  agent:
    name: harbor.agents.nop:NopAgent
    kwargs: {test_option: enabled}
    auth: api_key
    env: [SEARCH_CREDENTIAL]
    budget: 45m
evaluation: {install_timeout: 90, health_timeout: 5m}
out: results
"""
    )
    config = load_config(path)
    assert config.dataset == "owner/recordings"
    assert config.gpus == ("0", "GPU-example")
    assert config.tasks == TaskSelection(models=None, workloads=("S-N", "A-N"), types=None)
    assert config.agent.name == "codex"
    assert config.agent.model == "example-model"
    assert config.agent.kwargs == {"reasoning_effort": "high"}
    assert config.agent.auth == "api_key"
    assert config.agent.env == ("CUSTOM_API_KEY",)
    assert config.agent.budget_seconds == 1800
    assert config.search.mode == "single"
    assert config.search.agent.name == "harbor.agents.nop:NopAgent"
    assert config.search.agent.model is None
    assert config.search.agent.kwargs == {"test_option": "enabled"}
    assert config.search.agent.auth == "api_key"
    assert config.search.agent.env == ("SEARCH_CREDENTIAL",)
    assert config.search.agent.budget_seconds == 2700
    assert config.evaluation.install_timeout_seconds == 90
    assert config.evaluation.health_timeout_seconds == 300
    assert config.out == tmp_path / "results"
    assert not config.out.exists()
    custom = parse_config({"agent": {"name": "harbor.agents.nop:NopAgent"}})
    assert custom.agent.name == "harbor.agents.nop:NopAgent"
    assert custom.agent.model is None
    assert parse_config({"agent": {"name": "codex", "model": None}}).agent.model is None


def test_unknown_config_field_is_rejected():
    with pytest.raises(TypeError, match="ids"):
        parse_config({"tasks": {"ids": ["qwen36-35b-a3b_A-N"]}})


def test_empty_task_intersection_is_rejected():
    with pytest.raises(ValueError, match="select no tasks"):
        select_tasks(TaskSelection(models=("C2",), types=("NEW",)))
