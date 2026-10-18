"""Two review inputs/results and two invalid submission/evidence cases."""

import copy
import tomllib
from pathlib import Path

from rewardkit.models import JudgeTomlConfig

from eval.request_policy import request_fields
from eval.verify import judge


def task():
    return {
        "model": "Qwen3.6-35B-A3B",
        "workload": "A-MA",
        "assets": {"source": "Qwen/Qwen3.6-35B-A3B"},
        "target": {"source": "Qwen/Qwen3.6-35B-A3B"},
        "evaluation": {
            "agent": {
                "name": "claude-code",
                "model": "review-model",
                "budget_seconds": 30,
                "kwargs": {"version": "2.1.276", "reasoning_effort": "xhigh"},
            }
        },
    }


def decisions():
    rubric = tomllib.loads(Path(judge.__file__).with_name("rules.toml").read_text())
    return {
        "reward": {
            "kind": "agent",
            "criteria": [
                {"name": criterion["name"], "value": 1, "reasoning": "Source follows this rule."}
                for criterion in rubric["criterion"]
            ],
        }
    }


def test_review_uses_native_rewardkit_config_and_task_specific_prompt(tmp_path):
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "server.py").write_text("# Readable implementation\n")
    rubric = tmp_path / "rubric"
    judge.write_review(task(), submission, rubric)
    native = JudgeTomlConfig.model_validate(tomllib.loads((rubric / "rules.toml").read_text()))
    assert native.judge.judge == "claude-code" and native.judge.model == "review-model"
    assert native.judge.mode == "batched" and not native.judge.isolated
    assert native.judge.timeout == 30 and native.judge.version == "2.1.276"
    assert native.judge.reasoning_effort == "xhigh"
    assert native.scoring.aggregation == "all-pass" and len(native.criterion) == 7
    prompt = (rubric / "prompt.txt").read_text()
    assert "Qwen3.6-35B-A3B" in prompt and "A-MA" in prompt
    assert str(submission.resolve()) in prompt
    assert "/workspace/server.launch /models/target" in prompt
    assert "127.0.0.1:8000" in prompt and "{criteria}" in prompt
    assert str(request_fields("Qwen3.6-35B-A3B").fixed("A-MA")["temperature"]) in prompt
    assert (submission / "server.py").read_text() == "# Readable implementation\n"
    review_task = task()
    review_task["evaluation"]["agent"].update(
        name="codex", model=None, kwargs={"reasoning_effort": "high"}
    )
    judge.write_review(review_task, submission, rubric)
    updated = tomllib.loads((rubric / "rules.toml").read_text())
    assert updated["judge"]["judge"] == "codex" and "model" not in updated["judge"]
    assert updated["judge"]["reasoning_effort"] == "high"


def test_valid_submission_and_complete_native_decisions_pass(tmp_path):
    for name in ("install.sh",):
        (tmp_path / name).write_text("#!/bin/bash\nexit 0\n")
    assert judge.static_check(tmp_path) == {
        "ok": True,
        "missing": [],
        "external_paths": [],
        "reserved_paths": [],
    }
    result = judge.rules_result({"reward": 1}, decisions())
    assert result["ok"] is True and result["complete"] and result["kind"] == "agent"
    assert len(result["criteria"]) == 7 and all(item["value"] == 1 for item in result["criteria"])


def test_required_scripts_and_other_submission_paths_cannot_escape(tmp_path):
    missing = judge.static_check(tmp_path)
    assert not missing["ok"] and missing["missing"] == ["install.sh"]
    for name in ("install.sh",):
        (tmp_path / name).write_text("#!/bin/bash\n")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.sh"
    outside.write_text("#!/bin/bash\n")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/dependency.py").symlink_to(outside)
    invalid = judge.static_check(tmp_path)
    assert not invalid["ok"] and invalid["external_paths"] == ["nested/dependency.py"]
    (tmp_path / "nested/dependency.py").unlink()
    (tmp_path / "nested/dependency.py").symlink_to(tmp_path / "missing.sh")
    invalid = judge.static_check(tmp_path)
    assert invalid["ok"]  # A broken internal dependency is a preparation error, not an escape.
    hidden = tmp_path / ".harbor"
    hidden.mkdir()
    (hidden / "native-secret").write_text("not a submission")
    (tmp_path / "native-secret").symlink_to(hidden / "native-secret")
    assert judge.static_check(tmp_path)["external_paths"] == ["native-secret"]
    (tmp_path / "install.sh").unlink()
    assert judge.static_check(tmp_path)["missing"] == ["install.sh"]


def test_rule_violations_have_evidence_and_unusable_results_have_no_verdict():
    rejected = decisions()
    rejected["reward"]["criteria"][0].update(
        value=0, reasoning="server.py:9 returns stored answers rather than model output."
    )
    result = judge.rules_result({"reward": 0}, rejected)
    assert result["ok"] is False and result["complete"]
    assert "server.py:9" in result["criteria"][0]["reasoning"]
    malformed = [({}, {}), ({"reward": 1}, {"reward": {"criteria": []}}), ({"reward": 1}, rejected)]
    no_pass_evidence = decisions()
    no_pass_evidence["reward"]["criteria"][0]["reasoning"] = ""
    malformed.append(({"reward": 1}, no_pass_evidence))
    for field, value in (("reasoning", ""), ("error", "review timed out"), ("value", 2)):
        incomplete = copy.deepcopy(rejected)
        incomplete["reward"]["criteria"][0][field] = value
        malformed.append(({"reward": 0}, incomplete))
    duplicate = decisions()
    duplicate["reward"]["criteria"][0]["name"] = duplicate["reward"]["criteria"][1]["name"]
    malformed.append(({"reward": 1}, duplicate))
    for rewards, details in malformed:
        result = judge.rules_result(rewards, details)
        assert result["ok"] is None and not result["complete"]
