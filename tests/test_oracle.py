"""Opt-in complete native Oracle pipeline using production images and real weights.

Set LOCAL_EVAL_ORACLE_CONFIG to a normal config with one model/workload, a real
prepared dataset, sufficient GPU/disk and a real verification agent. Search and
coding are replaced with native OracleAgent; reference, preparation, actual
review, quality and timed replay all execute unchanged. Normal provider cleanup
applies and the run folder retains evidence. No metadata-only weights or skipped
verification steps are used.
"""

import asyncio
import json
import os
from pathlib import Path

import pytest
import yaml

from eval.run import run


@pytest.mark.skipif(
    not os.environ.get("LOCAL_EVAL_ORACLE_CONFIG"),
    reason="opt-in complete production Harbor jobs, real model inference and paid source review",
)
def test_native_oracle_uses_every_verification_gate(tmp_path):
    original = Path(os.environ["LOCAL_EVAL_ORACLE_CONFIG"]).resolve()
    settings = yaml.safe_load(original.read_text())
    # Resolve the ordinary output path before writing the temporary test config.
    settings["out"] = str((original.parent / settings.get("out", "runs")).resolve())
    settings["concurrency"] = 1
    settings["agent"] = {"name": "oracle", "budget": "24h", "kwargs": {"reasoning_effort": "xhigh"}}
    settings.setdefault("search", {})["agent"] = {
        "name": "oracle",
        "budget": "24h",
        "kwargs": {"reasoning_effort": "xhigh"},
    }
    config = tmp_path / "oracle.yaml"
    config.write_text(yaml.safe_dump(settings))
    folder = asyncio.run(run(config))
    results = json.loads((folder / "result.json").read_text())["tasks"]
    assert len(results) == 1, "supply one model/workload for this full integration check"
    result = next(iter(results.values()))
    assert result["status"] == "complete", result
    verified_path = Path(result["verification_result"])
    trial = json.loads(verified_path.read_text())
    assert trial["agent_info"]["name"] == "nop"
    assert [step["step_name"] for step in trial["step_results"]] == [
        "prepare",
        "review",
        "quality",
        "reference",
        "measure",
    ]
    assert not any(step.get("exception_info") for step in trial["step_results"])
    steps = verified_path.parent / "steps"
    reference = json.loads((steps / "reference/verifier/report.json").read_text())
    prepared = json.loads((steps / "prepare/verifier/report.json").read_text())
    reviewed = json.loads((steps / "review/verifier/report.json").read_text())
    measured = json.loads((steps / "measure/verifier/report.json").read_text())
    quality = json.loads((steps / "quality/verifier/report.json").read_text())
    assert reference["ok"] and reference["bench"]["valid"]
    assert prepared["install"]["ok"] and "obtain_weight" not in prepared
    assert (
        quality["ok"]
        and quality["quality"]["bound"]["correct"] >= quality["quality"]["bound"]["min_correct"]
    )
    assert reviewed["ok"] and reviewed["complete"] and len(reviewed["criteria"]) == 7
    assert measured["outcome"] == "passed" and measured["bench"]["candidate"]["valid"]


@pytest.mark.skipif(
    not os.environ.get("LOCAL_EVAL_ORACLE_CONFIG"),
    reason="opt-in complete production Harbor jobs, real model inference and paid source review",
)
def test_native_oracle_places_an_unsupported_pair_on_slo_tiers(tmp_path):
    from eval.baselines import mappings
    from eval.config import parse_config, select_tasks

    original = Path(os.environ["LOCAL_EVAL_ORACLE_CONFIG"]).resolve()
    settings = yaml.safe_load(original.read_text())
    settings["out"] = str((original.parent / settings.get("out", "runs")).resolve())
    settings["concurrency"] = 1
    settings["agent"] = {"name": "oracle", "budget": "24h", "kwargs": {"reasoning_effort": "xhigh"}}
    (task,) = select_tasks(parse_config(settings, base=original.parent).tasks)
    category = settings.get("environment", {}).get("device_category", "nvidia-cuda-x86")
    # A frozen unsupported verdict routes the pair to SLO on FreeToken's checkpoint;
    # the coding oracle then submits FreeToken, so tier placement has a working server.
    assert "freetoken" in {item.baseline for item in mappings(category, task.model)}
    verdicts = tmp_path / "verdicts"
    verdicts.mkdir()
    (verdicts / f"{task.task_id}.json").write_text(json.dumps({"supported": False}))
    settings.setdefault("search", {})["reference"] = str(verdicts)
    config = tmp_path / "oracle-slo.yaml"
    config.write_text(yaml.safe_dump(settings))
    folder = asyncio.run(run(config))
    result = json.loads((folder / "result.json").read_text())["tasks"][task.task_id]
    assert result["status"] == "complete" and result["type"] == "SLO", result
    trial = json.loads(Path(result["verification_result"]).read_text())
    assert [step["step_name"] for step in trial["step_results"]] == [
        "prepare",
        "review",
        "quality",
        "slo",
    ]
    steps = Path(result["verification_result"]).parent / "steps"
    report = json.loads((steps / "slo/verifier/report.json").read_text())
    assert report["outcome"] == "passed" and report["slo"]["coverage"]["started"] > 0
    assert result["slo"]["tier_met"] == report["slo"]["tier_met"]
