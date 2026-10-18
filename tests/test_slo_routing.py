"""Coordinator classification: empty mappings, support verdicts, scores and resume."""

import json
from dataclasses import asdict
from pathlib import Path

from test_run import write_trial
from test_tasks import fixture

from eval import run as runner
from eval.config import RunConfig, Task
from eval.tasks import freeze_image, render_task, select_image

VERDICT = {
    "supported": False,
    "attempts": [
        {"baseline": "freetoken", "stage": "launch", "error": "no kernel", "log": None},
        {"baseline": "ollama", "stage": "serve", "error": "OOM", "log": "attempts/01.log"},
    ],
}


def prepared_run(tmp_path, config):
    root, _, inputs, assets = fixture(tmp_path)
    folder = tmp_path / "run"
    image = freeze_image(select_image(root, "nvidia-cuda-x86", tuple(assets["baselines"])), folder)
    static = {
        "checkpoint": "nvidia/MiniMax-M3-NVFP4",
        "baselines": {},
    }
    planned = (
        (Task("searched_S-1", "Qwen3.6-35B-A3B", "S-1"), assets),
        (Task("static_S-1", "MiniMax-M3", "S-1"), static),
    )
    quality = inputs.pop("quality")
    paths = tuple(
        render_task(task, config, image, folder, {**inputs, "quality": quality}, value)
        for task, value in planned
    )
    manifest = {
        "schema": 6,
        "tasks": [{**asdict(task), "device_category": "nvidia-cuda-x86"} for task, _ in planned],
        "image": image.manifest,
        "inputs": {task.model: {task.workload: inputs} for task, _ in planned},
        "quality": quality,
        "assets": {task.model: value for task, value in planned},
        "task_paths": list(map(str, paths)),
    }
    (folder / "prepared.json").write_text(json.dumps(manifest))
    return folder, paths


def publish_verdict(folder, task_id):
    trial = folder / "jobs/search" / task_id
    write_trial(trial, folder / "tasks/search" / task_id, "search")
    published = trial / "artifacts/var/lib/local-eval/reference"
    (published / "reference.json").write_text(json.dumps(VERDICT))
    (published / "attempts").mkdir()
    (published / "attempts/01.log").write_text("CUDA out of memory\n")
    return trial


def test_unsupported_verdicts_and_static_pairs_become_slo_tasks(tmp_path):
    config = RunConfig()
    folder, paths = prepared_run(tmp_path, config)
    publish_verdict(folder, "searched_S-1")
    assert runner._coding_tasks(folder, paths, config=config) == paths
    targets = (
        ("searched_S-1", "nvidia/Qwen3.6-35B-A3B-NVFP4"),
        ("static_S-1", "nvidia/MiniMax-M3-NVFP4"),
    )
    for task_id, checkpoint in targets:
        for phase in ("agent", "evaluation"):
            environment = folder / "tasks" / phase / task_id / "environment"
            metadata = json.loads((environment / "task.json").read_text())
            runtime = json.loads((environment / "runtime.json").read_text())
            assert metadata["type"] == "SLO" and metadata["reference"] is None
            # Both kinds of SLO pair target FreeToken's checkpoint.
            assert metadata["target"] == runtime["target"]
            assert metadata["target"]["source"] == checkpoint
            # No reference handoff reaches an SLO coding or verification sandbox.
            assert not any("reference" in item["target"] for item in runtime["files"])
        steps = folder / "tasks/evaluation" / task_id / "steps"
        assert sorted(path.name for path in steps.iterdir()) == [
            "prepare",
            "quality",
            "review",
            "slo",
        ]
    prompt = (folder / "tasks/agent/searched_S-1/instruction.md").read_text()
    assert "--mode slo" in prompt and "reference.json" not in prompt
    # Classification happens once; later passes leave the rendered task alone.
    instruction = folder / "tasks/agent/searched_S-1/instruction.md"
    instruction.write_text("unchanged")
    runner._coding_tasks(folder, paths, config=config)
    assert instruction.read_text() == "unchanged"
    # An interrupted render has no final verification TOML; regenerate the pair.
    verification = folder / "tasks/evaluation/searched_S-1/task.toml"
    verification.unlink()
    runner._coding_tasks(folder, paths, config=config)
    assert verification.is_file() and instruction.read_text() != "unchanged"


def test_classification_preserves_all_tasks_and_the_unified_score(tmp_path):
    config = RunConfig()
    folder, paths = prepared_run(tmp_path, config)
    publish_verdict(folder, "searched_S-1")
    assert runner._coding_tasks(folder, paths, config=config) == paths
    task_id = "static_S-1"
    write_trial(folder / "jobs/coding" / task_id, folder / "tasks/agent" / task_id, "coding")
    trial = folder / "jobs/verify" / task_id
    rewards = {"score": 60.0}
    write_trial(trial, folder / "tasks/evaluation" / task_id, "verify", rewards=rewards)
    report = trial / "steps/slo/verifier/report.json"
    report.parent.mkdir(parents=True)
    slo = {
        "kind": "balanced",
        "tier": "T2",
        "tier_met": 2,
        "score": 60.0,
        "coverage": {"rows": 60, "repeats": 1, "started": 60, "fraction": 1.0},
        "tiers": [{"id": "T1", "met": True, "score": 1.0, "attainment": 1.0, "bins": {}}],
    }
    report.write_text(json.dumps({"outcome": "passed", "slo": slo}))
    results = runner._final_results(folder, paths)
    # The searched pair is SLO now; its missing coding attempt is not a zero score.
    assert results["searched_S-1"]["type"] == "SLO"
    assert results["searched_S-1"]["status"] == "infrastructure_error"
    assert results["searched_S-1"]["stage"] == "coding"
    static = results["static_S-1"]
    assert static["status"] == "complete" and static["type"] == "SLO"
    assert static["rewards"] == {"score": 60.0} and static["score"] == 60.0
    assert static["slo"]["tier"] == "T2"
    assert static["slo"]["tiers"] == [{"id": "T1", "met": True, "score": 1.0, "attainment": 1.0}]
    assert static["search_result"] is None


def test_resumed_search_restores_the_verdict_with_its_log_tails(tmp_path):
    config = RunConfig()
    folder, _ = prepared_run(tmp_path, config)
    trial = publish_verdict(folder, "searched_S-1")
    progress = {"agent_complete": True, "completed_steps": []}
    (trial / "progress.json").write_text(json.dumps(progress))
    task = folder / "tasks/search/searched_S-1"
    resumed, _ = runner._resume_task(folder, "search", task, trial, config.agent)
    runtime = json.loads((resumed / "environment/runtime.json").read_text())
    restored = {item["target"]: item for item in runtime["files"]}
    selection = json.loads(open(restored["/workspace/output/selection.json"]["source"]).read())
    assert selection == {"supported": False, "attempts": selection["attempts"]}
    logged = selection["attempts"][1]
    assert logged["log"] == "/workspace/output/reference/attempts/01.log"
    logs = restored["/workspace/output/reference"]
    assert logs["owner"] == "search" and not logs["read_only"]
    assert (Path(logs["source"]) / "attempts/01.log").read_text() == "CUDA out of memory\n"
    assert runtime["agent_complete"] is True
