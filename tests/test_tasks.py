"""Native three-job generation: two normal cases and two source boundaries."""

import hashlib
import json
import re
import subprocess
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
from harbor.models.task.config import TaskConfig
from harbor.models.task.task import Task as HarborTask

from eval.config import (
    AgentConfig,
    EnvironmentConfig,
    EvaluationConfig,
    RunConfig,
    SearchConfig,
    Task,
)
from eval.tasks import freeze_image, render_task, select_image


def fixture(tmp_path):
    root = tmp_path / "checkout"
    device = root / "images/nvidia-cuda-x86"
    device.mkdir(parents=True)
    (device / "Dockerfile").write_text("FROM ubuntu:22.04 AS toolchain\n")
    for phase in ("search", "coding", "verification"):
        (device / f"{phase}.Dockerfile").write_text("FROM toolchain\n")
    for baseline in ("freetoken", "ollama"):
        folder = device / "baselines" / baseline
        folder.mkdir(parents=True)
        for name in ("install.sh", "launch.sh"):
            (folder / name).write_text("#!/bin/sh\n")
    (device / "baselines/freetoken/sequential-read.patch").write_text("fixture patch\n")
    (root / "eval").mkdir()
    (root / "eval/__init__.py").write_text("")
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0.1'\n")
    run = tmp_path / "runs/fixture-id"
    inputs = {
        name: {"source": str(tmp_path / "dataset" / name)}
        for name in ("public", "private", "quality")
    }
    assets = {
        "checkpoint": "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "baselines": {
            "freetoken": "nvidia/Qwen3.6-35B-A3B-NVFP4",
            "ollama": "qwen:fixture",
        },
    }
    return root, run, inputs, assets


def embedded_python(script):
    """The Python program an oracle script feeds to the interpreter."""
    return script.read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


def read(folder):
    return (
        HarborTask(folder),
        TaskConfig.model_validate(tomllib.loads((folder / "task.toml").read_text())),
        json.loads((folder / "environment/runtime.json").read_text()),
    )


def test_local_tasks_use_phase_images_users_and_exact_dataset_scopes(tmp_path):
    root, run, inputs, assets = fixture(tmp_path)
    image = freeze_image(select_image(root, "nvidia-cuda-x86", tuple(assets["baselines"])), run)
    config = RunConfig(environment=EnvironmentConfig(kwargs={"gpu_ids": [0, "GPU-example"]}))
    render_task(
        Task("example_A-1", "Qwen3.6-35B-A3B", "A-1", "SOTA"), config, image, run, inputs, assets
    )
    records = [
        read(run / "tasks" / folder / "example_A-1") for folder in ("search", "agent", "evaluation")
    ]
    for phase, folder, (_native, parsed, runtime) in zip(
        ("search", "coding", "verification"),
        ("search", "agent", "evaluation"),
        records,
        strict=True,
    ):
        assert parsed.environment.docker_image is None
        assert (
            run / "tasks" / folder / "example_A-1/environment/Dockerfile"
        ).read_text() == image.manifest["images"][phase]["source"]
        assert runtime["gpu_ids"] == ["0", "GPU-example"]
        assert runtime["image"]["identity"] == image.manifest["images"][phase]["identity"]
        assert all(item["read_only"] for item in runtime["files"])
        assert not (run / "tasks" / folder / "example_A-1/environment/docker-compose.yaml").exists()
        for script in (run / "tasks" / folder / "example_A-1").rglob("*.sh"):
            subprocess.run(["bash", "-n", str(script)], check=True)
            for program in re.findall(r"<<'PY'\n(.*?)\nPY", script.read_text(), re.S):
                compile(program, str(script), "exec")
    search_folder = run / "tasks/search/example_A-1"
    for path in (search_folder / "instruction.md", search_folder / "solution/solve.sh"):
        assert "/var/lib/local-eval/private" in path.read_text()
    for oracle in (run / "tasks").glob("*/example_A-1/solution/solve.sh"):
        compile(embedded_python(oracle), str(oracle), "exec")
    for stage in ("search", "agent"):
        text = (run / "tasks" / stage / "example_A-1/instruction.md").read_text()
        assert "completion_mean" not in text and "TTFT" in text
    assert not (run / "tasks/agent/example_A-1/environment/run.md").exists()
    prompt = (run / "tasks/agent/example_A-1/instruction.md").read_text()
    assert "Build the best possible inference server on this machine." in prompt
    assert "/workspace/server.launch /models/target" in prompt
    assert "Time budget" not in prompt and "drafter" not in prompt
    assert "/workspace/output/submission/" in prompt and "/workspace/runtime/candidate/" in prompt
    assert "recovery" not in prompt.lower() and "--resume" not in prompt
    search, coding, verification = [record[1] for record in records]
    assert search.agent.user == "search" and coding.agent.user == "agent"
    assert search.verifier.user == search.verifier.collect[0].user == "root"
    assert coding.verifier.user == coding.verifier.collect[0].user == "root"
    for stage in ("search", "agent"):
        assert (run / "tasks" / stage / "example_A-1/tests").stat().st_mode & 0o777 == 0o700
    assert [step.name for step in verification.steps] == [
        "prepare",
        "review",
        "quality",
        "reference-prepare",
        "reference",
        "measure",
    ]
    assert [step.verifier.user for step in verification.steps] == [
        "root",
        "judge",
        "root",
        "root",
        "root",
        "root",
    ]
    assert all(step.min_reward == 1 for step in verification.steps[:5])
    assert verification.steps[3].verifier.network_mode is None
    assert all(
        verification.steps[index].verifier.network_mode.value == "no-network" for index in (2, 4, 5)
    )
    assert verification.steps[2].verifier.timeout_sec == (
        config.evaluation.quality_timeout_seconds + config.evaluation.health_timeout_seconds + 120
    )
    assert "CLAUDE_CODE_EFFORT_LEVEL" not in verification.steps[1].verifier.env
    scopes = [{item["target"] for item in record[2]["files"]} for record in records]
    assert scopes[0] == {
        "/var/lib/local-eval/task.json",
        "/var/lib/local-eval/private",
    }
    assert scopes[2] == {
        "/var/lib/local-eval/task.json",
        "/var/lib/local-eval/private",
        "/var/lib/local-eval/quality",
    }
    assert scopes[1] == {"/var/lib/local-eval/task.json", "/inputs/public"}
    assert not any("/workspace/data/S" in scope for scope in scopes)

    for workload in ("S-N", "A-N"):
        name = f"example_{workload}"
        render_task(
            Task(name, "Qwen3.6-35B-A3B", workload, "SOTA"), config, image, run, inputs, assets
        )
        prompt = (run / "tasks/agent" / name / "instruction.md").read_text()
        assert "serving throughput" in prompt and "including prefill" in prompt
        assert "completion_mean" not in prompt
        prompt = (run / "tasks/search" / name / "instruction.md").read_text()
        assert "Maximize serving throughput" in prompt and "ttft_mean" not in prompt
        prompt = (run / "tasks/agent" / name / "instruction.md").read_text()
        assert "Official verification uses **N=" in prompt


def test_cloud_source_builds_and_reviewer_credentials_are_native_placeholders(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "private-fixture-token")
    root, run, inputs, assets = fixture(tmp_path)
    image = freeze_image(select_image(root, "nvidia-cuda-x86", tuple(assets["baselines"])), run)
    config = RunConfig(
        environment=EnvironmentConfig(
            provider="daytona", kwargs={"gpus": 1, "gpu_types": ["RTX-4090"], "spot": True}
        ),
        evaluation=EvaluationConfig(
            agent=AgentConfig("claude-code", kwargs={"reasoning_effort": "xhigh"})
        ),
    )
    render_task(
        Task("example_A-MA", "Qwen3.6-35B-A3B", "A-MA", "SOTA"), config, image, run, inputs, assets
    )
    folder = run / "tasks/evaluation/example_A-MA"
    _, native, runtime = read(folder)
    assert not (folder / "steps/prepare/workdir").exists()
    assert native.environment.gpus == 1 and native.environment.gpu_types == ["RTX-4090"]
    assert native.environment.docker_image is None
    assert native.steps[1].verifier.env["CLAUDE_CODE_EFFORT_LEVEL"] == "xhigh"
    assert all(
        "CLAUDE_CODE_EFFORT_LEVEL" not in step.verifier.env
        for step in (native.steps[0], *native.steps[2:])
    )
    assert runtime["gpu_ids"] == [] and runtime["payload"]["identity"]
    assert "storage" not in runtime and "assets" not in runtime
    contents = (folder / "task.toml").read_text()
    assert "${CLAUDE_CODE_OAUTH_TOKEN}" in contents and "private-fixture-token" not in contents
    assert not (run / "build-context").exists()
    for phase, name in (("search", "search"), ("coding", "agent"), ("verification", "evaluation")):
        environment = run / "tasks" / name / "example_A-MA/environment"
        assert (environment / "Dockerfile").read_text() == image.manifest["images"][phase]["source"]
        setup = (environment / "setup.sh").read_text()
        assert "uv sync --locked" in setup and "-m eval.models" in setup
        assert "eval.providers.bootstrap" not in setup
    payload = Path(image.manifest["payload"]["source"])
    assert not (payload / "images").exists() and not (payload / "templates").exists()
    assert (payload / "baselines/nvidia-cuda-x86/freetoken/install.sh").exists()
    patch = "baselines/nvidia-cuda-x86/freetoken/sequential-read.patch"
    assert (payload / patch).read_text() == "fixture patch\n"
    assert (
        image.manifest["payload"]["hashes"][patch] == hashlib.sha256(b"fixture patch\n").hexdigest()
    )
    assert not (payload / ".venv").exists()


def test_recipe_changes_invalidate_payload_without_rebuilding_generic_image(tmp_path):
    root, _, _, assets = fixture(tmp_path)
    first = select_image(root, "nvidia-cuda-x86", tuple(assets["baselines"]))
    (root / "images/nvidia-cuda-x86/baselines/freetoken/install.sh").write_text(
        "#!/bin/sh\necho changed\n"
    )
    second = select_image(root, "nvidia-cuda-x86", tuple(assets["baselines"]))
    assert first.identity == second.identity
    assert first.manifest["payload"]["identity"] != second.manifest["payload"]["identity"]


def test_missing_authored_phase_or_launcher_stops_source_planning(tmp_path):
    root, _, _, assets = fixture(tmp_path)
    phase = root / "images/nvidia-cuda-x86/search.Dockerfile"
    text = phase.read_text()
    phase.unlink()
    with pytest.raises(FileNotFoundError, match="search.Dockerfile"):
        select_image(root, "nvidia-cuda-x86", tuple(assets["baselines"]))
    phase.write_text(text)
    (root / "images/nvidia-cuda-x86/baselines/freetoken/launch.sh").unlink()
    with pytest.raises(FileNotFoundError, match="launch.sh"):
        select_image(root, "nvidia-cuda-x86", tuple(assets["baselines"]))


def test_search_prompt_states_support_objective_resources_and_handoff(tmp_path):
    root, run, inputs, assets = fixture(tmp_path)
    image = freeze_image(select_image(root, "nvidia-cuda-x86", tuple(assets["baselines"])), run)
    config = RunConfig(
        search=SearchConfig(agent=AgentConfig("claude-code", budget_seconds=5400)),
        environment=EnvironmentConfig(
            provider="daytona", kwargs={"gpus": 2, "gpu_types": ["RTX-PRO-6000"]}
        ),
    )
    for workload, objective in (("S-1", "with equal weight"), ("S-N", "**N=32**")):
        name = f"example_{workload}"
        render_task(
            Task(name, "Qwen3.6-35B-A3B", workload, "SOTA"), config, image, run, inputs, assets
        )
        prompt = (run / "tasks/search" / name / "instruction.md").read_text()
        assert prompt.split("\n\n")[1].startswith(
            "Before anything else, comprehensively and rigorously"
        )
        assert "2 GPUs" in prompt and "RTX-PRO-6000" in prompt
        assert objective in prompt and "/var/lib/local-eval/private" in prompt
        assert "Choose your own search method" in " ".join(prompt.split())
        assert "Full replay is mandatory only during verification" in prompt
        assert '"supported": false' in prompt and "freetoken" in prompt and "ollama" in prompt
        assert "nvidia/Qwen3.6-35B-A3B-NVFP4" in prompt and "qwen:fixture" in prompt
        assert (
            "/workspace/output/selection.json" in prompt
            and "Transient infrastructure failures" in prompt
        )


def test_slo_tasks_render_without_search_reference_or_reference_timing(tmp_path):
    root, run, inputs, assets = fixture(tmp_path)
    image = freeze_image(select_image(root, "nvidia-cuda-x86", ()), run)
    config = RunConfig()
    static = {
        "checkpoint": "nvidia/MiniMax-M3-NVFP4",
        "baselines": {},
    }
    task = Task("static_S-1", "MiniMax-M3", "S-1", "SLO")
    render_task(task, config, image, run, inputs, static)
    assert not (run / "tasks/search/static_S-1").exists()
    coding = run / "tasks/agent/static_S-1"
    prompt = (coding / "instruction.md").read_text()
    assert "reference.json" not in prompt and "--mode slo" in prompt
    assert "/opt/eval/tables/slo.json" in prompt
    assert "Four tiers" in prompt and "0–100" in prompt
    assert "sqrt(min(1, tier_TTFT / mean_TTFT)" in prompt
    assert "Incomplete and unstarted units earn zero" in prompt
    assert not (coding / "environment/run.md").exists()
    # Every SLO task targets FreeToken's checkpoint; tokenizer files come from the source.
    target = {"source": "nvidia/MiniMax-M3-NVFP4"}
    for folder in (coding, run / "tasks/evaluation/static_S-1"):
        metadata = json.loads((folder / "environment/task.json").read_text())
        assert metadata["type"] == "SLO" and metadata["target"] == target
        assert metadata["reference"] is None
        assert json.loads((folder / "environment/runtime.json").read_text())["target"] == target
    oracle = coding / "solution/solve.sh"
    compile(embedded_python(oracle), str(oracle), "exec")
    assert "'baseline': None" in oracle.read_text()
    _, verification, _ = read(run / "tasks/evaluation/static_S-1")
    assert [step.name for step in verification.steps] == ["prepare", "review", "quality", "slo"]
    assert verification.steps[3].min_reward is None
    assert verification.steps[3].verifier.network_mode.value == "no-network"
    for script in (run / "tasks").rglob("static_S-1/**/*.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)

    # An unsupported search verdict re-renders a searched pair as SLO.
    task = Task("searched_A-N", "Qwen3.6-35B-A3B", "A-N", "SOTA")
    render_task(task, config, image, run, inputs, assets)
    allocation = {"gpus": 1, "cpu_prefixes": ["AMD EPYC 9355"]}
    for phase in ("agent", "evaluation"):
        runtime_path = run / "tasks" / phase / task.task_id / "environment/runtime.json"
        runtime = json.loads(runtime_path.read_text())
        runtime["allocation_check"] = allocation
        runtime_path.write_text(json.dumps(runtime))
    render_task(replace(task, type="SLO"), config, image, run, inputs, assets)
    for phase in ("agent", "evaluation"):
        runtime_path = run / "tasks" / phase / task.task_id / "environment/runtime.json"
        runtime = json.loads(runtime_path.read_text())
        assert runtime["allocation_check"] == allocation
        assert runtime["payload"] == image.manifest["payload"]
    evaluation = run / "tasks/evaluation/searched_A-N"
    assert sorted(path.name for path in (evaluation / "steps").iterdir()) == [
        "prepare",
        "quality",
        "review",
        "slo",
    ]
    metadata = json.loads((evaluation / "environment/task.json").read_text())
    assert metadata["target"] == {
        "source": "nvidia/Qwen3.6-35B-A3B-NVFP4",
    }
    prompt = (run / "tasks/agent/searched_A-N/instruction.md").read_text()
    assert (
        "min(1, throughput / tier_throughput)" in prompt
        and "Official verification uses **N=" in prompt
    )
    oracle = run / "tasks/agent/searched_A-N/solution/solve.sh"
    assert "'baseline': \"freetoken\"" in oracle.read_text()
    compile(embedded_python(oracle), str(oracle), "exec")


def test_sandbox_payload_leaves_out_host_only_recording_files(tmp_path):
    root, _, _, _ = fixture(tmp_path)
    for name in ("tables/data.csv", "tables/request_fields.json", "eval/bench/sources.py"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("fixture\n")
    targets = select_image(root, "nvidia-cuda-x86", ()).manifest["payload"]["sources"]
    assert "tables/request_fields.json" in targets and "eval/__init__.py" in targets
    assert "tables/data.csv" not in targets and "eval/bench/sources.py" not in targets


def test_gguf_reference_does_not_change_canonical_candidate_or_oracle(tmp_path):
    from eval import run as runner

    root, run, inputs, assets = fixture(tmp_path)
    assets["baselines"]["ollama"] = "owner/model-GGUF"
    image = freeze_image(select_image(root, "nvidia-cuda-x86", tuple(assets["baselines"])), run)
    task = Task("fixed_A-N", "Qwen3.6-35B-A3B", "A-N", "SOTA")
    path = render_task(task, RunConfig(), image, run, inputs, assets)
    supplied = tmp_path / "references"
    supplied.mkdir()
    (supplied / f"{task.task_id}.json").write_text(
        json.dumps({"baseline": "ollama", "args": [], "env": {}})
    )
    (run / "prepared.json").write_text(
        json.dumps(
            {
                "schema": 8,
                "tasks": [{"task_id": task.task_id, "model": task.model}],
                "assets": {task.model: assets},
            }
        )
    )
    assert runner._coding_tasks(run, (path,), reference=supplied) == (path,)
    for phase in ("agent", "evaluation"):
        folder = run / "tasks" / phase / task.task_id
        metadata = json.loads((folder / "environment/task.json").read_text())
        runtime = json.loads((folder / "environment/runtime.json").read_text())
        assert metadata["target"] == runtime["target"] == {"source": assets["checkpoint"]}
        assert metadata["assets"]["baselines"]["ollama"] == "owner/model-GGUF"
    from eval.verify.run import reference_checkpoint

    assert (
        reference_checkpoint(metadata, {"baseline": "ollama"})
        == "/workspace/runtime/reference/weights/target"
    )
    assert reference_checkpoint(metadata, {"baseline": "freetoken"}) == "/models/target"
    oracle = (path / "solution/solve.sh").read_text()
    assert "'baseline': \"freetoken\"" in oracle
