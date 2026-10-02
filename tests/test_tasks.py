"""Phase 2 rendering: two common cases and two edge cases, without image builds."""

import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml
from harbor.models.task.config import TaskConfig
from harbor.models.task.task import Task as HarborTask

from eval.config import AgentConfig, RunConfig, SearchConfig, Task
from eval.device import GPU, Machine
from eval.tasks import render_task, select_image


def _fixture(tmp_path):
    """Small authored-source stand-ins are only used for selector/renderer tests."""
    root = tmp_path / "checkout"
    dockerfile = root / "images/nvidia-cuda-x86/Dockerfile"
    installer = dockerfile.parent / "baselines/freetoken/install.sh"
    installer.parent.mkdir(parents=True)
    dockerfile.write_text("# Test-only supplied image; never built.\nFROM scratch\n")
    installer.write_text("#!/bin/sh\n# Test-only supplied installer; never executed.\n")
    shutil.copy(Path(__file__).parents[1] / ".dockerignore", root / ".dockerignore")
    (root / "eval").mkdir()
    (root / "eval/__init__.py").write_text('"""Test-only evaluator source."""\n')
    run = tmp_path / "runs" / "fixture-id"
    inputs = {}
    for key in ("public", "private", "calibration", "quality"):
        directory = run / "inputs" / key
        directory.mkdir(parents=True)
        inputs[key] = str(directory)
    models = run / "models/C2"
    source = models / "source"
    source.mkdir(parents=True)
    assets = {
        "root": str(models),
        "source": {
            "source": "Qwen/Qwen3.6-35B-A3B",
            "path": str(source),
            "checkpoint": str(source),
        },
        "baselines": {
            "freetoken": {
                "path": None,
                "source": "example/model",
                "checkpoint": "example/model",
            }
        },
    }
    machine = Machine("Linux", "x86_64", 8, 16 * 2**30, (GPU(2, "GPU-fixture", "fixture", 8192),))
    engines = ("freetoken", "ollama")
    other_installer = root / "images" / machine.category / "baselines" / engines[1] / "install.sh"
    other_installer.parent.mkdir(parents=True)
    other_installer.write_text("#!/bin/sh\n# Test-only supplied installer; never executed.\n")
    return root, machine, engines, run, inputs, assets


def test_task_parses_with_harbor_and_compose_and_protects_evaluator_inputs(tmp_path):
    root, machine, engines, run, inputs, assets = _fixture(tmp_path)
    image = select_image(root, machine, engines)
    folder = render_task(
        Task("example_S-1", "Qwen3.6-35B-A3B", "S-1", "SOTA"),
        RunConfig(
            agent=AgentConfig(
                "codex", "fixture-model", env=("FIXTURE_API_KEY",), budget_seconds=2700
            ),
            search=SearchConfig(
                "single", AgentConfig("claude-code", env=("SEARCH_API_KEY",), budget_seconds=1800)
            ),
        ),
        machine,
        image,
        run,
        inputs,
        assets,
    )
    task = TaskConfig.model_validate(tomllib.loads((folder / "task.toml").read_text()))
    assert task.schema_version == "1.4"
    assert task.verifier.user == "root" and task.verifier.environment_mode == "shared"
    assert task.multi_step_reward_strategy == "final"
    search, coding = task.steps
    assert (search.name, search.agent.user, search.agent.timeout_sec) == ("search", "search", 1800)
    assert (coding.name, coding.agent.user, coding.agent.timeout_sec) == ("coding", "agent", 2700)
    assert search.min_reward == 1 and search.verifier.timeout_sec == 600
    assert all(
        step.verifier.user == "root" and step.verifier.environment_mode == "shared"
        for step in task.steps
    )
    harbor_task = HarborTask(folder)
    assert harbor_task.has_steps and "Tune the reference" in harbor_task.step_instruction("search")
    assert "Implement serving" in harbor_task.step_instruction("coding")
    assert not (folder / "instruction.md").exists() and not (folder / "tests").exists()
    assert task.environment.healthcheck.command == "test -f /run/eval-ready"
    compose_path = folder / "environment/docker-compose.yaml"
    result = subprocess.run(
        ["docker", "compose", "-f", str(compose_path), "config", "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "MAIN_IMAGE_NAME": "hb__fixture"},
    )
    compose = json.loads(result.stdout)
    assert list(compose["services"]) == ["main"]
    service = compose["services"]["main"]
    assert service["image"] == image.tag
    assert service["build"]["context"] == str(root)
    assert service["build"]["dockerfile"] == str(machine.dockerfile)
    assert service["build"]["tags"] == ["hb__fixture"]
    assert service["command"] == ["bash", "/opt/eval/start.sh"]
    assert service["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] == [
        "GPU-fixture"
    ]
    targets = {mount["target"] for mount in service["volumes"]}
    assert {
        "/root/eval/private",
        "/home/search/data/calibration",
        "/root/eval/quality",
        "/root/eval/task.json",
        "/workspace/data/public",
        "/models",
    } <= targets
    assert all(mount["read_only"] for mount in service["volumes"])
    assert not (folder / "environment/Dockerfile").exists()
    metadata = json.loads((folder / "environment/task.json").read_text())
    assert metadata["agent"]["model"] == "fixture-model"
    assert metadata["agent"]["env"] == ["FIXTURE_API_KEY"]
    assert metadata["search"]["agent"]["model"] is None
    assert metadata["search"]["agent"]["env"] == ["SEARCH_API_KEY"]
    assert metadata["search"]["calibration_pool"] == "S"
    assert metadata["inputs"]["calibration"] == "/home/search/data/calibration"
    assert metadata["candidate"]["user"] == "agent"
    assert set(metadata["assets"]) == {"root", "source", "baselines"}
    assert metadata["assets"]["source"]["source"] == "Qwen/Qwen3.6-35B-A3B"
    assert metadata["assets"]["source"]["path"] == "/models/source"
    assert metadata["assets"]["source"]["checkpoint"] == "/models/source"
    assert metadata["assets"]["baselines"]["freetoken"]["path"] is None
    assert metadata["assets"]["baselines"]["freetoken"]["checkpoint"] == "example/model"
    coding_instructions = (folder / "steps/coding/instruction.md").read_text()
    search_instructions = (folder / "steps/search/instruction.md").read_text()
    assert "private" not in coding_instructions
    assert "/home/search/data/calibration" in search_instructions
    assert "single" in search_instructions
    assert "/models/source" in search_instructions
    assert "example/model" in search_instructions and "S request policy" in search_instructions
    assert "quantization" in search_instructions
    assert metadata["baselines"] == ["freetoken"]
    run_instructions = (folder / "environment/run.md").read_text()
    assert "Qwen/Qwen3.6-35B-A3B" in run_instructions
    assert "/models/source" in run_instructions
    assert "example/model" in run_instructions and "runtime to load" in run_instructions
    assert not (run / "models/C2/baselines").exists()
    assert "quantize" in run_instructions


def test_tasks_share_image_identity_independent_of_model_workload_and_host_secrets(tmp_path):
    root, machine, engines, run, inputs, assets = _fixture(tmp_path)
    image = select_image(root, machine, engines)
    first = render_task(
        Task("first_S-1", "Qwen3.6-35B-A3B", "S-1", "SOTA"),
        RunConfig(),
        machine,
        image,
        run,
        inputs,
        assets,
    )
    assets["baselines"]["ollama"] = {
        "source": "qwen:fixture",
        "path": None,
        "checkpoint": "qwen:fixture",
    }
    second = render_task(
        Task("second_A-N", "Qwen3.8-27B", "A-N", "SOTA"),
        RunConfig(),
        machine,
        image,
        run,
        inputs,
        assets,
    )
    assert (
        yaml.safe_load((first / "environment/docker-compose.yaml").read_text())["services"]["main"][
            "image"
        ]
        == yaml.safe_load((second / "environment/docker-compose.yaml").read_text())["services"][
            "main"
        ]["image"]
    )
    assert (
        json.loads((first / "environment/task.json").read_text())["workload"]
        != json.loads((second / "environment/task.json").read_text())["workload"]
    )
    assert image.build_args == {"BASELINES": "freetoken ollama"}
    assert json.loads((first / "environment/task.json").read_text())["baselines"] == ["freetoken"]
    assert json.loads((second / "environment/task.json").read_text())["baselines"] == [
        "freetoken",
        "ollama",
    ]
    second_service = yaml.safe_load((second / "environment/docker-compose.yaml").read_text())[
        "services"
    ]["main"]
    assert all(
        mount["type"] == "bind" and mount["read_only"] for mount in second_service["volumes"]
    )
    assert not (Path(assets["root"]) / "ollama").exists()
    assert (
        json.loads((second / "environment/task.json").read_text())["assets"]["baselines"]["ollama"][
            "checkpoint"
        ]
        == "qwen:fixture"
    )
    (root / ".env").write_text("TEST_ONLY_SECRET=must-never-enter-build-context\n")
    (root / "runs").mkdir()
    (root / "runs/private.json").write_text('{"answer": "test-only private value"}')
    assert select_image(root, machine, engines).identity == image.identity
    assert not any(path.startswith("runs/") or path == ".env" for path in image.manifest["sources"])
    (root / "images" / machine.category / "baselines" / engines[0] / "install.sh").write_text(
        "#!/bin/sh\n# Changed supplied source.\n"
    )
    assert select_image(root, machine, engines).identity != image.identity


def test_missing_supplied_image_assets_fail_before_a_task_or_build_exists(tmp_path):
    root, machine, engines, run, inputs, assets = _fixture(tmp_path)
    installer = root / "images" / machine.category / "baselines" / engines[0] / "install.sh"
    installer.unlink()
    with pytest.raises(FileNotFoundError, match="baselines/freetoken/install.sh"):
        select_image(root, machine, engines)
    installer.write_text("#!/bin/sh\n")
    (root / machine.dockerfile).unlink()
    with pytest.raises(FileNotFoundError, match="nvidia-cuda-x86/Dockerfile"):
        select_image(root, machine, engines)
    assert not (run / "tasks").exists()


def test_startup_is_readiness_only_and_step_hooks_use_the_isolated_evaluator(tmp_path):
    root, machine, engines, run, inputs, assets = _fixture(tmp_path)
    folder = render_task(
        Task("fixture_S-1", "Qwen3.6-35B-A3B", "S-1", "SOTA"),
        RunConfig(),
        machine,
        select_image(root, machine, engines),
        run,
        inputs,
        assets,
    )
    commands = tmp_path / "commands"
    commands.mkdir()
    events = tmp_path / "events"
    ready = tmp_path / "ready"
    shadow = commands / "python"
    shadow.write_text("#!/bin/sh\nexit 99\n")
    shadow.chmod(0o755)
    evaluator = tmp_path / "evaluator-python"
    evaluator.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$EVENTS"\n')
    evaluator.chmod(0o755)
    sleep = commands / "sleep"
    sleep.write_text('#!/bin/sh\nprintf "keepalive\\n" >> "$EVENTS"\n')
    sleep.chmod(0o755)
    touch = commands / "touch"
    touch.write_text(
        '#!/bin/sh\n[ "${EXIT_CODE:-0}" = 0 ] || exit "$EXIT_CODE"\nexec /usr/bin/touch "$@"\n'
    )
    touch.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{commands}:{os.environ['PATH']}",
        "EVENTS": str(events),
        "EXIT_CODE": "17",
    }
    # Replace only the container's fixed filesystem locations in this CPU test.
    startup_path = folder / "environment/start.sh"
    startup = startup_path.read_text().replace("/run/eval-ready", str(ready))
    assert "python" not in startup and "eval.search" not in startup
    ready.touch()
    failed = subprocess.run(["bash", "-c", startup], env=env, check=False)
    assert failed.returncode == 17 and not ready.exists()
    env["EXIT_CODE"] = "0"
    subprocess.run(["bash", "-c", startup], env=env, check=True)
    assert ready.exists()
    assert events.read_text().splitlines()[-1] == "keepalive"
    for phase in ("search", "coding"):
        hook = folder / "steps" / phase / "tests/test.sh"
        subprocess.run(["bash", "-n", str(hook)], check=True)
        script = hook.read_text()
        assert "exec /opt/eval/.venv/bin/python -I -m " in script
        subprocess.run(
            ["bash", "-c", script.replace("/opt/eval/.venv/bin/python", str(evaluator))],
            env=env,
            check=True,
        )
    subprocess.run(["bash", "-n", str(startup_path)], check=True)
    assert events.read_text().splitlines()[-2:] == [
        "-I -m eval.search.run --task /root/eval/task.json --finalize-reference",
        "-I -m eval.verify.run --task /root/eval/task.json",
    ]
