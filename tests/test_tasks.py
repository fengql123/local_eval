"""Native three-task rendering: two common and two edge cases, without image builds."""

import json
import os
import shutil
import subprocess
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from harbor.models.task.config import TaskConfig
from harbor.models.task.task import Task as HarborTask

from eval.config import AgentConfig, EvaluationConfig, RunConfig, SearchConfig, Task
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
    installer.with_name("launch.sh").write_text(
        "#!/bin/sh\n# Test-only supplied launcher; never executed.\n"
    )
    shutil.copy(Path(__file__).parents[1] / "images/prepare.sh", root / "images/prepare.sh")
    shutil.copy(Path(__file__).parents[1] / ".dockerignore", root / ".dockerignore")
    (root / "eval").mkdir()
    (root / "eval/__init__.py").write_text('"""Test-only evaluator source."""\n')
    run = tmp_path / "runs" / "fixture-id"
    inputs = {}
    for key in ("public", "public_s", "private", "calibration", "quality"):
        directory = run / "inputs" / key
        directory.mkdir(parents=True)
        inputs[key] = str(directory)
    models = run / "models/C2"
    source = models / "source"
    source.mkdir(parents=True)
    baseline = models / "baselines/freetoken"
    baseline.mkdir(parents=True)
    assets = {
        "root": str(models),
        "source": {
            "source": "Qwen/Qwen3.6-35B-A3B",
            "path": str(source),
            "checkpoint": str(source),
        },
        "baselines": {
            "freetoken": {
                "path": str(baseline),
                "source": "example/model",
                "checkpoint": str(baseline),
            }
        },
    }
    machine = Machine("Linux", "x86_64", 8, 16 * 2**30, (GPU(2, "GPU-fixture", "fixture", 8192),))
    engines = ("freetoken", "ollama")
    other_installer = root / "images" / machine.category / "baselines" / engines[1] / "install.sh"
    other_installer.parent.mkdir(parents=True)
    other_installer.write_text("#!/bin/sh\n# Test-only supplied installer; never executed.\n")
    other_installer.with_name("launch.sh").write_text(
        "#!/bin/sh\n# Test-only supplied launcher; never executed.\n"
    )
    return root, machine, engines, run, inputs, assets


def _task_config(folder):
    return TaskConfig.model_validate(tomllib.loads((folder / "task.toml").read_text()))


def _compose(folder):
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(folder / "environment/docker-compose.yaml"),
            "config",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "MAIN_IMAGE_NAME": "hb__fixture"},
    )
    return json.loads(result.stdout)["services"]["main"]


def test_native_three_tasks_scope_agents_data_and_immutable_weight_mounts(tmp_path):
    root, machine, engines, run, inputs, assets = _fixture(tmp_path)
    image = select_image(root, machine, engines)
    coding = render_task(
        Task("example_S-1", "Qwen3.6-35B-A3B", "S-1", "SOTA"),
        RunConfig(
            agent=AgentConfig("codex", "fixture-model", budget_seconds=2700),
            search=SearchConfig("single", AgentConfig("claude-code", budget_seconds=1800)),
            evaluation=EvaluationConfig(agent=AgentConfig("claude-code", budget_seconds=900)),
        ),
        machine,
        image,
        run,
        inputs,
        assets,
    )
    search, verification = (run / phase / coding.name for phase in ("search", "verification"))
    assert coding.parent == run / "coding"
    search_config, coding_config, verification_config = map(
        _task_config, (search, coding, verification)
    )
    for folder, config in ((search, search_config), (coding, coding_config)):
        assert config.schema_version == "1.4" and not config.steps
        assert config.verifier.user == "root" and config.verifier.environment_mode == "shared"
        native = HarborTask(folder)
        assert not native.has_steps and native.paths.instruction_path.is_file()
        assert native.paths.test_path_for(config.environment.os).is_file()
        solution = folder / "solution/solve.sh"
        assert solution.stat().st_mode & 0o111
        subprocess.run(["bash", "-n", str(solution)], check=True)
    assert (search_config.agent.user, search_config.agent.timeout_sec) == ("search", 1800)
    assert (coding_config.agent.user, coding_config.agent.timeout_sec) == ("agent", 2700)
    assert search_config.environment.workdir == "/home/search"
    assert coding_config.environment.workdir == "/workspace"
    assert search_config.environment.docker_image is None
    assert coding_config.environment.docker_image == image.tag
    assert search_config.artifacts == ["/root/eval/reference"]
    assert coding_config.artifacts == ["/workspace/submission"]
    assert len(coding_config.verifier.collect) == len(search_config.verifier.collect) == 1
    assert search_config.verifier.collect[0].command.endswith("--stage collect")
    assert search_config.verifier.collect[0].user == "root"
    assert coding_config.verifier.collect[0].command.endswith("--collect-submission")
    assert coding_config.verifier.collect[0].user == "root"
    prepare, review, measure = verification_config.steps
    assert [step.name for step in verification_config.steps] == ["prepare", "review", "measure"]
    assert [step.verifier.user for step in verification_config.steps] == ["root", "judge", "root"]
    assert all(step.verifier.environment_mode == "shared" for step in verification_config.steps)
    assert prepare.min_reward == review.min_reward == 1 and measure.min_reward is None
    assert review.verifier.timeout_sec == 1020
    assert verification_config.multi_step_reward_strategy == "final"
    assert verification_config.environment.docker_image == image.tag
    assert HarborTask(verification).has_steps
    assert (verification / "steps/prepare/workdir").is_dir()
    assert not (verification / "steps/prepare/workdir/submission").exists()
    services = {
        phase: _compose(run / phase / coding.name) for phase in ("search", "coding", "verification")
    }
    assert all(service["image"] == image.tag for service in services.values())
    assert services["search"]["build"]["context"] == str(root)
    assert services["search"]["build"]["tags"] == ["hb__fixture"]
    assert "build" not in services["coding"] and "build" not in services["verification"]
    assert all(service["deploy"] == services["search"]["deploy"] for service in services.values())
    assert services["verification"]["security_opt"] == ["no-new-privileges:true"]
    assert "security_opt" not in services["search"] and "security_opt" not in services["coding"]
    mounted = {
        phase: {item["target"]: item for item in service["volumes"]}
        for phase, service in services.items()
    }
    assert set(mounted["search"]) == {
        "/models/source",
        "/models/baselines/freetoken",
        "/home/search/data/S",
        "/root/eval/task.json",
        "/root/eval/reference",
        "/opt/eval/start.sh",
    }
    assert not mounted["search"]["/root/eval/reference"].get("read_only", False)
    assert all(
        item["read_only"]
        for target, item in mounted["search"].items()
        if target != "/root/eval/reference"
    )
    assert set(mounted["coding"]) == {
        "/models/source",
        "/models/reference",
        "/workspace/data/S",
        "/workspace/reference.json",
        "/root/eval/task.json",
        "/root/eval/reference",
        "/workspace/run.md",
        "/opt/eval/start.sh",
    }
    assert set(mounted["verification"]) == {
        "/models/source",
        "/models/reference",
        "/root/eval/private",
        "/root/eval/quality",
        "/root/eval/task.json",
        "/root/eval/reference",
        "/opt/eval/start.sh",
    }
    assert all(
        item["read_only"]
        for phase in ("coding", "verification")
        for item in mounted[phase].values()
    )
    selected_weights = coding / "reference/weights"
    assert not selected_weights.exists()
    assert mounted["coding"]["/models/reference"]["source"] == str(selected_weights)
    assert mounted["verification"]["/models/reference"]["source"] == str(selected_weights)
    for folder in (search, coding):
        assert (folder / "reference/metadata").stat().st_mode & 0o777 == 0o700
    assert json.loads((coding / "reference/metadata/public.json").read_text()) == {}
    metadata = json.loads((coding / "environment/task.json").read_text())
    assert metadata["agent"]["model"] == "fixture-model"
    assert metadata["assets"]["source"]["checkpoint"] == "/models/source"
    assert (
        metadata["assets"]["baselines"]["freetoken"]["checkpoint"] == "/models/baselines/freetoken"
    )
    assert "storage" not in metadata and "user" not in metadata["candidate"]
    instructions = (search / "instruction.md").read_text()
    assert "selection.json" in instructions and '"args": []' in instructions
    assert "read-only" in instructions and "new coding" in instructions
    assert "Path('selection.json')" in (search / "solution/solve.sh").read_text()
    assert "result['baseline']" in (search / "solution/solve.sh").read_text()


def test_shared_image_handles_source_equal_and_native_assets_with_review_only_credentials(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-api-secret")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fixture-oauth-secret")
    monkeypatch.setenv("CODEX_AUTH_JSON", '{"test_only":"fixture-codex-secret"}')
    monkeypatch.setenv("REWARDKIT_FORCE_SUBSCRIPTION", "1")
    root, machine, engines, run, inputs, assets = _fixture(tmp_path)
    assets["baselines"]["freetoken"] = dict(assets["source"])
    assets["baselines"]["ollama"] = {
        "source": "qwen:fixture",
        "path": None,
        "checkpoint": "qwen:fixture",
    }
    image = select_image(root, machine, engines)
    expected = (
        (AgentConfig("claude-code", auth="api_key"), {"ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"}),
        (AgentConfig("claude-code"), {"CLAUDE_CODE_OAUTH_TOKEN": "${CLAUDE_CODE_OAUTH_TOKEN}"}),
        (
            AgentConfig("codex"),
            {
                "CODEX_AUTH_JSON": "${CODEX_AUTH_JSON}",
                "REWARDKIT_FORCE_SUBSCRIPTION": "${REWARDKIT_FORCE_SUBSCRIPTION}",
            },
        ),
    )
    role_env = {
        "HOME": "/home/judge",
        "USER": "judge",
        "LOGNAME": "judge",
        "TMPDIR": "/home/judge/tmp",
        "XDG_CACHE_HOME": "/home/judge/.cache",
    }
    for index, (reviewer, credentials) in enumerate(expected):
        coding = render_task(
            Task(f"example_{index}", "Qwen3.6-35B-A3B", "A-N", "SOTA"),
            RunConfig(evaluation=EvaluationConfig(agent=reviewer)),
            replace(machine, gpus=()),
            image,
            run,
            inputs,
            assets,
        )
        verification = run / "verification" / coding.name
        task = _task_config(verification)
        assert task.steps[1].verifier.env == {**role_env, **credentials}
        assert task.steps[0].verifier.env == task.steps[2].verifier.env == {}
        metadata = json.loads((coding / "environment/task.json").read_text())
        assert metadata["assets"]["baselines"]["freetoken"]["checkpoint"] == "/models/source"
        assert metadata["assets"]["baselines"]["ollama"]["checkpoint"] == "qwen:fixture"
        assert metadata["evaluation"]["agent"]["model"] is None
        for phase in ("search", "coding", "verification"):
            folder = run / phase / coding.name
            service = yaml.safe_load((folder / "environment/docker-compose.yaml").read_text())[
                "services"
            ]["main"]
            assert (
                service["image"] == image.tag
                and "deploy" not in service
                and "environment" not in service
            )
            if phase == "search":
                assert not any(
                    item["target"].startswith("/models/baselines") for item in service["volumes"]
                )
            elif phase == "coding":
                assert {"/workspace/data/A", "/workspace/data/S"} <= {
                    item["target"] for item in service["volumes"]
                }
            if phase != "verification":
                assert _task_config(folder).verifier.env == {}
            for file in folder.rglob("*"):
                if file.is_file():
                    assert not any(
                        secret in file.read_text()
                        for secret in (
                            "fixture-api-secret",
                            "fixture-oauth-secret",
                            "fixture-codex-secret",
                        )
                    )
    assert image.build_args == {"BASELINES": "freetoken ollama"}
    (root / ".env").write_text("PRIVATE_SECRET=excluded\n")
    (root / "runs").mkdir()
    (root / "runs/private.json").write_text('{"answer":"private"}')
    assert select_image(root, machine, engines).identity == image.identity
    assert "images/prepare.sh" in image.manifest["sources"]
    (root / "images/prepare.sh").write_text("#!/bin/bash\n# Changed trusted preparation.\n")
    assert select_image(root, machine, engines).identity != image.identity


def test_missing_image_assets_fail_before_rendering_and_login_files_are_never_read(
    tmp_path, monkeypatch
):
    root, machine, engines, run, inputs, assets = _fixture(tmp_path)
    for relative in (
        machine.dockerfile,
        Path("images/prepare.sh"),
        Path("images") / machine.category / "baselines/freetoken/install.sh",
        Path("images") / machine.category / "baselines/freetoken/launch.sh",
    ):
        path = root / relative
        contents = path.read_bytes()
        path.unlink()
        with pytest.raises(FileNotFoundError, match=str(relative)):
            select_image(root, machine, engines)
        path.write_bytes(contents)
    assert not any((run / phase).exists() for phase in ("search", "coding", "verification"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "missing-login"))
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    coding = render_task(
        Task("missing-login_S-1", "Qwen3.6-35B-A3B", "S-1", "SOTA"),
        RunConfig(),
        machine,
        select_image(root, machine, engines),
        run,
        inputs,
        assets,
    )
    review = _task_config(run / "verification" / coding.name).steps[1]
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in review.verifier.env
    assert not (tmp_path / "missing-login").exists() and not (run / "judge-auth.json").exists()


def test_readiness_and_all_native_hooks_use_the_protected_interpreter(tmp_path):
    root, machine, engines, run, inputs, assets = _fixture(tmp_path)
    coding = render_task(
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
    events, ready = tmp_path / "events", tmp_path / "ready"
    (commands / "python").write_text("#!/bin/sh\nexit 99\n")
    evaluator = commands / "evaluator-python"
    evaluator.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$EVENTS"\n')
    (commands / "sleep").write_text('#!/bin/sh\nprintf "keepalive\\n" >> "$EVENTS"\n')
    (commands / "touch").write_text(
        '#!/bin/sh\n[ "${EXIT_CODE:-0}" = 0 ] || exit "$EXIT_CODE"\nexec /usr/bin/touch "$@"\n'
    )
    for command in commands.iterdir():
        command.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{commands}:{os.environ['PATH']}",
        "EVENTS": str(events),
        "EXIT_CODE": "17",
    }
    startup = (coding / "environment/start.sh").read_text().replace("/run/eval-ready", str(ready))
    assert "python" not in startup and "eval.search" not in startup
    ready.touch()
    assert subprocess.run(["bash", "-c", startup], env=env).returncode == 17
    assert not ready.exists()
    env["EXIT_CODE"] = "0"
    subprocess.run(["bash", "-c", startup], env=env, check=True)
    verification = run / "verification" / coding.name
    hooks = [run / "search" / coding.name / "tests/test.sh", coding / "tests/test.sh"]
    hooks.extend(
        verification / "steps" / stage / "tests/test.sh"
        for stage in ("prepare", "review", "measure")
    )
    for hook in hooks:
        subprocess.run(["bash", "-n", str(hook)], check=True)
        script = hook.read_text()
        assert "exec /opt/eval/.venv/bin/python -I -m " in script
        subprocess.run(
            ["bash", "-c", script.replace("/opt/eval/.venv/bin/python", str(evaluator))],
            env=env,
            check=True,
        )
    assert events.read_text().splitlines() == [
        "keepalive",
        "-I -m eval.search.run --task /root/eval/task.json --stage gate",
        "-I -m eval.verify.run --stage handoff --task /root/eval/task.json",
        "-I -m eval.verify.run --stage prepare --task /root/eval/task.json",
        "-I -m eval.verify.run --stage review",
        "-I -m eval.verify.run --stage measure --task /root/eval/task.json",
    ]
