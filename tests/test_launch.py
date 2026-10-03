"""Launch contract: two common compositions and two fixed-field override cases."""

import asyncio

import pytest

from eval.bench.serve import Server
from eval.launch import Launch


def test_fixed_inputs_and_tuning_compose_without_a_script_template(tmp_path):
    template = tmp_path / "launch.sh"
    params = {"args": ["--cache-size", "32", "--label=a b;$value"], "env": {"TUNING": "a b"}}
    launch = Launch(
        template,
        "/models/source with spaces",
        8000,
        tuple(params["args"]),
        dict(params["env"]),
    )
    command, environment = launch.prepare()
    assert command == ["bash", str(template), *params["args"]]
    assert environment == {
        "TUNING": "a b",
        "CHECKPOINT": "/models/source with spaces",
        "PORT": "8000",
    }
    assert params == {
        "args": ["--cache-size", "32", "--label=a b;$value"],
        "env": {"TUNING": "a b"},
    }
    assert not template.exists()


def test_no_tuning_leaves_backend_defaults_to_the_later_script(tmp_path):
    template = tmp_path / "launch.sh"
    launch = Launch(template, "qwen3.6:35b-a3b-mtp-q4_K_M", 8123)
    assert launch.prepare() == (
        ["bash", str(template)],
        {"CHECKPOINT": "qwen3.6:35b-a3b-mtp-q4_K_M", "PORT": "8123"},
    )
    assert not template.exists()


def test_fixed_option_overrides_fail_before_starting_or_opening_logs(tmp_path, monkeypatch):
    template = tmp_path / "launch.sh"
    log = tmp_path / "server.log"

    async def forbidden_spawn(*args, **kwargs):
        pytest.fail("fixed option overrides must not start a process")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_spawn)

    async def scenario():
        for arg in ("--model", "--model=other", "-m=other"):
            launch = Launch(
                template, "/models/source", 8000, (arg,), fixed_options=frozenset({"-m"})
            )
            with pytest.raises(ValueError, match=f"{arg.split('=', 1)[0]} is fixed"):
                async with Server(launch, "http://127.0.0.1:8000", log=log):
                    pytest.fail("invalid parameters must not enter the server context")
            assert not log.exists()

    asyncio.run(scenario())


def test_reserved_and_script_owned_environment_cannot_override_fixed_inputs(tmp_path):
    launch = Launch(
        tmp_path / "launch.sh", "/models/source", 8000, fixed_env=frozenset({"CUDA_HOME"})
    )
    for name in ("CHECKPOINT", "PORT", "CUDA_HOME"):
        launch.env[name] = "override"
        with pytest.raises(ValueError, match=f"environment {name} is fixed"):
            launch.prepare()
        launch.env.clear()
    assert launch.prepare()[1] == {"CHECKPOINT": "/models/source", "PORT": "8000"}
