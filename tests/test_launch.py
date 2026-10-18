"""Launch inputs: two common compositions and two rejected override cases."""

import asyncio

import pytest

from eval.bench.serve import Server
from eval.launch import ENDPOINT, Launch


def test_template_receives_mapped_weights_without_rewriting_the_script(tmp_path):
    template = tmp_path / "launch.sh"
    source = (
        ': "${CHECKPOINT:?checkpoint required}"\nexec engine serve "$CHECKPOINT" --port 8000 "$@"\n'
    )
    template.write_text(source)
    params = {"args": ["--workers", "2", "--label=a b;$value"], "env": {"TUNING": "a b"}}
    for checkpoint in ("/models/source with spaces", "qwen3.6:35b-a3b-mtp-q4_K_M"):
        launch = Launch(template, checkpoint, tuple(params["args"]), dict(params["env"]))
        assert launch.prepare() == (
            ["/bin/bash", str(template), *params["args"]],
            {"TUNING": "a b", "CHECKPOINT": checkpoint},
        )
        assert template.read_text() == source
    assert params == {"args": ["--workers", "2", "--label=a b;$value"], "env": {"TUNING": "a b"}}


def test_candidate_entrypoint_receives_only_the_fixed_checkpoint(tmp_path):
    script = tmp_path / "server.launch"
    launch = Launch(script, "/models/target with spaces")
    assert launch.prepare() == ([str(script), "/models/target with spaces"], {})
    assert Server(launch).endpoint == ENDPOINT
    for inputs in (
        {},
        {"checkpoint": "/models/target", "args": ("--tune",)},
        {"checkpoint": "/models/target", "env": {"MODEL": "other"}},
    ):
        with pytest.raises(ValueError, match="one fixed checkpoint"):
            Launch(script, **inputs).prepare()


def test_fixed_option_overrides_fail_before_starting_or_opening_logs(tmp_path, monkeypatch):
    template = tmp_path / "launch.sh"
    log = tmp_path / "server.log"

    async def forbidden_spawn(*args, **kwargs):
        pytest.fail("fixed option overrides must not start a process")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_spawn)

    async def scenario():
        for arg in ("--model", "--model=other", "--port=8001", "-m=other"):
            launch = Launch(template, "/models/source", (arg,), fixed_options=frozenset({"-m"}))
            with pytest.raises(ValueError, match=f"{arg.split('=', 1)[0]} is fixed"):
                async with Server(launch, log=log):
                    pytest.fail("invalid parameters must not enter the server context")
            assert not log.exists()

    asyncio.run(scenario())


def test_reserved_environment_is_rejected_without_interpreting_shell_source(tmp_path):
    script = tmp_path / "launch.sh"
    script.write_text('exec engine serve "$CHECKPOINT" --port 8000\n')
    launch = Launch(script, fixed_env=frozenset({"CUDA_HOME"}))
    for name in ("CHECKPOINT", "PORT", "BASH_ENV", "ENV", "CUDA_HOME"):
        launch.env[name] = "override"
        with pytest.raises(ValueError, match=f"environment {name} is fixed"):
            launch.prepare()
        launch.env.clear()
    assert launch.prepare() == (["/bin/bash", str(script)], {})
