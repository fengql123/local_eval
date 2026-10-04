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
        assert launch.needs_checkpoint
        assert launch.prepare() == (
            ["/bin/bash", str(template), *params["args"]],
            {"TUNING": "a b", "CHECKPOINT": checkpoint},
        )
        assert template.read_text() == source
    assert params == {"args": ["--workers", "2", "--label=a b;$value"], "env": {"TUNING": "a b"}}


def test_completed_scripts_keep_their_weights_and_optimized_defaults(tmp_path):
    script = tmp_path / "launch.sh"
    for source in (
        '# documentation: CHECKPOINT=$CHECKPOINT\nexec engine serve "/weights/optimized" --port 8000 --workers 4\n',
        'export CHECKPOINT="/weights/optimized with spaces"\nexec engine serve "${CHECKPOINT}" --port 8000 --workers 4\n',
        'exec engine serve "${CHECKPOINT:-/weights/optimized}" --port 8000 --workers 4\n',
    ):
        script.write_text(source)
        launch = Launch(script, "/ignored/source")
        assert not launch.needs_checkpoint
        assert launch.prepare() == (["/bin/bash", str(script)], {})
        assert script.read_text() == source
        assert Server(launch).endpoint == ENDPOINT


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


def test_reserved_environment_and_missing_template_checkpoint_are_rejected(tmp_path):
    script = tmp_path / "launch.sh"
    script.write_text('exec engine serve "$CHECKPOINT" --port 8000\n')
    launch = Launch(script, fixed_env=frozenset({"CUDA_HOME"}))
    for name in ("CHECKPOINT", "PORT", "BASH_ENV", "ENV", "CUDA_HOME"):
        launch.env[name] = "override"
        with pytest.raises(ValueError, match=f"environment {name} is fixed"):
            launch.prepare()
        launch.env.clear()
    with pytest.raises(ValueError, match="requires a mapped CHECKPOINT"):
        launch.prepare()
    script.write_text(
        'CHECKPOINT="" # still unfilled\nexec engine serve "$CHECKPOINT" --port 8000\n'
    )
    with pytest.raises(ValueError, match="requires a mapped CHECKPOINT"):
        launch.prepare()
