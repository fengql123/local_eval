"""Two completed/gated evaluations and two failed timed server evaluations."""

import asyncio
import json
import os
import subprocess
import sys
import tomllib
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from http_server import event, headers, server

from eval.bench.serve import Server, StartupError
from eval.launch import ENDPOINT
from eval.request_policy import request_fields
from eval.verify import run as verifier
from eval.verify.run import Backend, _backend, evaluate, rewards

TASK = {"model": "C2", "workload": "S-1"}
PREPARATION_TASK = {
    **TASK,
    "assets": {"source": {"checkpoint": "/models/source"}},
    "evaluation": {
        "install_timeout_seconds": 5,
        "agent": {"name": "claude-code", "auth": "login", "budget_seconds": 30, "kwargs": {}},
    },
}


def rows():
    return [
        {
            "id": "quality-letter",
            "model": "C2",
            "evaluation": json.dumps({"method": "letter", "answer": "B"}),
            "requests": [
                {
                    "id": "letter-request",
                    "messages": json.dumps(
                        [{"role": "user", "content": "Which letter follows A? Answer one letter."}]
                    ),
                    "tools": None,
                    "out": 8,
                    "parents": [],
                    "tool": 0,
                }
            ],
        }
    ]


def submission(tmp_path):
    folder = tmp_path / "submission"
    folder.mkdir()
    for name in ("install.sh", "launch.sh"):
        (folder / name).write_text("#!/bin/bash\nexit 0\n")
    return folder


PREPARED = {"ok": True, "stage": "prepare", "reason": "submission prepared"}
REVIEWED = {"ok": True}


async def reply(writer, *, valid=True):
    await headers(writer)
    writer.write(event({"content": "B"}))
    await writer.drain()
    await asyncio.sleep(0.002)
    writer.write(event(finish="stop"))
    if valid:
        writer.write(event(count=2))
    writer.write(b"data: [DONE]\n\n")
    await writer.drain()


def backend(name, endpoint, opened, closed):
    class TrackedServer(Server):
        async def _stop(self):
            await super()._stop()
            closed.append((name, self.stage))

    def factory(stage):
        managed = TrackedServer(None, endpoint, health_timeout=2)
        managed.stage = stage
        opened.append((name, stage, managed))
        return managed

    return Backend(name, request_fields("C2"), factory)


def test_passing_gate_uses_root_preparation_and_fresh_servers(tmp_path, monkeypatch):
    reference = {"throughput": 10, "ttft_mean": 2}
    assert rewards(reference, {"throughput": 20, "ttft_mean": 1}) == {
        "throughput": 50,
        "ttft": 50,
    }
    assert rewards(reference, {"throughput": 5, "ttft_mean": 4}) == {
        "throughput": 0,
        "ttft": 0,
    }
    source, out = submission(tmp_path), tmp_path / "results"
    opened, closed, operations = [], [], []
    monkeypatch.setattr(verifier, "RUBRIC", tmp_path / "rubric")
    monkeypatch.setattr(verifier, "REVIEW", tmp_path / "review")
    monkeypatch.setattr(verifier, "STATE", tmp_path / "private-state")
    monkeypatch.setattr(verifier, "VERIFIER", out)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only-reviewer-secret")
    monkeypatch.setenv("UNRELATED_PROVIDER_KEY", "test-only-coding-secret")
    # Preparation inherits the verifier UID; the real Harbor test asserts UID 0.
    environment = verifier.runtime_environment()
    assert environment["HOME"] == "/root"
    assert environment["USER"] == environment["LOGNAME"] == "root"
    assert "ANTHROPIC_API_KEY" not in environment and "UNRELATED_PROVIDER_KEY" not in environment
    assert asyncio.run(verifier.convert(source, "/models/source", tmp_path / "optional.log")) == {
        "ok": True,
        "skipped": True,
    }
    (source / "convert_weight.sh").write_text(
        '#!/bin/bash\nset -eu\nmkdir -p "$WEIGHTS" "$RUNTIME"\n'
        'echo "convert:$(id -u):$CHECKPOINT" > "$RUNTIME/order.txt"\n'
        'echo converted > "$WEIGHTS/checkpoint"\n'
    )
    (source / "install.sh").write_text(
        '#!/bin/bash\nset -eu\ntest -f "$WEIGHTS/checkpoint"\n'
        'echo "install:$(id -u):$CHECKPOINT" >> "$RUNTIME/order.txt"\n'
    )

    process = verifier._process

    async def image_preparation(command, **settings):
        if command[1] != "/opt/eval/images/prepare.sh":
            return await process(command, **settings)
        operations.append("image inputs")
        assert command[2:] == [str(source), str(verifier.RUBRIC), str(verifier.REVIEW)]
        assert "user" not in settings and "group" not in settings
        verifier.REVIEW.mkdir()
        return 0

    monkeypatch.setattr(verifier, "_process", image_preparation)
    task_path = tmp_path / "task.json"
    task_path.write_text(
        json.dumps({**PREPARATION_TASK, "candidate": {"launch": str(source / "launch.sh")}})
    )
    assert verifier.main(["--stage", "prepare", "--task", str(task_path)]) == 0
    prepared = json.loads((verifier.STATE / "prepare.json").read_text())
    assert prepared["ok"] and prepared["access"] == {"exit_code": 0}
    assert json.loads((out / "report.json").read_text()) == prepared
    assert json.loads((out / "reward.json").read_text()) == {"reward": 1}

    async def scenario():
        async def respond(reader, writer, body):
            await reply(writer)

        async with server(respond) as (reference_endpoint, reference_requests):
            async with server(respond) as (candidate_endpoint, candidate_requests):
                report = await evaluate(
                    TASK,
                    rows(),
                    rows(),
                    backend("reference", reference_endpoint, opened, closed),
                    backend("candidate", candidate_endpoint, opened, closed),
                    out=out,
                    prepared=prepared,
                    reviewed=REVIEWED,
                )
                assert len(reference_requests) == len(candidate_requests) == 3
                assert reference_requests == candidate_requests
        return report

    report = asyncio.run(scenario())
    assert report["outcome"] == "passed" and report["pre_run"]["ok"] is True
    assert report["bench"]["skipped"] is False
    assert operations == ["image inputs"]
    assert report["pre_run"]["convert"]["exit_code"] == 0
    assert (tmp_path / "runtime/order.txt").read_text().splitlines() == [
        f"convert:{os.geteuid()}:/models/source",
        f"install:{os.geteuid()}:/models/source",
    ]
    assert [(name, stage) for name, stage, _ in opened] == [
        ("reference", "quality"),
        ("candidate", "quality"),
        ("reference", "timed"),
        ("candidate", "timed"),
    ]
    assert len({id(managed) for _, _, managed in opened}) == 4
    assert closed == [(name, stage) for name, stage, _ in opened]
    reference, candidate = report["bench"]["reference"], report["bench"]["candidate"]
    assert reference["requests"] == candidate["requests"] == 1
    assert reference["output_tokens"] == candidate["output_tokens"] == 2
    assert report["rewards"] == {
        "throughput": round(100 * max(0, 1 - reference["throughput"] / candidate["throughput"]), 2),
        "ttft": round(100 * max(0, 1 - candidate["ttft_mean"] / reference["ttft_mean"]), 2),
    }
    assert json.loads((out / "report.json").read_text()) == report
    assert json.loads((out / "reward.json").read_text()) == report["rewards"]
    assert "throughput" in (out / "report.md").read_text().lower()


def test_failed_candidate_install_skips_quality_and_bench(tmp_path, monkeypatch):
    source, out = submission(tmp_path), tmp_path / "results"
    opened, closed = [], []
    (source / "install.sh").write_text(
        '#!/bin/bash\nsleep 60 >/dev/null 2>&1 &\necho "$!" > child.pid\nexit 7\n'
    )

    unrelated = subprocess.Popen(["sleep", "60"])
    try:
        prepared = asyncio.run(verifier.prepare_submission(PREPARATION_TASK, source, out))
        report = asyncio.run(
            evaluate(
                TASK,
                rows(),
                rows(),
                backend("reference", "http://127.0.0.1:1", opened, closed),
                backend("candidate", "http://127.0.0.1:1", opened, closed),
                out=out,
                prepared=prepared,
                reviewed={"ok": None},
            )
        )
        assert unrelated.poll() is None, "cleanup must not kill other root-owned processes"
        child = Path("/proc") / (source / "child.pid").read_text().strip() / "stat"
        assert not child.exists() or child.read_text().split()[2] == "Z"
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=2)
    assert report["outcome"] == "candidate_failure"
    assert report["pre_run"]["stage"] == "install" and report["pre_run"]["ok"] is False
    assert report["pre_run"]["install"]["exit_code"] == 7
    assert report["bench"]["skipped"] is True and not opened and not closed
    assert report["rewards"] == {"throughput": 0, "ttft": 0}
    assert json.loads((out / "reward.json").read_text()) == report["rewards"]

    # Constructing a candidate must not read an absent or pre-install launcher.
    monkeypatch.setattr(verifier, "runtime_environment", lambda: {"HOME": str(source)})
    monkeypatch.setattr(verifier, "offline", nullcontext)
    task = {
        **TASK,
        "category": "nvidia-cuda-x86",
        "candidate": {"launch": str(source / "launch.sh")},
        "evaluation": {"health_timeout_seconds": 1},
        "assets": {"baselines": {"ollama": {"checkpoint": "mapped-tag"}}},
    }
    (source / "launch.sh").unlink()
    candidate = _backend(task, "candidate", {}, out, reference={"baseline": "ollama"})
    missing_preparation = asyncio.run(verifier.prepare_submission(task, source, out))
    missing = asyncio.run(
        evaluate(
            task,
            rows(),
            rows(),
            candidate,
            candidate,
            out=out,
            prepared=missing_preparation,
            reviewed=REVIEWED,
        )
    )
    assert missing["pre_run"]["stage"] == "static" and missing["rewards"] == report["rewards"]

    @asynccontextmanager
    async def inspected_server(launch, endpoint, **settings):
        assert endpoint == ENDPOINT and "user" not in settings and "guard" not in settings
        yield SimpleNamespace(launch=launch)

    async def checkpoint(stage):
        async with candidate.server(stage) as managed:
            return managed.launch.checkpoint

    monkeypatch.setattr(verifier, "Server", inspected_server)
    (source / "launch.sh").write_text('exec engine serve "$CHECKPOINT" --port 8000\n')
    assert asyncio.run(checkpoint("quality")) == "mapped-tag"
    assert candidate.policy == request_fields("C2", "ollama")
    (source / "launch.sh").write_text("exec engine serve /workspace/quantized --port 8000\n")
    assert asyncio.run(checkpoint("timed")) is None
    assert candidate.policy == request_fields("C2", "candidate")


def test_reference_timed_failure_removes_stale_reward_and_stops_evaluation(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    (out / "reward.json").write_text('{"throughput": 100, "ttft": 100}')
    opened, closed = [], []

    async def scenario():
        reference_calls = 0

        async def reference_reply(reader, writer, body):
            nonlocal reference_calls
            reference_calls += 1
            await reply(writer, valid=reference_calls < 3)

        async def candidate_reply(reader, writer, body):
            await reply(writer)

        async with server(reference_reply) as (reference_endpoint, reference_requests):
            async with server(candidate_reply) as (candidate_endpoint, candidate_requests):
                report = await evaluate(
                    TASK,
                    rows(),
                    rows(),
                    backend("reference", reference_endpoint, opened, closed),
                    backend("candidate", candidate_endpoint, opened, closed),
                    out=out,
                    prepared=PREPARED,
                    reviewed=REVIEWED,
                )
                assert len(reference_requests) == 3 and len(candidate_requests) == 1
        return report

    report = asyncio.run(scenario())
    assert report["outcome"] == "infrastructure_error" and report["pre_run"]["ok"] is True
    assert "rewards" not in report and not (out / "reward.json").exists()
    assert [(name, stage) for name, stage, _ in opened] == [
        ("reference", "quality"),
        ("candidate", "quality"),
        ("reference", "timed"),
    ]
    assert closed == [(name, stage) for name, stage, _ in opened]
    assert "missing completion token usage" in report["reason"]
    assert json.loads((out / "report.json").read_text())["outcome"] == "infrastructure_error"

    def unavailable(stage):
        raise StartupError("reference exited", ENDPOINT)

    unavailable_reference = Backend("reference", request_fields("C2"), unavailable)
    failed = asyncio.run(
        evaluate(
            TASK,
            rows(),
            rows(),
            unavailable_reference,
            unavailable_reference,
            out=out,
            prepared=PREPARED,
            reviewed=REVIEWED,
        )
    )
    assert failed["pre_run"]["stage"] == "startup"
    assert failed["outcome"] == "infrastructure_error" and failed["bench"]["skipped"]
    assert "rewards" not in failed and not (out / "reward.json").exists()


def test_candidate_timed_protocol_failure_has_zero_rewards_and_cleanup(tmp_path, monkeypatch):
    source, out = submission(tmp_path), tmp_path / "results"
    opened, closed = [], []

    async def scenario():
        candidate_calls = 0

        async def reference_reply(reader, writer, body):
            await reply(writer)

        async def candidate_reply(reader, writer, body):
            nonlocal candidate_calls
            candidate_calls += 1
            await reply(writer, valid=candidate_calls < 3)

        async with server(reference_reply) as (reference_endpoint, reference_requests):
            async with server(candidate_reply) as (candidate_endpoint, candidate_requests):
                report = await evaluate(
                    TASK,
                    rows(),
                    rows(),
                    backend("reference", reference_endpoint, opened, closed),
                    backend("candidate", candidate_endpoint, opened, closed),
                    out=out,
                    prepared=PREPARED,
                    reviewed=REVIEWED,
                )
                assert len(reference_requests) == len(candidate_requests) == 3
        return report

    report = asyncio.run(scenario())
    assert report["outcome"] == "candidate_failure" and report["pre_run"]["ok"] is True
    assert report["bench"]["reference"]["valid"] and report["bench"]["skipped"] is False
    assert report["rewards"] == {"throughput": 0, "ttft": 0}
    assert json.loads((out / "reward.json").read_text()) == report["rewards"]
    assert len(opened) == 4 and closed == [(name, stage) for name, stage, _ in opened]
    assert "missing completion token usage" in report["reason"]

    async def startup_case(error):
        async def respond(reader, writer, body):
            await reply(writer)

        def unavailable(stage):
            raise error

        async with server(respond) as (endpoint, received):
            return await evaluate(
                TASK,
                rows(),
                rows(),
                backend("reference", endpoint, [], []),
                Backend("candidate", request_fields("C2"), unavailable),
                out=out,
                prepared=PREPARED,
                reviewed=REVIEWED,
            )

    failed = asyncio.run(startup_case(StartupError("wrong port", ENDPOINT)))
    assert failed["outcome"] == "candidate_failure" and failed["pre_run"]["stage"] == "startup"
    assert failed["bench"]["skipped"] and failed["rewards"] == {"throughput": 0, "ttft": 0}
    unavailable = asyncio.run(startup_case(verifier.Infrastructure("firewall unavailable")))
    assert unavailable["outcome"] == "infrastructure_error"
    assert "rewards" not in unavailable and not (out / "reward.json").exists()

    # An incomplete offline boundary is infrastructure failure, and must roll back.
    calls = []

    def execute(command, **settings):
        calls.append(command)
        assert settings["check"]
        if command[:2] == ["ip6tables", "-I"]:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(verifier.subprocess, "run", execute)
    with pytest.raises(verifier.Infrastructure, match="offline"):
        with verifier.offline():
            pytest.fail("an incomplete firewall must not start measurement")
    assert [call[:2] for call in calls] == [
        ["iptables", "-I"],
        ["ip6tables", "-I"],
        ["iptables", "-D"],
    ]
    assert all("owner" not in call and "pkill" not in call for call in calls)

    # This unit substitutes the subprocess boundary only. Live tests run Claude.
    monkeypatch.setattr(verifier, "RUBRIC", tmp_path / "rubric")
    monkeypatch.setattr(verifier, "REVIEW", tmp_path / "review")
    verifier.REVIEW.mkdir()
    verifier.judge.write_review(PREPARATION_TASK, source, verifier.RUBRIC)
    monkeypatch.setenv("HOME", str(tmp_path / "judge-home"))
    monkeypatch.setenv("USER", "judge")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "test-only-native-token")
    exit_code = 1
    executions = []

    async def native_process(command, **settings):
        executions.append(command)
        assert command[:6] == [
            sys.executable,
            "-I",
            "-m",
            "rewardkit",
            str(verifier.RUBRIC),
            "--workspace",
        ]
        assert command[6] in (str(source), "/workspace/submission")
        assert command[7:] == ["--output", str(verifier.REVIEW / "reward.json")]
        assert settings["cwd"] == Path(os.environ["HOME"]) and settings["timeout"] == 30
        assert settings["environment"]["USER"] == "judge"
        assert settings["environment"]["CLAUDE_CODE_OAUTH_TOKEN"] == "test-only-native-token"
        assert "user" not in settings and "group" not in settings
        if exit_code == 0:
            criteria = tomllib.loads((verifier.RUBRIC / "rules.toml").read_text())["criterion"]
            (verifier.REVIEW / "reward.json").write_text('{"reward": 0}')
            (verifier.REVIEW / "reward-details.json").write_text(
                json.dumps(
                    {
                        "reward": {
                            "kind": "agent",
                            "criteria": [
                                {
                                    "name": item["name"],
                                    "value": 0 if number == 0 else 1,
                                    "reasoning": "server.py:9 returns recorded answers."
                                    if number == 0
                                    else "Rule passes.",
                                }
                                for number, item in enumerate(criteria)
                            ],
                        }
                    }
                )
            )
        return exit_code

    monkeypatch.setattr(verifier, "_process", native_process)
    forbidden = Backend("unstarted", request_fields("C2"), lambda stage: pytest.fail(stage))

    async def review_gate():
        reviewed = await verifier.review_rules(source)
        return await evaluate(
            TASK,
            rows(),
            rows(),
            forbidden,
            forbidden,
            out=out,
            prepared=PREPARED,
            reviewed=reviewed,
        )

    provider_failure = asyncio.run(review_gate())
    assert provider_failure["pre_run"]["rules"]["ok"] is None
    assert provider_failure["pre_run"]["rules"]["exit_code"] == 1
    assert (
        provider_failure["outcome"] == "infrastructure_error"
        and provider_failure["bench"]["skipped"]
    )
    assert "rewards" not in provider_failure and not (out / "reward.json").exists()
    exit_code = 0
    rule_violation = asyncio.run(review_gate())
    assert rule_violation["pre_run"]["stage"] == "rules"
    assert rule_violation["pre_run"]["rules"]["ok"] is False
    assert rule_violation["pre_run"]["rules"]["complete"]
    assert rule_violation["outcome"] == "candidate_failure" and rule_violation["bench"]["skipped"]
    assert rule_violation["rewards"] == {"throughput": 0, "ttft": 0}
    assert len(executions) == 2
    verifier.REVIEW.chmod(0o700)
    transcript = verifier.REVIEW / "rules.log"
    transcript.write_text("Native reviewer evidence.\n")
    transcript.chmod(0o600)
    monkeypatch.setattr(verifier, "VERIFIER", tmp_path / "native-review")
    assert (
        verifier.main(["--stage", "review", "--task", str(tmp_path / "unreadable-private.json")])
        == 0
    )
    durable = json.loads((verifier.REVIEW / "review.json").read_text())
    assert durable["ok"] is False and durable["complete"]
    assert json.loads((verifier.VERIFIER / "report.json").read_text()) == durable
    assert json.loads((verifier.VERIFIER / "reward.json").read_text()) == {"reward": 0}
    exported = verifier.VERIFIER / "rules.log"
    assert exported.read_text() == transcript.read_text()
    assert exported.stat().st_mode & 0o004
    assert not any(path.is_dir() for path in verifier.VERIFIER.iterdir())
    assert verifier.REVIEW.stat().st_mode & 0o777 == 0o700
