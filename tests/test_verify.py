"""Two complete verification procedures and two failed gate/transport cases."""

import asyncio
import json
import subprocess
from pathlib import Path

import pytest
from datasets import Dataset
from http_server import event, headers, server

from eval.bench.serve import Server, StartupError
from eval.launch import ENDPOINT
from eval.request_policy import request_fields
from eval.verify import access
from eval.verify import run as verifier
from eval.verify.run import Backend, evaluate, install_reference, pre_run, rewards

TASK = {
    "model": "Qwen3.6-35B-A3B",
    "workload": "S-1",
    "device_category": "nvidia-cuda-x86",
    "assets": {
        "checkpoint": "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "baselines": {"freetoken": "nvidia/Qwen3.6-35B-A3B-NVFP4"},
    },
    "target": {"source": "nvidia/Qwen3.6-35B-A3B-NVFP4"},
    "evaluation": {
        "install_timeout_seconds": 5,
        "health_timeout_seconds": 2,
        "agent": {"name": "claude-code", "budget_seconds": 30},
    },
}
PREPARED = {"ok": True, "stage": "prepare", "reason": "submission prepared"}
BOUND = {"id": "frozen-profile", "question_ids": ["quality-letter"], "min_correct": 1}


def rows():
    return [
        {
            "id": "quality-letter",
            "model": TASK["model"],
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


def inputs(tmp_path):
    source = tmp_path / "submission"
    source.mkdir()
    (source / "install.sh").write_text(
        "#!/bin/bash\nset -eu\n"
        'printf "#!/bin/bash\\nexit 0\\n" > "./server.launch"\n'
        'chmod +x "./server.launch"\n'
        'printf "%s\\n" "/models/target" > ./installed-checkpoint\n'
    )
    data = tmp_path / "data"
    Dataset.from_list(rows()).save_to_disk(str(data))
    selected = tmp_path / "reference.json"
    selected.write_text(json.dumps({"baseline": "freetoken", "args": [], "env": {}}))
    return {
        **TASK,
        "reference": str(selected),
        "candidate": {
            "submission": str(source),
            "launch": str(source / "server.launch"),
        },
        "inputs": {"private": str(data), "quality": str(data)},
    }


@pytest.fixture(autouse=True)
def boundaries(tmp_path, monkeypatch):
    for name, folder in (
        ("STATE", "state"),
        ("REFERENCE", "reference"),
        ("RUBRIC", "rubric"),
        ("REVIEW", "review"),
    ):
        monkeypatch.setattr(verifier, name, tmp_path / folder)
    monkeypatch.setattr(verifier, "CANDIDATE_UID", None)
    # UID and immutable-path enforcement are exercised in production-container tests.
    monkeypatch.setattr(access, "seal_submission", lambda _: None)
    monkeypatch.setattr(access, "readonly", lambda _: None)
    monkeypatch.setattr(access, "reset_runtime", lambda _: tmp_path)

    async def install(task, selection, out):
        (verifier.REFERENCE / "baselines").mkdir(parents=True, exist_ok=True)
        return "/models/target"

    async def stopped():
        return None

    monkeypatch.setattr(verifier, "install_reference", install)
    monkeypatch.setattr(verifier, "stop_runtime_users", stopped)


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
        opened.append((name, stage))
        return managed

    return Backend(name, request_fields(TASK["model"]), factory)


def saved_reference():
    return {
        "ok": True,
        "baseline": "freetoken",
        "checkpoint": "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "bench": {
            "valid": True,
            "throughput": 10,
            "ttft_mean": 2,
            "records": [
                {
                    "ok": True,
                    "t_send": 0,
                    "t_first": 2,
                    "t_last": 3,
                    "t_end": 4,
                    "output_tokens": 10,
                }
            ],
            "warmup": {},
        },
    }


def test_install_review_quality_reference_candidate_keep_the_shared_target(tmp_path, monkeypatch):
    task = inputs(tmp_path)
    source, out = Path(task["candidate"]["submission"]), tmp_path / "results"
    shared = tmp_path / "target.safetensors"
    shared.write_bytes(b"unchanged fixed target fixture")
    opened, closed = [], []

    async def scenario():
        prepared = await verifier.prepare_submission(task, source, out)
        assert prepared["ok"]
        assert not verifier.REFERENCE.exists()  # No baseline weights during candidate setup.
        rubric = verifier.RUBRIC / "prompt.txt"
        expected_prompt = rubric.read_text()
        rubric.write_text("obsolete reviewer instructions")
        installed = source / "installed-checkpoint"
        installed_at = installed.stat().st_mtime_ns
        assert await verifier.prepare_submission(task, source, out) == prepared
        assert rubric.read_text() == expected_prompt
        assert installed.stat().st_mtime_ns == installed_at
        assert not (verifier.STATE / "report-space.reserve").exists()
        assert (source / "installed-checkpoint").read_text().strip() == "/models/target"
        assert "obtain_weight" not in prepared and (verifier.RUBRIC / "prompt.txt").is_file()

        async def respond(_reader, writer, _body):
            await reply(writer)

        async with server(respond) as (endpoint, requests):
            candidate = backend("candidate", endpoint, opened, closed)
            gate = await pre_run(rows(), BOUND, candidate)
            assert gate["ok"] and gate["quality"]["bound"]["correct"] == 1
            assert not verifier.REFERENCE.exists()  # Failed gates never acquire a reference.
            monkeypatch.setattr(
                verifier, "_backend", lambda *a, **k: backend("freetoken", endpoint, opened, closed)
            )
            assert (await verifier.prepare_reference(task, source, out))["ok"]

            async def no_downloads(*args):
                raise AssertionError("offline timing must not install or acquire reference assets")

            monkeypatch.setattr(verifier, "install_reference", no_downloads)
            reference = await verifier.reference(task, source, out)
            assert reference["ok"] and "quality" not in reference
            assert await verifier.reference(task, source, out) == reference
            assert (await verifier.prepare_reference(task, source, out))["ok"]
            report = await evaluate(
                task,
                rows(),
                reference,
                candidate,
                out=out,
                prepared=prepared,
                reviewed={"ok": True},
                quality_gate=gate,
            )
            assert len(requests) == 5  # One quality, then warmup + one timed row on each server.
            return report

    report = asyncio.run(scenario())
    assert (
        report["outcome"] == "passed" and shared.read_bytes() == b"unchanged fixed target fixture"
    )
    assert verifier.REFERENCE.exists()
    assert (
        opened
        == closed
        == [("candidate", "quality"), ("freetoken", "timed"), ("candidate", "timed")]
    )
    assert json.loads((out / "report.json").read_text()) == report
    assert json.loads((out / "reward.json").read_text()) == report["rewards"]
    assert rewards("S-1", saved_reference()["bench"], {"throughput": 20, "ttft_mean": 1}) == {
        "score": 50.0,
    }


def test_reference_protocol_failure_keeps_target_and_has_no_candidate_score(tmp_path, monkeypatch):
    task = inputs(tmp_path)
    source, out = Path(task["candidate"]["submission"]), tmp_path / "results"
    opened, closed = [], []

    async def scenario():
        await verifier.prepare_submission(task, source, out)
        calls = 0

        async def respond(_reader, writer, _body):
            nonlocal calls
            calls += 1
            await reply(writer, valid=calls == 1)

        async with server(respond) as (endpoint, _):
            monkeypatch.setattr(
                verifier, "_backend", lambda *a, **k: backend("freetoken", endpoint, opened, closed)
            )
            assert (await verifier.prepare_reference(task, source, out))["ok"]
            return await verifier.reference(task, source, out)

    result = asyncio.run(scenario())
    assert result["ok"] is None and "missing completion token usage" in result["reason"]
    assert verifier.REFERENCE.exists() and not (verifier.STATE / "reference.json").exists()
    assert opened == closed == [("freetoken", "timed")]
    verifier.save_gate(out, result)
    assert not (out / "reward.json").exists()


def test_failed_installer_kills_its_children_and_stops_before_review(tmp_path):
    task = inputs(tmp_path)
    source, out = Path(task["candidate"]["submission"]), tmp_path / "results"
    (source / "install.sh").write_text(
        '#!/bin/bash\nsleep 60 >/dev/null 2>&1 &\necho "$!" > child.pid\nexit 7\n'
    )
    unrelated = subprocess.Popen(["sleep", "60"])
    try:
        result = asyncio.run(verifier.prepare_submission(task, source, out))
        assert result["ok"] is False and result["install"]["exit_code"] == 7
        assert not verifier.REFERENCE.exists()
        assert not (verifier.STATE / "report-space.reserve").exists()
        assert not (verifier.STATE / "install.json").exists() and not verifier.RUBRIC.exists()
        assert unrelated.poll() is None
        child = Path("/proc") / (source / "child.pid").read_text().strip() / "stat"
        assert not child.exists() or child.read_text().split()[2] == "Z"
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=2)
    verifier.save_gate(out, result)
    assert json.loads((out / "reward.json").read_text()) == {"reward": 0}


@pytest.mark.parametrize("workload", ["S-1", "S-N"])
def test_bad_candidate_timing_and_wrong_port_stop_with_zero_rewards(tmp_path, workload):
    task = {**TASK, "workload": workload}
    opened, closed = [], []
    out = tmp_path / "results"

    async def scenario():
        calls = 0

        async def respond(_reader, writer, _body):
            nonlocal calls
            calls += 1
            await reply(writer, valid=calls == 1)

        async with server(respond) as (endpoint, _):
            return await evaluate(
                task,
                (
                    [dict(rows()[i % len(rows())], id=f"item-{i}") for i in range(8)]
                    if workload == "S-N"
                    else rows()
                ),
                saved_reference(),
                backend("candidate", endpoint, opened, closed),
                out=out,
                prepared=PREPARED,
                reviewed={"ok": True},
                quality_gate={"ok": True, "stage": "quality"},
            )

    report = asyncio.run(scenario())
    assert report["outcome"] == "candidate_failure" and report["rewards"] == verifier.zero_rewards(
        workload
    )
    assert "missing completion token usage" in report["reason"]
    assert closed == opened == [("candidate", "timed")]

    def wrong_port(stage):
        raise StartupError("wrong port", ENDPOINT)

    candidate = Backend("candidate", request_fields(TASK["model"]), wrong_port)
    gate = asyncio.run(pre_run(rows(), BOUND, candidate))
    failed = asyncio.run(
        evaluate(
            task,
            rows(),
            saved_reference(),
            candidate,
            out=out,
            prepared=PREPARED,
            reviewed={"ok": True},
            quality_gate=gate,
        )
    )
    assert failed["pre_run"]["stage"] == "startup" and failed["bench"]["skipped"]
    assert failed["rewards"] == verifier.zero_rewards(workload)


def test_resume_reuses_a_completed_reviewer_decision(tmp_path, monkeypatch):
    from eval.verify import run as verifier

    decision = {"ok": True, "rules": {"selected_model": True}}
    monkeypatch.setattr(verifier, "REVIEW", tmp_path)
    (tmp_path / "review.json").write_text(json.dumps(decision))

    async def unexpected(*args, **kwargs):
        raise AssertionError("A finished review must not start another agent")

    monkeypatch.setattr(verifier, "_process", unexpected)
    assert asyncio.run(verifier.review_rules(tmp_path / "submission")) == decision


@pytest.mark.parametrize("separate", [False, True])
def test_reference_acquisition_uses_its_mapping_and_reuses_identical_target(
    tmp_path, monkeypatch, separate
):
    from types import SimpleNamespace

    task = inputs(tmp_path)
    canonical = task["target"]["source"]
    selected = {"baseline": "freetoken", "args": [], "env": {}}
    task["assets"]["baselines"] = {"freetoken": "owner/reference-GGUF" if separate else canonical}
    calls = []
    protected = []

    def workspace():
        verifier.REFERENCE.mkdir()

    def acquire(source, destination):
        calls.append((source, destination))
        destination.mkdir(parents=True)
        (destination / "model.gguf").write_text("reference-only fixture")
        return SimpleNamespace(checkpoint=str(destination))

    async def installed(command, **kwargs):
        assert kwargs["user"] == verifier.REFERENCE_UID
        return 0

    monkeypatch.setattr(access, "reference_workspace", workspace)
    monkeypatch.setattr(access, "readonly", lambda path: protected.append(path))
    monkeypatch.setattr("eval.models.prepare_asset", acquire)
    monkeypatch.setattr(verifier, "_process", installed)
    result = asyncio.run(install_reference(task, selected, tmp_path / "logs"))
    if separate:
        destination = verifier.REFERENCE / "weights/target"
        assert result == str(destination)
        assert calls == [("owner/reference-GGUF", destination)]
        assert protected == [destination]
    else:
        assert result == "/models/target" and not calls and not protected
    assert task["target"]["source"] == canonical


def test_coding_export_preserves_source_without_native_state(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "install.sh").write_text("#!/bin/bash\n")
    (project / "server.py").write_text("print('serving')\n")
    state = tmp_path / "runtime"
    state.mkdir()
    (state / "installed.so").write_text("generated runtime")
    (state / "operation.json").write_text("replay checkpoint")
    target = tmp_path / "published"
    verifier.collect_submission(target, project)
    assert {path.name for path in target.iterdir()} == {"install.sh", "server.py"}
    assert (state / "installed.so").is_file()
    assert (target / "server.py").read_text() == (project / "server.py").read_text()


def test_failed_coding_export_cannot_publish_a_stale_submission(tmp_path, monkeypatch):
    project, target = tmp_path / "project", tmp_path / "published"
    project.mkdir()
    target.mkdir()
    (target / "install.sh").write_text("stale submission")

    def interrupted(*args, **kwargs):
        raise OSError("source transfer interrupted")

    monkeypatch.setattr(verifier.shutil, "copytree", interrupted)
    with pytest.raises(OSError, match="transfer interrupted"):
        verifier.collect_submission(target, project)
    assert not target.exists()


@pytest.mark.parametrize(
    "values, expected",
    [
        (
            {"TRITON_CACHE_DIR": "/workspace/runtime/search/cache/triton", "OMP_NUM_THREADS": "8"},
            {"TRITON_CACHE_DIR": "{reference}/cache/triton", "OMP_NUM_THREADS": "8"},
        ),
        ({}, {}),
        ({"CUDA_CACHE_PATH": "/mnt/cache/cuda"}, {"CUDA_CACHE_PATH": "/mnt/cache/cuda"}),
        (
            {"ROOT": "/workspace/runtime/search", "OTHER": "/workspace/runtime/search-other/cache"},
            {"ROOT": "{reference}", "OTHER": "/workspace/runtime/search-other/cache"},
        ),
    ],
)
def test_reference_relocates_only_search_runtime_environment(
    tmp_path, monkeypatch, values, expected
):
    observed = []
    params = {"args": ["--max-running-requests", "8"], "env": dict(values)}

    class StartedServer:
        def __init__(self, launch, *args, **kwargs):
            observed.append(launch.prepare())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    monkeypatch.setattr(verifier, "Server", StartedServer)

    async def scenario():
        backend = verifier._backend(
            TASK, "freetoken", params, tmp_path, checkpoint="/models/reference"
        )
        async with backend.server("measure"):
            pass

    asyncio.run(scenario())
    command, environment = observed[0]
    assert command[-2:] == params["args"]
    assert environment == {
        **{key: value.format(reference=verifier.REFERENCE) for key, value in expected.items()},
        "CHECKPOINT": "/models/reference",
    }
    assert params["env"] == values


def test_failed_online_reference_preparation_has_no_candidate_score(tmp_path, monkeypatch):
    task = inputs(tmp_path)
    out = tmp_path / "results"

    async def failed(*args):
        raise verifier.Infrastructure("reference acquisition failed")

    monkeypatch.setattr(verifier, "install_reference", failed)
    result = asyncio.run(
        verifier.prepare_reference(task, Path(task["candidate"]["submission"]), out)
    )
    assert result["ok"] is None and result["stage"] == "reference-prepare"
    verifier.save_gate(out, result)
    assert not (out / "reward.json").exists()
    assert not (verifier.STATE / "reference.json").exists()
