"""Selections, unsupported verdicts and invalid tuning/selection boundaries."""

import json
import os
from pathlib import Path

import pytest

from eval.search import run as search
from eval.search.run import finalize


def arrange(tmp_path, selection):
    recipes = tmp_path / "recipes"
    script = recipes / "device" / "engine" / "launch.sh"
    script.parent.mkdir(parents=True)
    script.write_text('#!/bin/bash\nexec server --model "$CHECKPOINT" "$@"\n')
    private = tmp_path / "reference.json"
    task = {
        "device_category": "device",
        "model": "model",
        "assets": {"baselines": {"engine": "owner/model"}},
        "reference": str(private),
    }
    chosen = tmp_path / "selection.json"
    chosen.write_text(json.dumps(selection))
    return task, chosen, recipes


def test_default_selection_publishes_only_parameters(tmp_path):
    task, chosen, recipes = arrange(tmp_path, {"baseline": "engine", "checkpoint": "/untrusted"})
    reference = finalize(task, chosen, recipes=recipes)
    assert reference == {"baseline": "engine", "args": [], "env": {}}
    assert not (tmp_path / "public.json").exists()
    assert json.loads((tmp_path / "reference.json").read_text()) == reference


def test_bench_report_parameters_and_native_tag_are_retained(tmp_path):
    params = {"args": ["--batch-size", "32"], "env": {"THREADS": "8"}}
    task, chosen, recipes = arrange(tmp_path, {"baseline": "engine", "parameters": params})
    task["assets"]["baselines"]["engine"] = "model:quant"
    reference = finalize(task, chosen, recipes=recipes)
    assert "checkpoint" not in reference
    assert reference["args"] == params["args"] and reference["env"] == params["env"]


def test_invalid_or_missing_selection_is_a_gate_failure(tmp_path, monkeypatch):
    task, chosen, recipes = arrange(tmp_path, {"baseline": "other"})
    with pytest.raises(ValueError, match="not mapped"):
        finalize(task, chosen, recipes=recipes)
    assert not (tmp_path / "reference.json").exists()

    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task))
    out = tmp_path / "verifier"
    routed = {"/workspace/output/selection.json": chosen, "/logs/verifier": out}
    monkeypatch.setattr(search, "Path", lambda value: routed.get(str(value), Path(value)))
    monkeypatch.setattr(
        search, "finalize", lambda task, path: finalize(task, path, recipes=recipes)
    )
    chosen.unlink()
    arguments = ["--task", str(task_path), "--stage"]
    assert search.main([*arguments, "collect"]) == 0
    report = tmp_path / "report.json"
    assert json.loads(report.read_text())["ok"] is False
    assert search.main([*arguments, "gate"]) == 0
    assert json.loads((out / "reward.json").read_text()) == {"reward": 0}
    assert not (tmp_path / "reference.json").exists()

    chosen.write_text(json.dumps({"baseline": "engine"}))
    assert search.main([*arguments, "collect"]) == 0
    assert json.loads(report.read_text())["ok"] is True
    assert (tmp_path / "reference.json").is_file()
    assert not (tmp_path / "public.json").exists()


def test_fixed_launch_inputs_cannot_be_tuned(tmp_path):
    task, chosen, recipes = arrange(tmp_path, {"baseline": "engine", "args": ["--port=9000"]})
    with pytest.raises(ValueError, match="--port is fixed"):
        finalize(task, chosen, recipes=recipes)
    chosen.write_text(json.dumps({"baseline": "engine", "env": {"CHECKPOINT": "/other"}}))
    with pytest.raises(ValueError, match="CHECKPOINT is fixed"):
        finalize(task, chosen, recipes=recipes)
    assert not (tmp_path / "reference.json").exists()


def test_unsupported_verdict_keeps_bounded_search_owned_log_tails(tmp_path):
    install = tmp_path / "install.log"
    install.write_bytes(b"x" * 70_000 + b"CUDA arch sm_120 unsupported\n")
    (tmp_path / "link.log").symlink_to("/etc/hostname")
    verdict = {
        "supported": False,
        "attempts": [
            {"baseline": "engine", "stage": "launch", "error": "no kernel", "log": str(install)},
            {
                "baseline": "engine",
                "stage": "serve",
                "error": "OOM",
                "log": str(tmp_path / "link.log"),
            },
            {"baseline": "engine", "stage": "precheck", "error": "too large"},
        ],
    }
    task, chosen, recipes = arrange(tmp_path, verdict)
    reference = finalize(task, chosen, recipes=recipes, uid=os.getuid(), roots=(tmp_path,))
    # The verdict picks no checkpoint: every SLO task targets FreeToken's.
    assert set(reference) == {"supported", "attempts"} and reference["supported"] is False
    first, linked, unlogged = reference["attempts"]
    tail = (tmp_path / first["log"]).read_bytes()
    assert len(tail) == search.LOG_LIMIT and tail.endswith(b"sm_120 unsupported\n")
    # Links leaving the search roots never reach the artifact; the error still counts.
    assert linked["log"] is None and "outside" in linked["log_error"]
    assert unlogged == {
        "baseline": "engine",
        "stage": "precheck",
        "error": "too large",
        "log": None,
    }
    assert json.loads((tmp_path / "reference.json").read_text()) == reference
    # Only files owned by the search account are copied.
    foreign = finalize(task, chosen, recipes=recipes, uid=os.getuid() + 1, roots=(tmp_path,))
    assert "owned by search" in foreign["attempts"][0]["log_error"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"attempts": []}, "attempts"),
        ({"attempts": [{"baseline": "other", "stage": "install", "error": "x"}]}, "cover"),
        ({"attempts": [{"baseline": "engine", "stage": "tuning", "error": "x"}]}, "stage"),
        ({"attempts": [{"baseline": "engine", "stage": "install", "error": " "}]}, "error"),
        ({"supported": "no"}, "true or false"),
    ],
)
def test_malformed_unsupported_verdicts_fail_the_gate(tmp_path, change, message):
    verdict = {
        "supported": False,
        "attempts": [{"baseline": "engine", "stage": "install", "error": "failed"}],
    }
    task, chosen, recipes = arrange(tmp_path, verdict | change)
    with pytest.raises((ValueError, TypeError), match=message):
        finalize(task, chosen, recipes=recipes)
    assert not (tmp_path / "reference.json").exists()
