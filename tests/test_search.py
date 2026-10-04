"""Two successful selections and two invalid tuning/selection boundaries."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.search import run as search
from eval.search.run import finalize


def arrange(tmp_path, selection):
    images = tmp_path / "images"
    script = images / "device" / "baselines" / "engine" / "launch.sh"
    script.parent.mkdir(parents=True)
    script.write_text('#!/bin/bash\nexec server --model "$CHECKPOINT" "$@"\n')
    private = tmp_path / "reference.json"
    public = tmp_path / "public.json"
    task = {
        "category": "device",
        "model": "model",
        "assets": {
            "baselines": {
                "engine": {
                    "path": "/models/baselines/engine",
                    "checkpoint": "/models/baselines/engine",
                }
            }
        },
        "reference": str(private),
        "public_reference": str(public),
    }
    chosen = tmp_path / "selection.json"
    chosen.write_text(json.dumps(selection))
    return task, chosen, images


def test_default_selection_publishes_mapped_weights(tmp_path):
    task, chosen, images = arrange(tmp_path, {"baseline": "engine", "checkpoint": "/untrusted"})
    reference = finalize(task, chosen, images=images)
    assert reference == {
        "category": "device",
        "model": "model",
        "baseline": "engine",
        "checkpoint": "/models/reference",
        "args": [],
        "env": {},
    }
    assert json.loads((tmp_path / "public.json").read_text()) == reference
    assert json.loads((tmp_path / "reference.json").read_text()) == reference


def test_bench_report_parameters_and_native_tag_are_retained(tmp_path):
    params = {"args": ["--batch-size", "32"], "env": {"THREADS": "8"}}
    task, chosen, images = arrange(tmp_path, {"baseline": "engine", "parameters": params})
    task["assets"]["baselines"]["engine"] = {"path": None, "checkpoint": "model:quant"}
    reference = finalize(task, chosen, images=images)
    assert reference["checkpoint"] == "model:quant"
    assert reference["args"] == params["args"] and reference["env"] == params["env"]


def test_invalid_selection_is_a_gate_failure_but_missing_image_assets_are_not(
    tmp_path, monkeypatch
):
    task, chosen, images = arrange(tmp_path, {"baseline": "other"})
    with pytest.raises(ValueError, match="not mapped"):
        finalize(task, chosen, images=images)
    assert not (tmp_path / "reference.json").exists()

    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task))
    out = tmp_path / "verifier"
    routed = {"/home/search/selection.json": chosen, "/logs/verifier": out}
    monkeypatch.setattr(search, "Path", lambda value: routed.get(str(value), Path(value)))
    monkeypatch.setattr(search.subprocess, "run", lambda _: SimpleNamespace(returncode=1))
    monkeypatch.setattr(search, "finalize", lambda task, path: finalize(task, path, images=images))
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
    assert (tmp_path / "public.json").is_file()
    launcher = images / "device" / "baselines" / "engine" / "launch.sh"
    launcher.unlink()
    with pytest.raises(FileNotFoundError) as error:
        search.main([*arguments, "collect"])
    assert error.value.filename == str(launcher)
    assert not report.exists()
    assert not (tmp_path / "reference.json").exists()
    assert not (tmp_path / "public.json").exists()
    with pytest.raises(FileNotFoundError) as error:
        search.main([*arguments, "gate"])
    assert error.value.filename == str(report)


def test_fixed_launch_inputs_cannot_be_tuned(tmp_path):
    task, chosen, images = arrange(tmp_path, {"baseline": "engine", "args": ["--port=9000"]})
    with pytest.raises(ValueError, match="--port is fixed"):
        finalize(task, chosen, images=images)
    chosen.write_text(json.dumps({"baseline": "engine", "env": {"CHECKPOINT": "/other"}}))
    with pytest.raises(ValueError, match="CHECKPOINT is fixed"):
        finalize(task, chosen, images=images)
    assert not (tmp_path / "reference.json").exists()
