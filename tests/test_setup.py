"""Phase 2 acquisition mapping with synthetic HF rows; no dataset or model download."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from datasets import Dataset

from eval import setup
from eval.baselines import needed
from eval.config import CALIBRATION_SUBSET, QUALITY_SUBSET, SEED, parse_config
from eval.device import GPU, Machine
from eval.models import ModelAsset


def _source(pool, split):
    def row(identity, model="C2"):
        content = [{"type": "text", "text": "Question"}]
        tools = [
            {"type": "function", "function": {"name": "tool", "parameters": {"type": "object"}}}
        ]
        return {
            "id": identity,
            "model": model,
            "requests": [
                {
                    "id": "r1",
                    "parents": [],
                    "tool": 0.0,
                    "out": 12,
                    "messages": [{"role": "user", "content": content}],
                    "tools": tools if identity == f"{pool}-1" else [],
                }
            ],
            "evaluation": {"method": "choice", "answer": "A"},
            "extra": '{"prepared_metadata": "preserve as supplied"}',
        }

    rows = (
        [row(f"{pool}-public")]
        if split == "public"
        else [row(f"{pool}-{index}") for index in range(60)]
    )
    return Dataset.from_list([*rows, row("other-model", model="C1")])


def _read(path):
    return [json.loads(line) for line in (Path(path) / "rows.jsonl").read_text().splitlines()]


def _arrange(tmp_path, monkeypatch, workloads=("S-1",)):
    config = parse_config(
        {"dataset": "owner/recordings", "tasks": {"models": ["C2"], "workloads": list(workloads)}}
    )
    machine = Machine("Linux", "x86_64", 8, 32 * 2**30, (GPU(0, "GPU-fixture", "GPU", 81920),))
    repo = tmp_path / "repo"
    dockerfile = repo / machine.dockerfile
    dockerfile.parent.mkdir(parents=True)
    dockerfile.write_text("# Synthetic source fixture; never built.\nFROM scratch\n")
    (repo / ".dockerignore").write_text("runs/\n")
    for engine in needed(machine.category, ("C2",)):
        path = repo / "images" / machine.category / "baselines" / engine / "install.sh"
        path.parent.mkdir(parents=True)
        path.write_text("#!/bin/sh\n# Synthetic installer fixture; never executed.\n")
        path.with_name("launch.sh").write_text(
            "#!/bin/sh\n# Synthetic launcher fixture; never executed.\n"
        )
    calls, rendered, acquired = [], [], []

    def load(source, pool, *, split):
        calls.append((source, pool, split))
        return _source(pool, split)

    def asset(source, destination):
        acquired.append((source, destination))
        if ":" in source:
            return ModelAsset(source, None, source)
        destination.mkdir(parents=True, exist_ok=True)
        return ModelAsset(source, destination, str(destination))

    def render(task, config, machine, image, run_folder, inputs, assets):
        rendered.append((task, inputs, assets))
        path = run_folder / "tasks" / task.task_id
        path.mkdir(parents=True)
        return path

    monkeypatch.setattr(setup, "load_dataset", load)
    monkeypatch.setattr(setup, "prepare_asset", asset)
    monkeypatch.setattr(setup, "render_task", render)
    return config, machine, repo, calls, rendered, acquired


def test_s_views_preserve_rows_and_sample_shared_calibration(tmp_path, monkeypatch):
    config, machine, repo, calls, rendered, acquired = _arrange(tmp_path, monkeypatch)
    previous_umask = os.umask(0o077)
    try:
        result = setup.prepare(config, tmp_path / "run", repo_root=repo, machine=machine)
    finally:
        os.umask(previous_umask)
    views = result.inputs["C2"]["S"]
    assert set(result.assets["C2"]) == {"root", "source", "baselines"}
    source_asset = result.assets["C2"]["source"]
    assert source_asset["source"] == "Qwen/Qwen3.6-35B-A3B"
    assert Path(source_asset["path"]) == tmp_path / "run" / "models" / "C2" / "source"
    assert source_asset["checkpoint"] == source_asset["path"]
    baseline_assets = result.assets["C2"]["baselines"]
    assert baseline_assets["freetoken"]["source"] == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert baseline_assets["freetoken"]["path"] is None
    assert baseline_assets["freetoken"]["checkpoint"] == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert acquired == [(source_asset["source"], Path(source_asset["path"]))]
    assert baseline_assets["ollama"]["path"] is None
    assert Path(result.assets["C2"]["root"]).stat().st_mode & 0o777 == 0o755
    assert Path(views["public"]).stat().st_mode & 0o777 == 0o755
    assert (Path(views["public"]) / "rows.jsonl").stat().st_mode & 0o777 == 0o644
    assert Path(views["calibration"]).stat().st_mode & 0o777 == 0o755
    assert (Path(views["calibration"]) / "rows.jsonl").stat().st_mode & 0o777 == 0o644
    for protected in ("private", "quality"):
        assert Path(views[protected]).stat().st_mode & 0o777 == 0o700
        assert (Path(views[protected]) / "rows.jsonl").stat().st_mode & 0o777 == 0o600
    assert [row["id"] for row in _read(views["public"])] == ["S-public"]
    private = _read(views["private"])
    expected = _source("S", "private").filter(lambda row: row["model"] == "C2")
    assert private == list(expected)
    assert private[0]["requests"][0]["messages"][0]["content"] == [
        {"type": "text", "text": "Question"}
    ]
    assert private[1]["requests"][0]["tools"][0]["function"]["parameters"] == {"type": "object"}
    assert private[0]["evaluation"] == {"method": "choice", "answer": "A"}
    assert private[0]["extra"] == '{"prepared_metadata": "preserve as supplied"}'
    assert all("calibration" not in row for row in private)
    calibration = [row["id"] for row in _read(views["calibration"])]
    quality = [row["id"] for row in _read(views["quality"])]
    assert len(set(calibration)) == CALIBRATION_SUBSET
    assert len(set(quality)) == QUALITY_SUBSET
    assert calibration == expected.shuffle(seed=SEED).select(range(CALIBRATION_SUBSET))["id"]
    assert quality == expected.shuffle(seed=SEED + 1).select(range(QUALITY_SUBSET))["id"]
    assert set(calibration + quality) <= {row["id"] for row in private}
    # Independent samples may overlap; the private timing membership stays intact.
    assert calls == [(config.dataset, "S", "public"), (config.dataset, "S", "private")]
    assert rendered[0][1] == views
    repeated = setup.prepare(config, tmp_path / "repeat", repo_root=repo, machine=machine)
    repeated_views = repeated.inputs["C2"]["S"]
    assert [row["id"] for row in _read(repeated_views["quality"])] == quality
    assert [row["id"] for row in _read(repeated_views["calibration"])] == calibration
    marker = json.loads((tmp_path / "run" / "prepared.json").read_text())
    assert marker["task_paths"] == [str(path) for path in result.task_paths]


def test_agentic_only_tasks_share_s_calibration_and_quality_with_source_reuse(
    tmp_path, monkeypatch
):
    config, machine, repo, calls, rendered, acquired = _arrange(
        tmp_path, monkeypatch, ("A-N", "A-MA")
    )
    entries = setup.mappings(machine.category, "C2")
    entries = tuple(
        replace(entry, checkpoint=entry.source_model_checkpoint)
        if entry.baseline == "freetoken"
        else entry
        for entry in entries
    )
    monkeypatch.setattr(setup, "mappings", lambda category, name: entries)
    result = setup.prepare(config, tmp_path / "run", repo_root=repo, machine=machine)
    views = result.inputs["C2"]
    assert result.assets["C2"]["source"] == result.assets["C2"]["baselines"]["freetoken"]
    assert not (tmp_path / "run" / "models" / "C2" / "baselines" / "freetoken").exists()
    assert set(views) == {"A", "A-MA", "S"}
    assert set(views["S"]) == {"calibration", "quality"}
    assert {(pool, split) for _, pool, split in calls} == {
        ("A", "public"),
        ("A", "private"),
        ("A-MA", "public"),
        ("A-MA", "private"),
        ("S", "private"),
    }
    assert len(calls) == 5
    quality = views["S"]["quality"]
    assert len(_read(quality)) == QUALITY_SUBSET
    assert len(rendered) == 2 and all(inputs["quality"] == quality for _, inputs, _ in rendered)
    calibration = views["S"]["calibration"]
    assert len(_read(calibration)) == CALIBRATION_SUBSET
    assert all(inputs["calibration"] == calibration for _, inputs, _ in rendered)
    assert all(row["id"].startswith("S-") for row in _read(calibration))
    assert acquired == [
        (entries[0].source_model_checkpoint, Path(result.assets["C2"]["source"]["path"]))
    ]
    for pool in ("A", "A-MA"):
        assert set(views[pool]) == {"public", "private"}
        assert len(_read(views[pool]["private"])) == 60


def test_missing_supplied_scripts_stop_before_any_acquisition(tmp_path, monkeypatch):
    config, machine, repo, calls, rendered, acquired = _arrange(tmp_path, monkeypatch)
    installer = (
        Path("images")
        / machine.category
        / "baselines"
        / needed(machine.category, ("C2",))[0]
        / "install.sh"
    )
    (repo / installer).unlink()
    monkeypatch.setattr(setup, "prepare_asset", lambda *args: pytest.fail("must not download"))
    with pytest.raises(FileNotFoundError, match="install.sh"):
        setup.prepare(config, tmp_path / "run", repo_root=repo, machine=machine)
    (repo / installer).write_text("#!/bin/sh\n")
    (repo / installer.with_name("launch.sh")).unlink()
    with pytest.raises(FileNotFoundError, match="launch.sh"):
        setup.prepare(config, tmp_path / "run", repo_root=repo, machine=machine)
    assert calls == [] and rendered == []
    assert not (tmp_path / "run" / "prepared.json").exists()


def test_small_s_pool_caps_samples_and_interrupted_preparation_can_resume(tmp_path, monkeypatch):
    config, machine, repo, _, rendered, acquired = _arrange(tmp_path, monkeypatch, ("S-1", "S-N"))
    load = setup.load_dataset

    def small_pool(source, pool, *, split):
        rows = load(source, pool, split=split)
        return rows.select(range(3)) if split == "private" else rows

    monkeypatch.setattr(setup, "load_dataset", small_pool)
    original = setup.render_task

    def interrupt(*args):
        if rendered:
            raise InterruptedError("render interrupted")
        return original(*args)

    monkeypatch.setattr(setup, "render_task", interrupt)
    with pytest.raises(InterruptedError, match="render interrupted"):
        setup.prepare(config, tmp_path / "run", repo_root=repo, machine=machine)
    assert len(rendered) == 1
    assert (tmp_path / "run" / "image.json").exists()
    for split in ("private", "calibration", "quality"):
        assert len(_read(tmp_path / "run" / "input" / "C2" / "S" / split)) == 3
    assert not (tmp_path / "run" / "prepared.json").exists()
    retained = tmp_path / "run" / "models" / "C2" / "source" / "retained.json"
    retained.write_text("{}")
    monkeypatch.setattr(setup, "render_task", original)
    result = setup.prepare(config, tmp_path / "run", repo_root=repo, machine=machine)
    assert len(result.task_paths) == 2 and all(path.is_dir() for path in result.task_paths)
    assert (tmp_path / "run" / "prepared.json").is_file()
    assert retained.read_text() == "{}"
