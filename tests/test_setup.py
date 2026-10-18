"""Prepared HF inputs: two common cases and two preparation boundaries."""

import asyncio
import json
from dataclasses import replace

import pytest
from datasets import Dataset, load_from_disk
from test_quality_suite import questions

from eval import setup
from eval.config import parse_config

MODEL = "Qwen3.6-35B-A3B"


def arrange(tmp_path, monkeypatch, workloads=("A-1", "A-MA"), *, private_s_rows=8):
    repo = tmp_path / "checkout"
    device = repo / "images/nvidia-cuda-x86"
    device.mkdir(parents=True)
    (device / "Dockerfile").write_text("FROM ubuntu:22.04 AS toolchain\n")
    for phase in ("search", "coding", "verification"):
        (device / f"{phase}.Dockerfile").write_text("FROM toolchain\n")
    for engine in ("freetoken", "llama.cpp", "ollama"):
        folder = device / "baselines" / engine
        folder.mkdir(parents=True)
        for script in ("install.sh", "launch.sh"):
            (folder / script).write_text("#!/bin/sh\n")
    config = parse_config(
        {
            "dataset": "owner/prepared",
            "search": {"agent": {"name": "codex", "kwargs": {"reasoning_effort": "xhigh"}}},
            "evaluation": {"agent": {"name": "codex", "kwargs": {"reasoning_effort": "xhigh"}}},
            "concurrency": 2,
            "tasks": {"models": [MODEL], "workloads": list(workloads)},
        },
        base=tmp_path,
    )
    calls = []

    def load(source, name, split, **kwargs):
        calls.append((name, split))
        return Dataset.from_list(
            [
                {
                    **row,
                    "public": row["selection"]["workloads"] if role == "public" else [],
                    "private": row["selection"]["workloads"] if role == "private" else [],
                }
                for role in ("public", "private")
                for row in split_rows(role)
            ]
        ).remove_columns(["pool", "selection"])

    def split_rows(split):
        return Dataset.from_list(
            [
                {
                    "id": f"{pool}-{split}-{i}",
                    "model": MODEL,
                    "pool": pool,
                    "selection": {
                        "version": "stratified-v3",
                        "workloads": (
                            ["S-1", "S-N"]
                            if pool == "S"
                            else ["A-MA"]
                            if pool == "A-MA"
                            else ["A-1"]
                            if i < (10 if split == "public" else 8)
                            else ["A-N"]
                        ),
                    },
                    "requests": [],
                    "extra": "unchanged",
                }
                for pool in ("S", "A", "A-MA")
                for i in range(
                    private_s_rows
                    if pool == "S" and split == "private"
                    else 20
                    if pool == "A" and split == "public"
                    else 16
                    if pool == "A"
                    else 10
                    if split == "public"
                    else 8
                )
            ]
        )

    monkeypatch.setattr(setup, "load_dataset", load)
    monkeypatch.setattr("eval.quality.load_dataset", lambda *a, **k: Dataset.from_list(questions()))
    return config, repo, calls


def test_local_prepared_model_splits_are_reused_without_copy_or_sampling(tmp_path, monkeypatch):
    config, repo, calls = arrange(tmp_path, monkeypatch, workloads=("A-1", "A-N", "A-MA"))
    result = asyncio.run(setup.prepare(config, tmp_path / "run", repo_root=repo))
    assert calls == [("workloads", "records")]
    assert (
        load_from_disk(result.inputs[MODEL]["A-MA"]["public"]["source"])[0]["id"] == "A-MA-public-0"
    )
    assert not tuple((tmp_path / "run").rglob("*.jsonl"))
    assert (tmp_path / "run/prepared.json").exists()
    metadata = json.loads(
        (
            tmp_path / "run/tasks/search" / result.task_paths[0].name / "environment/task.json"
        ).read_text()
    )
    assert metadata["device_category"] == "nvidia-cuda-x86" and "machine" not in metadata
    assert "public_s" not in metadata["inputs"]
    a1 = load_from_disk(result.inputs[MODEL]["A-1"]["private"]["source"])
    an = load_from_disk(result.inputs[MODEL]["A-N"]["private"]["source"])
    assert len(a1) == len(an) == 8
    assert a1[0]["id"] == "A-private-0" and an[0]["id"] == "A-private-8"
    assert (
        result.inputs[MODEL]["A-1"]["private"]["identity"]
        != (result.inputs[MODEL]["A-N"]["private"]["identity"])
    )


def test_cloud_maps_selected_hf_views_on_coordinator_without_weight_acquisition(
    tmp_path, monkeypatch
):
    config, repo, calls = arrange(tmp_path, monkeypatch)
    config = replace(
        config,
        environment=replace(
            config.environment,
            provider="daytona",
            kwargs={"gpus": 1, "gpu_types": ["RTX-4090"]},
        ),
    )

    result = asyncio.run(setup.prepare(config, tmp_path / "run", repo_root=repo))
    assert calls == [("workloads", "records")]
    assert "S" not in result.inputs[MODEL] and "quality" not in result.inputs[MODEL]
    assert not (tmp_path / "run/assets").exists()
    assert not (tmp_path / "run/build-context/setup.Dockerfile").exists()


def test_missing_baseline_script_stops_before_acquisition(tmp_path, monkeypatch):
    config, repo, calls = arrange(tmp_path, monkeypatch)
    (repo / "images/nvidia-cuda-x86/baselines/freetoken/launch.sh").unlink()
    with pytest.raises(FileNotFoundError, match="launch.sh"):
        asyncio.run(setup.prepare(config, tmp_path / "run", repo_root=repo))
    assert not calls and not (tmp_path / "run/prepared.json").exists()


def test_official_concurrency_only_constrains_private_batches(tmp_path, monkeypatch):
    for count in (32, 40):
        folder = tmp_path / str(count)
        config, repo, _ = arrange(
            folder, monkeypatch, workloads=("S-N", "A-N"), private_s_rows=count
        )
        config = replace(
            config,
            evaluation=replace(config.evaluation, concurrency={"S-N": 32, "A-N": 8}),
        )
        if count == 40:
            with pytest.raises(ValueError, match="S-N/private"):
                asyncio.run(setup.prepare(config, folder / "run", repo_root=repo))
            continue
        result = asyncio.run(setup.prepare(config, folder / "run", repo_root=repo))
        assert [
            json.loads(
                (folder / "run/tasks/search" / task.name / "environment/task.json").read_text()
            )["evaluation"]["concurrency"]
            for task in result.task_paths
        ] == [32, 8]


def test_interrupted_preparation_reuses_frozen_batches_and_mapped_inputs(tmp_path, monkeypatch):
    config, repo, calls = arrange(tmp_path, monkeypatch)
    original = setup.map_inputs

    def interrupt(*args, **kwargs):
        raise InterruptedError("input mapping interrupted")

    monkeypatch.setattr(setup, "map_inputs", interrupt)
    with pytest.raises(InterruptedError):
        asyncio.run(setup.prepare(config, tmp_path / "run", repo_root=repo))
    assert not (tmp_path / "run/prepared.json").exists()
    monkeypatch.setattr(setup, "map_inputs", original)
    result = asyncio.run(setup.prepare(config, tmp_path / "run", repo_root=repo))
    observed = list(calls)
    setup_result = asyncio.run(setup.prepare(config, tmp_path / "run", repo_root=repo))
    assert calls == observed and setup_result.task_paths == result.task_paths
    assert (
        load_from_disk(result.inputs[MODEL]["A-1"]["private"]["source"])[0]["id"] == "A-private-0"
    )


def test_planning_keeps_all_types_unset_until_reference_resolution(tmp_path):
    repo = tmp_path / "checkout"
    for category in ("fixture-device", "amd-rocm-x86"):
        device = repo / "images" / category
        device.mkdir(parents=True)
        (device / "Dockerfile").write_text("FROM ubuntu:22.04 AS toolchain\n")
        for phase in ("search", "coding", "verification"):
            (device / f"{phase}.Dockerfile").write_text("FROM toolchain\n")
    for engine in ("llama.cpp", "ollama"):
        folder = repo / "images/amd-rocm-x86/baselines" / engine
        folder.mkdir(parents=True)
        for script in ("install.sh", "launch.sh"):
            (folder / script).write_text("#!/bin/sh\n")

    def plan(category, folder, **tasks):
        agent = {"name": "codex", "kwargs": {"reasoning_effort": "xhigh"}}
        settings = {
            "environment": {"device_category": category},
            "tasks": tasks,
            "search": {"agent": agent},
            "evaluation": {"agent": agent},
        }
        return setup.plan(parse_config(settings, base=tmp_path), tmp_path / folder, repo_root=repo)

    manifest = plan("fixture-device", "all")
    assert len(manifest["tasks"]) == 35
    assert all("type" not in task for task in manifest["tasks"])
    assert manifest["assets"]["MiniMax-M3"] == {
        "checkpoint": "nvidia/MiniMax-M3-NVFP4",
        "baselines": {},
    }
    mixed = plan("amd-rocm-x86", "mixed", models=["MiniMax-M3", MODEL], workloads=["S-1"])
    assert [task["model"] for task in mixed["tasks"]] == ["MiniMax-M3", MODEL]
    assert all("type" not in task for task in mixed["tasks"])
    assert not (tmp_path / "mixed/image.json").exists()
