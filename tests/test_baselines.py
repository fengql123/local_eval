"""Baseline selection, installation and explicit acquired checkpoint paths."""

from pathlib import Path

import pytest

from eval.baselines import install, mapping, mappings, needed, resolve_launch


def recipe(root, name, body="echo installed >> calls.txt"):
    folder = root / "nvidia-cuda-x86" / name
    folder.mkdir(parents=True)
    (folder / "install.sh").write_text("#!/bin/bash\nset -eu\n" + body + "\n")
    (folder / "launch.sh").write_text('exec engine "$CHECKPOINT"\n')
    return folder


@pytest.mark.parametrize(
    "model,source,checkpoint",
    [
        ("Qwen3.8-Flash-Next", "Qwen/Qwen3.8-Flash-Next", "RadixArk/Qwen3.8-Flash-Next-NVFP4"),
        (
            "DeepSeek-V4.1-Flash",
            "deepseek-ai/DeepSeek-V4.1-Flash",
            "deepseek-ai/DeepSeek-V4.1-Flash",
        ),
    ],
)
def test_mapped_baseline_uses_explicit_local_checkpoint(
    tmp_path, monkeypatch, model, source, checkpoint
):
    item = mapping("nvidia-cuda-x86", model, "freetoken")
    assert item.checkpoint == checkpoint
    launch = tmp_path / item.launch_script
    assert resolve_launch(launch, item.model, "A-1") == ("freetoken", item.checkpoint)
    assert {
        entry.baseline: entry.checkpoint for entry in mappings("nvidia-cuda-x86", "MiniMax-M3")
    } == {
        "freetoken": "nvidia/MiniMax-M3-NVFP4",
        "llama.cpp": "AesSedai/MiniMax-M3-GGUF",
        "ollama": "AesSedai/MiniMax-M3-GGUF",
    }
    monkeypatch.setenv("CHECKPOINT", "/workspace/weights/fp4")
    assert resolve_launch(launch, item.model, "A-1") == ("freetoken", "/workspace/weights/fp4")


def test_selected_recipe_installs_once_and_carries_patches(tmp_path):
    recipes, workspace = tmp_path / "recipes", tmp_path / "baselines"
    original = recipe(recipes, "freetoken", "cat fix.patch >> calls.txt")
    (original / "fix.patch").write_text("installed\n")
    task = {
        "device_category": "nvidia-cuda-x86",
        "assets": {"baselines": {"freetoken": "owner/fp4"}},
    }
    install(task, recipes=recipes, workspace=workspace)
    install(task, recipes=recipes, workspace=workspace)
    assert (workspace / "nvidia-cuda-x86/freetoken/calls.txt").read_text() == "installed\n"
    assert needed("nvidia-cuda-x86", ("Qwen3.8-27B", "Qwen3.6-35B-A3B")) == (
        "freetoken",
        "llama.cpp",
        "ollama",
    )


def test_candidate_and_unmapped_device_do_not_guess_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKPOINT", "/unexpected/reference")
    for script in (tmp_path / "submission/launch.sh", Path("/launch.sh"), Path("/app/launch.sh")):
        assert resolve_launch(script, "Qwen3.8-27B", "A-MA") == ("candidate", None)
    assert mappings("unknown", "Qwen3.8-27B") == ()
    assert needed("nvidia-cuda-x86", ()) == ()


def test_partial_install_retries_but_failed_install_is_not_complete(tmp_path):
    recipes, workspace = tmp_path / "recipes", tmp_path / "baselines"
    original = recipe(
        recipes,
        "freetoken",
        "echo compiler-failure >&2; printf '%05000d\\n' 0 >&2; exit 17",
    )
    task = {
        "device_category": "nvidia-cuda-x86",
        "assets": {"baselines": {"freetoken": "owner/fp4"}},
    }
    target = workspace / "nvidia-cuda-x86/freetoken"
    target.mkdir(parents=True)
    (target / "partial").touch()
    with pytest.raises(RuntimeError, match="installation exited 17") as failure:
        install(task, recipes=recipes, workspace=workspace)
    assert "compiler-failure" in str(failure.value)
    assert not (target / ".installed").exists() and not (target / "partial").exists()
    (original / "install.sh").write_text("echo success > result\n")
    install(task, recipes=recipes, workspace=workspace)
    assert (target / "result").read_text() == "success\n"


def test_model_catalog_has_canonical_checkpoints_independent_of_device():
    from eval.config import model, models

    assert len(models()) == 7
    assert model("MiniMax-M3").checkpoint == "nvidia/MiniMax-M3-NVFP4"
    assert model("GLM-5.3-Flash").checkpoint == "RedHatAI/GLM-5.3-Flash-NVFP4"
    assert all(item.checkpoint for item in models())
