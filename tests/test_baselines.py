"""Phase 1 baseline metadata: two common cases and two failure cases."""

from pathlib import Path

import pytest

from eval.baselines import mapping, needed, recipe, validate_catalog


def test_catalog_selects_shared_image_installers_and_deduplicates_engines():
    validate_catalog()
    engines = needed("nvidia-cuda-x86", ("C2", "C1"))
    assert [item.baseline for item in engines] == ["llama.cpp", "ollama", "freetoken"]
    freetoken = engines[-1]
    assert freetoken.prefix == Path("/opt/baselines/freetoken")
    assert freetoken.install_script == Path("images/nvidia-cuda-x86/baselines/freetoken/install.sh")
    assert freetoken.environment == "venv"
    assert freetoken.status == "pending-assets"
    model = mapping("nvidia-cuda-x86", "C2", "llama.cpp")
    assert (model.checkpoint_kind, model.checkpoint, model.file_pattern) == (
        "gguf",
        "ggml-org/Qwen3.6-35B-A3B-GGUF",
        "*Q4_K_M*.gguf",
    )


def test_launch_keeps_arguments_separate_and_accepts_environment_overrides():
    launch = recipe("freetoken", "nvidia-cuda-x86").launch(
        "/models/path with spaces", 8000, args=("--max-model-len", "4096")
    )
    assert launch.argv[3] == "/models/path with spaces"
    assert launch.argv[-2:] == ("--max-model-len", "4096")
    ollama = recipe("ollama", "nvidia-cuda-x86")
    native = ollama.launch(
        "qwen3.6:35b-a3b-mtp-q4_K_M",
        1919,
        env={"OLLAMA_HOST": "127.0.0.1:2222", "USER_TUNING_SETTING": "8192"},
    )
    assert native.argv == ("/opt/baselines/ollama/bin/ollama", "serve")
    assert native.env == {
        "OLLAMA_HOST": "127.0.0.1:2222",
        "OLLAMA_MODELS": "/models/ollama",
        "USER_TUNING_SETTING": "8192",
    }
    assert native.served_model == "qwen3.6:35b-a3b-mtp-q4_K_M"
    assert ollama.launch("another-model", 1919).env["OLLAMA_HOST"] == "127.0.0.1:1919"


def test_selection_reports_a_model_without_any_catalog_mapping():
    with pytest.raises(ValueError, match="DeepSeek-V4.1-Flash"):
        needed("nvidia-cuda-x86", ("C2", "N4"))


def test_planned_baseline_cannot_compose_a_launch_without_launch_metadata():
    item = recipe("lm-studio", "apple-silicon")
    assert item.install_script == Path("images/apple-silicon/baselines/lm-studio/install.sh")
    with pytest.raises(ValueError, match="planned"):
        item.launch("example", 8000)
