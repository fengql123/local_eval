"""Baseline mappings: two common selections and two empty-selection edges."""

import json

from eval.baselines import mapping, mappings, needed, resolve_launch


def test_mapping_supplies_source_and_baseline_checkpoints(tmp_path):
    item = mapping("nvidia-cuda-x86", "C2", "freetoken")
    assert item.source_model_checkpoint == "Qwen/Qwen3.6-35B-A3B"
    assert item.checkpoint == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert str(item.launch_script) == "images/nvidia-cuda-x86/baselines/freetoken/launch.sh"
    assert mappings("nvidia-cuda-x86", item.model)[0] == item
    launch = tmp_path / item.launch_script
    launch.parent.mkdir(parents=True)
    launch.write_text('exec engine serve "$CHECKPOINT" --port 8000\n')
    assert resolve_launch(launch, "C2", "S-1") == ("freetoken", item.checkpoint)


def test_needed_deduplicates_shared_engines_and_copied_template_uses_task_reference(tmp_path):
    assert needed("nvidia-cuda-x86", ("C2", "C1")) == ("freetoken", "llama.cpp", "ollama")
    assert needed("apple-silicon", ("C2", "C1")) == ("llama.cpp", "ollama", "mlx-serve")
    launch = tmp_path / "launch.sh"
    launch.write_text('exec engine serve "$CHECKPOINT" --port 8000\n')
    task = {
        "category": "nvidia-cuda-x86",
        "model": "C2",
        "workload": "A-N",
        "assets": {"baselines": {"freetoken": {"checkpoint": "/models/nvfp4"}}},
    }
    assert resolve_launch(launch, "C2", "A-N", task=task, reference={"baseline": "freetoken"}) == (
        "freetoken",
        "/models/nvfp4",
    )
    selected = {"baseline": "freetoken", "checkpoint": "/models/reference"}
    assert resolve_launch(launch, "C2", "A-N", task=task, reference=selected) == (
        "freetoken",
        "/models/reference",
    )


def test_category_without_rows_and_complete_candidate_need_no_mapping(tmp_path):
    assert mappings("no-device", "C2") == ()
    assert needed("no-device", ("C2",)) == ()
    launch = tmp_path / "launch.sh"
    launch.write_text("exec engine serve /workspace/weights/optimized --port 8000 --workers 4\n")
    assert resolve_launch(launch, "C2", "A-MA") == ("candidate", None)


def test_model_without_category_mapping_and_native_checkpoint_identifier(tmp_path, monkeypatch):
    assert mappings("nvidia-cuda-x86", "N4") == ()
    assert needed("nvidia-cuda-x86", ()) == ()
    launch = tmp_path / "launch.sh"
    launch.write_text('exec engine serve "${CHECKPOINT}" --port 8000\n')
    native = mapping("nvidia-cuda-x86", "C2", "ollama").checkpoint
    (tmp_path / "reference.json").write_text(
        json.dumps(
            {
                "category": "nvidia-cuda-x86",
                "baseline": "ollama",
                "checkpoint": native,
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    assert resolve_launch(launch, "C2", "S-N") == ("ollama", native)
