"""Baseline mappings: two common selections and two empty-selection edges."""

from eval.baselines import mapping, mappings, needed


def test_mapping_supplies_source_and_baseline_checkpoints():
    item = mapping("nvidia-cuda-x86", "C2", "freetoken")
    assert item.source_model_checkpoint == "Qwen/Qwen3.6-35B-A3B"
    assert item.checkpoint == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert mappings("nvidia-cuda-x86", item.model)[0] == item


def test_needed_deduplicates_shared_engines_in_table_order():
    assert needed("nvidia-cuda-x86", ("C2", "C1")) == ("freetoken", "llama.cpp", "ollama")


def test_category_without_rows_selects_no_engines():
    assert mappings("no-device", "C2") == ()
    assert needed("no-device", ("C2",)) == ()


def test_model_without_category_mapping_selects_no_engines():
    assert mappings("nvidia-cuda-x86", "N4") == ()
    assert needed("nvidia-cuda-x86", ()) == ()
