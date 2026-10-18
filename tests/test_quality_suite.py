"""One dataset release: two preparation paths and two hidden-suite boundaries."""

import copy
import json
from pathlib import Path

import pytest
from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import DatasetCard

from eval import quality


def questions():
    return [
        {
            "id": str(number),
            "category": category,
            "source": "fixture",
            "item": str(number),
            "evaluation": json.dumps(
                {"method": next(iter(sorted(quality.METHODS[category]))), "answer": "B"}
            ),
            "requests": [
                {
                    "id": 0,
                    "parents": [],
                    "out": 32768,
                    "tools": None,
                    "messages": json.dumps([{"role": "user", "content": str(number)}]),
                }
            ],
        }
        for number, category in enumerate(
            category for category, count in quality.MIX.items() for _ in range(count)
        )
    ]


def workloads(path, row=None):
    path.mkdir(parents=True, exist_ok=True)
    row = row or {
        **questions()[0],
        "source": "public",
        "item": "other",
        "requests": [
            {**questions()[0]["requests"][0], "messages": '[{"role":"user","content":"public"}]'}
        ],
    }
    Dataset.from_list([row]).to_parquet(str(path / "workload.parquet"))
    card = DatasetCard("---\nconfigs: []\n---\n# Fixture\n")
    card.data["configs"] = [
        {
            "config_name": "workloads",
            "data_files": [{"split": s, "path": "workload.parquet"} for s in ("records",)],
        }
    ]
    card.save(path / "README.md")
    return path


def test_quality_is_a_configuration_in_the_same_release(tmp_path):
    source = workloads(tmp_path / "dataset")
    before = (source / "workload.parquet").read_bytes()
    rows = questions()
    result = quality.prepare(Dataset.from_list(rows), [r["id"] for r in rows], source, source)
    assert result["distribution"] == quality.MIX
    assert (source / "workload.parquet").read_bytes() == before
    assert load_dataset(str(source), "quality", split="questions").to_list() == rows
    descriptor = quality.map_suite(str(source), tmp_path / "run/input/quality")
    assert load_from_disk(str(Path(descriptor["source"]) / "questions")).to_list() == rows


def test_hf_quality_mapping_selects_only_the_quality_configuration(tmp_path, monkeypatch):
    calls = []

    def load(source, config, *, split, **kwargs):
        calls.append((source, config, split))
        return Dataset.from_list(questions())

    monkeypatch.setattr(quality, "load_dataset", load)
    quality.map_suite("owner/benchmark", tmp_path / "view")
    quality.map_suite("owner/benchmark", tmp_path / "view")
    assert calls == [("owner/benchmark", "quality", "questions")]


def test_overlap_with_a_workload_prompt_is_rejected_before_publication(tmp_path):
    rows = questions()
    source = workloads(tmp_path / "dataset", rows[0])
    with pytest.raises(ValueError, match="overlap"):
        quality.prepare(
            Dataset.from_list(rows), [r["id"] for r in rows], tmp_path / "output", source
        )
    assert not (tmp_path / "output").exists()


def test_wrong_distribution_or_duplicate_prompts_cannot_form_a_suite():
    rows = questions()
    with pytest.raises(ValueError, match="mix"):
        quality.validate(rows[:-1])
    altered = copy.deepcopy(rows)
    altered[-1]["requests"] = altered[0]["requests"]
    with pytest.raises(ValueError, match="distinct"):
        quality.validate(altered)


@pytest.mark.parametrize("interrupted", ["questions", "questions.partial"])
def test_interrupted_quality_export_is_replaced_atomically(tmp_path, monkeypatch, interrupted):
    from datasets import Dataset, load_from_disk

    from eval.quality import map_suite

    partial = tmp_path / interrupted
    partial.mkdir()
    (partial / "unfinished").write_text("interrupted export")
    rows = questions()
    monkeypatch.setattr("eval.quality.load_dataset", lambda *a, **k: Dataset.from_list(rows))
    descriptor = map_suite("owner/prepared", tmp_path)
    assert descriptor["source"] == str(tmp_path)
    assert load_from_disk(str(tmp_path / "questions")).to_list() == rows
    assert not (tmp_path / "questions/unfinished").exists()
    assert not (tmp_path / "questions.partial").exists()
