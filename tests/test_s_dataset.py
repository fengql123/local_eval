"""Length-stratified S views, opaque public rows and sandbox payloads without sources."""

import csv
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers

from eval import s_dataset
from eval.tasks import HOST_ONLY, select_image

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = {"B1": 100, "B2": 1000, "B3": 3000, "B4": 6000, "B5": 12000, "B6": 20000}


class Words:
    """A reference tokenizer stand-in: one token per whitespace-separated word."""

    def encode(self, text):
        return SimpleNamespace(ids=text.split())


def prompt(words, tag):
    return [{"role": "user", "content": f"{tag} " + "w " * (words - 1)}]


def pool(per_bin=12):
    rows = []
    for name, words in LENGTHS.items():
        for item in range(per_bin):
            source = {"B1": "math", "B5": "doc", "B6": "doc"}.get(name, "chat")
            rows.append(
                {
                    "source": f"{source}-source",
                    "item": f"{name}-{item}",
                    "category": source,
                    "messages": prompt(words + item, f"{name}-{item}"),
                    "evaluation": {"method": "letter", "answer": "A"} if source == "doc" else None,
                }
            )
    return rows


def authored():
    return [
        {"category": "math", "messages": prompt(words + extra, f"public-{name}-{extra}")}
        for name, words in LENGTHS.items()
        for extra in range(3 if name in ("B1", "B2", "B3", "B4") else 2)
    ]


def test_private_rows_are_sampled_by_length_and_cycle_through_bins():
    excluded = s_dataset.identity(pool()[0]["messages"])
    selected = s_dataset.stratify(pool(), Words(), seed=7, exclude={excluded})
    assert len(selected) == 64 and Counter(row["bin"] for row in selected) == dict(
        zip(LENGTHS, (11, 11, 11, 11, 10, 10), strict=True)
    )
    assert [row["bin"] for row in selected[:7]] == [*LENGTHS, "B1"]
    assert all(row["item"] != "B1-0" for row in selected)
    assert selected == s_dataset.stratify(pool(), Words(), seed=7, exclude={excluded})
    assert selected != s_dataset.stratify(pool(), Words(), seed=8, exclude={excluded})
    with pytest.raises(ValueError, match="too few"):
        s_dataset.stratify(pool(per_bin=9), Words(), seed=7)


def test_source_views_keep_every_selected_prompt_without_calibration():
    selected = s_dataset.stratify(pool(), Words(), seed=7)
    views = s_dataset.private_views(selected, models=("Qwen3.6-35B-A3B", "MiniMax-M3"))
    one = [row for row in views["private"] if row["model"] == "MiniMax-M3"]
    assert len(one) == 64 and len({row["id"] for row in one}) == 64
    assert set(views) == {"private"}
    assert all("calibration" not in row for row in one)
    request = one[0]["requests"][0]
    assert request["out"] == 32768 and json.loads(request["messages"])[0]["role"] == "user"
    # Timing-only prompts need no answer key.
    assert {row["evaluation"] is None for row in one} == {True, False}
    with pytest.raises(ValueError, match="per-bin quotas"):
        s_dataset.private_views(selected[:-1])


def test_public_rows_reveal_no_source():
    public = s_dataset.public_view(authored(), Words(), models=("MiniMax-M3",))
    assert [row["id"] for row in public] == [f"S.pub.{n:02d}" for n in range(1, 17)]
    assert {row["source"] for row in public} == {"math"} and public[0]["item"] is None
    assert Counter(row["bin"] for row in public) == dict(
        zip(LENGTHS, (3, 3, 3, 3, 2, 2), strict=True)
    )
    with pytest.raises(ValueError, match="per-bin quotas"):
        s_dataset.public_view(authored()[:-1], Words())
    with pytest.raises(ValueError, match="category"):
        s_dataset.public_view([{**authored()[0], "category": "aime"}], Words())


def test_builder_writes_views_and_a_reproducible_manifest(tmp_path):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "w": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    for name, rows in (("pool.jsonl", pool()), ("public.jsonl", authored())):
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "views"
    s_dataset.main(
        [
            str(tmp_path / "pool.jsonl"),
            str(tmp_path / "public.jsonl"),
            "--tokenizer",
            str(tmp_path / "tokenizer.json"),
            "--seed",
            "3",
            "--output",
            str(output),
        ]
    )
    summary = json.loads((output / "manifest.json").read_text())
    assert summary["seed"] == 3 and summary["private_bins"] == dict(
        zip(LENGTHS, (11, 11, 11, 11, 10, 10), strict=True)
    )
    assert summary["rows"]["private"] == 64 * 7 and summary["rows"]["public"] == 16 * 7
    # The same per-model layout and schema as the prepared dataset.
    assert summary["counts"]["MiniMax-M3"] == {
        "private/S": 64,
        "public/S": 16,
    }
    records = load_dataset(
        str(output), "workloads", split="records", cache_dir=tmp_path / "hf-cache"
    )
    private = records.filter(lambda r: r["model"] == "MiniMax-M3" and "S-1" in r["private"])
    public = records.filter(lambda r: r["model"] == "MiniMax-M3" and "S-1" in r["public"])
    assert {row["id"] for row in public}.isdisjoint({row["id"] for row in private})
    assert private.features == s_dataset.features()
    assert private.column_names == ["id", "model", "requests", "public", "private"]


def test_sandbox_payload_names_no_private_source():
    with (ROOT / "tables/data.csv").open(newline="") as handle:
        names = {
            value.lower()
            for row in csv.DictReader(handle)
            for value in (row["source"], row["data_source"], row["source_id"].split("/")[-1])
        }
    names |= {"wildchat", "longbench"}
    payload = select_image(ROOT, "nvidia-cuda-x86", ())
    assert not HOST_ONLY & set(payload.manifest["payload"]["sources"])
    for source in payload.manifest["payload"]["sources"]:
        text = (ROOT / source).read_text(errors="replace").lower()
        assert not [name for name in names if name in text], source
