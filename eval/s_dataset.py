"""Build length-stratified S views: 64 private rows and a public set that reveals no source.

Prompts are binned by length under one fixed reference tokenizer, so every TTFT
range carries the same weight in SLO placement. Private rows keep their source
for audit; public rows get opaque IDs and a generic category instead.

Pool and authored rows are JSON lines with ``source``, ``item``, ``category``
(math, chat or document), ``messages`` and an optional ``evaluation``. Timing
rows need no answer key: answer quality is checked by the separate quality suite.
This is maintainer tooling; it runs on the host and never ships to sandboxes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter
from pathlib import Path

from .config import OUTPUT_ALLOWANCE
from .config import models as catalog_models

# Log-spaced prompt-length bins (reference-tokenizer tokens), lower bound inclusive.
BINS = (
    ("B1", 0, 512),
    ("B2", 512, 2048),
    ("B3", 2048, 4096),
    ("B4", 4096, 8192),
    ("B5", 8192, 16384),
    ("B6", 16384, 32768),
)
PRIVATE_PER_BIN = (11, 11, 11, 11, 10, 10)
PUBLIC_PER_BIN = (3, 3, 3, 3, 2, 2)
CATEGORIES = ("math", "chat", "document")


def features():
    """The prepared S schema, plus each row's length bin and prompt tokens."""
    from datasets import Features, List, Value

    return Features(
        {
            "id": Value("string"),
            "model": Value("string"),
            "public": List(Value("string")),
            "private": List(Value("string")),
            "requests": List(
                {
                    "id": Value("int32"),
                    "parents": List(Value("int32")),
                    "tool": Value("float64"),
                    "out": Value("int32"),
                    "messages": Value("string"),
                    "tools": Value("string"),
                }
            ),
        }
    )


def prompt_tokens(messages: list[dict], tokenizer) -> int:
    """Prompt length under the reference tokenizer, without any chat template."""
    text = "\n".join(str(message.get("content") or "") for message in messages)
    return len(tokenizer.encode(text).ids)


def length_bin(tokens: int) -> str | None:
    return next((name for name, low, high in BINS if low <= tokens < high), None)


def identity(messages: list[dict]) -> str:
    """The prompt identity the quality suite uses to reject overlap."""
    from .quality import prompt_identity

    return prompt_identity({"requests": [{"messages": json.dumps(messages)}]})


def _interleave(selected: dict[str, list[dict]]) -> list[dict]:
    layers = itertools.zip_longest(*(selected[name] for name, _, _ in BINS if name in selected))
    return [row for layer in layers for row in layer if row is not None]


def stratify(
    pool: list[dict],
    tokenizer,
    *,
    seed: int,
    per_bin: int | None = None,
    exclude: set[str] = frozenset(),
) -> list[dict]:
    """Sample ``per_bin`` rows uniformly within each length bin, ordered across bins."""
    candidates: dict[str, list[dict]] = {name: [] for name, _, _ in BINS}
    seen = set(exclude)
    for row in sorted(pool, key=lambda item: (item["source"], str(item["item"]))):
        key = identity(row["messages"])
        tokens = prompt_tokens(row["messages"], tokenizer)
        name = length_bin(tokens)
        if name is None or key in seen:
            continue
        seen.add(key)
        candidates[name].append({**row, "bin": name, "prompt_tokens": tokens})
    quotas = dict(
        zip(candidates, PRIVATE_PER_BIN if per_bin is None else [per_bin] * len(BINS), strict=True)
    )
    short = {name: len(rows) for name, rows in candidates.items() if len(rows) < quotas[name]}
    if short:
        raise ValueError(f"too few distinct prompts for per-bin quotas: {short}")
    generator = random.Random(seed)
    selected = {name: generator.sample(rows, quotas[name]) for name, rows in candidates.items()}
    return _interleave(selected)


def _row(entry: dict, model: str, *, row_id: str, canary: str | None) -> dict:
    evaluation = entry.get("evaluation")
    request = {
        "id": 0,
        "parents": [],
        "tool": 0.0,
        "out": OUTPUT_ALLOWANCE,
        "messages": json.dumps(entry["messages"], ensure_ascii=False),
        "tools": None,
    }
    return {
        "id": row_id,
        "model": model,
        "source": entry["source"],
        "item": entry.get("item"),
        "harness": None,
        "evaluation": json.dumps(evaluation) if evaluation is not None else None,
        "requests": [request],
        "extra": json.dumps({"request_extra": [{}]}),
        "canary": canary,
        "bin": entry["bin"],
        "prompt_tokens": entry["prompt_tokens"],
    }


def private_views(
    selected: list[dict], *, models=None, canary: str | None = None
) -> dict[str, list[dict]]:
    """Expand the source prompt selection per model without a calibration split."""
    models = models if models is not None else tuple(m.name for m in catalog_models())
    counts = Counter(row["bin"] for row in selected)
    if counts != dict(zip((name for name, _, _ in BINS), PRIVATE_PER_BIN, strict=True)):
        raise ValueError(f"private S needs per-bin quotas {PRIVATE_PER_BIN}, not {dict(counts)}")
    private = [
        _row(
            row,
            model,
            row_id=f"S.{row['source']}.{row['item']}",
            canary=canary,
        )
        for model in models
        for row in selected
    ]
    return {"private": private}


def public_view(
    authored: list[dict], tokenizer, *, models=None, canary: str | None = None
) -> list[dict]:
    """Authored rows, with fixed per-bin quotas, with opaque IDs and only a generic category."""
    models = models if models is not None else tuple(m.name for m in catalog_models())
    rows = []
    for row in authored:
        if row.get("category") not in CATEGORIES:
            raise ValueError(f"public rows need a category in {CATEGORIES}")
        tokens = prompt_tokens(row["messages"], tokenizer)
        rows.append({**row, "bin": length_bin(tokens), "prompt_tokens": tokens})
    counts = Counter(row["bin"] for row in rows)
    expected = dict(zip((name for name, _, _ in BINS), PUBLIC_PER_BIN, strict=True))
    if counts != expected:
        raise ValueError(f"public S needs per-bin quotas {PUBLIC_PER_BIN}, not {dict(counts)}")
    grouped = {name: [row for row in rows if row["bin"] == name] for name, _, _ in BINS}
    ordered = _interleave(grouped)
    # The generic category replaces the source, and there is no item.
    return [
        _row(
            {**row, "source": row["category"], "item": None},
            model,
            row_id=f"S.pub.{number:02d}",
            canary=canary,
        )
        for model in models
        for number, row in enumerate(ordered, start=1)
    ]


def write_views(
    views: dict[str, list[dict]], output: Path, *, other_pools: Path | None = None
) -> dict[str, dict[str, int]]:
    """Write the workloads configuration, preserving other pools from this HF release."""
    from datasets import Dataset, load_dataset
    from huggingface_hub import DatasetCard

    counts = {}
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, entries in views.items():
        for row in entries:
            rows.append(
                {
                    **{key: row[key] for key in ("id", "model", "requests")},
                    "public": ["S-1", "S-N"] if split == "public" else [],
                    "private": ["S-1", "S-N"] if split == "private" else [],
                }
            )
            counts.setdefault(row["model"], {}).setdefault(f"{split}/S", 0)
            counts[row["model"]][f"{split}/S"] += 1
    if other_pools:
        rows.extend(
            row
            for row in load_dataset(str(other_pools), "workloads", split="records")
            if not row["id"].startswith("S.")
        )
    folder = output / "workloads"
    folder.mkdir(exist_ok=True)
    for model in dict.fromkeys(row["model"] for row in rows):
        selected = [row for row in rows if row["model"] == model]
        Dataset.from_list(selected, features=features()).to_parquet(
            str(folder / f"{model}.parquet")
        )
    readme = output / "README.md"
    card = (
        DatasetCard.load(readme)
        if readme.exists()
        else DatasetCard("---\nconfigs: []\n---\n# Benchmark dataset\n")
    )
    configs = [
        entry
        for entry in card.data.to_dict().get("configs", [])
        if entry["config_name"] != "workloads"
    ]
    configs.insert(
        0,
        {
            "config_name": "workloads",
            "default": True,
            "data_files": [{"split": "records", "path": "workloads/*.parquet"}],
        },
    )
    card.data["configs"] = configs
    card.save(readme)
    return counts


def manifest(views: dict[str, list[dict]], *, seed: int, tokenizer: str) -> dict:
    """Record how the views were drawn so a new version can be reproduced."""
    private = [row for row in views["private"] if row["model"] == views["private"][0]["model"]]
    return {
        "tokenizer": tokenizer,
        "seed": seed,
        "bins": [{"name": name, "min_tokens": low, "max_tokens": high} for name, low, high in BINS],
        "rows": {split: len(rows) for split, rows in views.items()},
        "private_sources": dict(Counter(row["source"] for row in private)),
        "private_bins": dict(Counter(row["bin"] for row in private)),
    }


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> None:
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool", type=Path, help="JSON lines of candidate private prompts")
    parser.add_argument("public", type=Path, help="JSON lines of 16 authored public prompts")
    parser.add_argument("--tokenizer", required=True, help="reference tokenizer.json or HF ID")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[], help="JSON lines to avoid")
    parser.add_argument("--canary", help="canary string stored in every row")
    parser.add_argument(
        "--other-pools", type=Path, help="prepared dataset whose A and A-MA views are copied"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    tokenizer = (
        Tokenizer.from_file(args.tokenizer)
        if Path(args.tokenizer).is_file()
        else Tokenizer.from_pretrained(args.tokenizer)
    )
    authored = _jsonl(args.public)
    exclude = {identity(row["messages"]) for row in authored}
    for path in args.exclude:
        exclude.update(identity(row["messages"]) for row in _jsonl(path))
    selected = stratify(_jsonl(args.pool), tokenizer, seed=args.seed, exclude=exclude)
    views = private_views(selected, canary=args.canary)
    views["public"] = public_view(authored, tokenizer, canary=args.canary)
    counts = write_views(views, args.output, other_pools=args.other_pools)
    summary = {**manifest(views, seed=args.seed, tokenizer=args.tokenizer), "counts": counts}
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
