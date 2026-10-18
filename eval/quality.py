"""Prepare an independent, hidden quality suite without changing workload datasets.

Input pools use the normal requests/evaluation columns plus category, source and
item. A selection names the reviewed IDs; pilot results help choose those IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import Dataset, load_dataset, load_from_disk

from .config import TABLES
from .config import model as get_model

MIX = {"math": 8, "science": 8, "code": 6, "long_context": 8}
METHODS = {
    "math": {"math_verify"},
    "science": {"letter"},
    "code": {"code"},
    "long_context": {"letter", "exact"},
}


def prompt_identity(row: dict) -> str:
    def normalize(value):
        if isinstance(value, str):
            return re.sub(r"\s+", " ", value).strip()
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        return value

    value = normalize(json.loads(row["requests"][0]["messages"]))
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def validate(rows: list[dict]) -> None:
    """Enforce the selected suite, not the size or distribution of any S workload."""
    if Counter(row["category"] for row in rows) != MIX:
        raise ValueError(f"quality requires the standalone mix {MIX}")
    if len({row["id"] for row in rows}) != len(rows) or len(
        {prompt_identity(r) for r in rows}
    ) != len(rows):
        raise ValueError("quality questions must have distinct IDs and prompts")
    for row in rows:
        requests = row["requests"]
        if len(requests) != 1 or requests[0]["parents"] or requests[0]["tools"]:
            raise ValueError("quality questions must be independent single-turn requests")
        if json.loads(row["evaluation"])["method"] not in METHODS[row["category"]]:
            raise ValueError(f"unexpected checker for quality category {row['category']}")


def prepare(pool: Dataset, selected: list[str], output: Path, workloads: Path) -> dict:
    """Freeze approved questions, rejecting overlap without rewriting any workload."""
    by_id = {row["id"]: row for row in pool}
    rows = [by_id[key] for key in selected]
    validate(rows)
    exposed, origins = set(), set()
    with TemporaryDirectory() as cache:
        for split in ("records",):
            for row in load_dataset(str(workloads), "workloads", split=split, cache_dir=cache):
                for request in row["requests"]:
                    exposed.add(prompt_identity({"requests": [request]}))
                if "source" in row and "item" in row:
                    origins.add((row["source"], row["item"]))
    overlap = [
        row["id"]
        for row in rows
        if prompt_identity(row) in exposed or (row["source"], row["item"]) in origins
    ]
    if overlap:
        raise ValueError(f"quality questions overlap workload rows: {overlap}")
    from .verify.profile import questions_identity

    manifest = {
        "schema": 1,
        "distribution": dict(Counter(row["category"] for row in rows)),
        "question_ids": selected,
        "questions_identity": questions_identity(rows),
        "workloads": str(workloads),
        "selection": "explicit IDs; profile with fresh rounds after selection",
    }
    from huggingface_hub import DatasetCard

    (output / "quality").mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(output / "quality/questions.parquet"))
    readme = output / "README.md"
    card = (
        DatasetCard.load(readme)
        if readme.exists()
        else DatasetCard("---\nconfigs: []\n---\n# Benchmark dataset\n")
    )
    configs = [
        entry
        for entry in card.data.to_dict().get("configs", [])
        if entry["config_name"] != "quality"
    ]
    configs.append(
        {
            "config_name": "quality",
            "data_files": [{"split": "questions", "path": "quality/questions.parquet"}],
        }
    )
    card.data["configs"] = configs
    card.save(readme)
    (output / "quality-selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def map_suite(source: str, destination: Path) -> dict:
    """Export the hidden quality configuration from the same HF dataset release."""
    questions = destination / "questions"
    partial = destination / "questions.partial"
    if not all((questions / name).is_file() for name in ("state.json", "dataset_info.json")):
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        for folder in (questions, partial):
            if folder.exists():
                shutil.rmtree(folder)
        load_dataset(
            source, "quality", split="questions", cache_dir=destination.parent / "hf-cache"
        ).save_to_disk(str(partial))
        partial.rename(questions)
    return {"source": str(destination.resolve()), "dataset": source}


def model_range(model: str, rows: list[dict], *, table: Path | None = None) -> dict:
    """Read the one benchmark bound for a model, shared across its baselines."""
    from .request_policy import request_fields
    from .verify.profile import questions_identity

    table = table or TABLES / "quality.csv"
    with table.open(newline="") as handle:
        row = next(row for row in csv.DictReader(handle) if row["model"] == model)
    if row["checkpoint"] != get_model(model).checkpoint:
        raise ValueError(f"quality checkpoint differs from models.csv for {model}")
    if json.loads(row["policy"]) != request_fields(model).fixed("S-1"):
        raise ValueError(f"quality sampling policy differs from the measured range for {model}")
    if row["questions_identity"] != questions_identity(rows):
        raise ValueError(f"quality questions differ from the measured range for {model}")
    rounds = json.loads(row["rounds"])
    if (
        len(rounds) != 32
        or min(rounds) != int(row["min_correct"])
        or max(rounds) != int(row["max_correct"])
    ):
        raise ValueError(f"quality table must contain a complete 32-round range for {model}")
    return {
        **row,
        "id": f"{model}:{row['questions_identity']}",
        "question_ids": [r["id"] for r in rows],
        "min_correct": int(row["min_correct"]),
        "max_correct": int(row["max_correct"]),
        "rounds": rounds,
    }


def publish(profile_path: Path, dataset: str, *, table: Path | None = None) -> Path:
    """Update the model's table row after validating a complete offline profile."""
    from .request_policy import request_fields
    from .verify.profile import validate_profile

    profile = json.loads(profile_path.read_text())
    rows = load_dataset(dataset, "quality", split="questions").to_list()
    validate_profile(
        profile, get_model(profile["model"]).checkpoint, request_fields(profile["model"]), rows
    )
    row = {
        "model": profile["model"],
        "checkpoint": profile["checkpoint"],
        "questions": len(rows),
        "questions_identity": profile["questions_identity"],
        "min_correct": profile["min_correct"],
        "max_correct": max(entry["correct"] for entry in profile["rounds"]),
        "rounds": json.dumps([entry["correct"] for entry in profile["rounds"]]),
        "policy": json.dumps(profile["policy"], sort_keys=True),
    }
    table = table or TABLES / "quality.csv"
    existing = list(csv.DictReader(table.open(newline=""))) if table.exists() else []
    entries = [entry for entry in existing if entry["model"] != profile["model"]] + [row]
    table.parent.mkdir(parents=True, exist_ok=True)
    partial = table.with_suffix(".partial")
    with partial.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(entries)
    partial.replace(table)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare", help="freeze reviewed IDs from a prepared question pool")
    build.add_argument("pool", help="HF repository (questions split) or absolute saved dataset")
    build.add_argument(
        "--selection", type=Path, required=True, help="JSON list of reviewed question IDs"
    )
    build.add_argument(
        "--workloads", type=Path, required=True, help="prepared workload root to exclude"
    )
    build.add_argument("--output", type=Path, required=True)
    release = commands.add_parser("publish", help="release a complete offline profile")
    release.add_argument("profile", type=Path)
    release.add_argument("--dataset", required=True)
    release.add_argument("--table", type=Path, default=TABLES / "quality.csv")
    args = parser.parse_args()
    if args.command == "publish":
        print(publish(args.profile, args.dataset, table=args.table))
    else:
        pool = (
            load_from_disk(args.pool)
            if Path(args.pool).is_absolute()
            else load_dataset(args.pool, split="questions")
        )
        result = prepare(pool, json.loads(args.selection.read_text()), args.output, args.workloads)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
