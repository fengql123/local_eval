"""Build a private pilot pool from official sources; never modify S workloads.

Run as python -m examples.prepare_quality_pool. GPQA is skipped unless --gpqa
points to an authorized official gpqa_diamond.csv supplied locally.
Selection and 32-round profiling are separate, explicit steps.
"""

import argparse
import base64
import csv
import hashlib
import json
import pickletools
import random
import zlib
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download

from eval.quality import MIX


def row(source, item, category, prompt, method, answer):
    return {
        "id": f"quality.{source}.{item}",
        "source": source,
        "item": str(item),
        "category": category,
        "evaluation": json.dumps({"method": method, "answer": answer}),
        "requests": [
            {
                "id": 0,
                "parents": [],
                "tool": 0,
                "tools": None,
                "out": 32768,
                "messages": json.dumps([{"role": "user", "content": prompt}]),
            }
        ],
    }


def gpqa(records):
    for item in records:
        question = item["Question"].strip()
        key = hashlib.sha256(question.encode()).hexdigest()[:16]
        choices = [item["Correct Answer"], *(item[f"Incorrect Answer {n}"] for n in (1, 2, 3))]
        order = list(range(4))
        random.Random(key).shuffle(order)
        prompt = (
            question
            + "\n"
            + "\n".join(
                f"{letter}. {choices[index]}" for letter, index in zip("ABCD", order, strict=True)
            )
        )
        yield row(
            "gpqa_diamond",
            key,
            "science",
            prompt + "\nEnd with Final answer: LETTER.",
            "letter",
            "ABCD"[order.index(0)],
        )


def code_cases(encoded):
    """Read LiveCodeBench's pickled JSON string without executing a pickle."""
    operations = list(pickletools.genops(zlib.decompress(base64.b64decode(encoded))))
    safe = {"PROTO", "FRAME", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8", "MEMOIZE", "STOP"}
    if any(op.name not in safe for op, _, _ in operations):
        raise ValueError("expected LiveCodeBench's serialized JSON string")
    strings = [value for op, value, _ in operations if "UNICODE" in op.name]
    if len(strings) != 1:
        raise ValueError("expected one LiveCodeBench test-case JSON string")
    return json.loads(strings[0])


def livecodebench(records):
    for item in records:
        if item["difficulty"] not in ("medium", "hard") or item["starter_code"].strip():
            continue
        tests = json.loads(item["public_test_cases"]) + code_cases(item["private_test_cases"])
        if not tests or any(test["testtype"] != "stdin" for test in tests):
            continue
        yield row(
            "livecodebench",
            f"{item['platform']}-{item['question_id']}",
            "code",
            item["question_content"] + "\nWrite a complete Python 3 program using only the "
            "standard library, reading stdin and writing stdout. Return one python code block.",
            "code",
            {"tests": [{"input": case["input"], "output": case["output"]} for case in tests]},
        )


def longbench(records):
    for item in records:
        if item["difficulty"] != "hard" or len(item["context"]) > 100000:
            continue
        prompt = (
            item["context"]
            + "\n\n"
            + item["question"]
            + "\n"
            + "\n".join(f"{letter}. {item['choice_' + letter]}" for letter in "ABCD")
        )
        yield row(
            "longbench2",
            item["_id"],
            "long_context",
            prompt + "\nEnd with Final answer: LETTER.",
            "letter",
            item["answer"],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gpqa", type=Path, help="opt in with an authorized official GPQA-Diamond CSV"
    )
    parser.add_argument(
        "--ruler", type=Path, help="optional prepared RULER questions saved with HF"
    )
    args = parser.parse_args()
    rows = []
    mix = MIX
    if args.gpqa:
        with args.gpqa.open(newline="") as source:
            rows += list(gpqa(csv.DictReader(source)))
        mix = MIX | {"science": 8}
    math = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows += [
        row(
            "math500",
            item["unique_id"],
            "math",
            item["problem"] + "\nReason step by step and put your final answer in \\boxed{}.",
            "math_verify",
            item["answer"],
        )
        for item in math
        if item["level"] >= 4
    ]
    aime = load_dataset("MathArena/aime_2025", split="train")
    rows += [
        row(
            "aime2025",
            f"{int(item['problem_idx']):02d}",
            "math",
            item["problem"] + "\nReason step by step and put your final answer in \\boxed{}.",
            "math_verify",
            str(int(item["answer"])),
        )
        for item in aime
    ]
    coding = Path(
        hf_hub_download("livecodebench/code_generation_lite", "test6.jsonl", repo_type="dataset")
    )
    with coding.open() as source:
        rows += list(livecodebench(json.loads(line) for line in source))
    context = Path(hf_hub_download("zai-org/LongBench-v2", "data.json", repo_type="dataset"))
    rows += list(longbench(json.loads(context.read_text())))
    if args.ruler:
        from datasets import load_from_disk

        rows += load_from_disk(str(args.ruler)).to_list()
    args.output.mkdir(parents=True, mode=0o700)
    Dataset.from_list(rows).save_to_disk(str(args.output / "questions"))
    (args.output / "pool.json").write_text(
        json.dumps(
            {
                "kind": "pilot pool, not a verification suite",
                "target_distribution": mix,
                "counts": {
                    category: sum(r["category"] == category for r in rows) for category in mix
                },
                "selection": f"choose a manageable pilot subset, then freeze {sum(mix.values())} reviewed IDs",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Prepared {len(rows)} pilot questions at {args.output}; no inference launched")


if __name__ == "__main__":
    main()
