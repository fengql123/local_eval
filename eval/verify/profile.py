"""Prepare a standalone quality bound from 32 rounds of the frozen questions.

This is trusted offline preparation. Candidate evaluation only reads its result;
it never changes the questions or learns a bound from a live performance baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import time
from pathlib import Path

from ..bench.clients import ChatClient
from ..bench.replay import replay
from ..request_policy import RequestPolicy

ROUNDS = 32
LOWER_RANK = 1


def identity(source: str, policy: RequestPolicy, rows: list[dict]) -> str:
    value = {
        "checkpoint": source,
        "policy": policy.fixed("S-1"),
        "questions_identity": questions_identity(rows),
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def questions_identity(rows: list[dict]) -> str:
    """Bind the questions and checkers; quality never uses tool scheduling."""
    value = [
        {
            **{key: row[key] for key in ("id", "category", "source", "item", "evaluation")},
            "requests": [
                {key: value for key, value in request.items() if key != "tool"}
                for request in row["requests"]
            ],
        }
        for row in rows
    ]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(
    folder: Path,
    source: str,
    policy: RequestPolicy,
    rows: list[dict],
    *,
    target: dict | None = None,
) -> dict:
    from ..quality import model_range

    return model_range(policy.model, rows)


def validate_profile(profile, source, policy, rows, *, target=None) -> None:
    """A partial, stale or mismatched profile cannot authorize verification."""
    from ..quality import validate

    validate(rows)
    if profile.get("kind") == "pilot":
        raise ValueError("a pilot cannot authorize verification")
    ids = [row["id"] for row in rows]
    rounds = profile["rounds"]
    if (
        profile["id"] != identity(source, policy, rows)
        or profile["checkpoint"] != source
        or profile["model"] != policy.model
        or profile["policy"] != policy.fixed("S-1")
        or profile["questions_identity"] != questions_identity(rows)
        or profile["question_ids"] != ids
        or len(rounds) != ROUNDS
        or [entry["round"] for entry in rounds] != list(range(ROUNDS))
        or any(
            set(entry["items"]) != set(ids)
            or any(type(value) is not bool for value in entry["items"].values())
            or entry["correct"] != sum(entry["items"].values())
            or any(entry["items"].get(key, False) for key in entry.get("truncated", []))
            for entry in rounds
        )
        or profile["lower_rank"] != LOWER_RANK
        or profile["min_correct"] != min(entry["correct"] for entry in rounds)
        or not profile["tokenizer_identity"]
        or (
            target is not None
            and any(
                profile[key] != target[key] for key in ("tokenizer_identity", "checkpoint_revision")
            )
        )
    ):
        raise ValueError(
            "quality profile does not match the fixed checkpoint and selected questions"
        )


def main() -> None:
    """Run a trusted profiling server independently of searched performance baselines."""
    from datasets import load_from_disk

    from ..bench.serve import Server
    from ..launch import ENDPOINT, Launch
    from ..request_policy import request_fields
    from .access import reset_runtime
    from .run import runtime_environment, stop_runtime_users

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", type=Path, required=True, help="trusted model, rows and profiler configuration"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pilot", action="store_true", help="two exploratory rounds; never publish a gate"
    )
    args = parser.parse_args()
    task = json.loads(args.task.read_text())
    target = json.loads(Path("/var/lib/local-eval/target.json").read_text())
    settings = task["profiler"]
    policy = request_fields(task["model"])
    policy.overrides = {"skip_special_tokens": False}
    reset_runtime("reference")
    script = Path(settings["launch"])
    owned = Server(
        Launch(script, "/models/target", tuple(settings["args"]), dict(settings.get("env", {}))),
        ENDPOINT,
        cwd=script.parent,
        log=args.output / "server.log",
        health_timeout=task["evaluation"]["health_timeout_seconds"],
        environment=runtime_environment("reference"),
        user=1005,
    )

    async def execute():
        try:
            return await build(
                load_from_disk(str(Path(task["inputs"]["quality"]) / "questions")).to_list(),
                target["source"],
                policy,
                server=owned,
                output=args.output,
                implementation={
                    **settings["implementation"],
                    "args": settings["args"],
                    "environment": task["environment"],
                },
                tokenizer_identity=target["tokenizer_identity"],
                checkpoint_revision=target["checkpoint_revision"],
                concurrency=settings.get("concurrency", -1),
                pilot=args.pilot,
            )
        finally:
            await stop_runtime_users()

    result = asyncio.run(execute())
    print(
        f"Saved {'pilot' if args.pilot else 'profile'} {result['id']} to {args.output}", flush=True
    )


async def build(
    rows: list[dict],
    source: str,
    policy: RequestPolicy,
    *,
    server,
    output: Path,
    implementation: dict,
    tokenizer_identity: str,
    checkpoint_revision: str | None = None,
    tokenizer=None,
    concurrency: int = -1,
    pilot: bool = False,
) -> dict:
    """Continuously feed requests across rounds; retain answers and publish all 32 rounds."""
    from ..accuracy import grade
    from ..quality import validate

    if not pilot:
        validate(rows)
    count = 2 if pilot else ROUNDS
    if concurrency != -1 and concurrency < 1:
        raise ValueError("concurrency must be positive or -1")
    output.mkdir(parents=True, exist_ok=True)
    completed = [None] * count
    manifest = {
        "checkpoint": source,
        "questions_identity": questions_identity(rows),
        "tokenizer_identity": tokenizer_identity,
        "checkpoint_revision": checkpoint_revision,
        "policy": policy.fixed("S-1"),
        "implementation": implementation,
        "rounds": count,
        "kind": "pilot" if pilot else "profile",
        "lower_rank": LOWER_RANK,
        "concurrency": concurrency,
    }
    plan = output / "preparation.json"
    if plan.exists() and json.loads(plan.read_text()) != manifest:
        raise ValueError("completed quality rounds belong to a different preparation")
    plan.write_text(json.dumps(manifest, indent=2) + "\n")
    saved = output / ("pilot.json" if pilot else "profile.json")
    if saved.is_file():
        result = json.loads(saved.read_text())
        if not pilot:
            validate_profile(result, source, policy, rows)
        return result
    async with server as running:
        async with ChatClient(
            running.endpoint, policy, mode="quality", total_timeout=108000, tokenizer=tokenizer
        ) as client:
            limit = asyncio.Semaphore(count * len(rows) if concurrency == -1 else concurrency)

            async def run_round(number):
                path = output / f"round-{number:03d}.json"
                if path.exists():
                    completed[number] = json.loads(path.read_text())
                    return
                parts = output / f"round-{number:03d}"
                parts.mkdir(exist_ok=True)
                records = [None] * len(rows)

                async def request(index, row):
                    saved = parts / f"{index:03d}.json"
                    if saved.is_file():
                        records[index] = json.loads(saved.read_text())
                        return
                    async with limit:
                        (record,) = await replay([row], "S-1", client, concurrency=-1)
                    partial = saved.with_suffix(".partial")
                    partial.write_text(json.dumps(record) + "\n")
                    partial.replace(saved)
                    records[index] = record

                started = time.monotonic()
                async with asyncio.TaskGroup() as pending:
                    for index, row in enumerate(rows):
                        pending.create_task(request(index, row))
                items = {
                    row["id"]: record["finish_reason"] == "stop"
                    and grade(row["evaluation"], record.get("raw_text", record["text"]))
                    for row, record in zip(rows, records, strict=True)
                }
                result = {
                    "round": number,
                    "correct": sum(items.values()),
                    "items": items,
                    "truncated": [
                        record["row_id"]
                        for record in records
                        if record.get("finish_reason") == "length"
                    ],
                }
                partial = path.with_suffix(".partial")
                partial.write_text(json.dumps({**result, "records": records}) + "\n")
                partial.replace(path)
                shutil.rmtree(parts)
                completed[number] = result
                print(
                    f"Offline quality: {sum(item is not None for item in completed)}/{count} "
                    f"rounds, round {number}: {result['correct']}/{len(rows)} correct, "
                    f"{time.monotonic() - started:.1f}s",
                    flush=True,
                )

            async with asyncio.TaskGroup() as group:
                for number in range(count):
                    group.create_task(run_round(number))
    profile = {
        "id": identity(source, policy, rows),
        "model": policy.model,
        "checkpoint": source,
        "tokenizer_identity": tokenizer_identity,
        "checkpoint_revision": checkpoint_revision,
        "policy": policy.fixed("S-1"),
        "question_ids": [row["id"] for row in rows],
        "questions_identity": questions_identity(rows),
        "rounds": [{k: v for k, v in round.items() if k != "records"} for round in completed],
        "lower_rank": LOWER_RANK,
        "min_correct": sorted(round["correct"] for round in completed)[LOWER_RANK - 1],
        "implementation": implementation,
    }
    profile["questions"] = {
        row["id"]: {
            "category": row["category"],
            "correct": sum(entry["items"][row["id"]] for entry in completed),
            "attempts": count,
            "truncated": sum(row["id"] in entry["truncated"] for entry in completed),
        }
        for row in rows
    }
    if not pilot:
        validate_profile(profile, source, policy, rows)
    if pilot:
        profile.pop("min_correct")
        profile["kind"] = "pilot"
    path = output / ("pilot.json" if pilot else "profile.json")
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(profile, indent=2) + "\n")
    partial.replace(path)
    return profile


if __name__ == "__main__":
    main()
