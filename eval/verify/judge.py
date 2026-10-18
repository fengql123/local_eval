"""Submission rules, native reviewer inputs and stored quality decisions.

Execution and procedure order belong to verify.run; image sources define the
reviewer's access. Answer grading belongs to the shared accuracy functions.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from rewardkit.models import JudgeConfig

from ..config import AgentConfig
from ..request_policy import request_fields


def static_check(submission: Path) -> dict:
    """Check the reproducible submission contract, not implementation semantics."""
    scripts = ["install.sh"]
    missing = [name for name in scripts if not (submission / name).is_file()]
    reserved = [
        name
        for name in ("runtime", "output", ".harbor")
        if submission != Path("/workspace") and (submission / name).exists()
    ]

    def project():
        for folder, dirs, names in submission.walk():
            if folder == submission and submission == Path("/workspace"):
                dirs[:] = [name for name in dirs if name not in (".harbor", "runtime", "output")]
            yield from (folder / name for name in dirs + names)

    escaped = [
        str(path.relative_to(submission))
        for path in project()
        if path.is_symlink()
        and (
            not path.resolve().is_relative_to(submission.resolve())
            or path.resolve().is_relative_to(submission.resolve() / ".harbor")
        )
    ]
    return {
        "ok": not (missing or escaped or reserved),
        "missing": missing,
        "external_paths": escaped,
        "reserved_paths": reserved,
    }


def write_review(task: dict, submission: Path, rubric: Path) -> None:
    """Write Rewardkit's native configuration and task-specific inspection prompt."""
    agent = AgentConfig(**task["evaluation"]["agent"])
    policy = request_fields(task["model"])
    slo = task.get("type") == "SLO"
    prompt = (
        Path(__file__)
        .with_name("prompt.txt")
        .read_text()
        .format(
            model=task["model"],
            target=task["target"]["source"],
            workload=task["workload"],
            submission=submission.resolve(),
            request_policy=json.dumps(policy.fixed(task["workload"])),
            comparison="No baseline serves this model here; fixed SLO tiers score the candidate."
            if slo
            else "The candidate uses its fixed model checkpoint; the reference may use another format.",
            timing="" if slo else "; the selected reference has not yet been timed",
            criteria="{criteria}",
        )
    )
    options = JudgeConfig(
        judge=agent.name,
        model=agent.model,
        mode="batched",
        timeout=agent.budget_seconds,
        isolated=False,
        prompt_template="prompt.txt",
        **agent.kwargs,
    ).model_dump(mode="json", by_alias=True, exclude_unset=True, exclude_none=True)

    def toml(value):
        if isinstance(value, dict):
            return "{" + ", ".join(f"{json.dumps(k)} = {toml(v)}" for k, v in value.items()) + "}"
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(map(toml, value)) + "]"
        return json.dumps(value)

    scoring = Path(__file__).with_name("rules.toml").read_text().partition("[scoring]")[2]
    rubric.mkdir(parents=True, exist_ok=True)
    (rubric / "rules.toml").write_text(
        "[judge]\n"
        + "\n".join(f"{key} = {toml(value)}" for key, value in options.items())
        + "\n\n[scoring]"
        + scoring
    )
    (rubric / "prompt.txt").write_text(prompt)


def rules_result(rewards: dict, details: dict) -> dict:
    """Separate real rule violations from missing or errored Rewardkit decisions."""
    rubric = tomllib.loads(Path(__file__).with_name("rules.toml").read_text())
    expected = {criterion["name"] for criterion in rubric["criterion"]}
    detail = details.get("reward", {})
    decisions = detail.get("criteria", [])
    complete = (
        len(decisions) == len(expected)
        and {entry.get("name") for entry in decisions} == expected
        and all(entry.get("value") in (0, 1) and not entry.get("error") for entry in decisions)
        and all(
            isinstance(entry.get("reasoning"), str) and entry["reasoning"].strip()
            for entry in decisions
        )
        and rewards.get("reward") in (0, 1)
        and rewards["reward"] == int(all(entry["value"] == 1 for entry in decisions))
    )
    return {
        **detail,
        "ok": rewards["reward"] == 1 if complete else None,
        "complete": complete,
    }


def quality_result(rows: list[dict], profile: dict, candidate: dict[str, bool]) -> dict:
    """Apply the independently prepared lower bound to all selected questions."""
    expected = {row["id"] for row in rows}
    missing = sorted(expected - candidate.keys())
    unexpected = sorted(candidate.keys() - expected)
    if not expected or missing or unexpected or set(profile["question_ids"]) != expected:
        return {
            "ok": False,
            "reason": "quality answers must cover exactly the selected rows",
            "missing": missing,
            "unexpected": unexpected,
        }
    correct = sum(candidate.values())
    result = {
        "ok": correct >= profile["min_correct"],
        "bound": {
            "items": len(expected),
            "correct": correct,
            "min_correct": profile["min_correct"],
            "accuracy": correct / len(expected),
            "profile_id": profile["id"],
        },
    }
    categories = sorted({row["category"] for row in rows if "category" in row})
    if categories:
        result["categories"] = {
            category: {
                "items": sum(row["category"] == category for row in rows),
                "correct": sum(candidate[row["id"]] for row in rows if row["category"] == category),
            }
            for category in categories
        }
    return result
