"""Submission rules, native reviewer inputs and paired quality decisions.

Execution and procedure order belong to verify.run; image sources define the
reviewer's access. Answer grading belongs to the shared accuracy functions.
"""

from __future__ import annotations

import json
import tomllib
from math import comb
from pathlib import Path

from rewardkit.models import JudgeConfig

from ..config import AgentConfig
from ..request_policy import request_fields

ALPHA = 0.05


def static_check(submission: Path) -> dict:
    """Require install/launch; an optional conversion script stays in the submission."""
    scripts = ["install.sh", "launch.sh"]
    if (submission / "convert_weight.sh").exists() or (
        submission / "convert_weight.sh"
    ).is_symlink():
        scripts.append("convert_weight.sh")
    missing = [name for name in scripts if not (submission / name).is_file()]
    escaped = [
        name
        for name in scripts
        if (submission / name).is_symlink()
        and not (submission / name).resolve().is_relative_to(submission.resolve())
    ]
    return {"ok": not (missing or escaped), "missing": missing, "external_scripts": escaped}


def write_review(task: dict, submission: Path, rubric: Path) -> None:
    """Write Rewardkit's native configuration and task-specific inspection prompt."""
    agent = AgentConfig(**task["evaluation"]["agent"])
    policy = request_fields(task["model"])
    prompt = (
        Path(__file__)
        .with_name("prompt.txt")
        .read_text()
        .format(
            model=task["model"],
            workload=task["workload"],
            submission=submission.resolve(),
            weights=(submission.parent / "weights").resolve(),
            request_policy=json.dumps(policy.fixed(task["workload"])),
            criteria="{criteria}",
        )
    )
    unknown = set(agent.kwargs) - {"version", "mcp_servers"}
    if unknown:
        raise ValueError(f"unsupported Rewardkit agent kwargs: {', '.join(sorted(unknown))}")
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
        and all(entry.get("reasoning") for entry in decisions if entry["value"] == 0)
        and rewards.get("reward") in (0, 1)
    )
    return {
        **detail,
        "ok": rewards["reward"] == 1 if complete else None,
        "complete": complete,
    }


def paired(rows: list[dict], reference: dict[str, bool], candidate: dict[str, bool]) -> dict:
    """Exact one-sided McNemar test, using every expected quality row."""
    expected = {row["id"] for row in rows}
    missing = {
        "reference": sorted(expected - reference.keys()),
        "candidate": sorted(expected - candidate.keys()),
    }
    unexpected = {
        "reference": sorted(reference.keys() - expected),
        "candidate": sorted(candidate.keys() - expected),
    }
    if not expected or any(missing.values()) or any(unexpected.values()):
        return {
            "ok": False,
            "reason": "quality answers must cover exactly the selected rows",
            "missing": missing,
            "unexpected": unexpected,
        }
    lost = sum(reference[key] and not candidate[key] for key in expected)
    won = sum(candidate[key] and not reference[key] for key in expected)
    discordant = lost + won
    probability = sum(comb(discordant, k) for k in range(lost, discordant + 1)) / 2**discordant
    return {
        "ok": probability >= ALPHA,
        "mcnemar": {
            "items": len(expected),
            "lost": lost,
            "won": won,
            "p": probability,
            "ref_accuracy": sum(reference.values()) / len(expected),
            "accuracy": sum(candidate.values()) / len(expected),
        },
    }
