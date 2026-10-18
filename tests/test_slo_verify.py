"""SLO verification: candidate-only preparation, bounded timing and tier rewards."""

import asyncio
import json
from pathlib import Path

from http_server import headers, server
from test_verify import BOUND, PREPARED, TASK, backend, boundaries, inputs, reply, rows

from eval.bench import slo
from eval.verify import run as verifier

SLO_EVALUATION = {
    **TASK["evaluation"],
    "slo_budget_seconds": 30,
    "slo_request_cap_seconds": 5,
    "slo_repeats": 2,
    "quality_timeout_seconds": 30,
}
GATE = {"ok": True, "stage": "quality"}

__all__ = ["boundaries"]


def slo_task(tmp_path=None, **changes):
    base = inputs(tmp_path) if tmp_path is not None else TASK
    return {**base, "type": "SLO", "reference": None, "evaluation": SLO_EVALUATION, **changes}


def test_slo_preparation_installs_only_the_candidate(tmp_path, monkeypatch):
    task = slo_task(tmp_path)

    async def forbidden(*args):
        raise AssertionError("SLO tasks have no reference to install")

    monkeypatch.setattr(verifier, "install_reference", forbidden)
    source = Path(task["candidate"]["submission"])
    prepared = asyncio.run(verifier.prepare_submission(task, source, tmp_path / "results"))
    assert prepared["ok"] and prepared["install"]["ok"]
    prompt = (verifier.RUBRIC / "prompt.txt").read_text()
    assert "fixed SLO tiers score the candidate" in prompt and "selected reference" not in prompt


def test_slo_timing_repeats_fresh_servers_and_writes_tier_rewards(tmp_path):
    opened, closed = [], []
    out = tmp_path / "results"

    async def scenario():
        async def respond(_reader, writer, _body):
            await reply(writer)

        async with server(respond) as (endpoint, requests):
            report = await verifier.slo_measure(
                slo_task(),
                rows(),
                backend("candidate", endpoint, opened, closed),
                out=out,
                prepared=PREPARED,
                reviewed={"ok": True},
                quality_gate=GATE,
            )
            # Each repeat starts a fresh server: one warmup and one timed row.
            assert len(requests) == 4
            return report

    report = asyncio.run(scenario())
    assert report["outcome"] == "passed" and report["type"] == "SLO"
    assert opened == closed == [("candidate", "slo"), ("candidate", "slo")]
    placed = report["slo"]
    assert placed["coverage"] == {
        "rows": 1,
        "repeats": 2,
        "started": 2,
        "fraction": 1.0,
        "completed": 2,
        "complete_fraction": 1.0,
    }
    assert placed["tier_met"] >= 1 and placed["tiers"][0]["met"]
    assert report["rewards"] == {"score": report["score"]}
    assert 0 <= report["score"] <= 100
    assert json.loads((out / "reward.json").read_text()) == report["rewards"]
    assert "SLO tier" in (out / "report.md").read_text()


def test_slo_failures_score_by_workload_kind(tmp_path):
    async def measure(task, quality_gate=GATE):
        opened, closed = [], []
        calls = 0

        async def respond(_reader, writer, _body):
            nonlocal calls
            calls += 1
            await reply(writer, valid=calls == 1)

        async with server(respond) as (endpoint, _):
            return await verifier.slo_measure(
                task,
                rows(),
                backend("candidate", endpoint, opened, closed),
                out=tmp_path / task["workload"],
                prepared=PREPARED,
                reviewed={"ok": True},
                quality_gate=quality_gate,
            )

    # A failed single-stream request is a miss with zero credit, not a failed run.
    single = asyncio.run(measure(slo_task(evaluation={**SLO_EVALUATION, "slo_repeats": 1})))
    assert single["outcome"] == "passed" and single["slo"]["requests"]["failed"] == 1
    assert single["rewards"] == slo.zero("S-1")
    # Parallel workloads keep SOTA's strictness: any failure fails the candidate.
    parallel = asyncio.run(measure(slo_task(workload="S-N")))
    assert parallel["outcome"] == "candidate_failure"
    assert parallel["rewards"] == slo.zero("S-N")
    gated = asyncio.run(
        measure(slo_task(), {"ok": False, "stage": "quality", "reason": "below bound"})
    )
    assert gated["bench"]["skipped"] and gated["rewards"] == slo.zero("S-1")


def test_quality_answers_unfinished_at_the_deadline_count_as_wrong(tmp_path):
    opened, closed = [], []

    async def scenario():
        async def respond(_reader, writer, _body):
            await headers(writer)
            await asyncio.sleep(2)

        async with server(respond) as (endpoint, _):
            candidate = backend("candidate", endpoint, opened, closed)
            return await verifier.pre_run(rows(), BOUND, candidate, budget=0.3)

    gate = asyncio.run(scenario())
    assert gate["ok"] is False and gate["quality"]["bound"]["correct"] == 0
    assert gate["reason"] == "candidate accuracy is below the offline lower bound"
