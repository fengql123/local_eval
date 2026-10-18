"""Replay commits: batching, quality, cancellation and changed input."""

import asyncio
import hashlib
import json
from collections import Counter

import pytest

from eval.bench.progress import Progress
from eval.bench.replay import replay


def rows(count):
    return [
        {
            "id": str(i),
            "requests": [{"id": 0, "parents": [], "out": 3, "messages": f"[{i}]", "tools": None}],
        }
        for i in range(count)
    ]


def test_n_batches_commit_with_a_barrier_and_resume_only_the_interrupted_batch(tmp_path):
    async def scenario(workload):
        folder = tmp_path / workload
        counts = Counter()
        release = asyncio.Event()
        first = asyncio.Event()

        class Client:
            async def request(self, request, _):
                key = int(request["messages"].strip("[]"))
                counts[key] += 1
                if key >= 8:
                    first.set()
                    await release.wait()
                else:
                    assert not any(k >= 8 for k in counts)
                return {"ok": True, "errors": [], "text": "answer"}

        pending = asyncio.create_task(
            replay(rows(16), workload, Client(), progress=Progress(folder, {"mode": "timing"}))
        )
        await asyncio.wait_for(first.wait(), 2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert (folder / "units/000000.json").exists()
        assert not (folder / "units/000001.json").exists()
        release.set()
        records = await replay(
            rows(16), workload, Client(), progress=Progress(folder, {"mode": "timing"})
        )
        assert len(records) == 16
        assert all(counts[i] == 1 for i in range(8))
        assert all(counts[i] == 2 for i in range(8, 16))
        assert len((folder / "events.jsonl").read_text().splitlines()) > 16

    for workload in ("S-N", "A-N"):
        asyncio.run(scenario(workload))


def test_quality_retains_completed_wrong_answers_and_resumes_unfinished_ones(tmp_path):
    async def scenario():
        counts = Counter()
        started = asyncio.Event()
        release = asyncio.Event()

        class Client:
            async def request(self, request, _):
                key = request["messages"]
                counts[key] += 1
                if key == "[1]":
                    started.set()
                    await release.wait()
                return {"ok": True, "errors": [], "text": "wrong", "finish_reason": "stop"}

        pending = asyncio.create_task(
            replay(
                rows(2),
                "S-1",
                Client(),
                quality=True,
                concurrency=-1,
                progress=Progress(tmp_path, {"mode": "quality"}),
            )
        )
        await asyncio.wait_for(started.wait(), 2)
        while not (tmp_path / "units/000000.json").exists():
            await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        release.set()
        records = await replay(
            rows(2),
            "S-1",
            Client(),
            quality=True,
            concurrency=-1,
            progress=Progress(tmp_path, {"mode": "quality"}),
        )
        assert [r["text"] for r in records] == ["wrong", "wrong"]
        assert counts == {"[0]": 1, "[1]": 2}

    asyncio.run(scenario())


def test_public_concurrency_is_adjustable_and_partial_batches_finish():
    async def scenario():
        class Client:
            async def request(self, request, _):
                return {"ok": True, "errors": []}

        records = await replay(rows(10), "S-N", Client(), concurrency=3)
        assert [sum(r["batch"] == i for r in records) for i in range(4)] == [3, 3, 3, 1]
        assert len(await replay(rows(10), "A-N", Client(), concurrency=-1)) == 10
        for n in (0, -2):
            with pytest.raises(ValueError, match="concurrency"):
                await replay(rows(10), "S-N", Client(), concurrency=n)

    asyncio.run(scenario())


def test_resume_refuses_changed_inputs_and_discards_uncommitted_timing(tmp_path):
    p = Progress(tmp_path, {"mode": "timing"})
    p.prior_elapsed = 12
    p.committed = 4
    p.checkpoint()
    assert Progress(tmp_path, {"mode": "timing"}).elapsed == 4
    with pytest.raises(ValueError, match="configuration changed"):
        Progress(tmp_path, {"mode": "quality"})


def test_resume_preserves_units_when_only_the_cold_warmup_deadline_changes(tmp_path):
    definition = {"mode": "timing", "warmup": {"max_tokens": 16, "timeout_seconds": 60}}
    saved = Progress(tmp_path, definition)
    saved.commit(0, [{"ok": True, "output_tokens": 3}])
    # Reproduce the identity format written before deadlines were excluded.
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "identity": hashlib.sha256(
                    json.dumps(definition, sort_keys=True).encode()
                ).hexdigest(),
                **definition,
            }
        )
    )
    resumed = Progress(
        tmp_path, {**definition, "warmup": {"max_tokens": 16, "timeout_seconds": 300}}
    )
    assert resumed.saved(0) == [{"ok": True, "output_tokens": 3}]
    assert json.loads(config.read_text())["warmup"]["timeout_seconds"] == 300
    with pytest.raises(ValueError, match="configuration changed"):
        Progress(tmp_path, {**definition, "warmup": {"max_tokens": 17, "timeout_seconds": 300}})


def test_extending_quality_budget_replays_only_deadline_cuts_and_keeps_wrong_answers(tmp_path):
    definition = {"mode": "quality", "budget": 7200, "rows": rows(3)}
    saved = Progress(tmp_path, definition)
    saved.prior_elapsed = 7200
    wrong = {"row_id": "0", "ok": True, "errors": [], "text": "wrong"}
    right = {"row_id": "1", "ok": True, "errors": [], "text": "right"}
    saved.commit(0, [wrong])
    saved.commit(1, [right])
    saved.commit(2, [{"row_id": "2", "cut": "budget", "ok": False}])
    (tmp_path / "answers.json").write_text(json.dumps({"0": False, "1": True, "2": False}))
    (tmp_path / "result.json").write_text(json.dumps({"accuracy": 1 / 3}))
    (tmp_path / "cutoff.json").write_text(
        json.dumps({"reason": "remaining answers cannot meet the required accuracy"})
    )
    resumed = Progress(tmp_path, {**definition, "budget": 18000})
    assert resumed.saved(0) == [wrong] and resumed.saved(1) == [right]
    assert resumed.saved(2) is None
    assert json.loads((tmp_path / "answers.json").read_text()) == {"0": False, "1": True}
    assert not (tmp_path / "result.json").exists() and not (tmp_path / "cutoff.json").exists()
    assert 7200 <= resumed.elapsed < 7201
    assert json.loads((tmp_path / "checkpoint.json").read_text())["completed_units"] == [0, 1]

    async def scenario():
        class Client:
            async def request(self, request, _, result=None):
                assert request["messages"] == "[2]"
                return {"ok": True, "errors": [], "text": "new answer"}

        records = await replay(
            rows(3), "S-1", Client(), quality=True, concurrency=-1, budget=18000, progress=resumed
        )
        assert [record["text"] for record in records] == ["wrong", "right", "new answer"]

    asyncio.run(scenario())
    with pytest.raises(ValueError, match="configuration changed"):
        Progress(tmp_path, {**definition, "budget": 20000, "rows": rows(4)})


def test_extending_quality_budget_keeps_a_completed_result(tmp_path):
    saved = Progress(tmp_path, {"mode": "quality", "budget": 7200})
    saved.commit(0, [{"row_id": "done", "ok": True}])
    (tmp_path / "result.json").write_text(json.dumps({"accuracy": 1}))
    resumed = Progress(tmp_path, {"mode": "quality", "budget": 18000})
    assert resumed.saved(0) == [{"row_id": "done", "ok": True}]
    assert json.loads((tmp_path / "result.json").read_text()) == {"accuracy": 1}


@pytest.mark.parametrize("mode,budget", [("quality", 7199), ("timing", 18000)])
def test_budget_resume_refuses_reduced_quality_time_or_changed_timing_work(mode, budget, tmp_path):
    Progress(tmp_path, {"mode": mode, "budget": 7200})
    with pytest.raises(ValueError, match="configuration changed"):
        Progress(tmp_path, {"mode": mode, "budget": budget})
