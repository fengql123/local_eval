"""Run summaries: SOTA gains, coding cost and SLO placement grouped by machine."""

import json

import pytest
import yaml

from eval import report
from eval.costs import Attempt, Usage, update_ledger

MACHINE = {
    "category": "nvidia-cuda-x86",
    "gpus": [{"name": "RTX 4090", "memory_mib": 24564, "driver": "580.1"}],
    "cpus": 8,
    "memory_mb": 65536,
    "fingerprint": "abc123",
}


def run_folder(root, name, agent, tasks):
    folder = root / name
    folder.mkdir(parents=True)
    settings = {
        "agent": {"name": "claude-code", "model": agent},
        "environment": {
            "device_category": "nvidia-cuda-x86",
            "cpus": 8,
            "memory_mb": 65536,
            "kwargs": {"gpu_ids": [0]},
        },
    }
    (folder / "run.yaml").write_text(yaml.safe_dump(settings))
    (folder / "result.json").write_text(json.dumps({"tasks": tasks}))
    attempts = [
        Attempt(
            f"{name}-{task}",
            f"local-eval-coding/{task}",
            "trial",
            "implementation",
            "claude-code",
            None,
            agent,
            None,
            Usage(cost_usd=12.5, cost_kind="reported"),
            "fixture",
        )
        for task in tasks
    ]
    update_ledger(folder / "costs.json", attempts)
    return folder


def test_runs_group_by_machine_with_gains_costs_and_tiers(tmp_path):
    sota = {
        "status": "complete",
        "type": "SOTA",
        "rewards": {"score": 29.2893},
        "score": 29.2893,
        "gains": {"throughput": 2.0, "ttft": 1.0},
        "machine": MACHINE,
        "timings": {"coding": {"recorded_seconds": {"agent_execution": 3600}}},
    }
    placed = {
        "status": "complete",
        "type": "SLO",
        "rewards": {"score": 60},
        "score": 60,
        "machine": MACHINE,
        "slo": {"tier": "T3", "tier_met": 3, "score": 60},
    }
    run_folder(tmp_path, "one", "model-a", {"qwen_S-1": sota, "minimax_A-N": placed})
    legacy = {"status": "complete", "rewards": {"throughput": 40.0}, "gains": {"throughput": 1.5}}
    run_folder(tmp_path, "two", "model-b", {"qwen_S-N": legacy})
    rows = report.collect([tmp_path])
    by_task = {row["task"]: row for row in rows}
    assert by_task["qwen_S-1"]["machine"] == "abc123" and by_task["qwen_S-1"]["cost_usd"] == 12.5
    assert by_task["qwen_S-1"]["timings"]["coding"]["recorded_seconds"]["agent_execution"] == 3600
    assert by_task["minimax_A-N"]["type"] == "SLO" and by_task["minimax_A-N"]["gains"] is None
    # Older runs have no fingerprint; their configured resources identify the machine.
    assert (
        by_task["qwen_S-N"]["type"] == "SOTA"
        and "(configured)" in by_task["qwen_S-N"]["machine_label"]
    )
    page = report.render(rows)
    assert page.count("<h2>") == 2 and "<svg" in page and "<circle" in page
    assert "<b>T3</b> 60.00/100" in page and "T1 &lt; T2" in page
    assert by_task["qwen_S-1"]["score"] == pytest.approx(29.2893)
    assert "Task scores and completion" in page
    output = tmp_path / "out"
    report.main([str(tmp_path / "one"), str(tmp_path / "two"), "--output", str(output)])
    assert len(json.loads((output / "report.json").read_text())) == 3


def test_imported_results_keep_their_protocol_and_rejected_zero(tmp_path):
    common = {"type": "SOTA", "machine": MACHINE, "gains": {"throughput": 2.0}}
    folder = run_folder(
        tmp_path,
        "continued",
        "model-a",
        {
            "qwen_S-1": {
                **common,
                "status": "complete",
                "rewards": {"throughput": 60.0},
                "evaluation_protocol": {"dataset": "dataset-a"},
            },
            "qwen_S-N": {
                **common,
                "type": "SLO",
                "status": "candidate_failed",
                "slo": {"tier": "L8", "tier_met": 8},
                "evaluation_protocol": {"dataset": "dataset-b"},
            },
        },
    )
    (folder / "zero-audits.json").write_text(
        json.dumps(
            {
                "qwen_S-1": {"reason": "Superseded old failure"},
                "qwen_S-N": {"reason": "Fixed weights changed & were rejected"},
            }
        )
    )
    rows = report.collect([folder])
    assert rows[0]["reason"] is None
    assert rows[1]["reason"] == "Fixed weights changed & were rejected"
    assert "Fixed weights changed &amp; were rejected" in report.render(rows)
    assert [r["dataset"] for r in rows] == ["dataset-a", "dataset-b"]
    assert all("replay_version" not in row for row in rows)
    assert rows[1]["rewards"] == {"score": 0} and rows[1]["score"] == 0
    assert report.render(rows).count("<h2>") == 2

    (folder / "result.json").write_text(json.dumps({"tasks": {"qwen_S-1": {"status": "pending"}}}))
    assert report.collect([folder])[0]["type"] is None


def test_final_averages_include_rejected_zeros_and_wait_for_pending_cells(tmp_path):
    folder = run_folder(
        tmp_path,
        "pending",
        "agent",
        {
            "model_S-1": {"type": "SOTA", "status": "complete", "score": 50, "machine": MACHINE},
            "model_A-1": {"type": "SOTA", "status": "candidate_failed"},
            "model_S-N": {"type": "SLO", "status": "pending"},
        },
    )
    collected = report.collect([folder])
    summary = report.averages(collected)[0]
    assert summary["score"] is None and summary["scored"] == 2 and summary["tasks"] == 3
    assert summary["SOTA"] == 25 and summary["rejected"] == 1
    assert "1/3" in report.render(collected)
    collected[-1]["score"] = 100
    collected[-1]["status"] = "complete"
    assert report.averages(collected)[0]["score"] == 50
    report.build([folder], tmp_path / "output")
    saved = json.loads((tmp_path / "output/report.json").read_text())
    assert len(saved) == 3 and saved[1]["score"] == 0 and saved[2]["score"] is None


def test_saved_n_timestamps_recover_the_score_including_prefill(tmp_path):
    folder = run_folder(tmp_path, "saved", "agent", {})
    trial = folder / "jobs/verify/model_S-N"
    source = trial / "steps/measure/verifier/report.json"
    source.parent.mkdir(parents=True)

    def timed(first, last, end):
        return {
            "ok": True,
            "t_send": 0,
            "t_first": first,
            "t_last": last,
            "t_end": end,
            "output_tokens": 100,
        }

    source.write_text(
        json.dumps(
            {
                "outcome": "passed",
                "bench": {
                    "reference": {"records": [timed(10, 20, 20)]},
                    "candidate": {"records": [timed(5, 10, 10)]},
                },
            }
        )
    )
    entry = {
        "status": "complete",
        "type": "SOTA",
        "gains": {"throughput": 99},
        "verification_result": str(trial / "result.json"),
    }
    (folder / "result.json").write_text(json.dumps({"tasks": {"model_S-N": entry}}))
    result = report.collect([folder])[0]
    assert result["gains"] == {"throughput": 2}
    assert result["score"] == 50
    assert result["measurements"]["reference"]["throughput"] == 5
    assert result["measurements"]["candidate"]["throughput"] == 10
    # Old decode-only summaries cannot recover N serving throughput without timestamps.
    source.write_text(
        json.dumps(
            {
                "outcome": "passed",
                "bench": {"reference": {"throughput": 5}, "candidate": {"throughput": 10}},
            }
        )
    )
    assert report.collect([folder])[0]["score"] is None
    entry["status"] = "candidate_failed"
    (folder / "result.json").write_text(json.dumps({"tasks": {"model_S-N": entry}}))
    assert report.collect([folder])[0]["score"] == 0


def test_saved_slo_uses_the_original_rows_including_unstarted_graphs(tmp_path):
    from datasets import Dataset

    folder = run_folder(tmp_path, "saved-slo", "agent", {})
    rows = [
        {"id": name, "requests": [{"id": 0, "messages": "[]", "tools": None}]}
        for name in ("one", "two")
    ]
    source = folder / "input/private"
    Dataset.from_list(rows).save_to_disk(source)
    (folder / "prepared.json").write_text(
        json.dumps(
            {
                "tasks": [{"task_id": "model_A-1", "model": "model"}],
                "inputs": {"model": {"A-1": {"private": {"source": str(source)}}}},
            }
        )
    )
    settings = yaml.safe_load((folder / "run.yaml").read_text())
    settings["evaluation"] = {"concurrency": {"S-N": 32, "A-N": 8}}
    (folder / "run.yaml").write_text(yaml.safe_dump(settings))
    trial = folder / "jobs/verify/model_A-1"
    output = trial / "steps/slo/verifier/report.json"
    output.parent.mkdir(parents=True)
    record = {
        "row_id": "one",
        "request_id": 0,
        "ok": True,
        "errors": [],
        "t_send": 0,
        "t_first": 0.1,
        "t_last": 1.1,
        "t_end": 1.1,
        "output_tokens": 100,
    }
    output.write_text(
        json.dumps(
            {
                "outcome": "passed",
                "bench": {"candidate": [{"records": [record]}]},
            }
        )
    )
    (folder / "result.json").write_text(
        json.dumps(
            {
                "tasks": {
                    "model_A-1": {
                        "status": "complete",
                        "type": "SLO",
                        "rewards": {"slo_score": 7.9},
                        "verification_result": str(trial / "result.json"),
                    }
                }
            }
        )
    )
    saved = report.collect([folder])[0]
    assert saved["score"] == 50
    assert saved["slo"]["coverage"]["complete_fraction"] == 0.5
    assert len(saved["slo"]["detail"]) == 2


@pytest.mark.parametrize(
    "run_status,run_cycle,task_status,task_cycle,phase,expected",
    [
        ("running", 1, "running", 1, "verify", "running"),
        ("running", 2, "running", 1, "coding", "queued"),
        ("interrupted", 1, "running", 1, "verify", "interrupted"),
        ("running", 1, "complete", 1, "search", "incomplete"),
    ],
)
def test_unfinished_results_show_coordinator_progress(
    tmp_path, run_status, run_cycle, task_status, task_cycle, phase, expected
):
    folder = run_folder(
        tmp_path, "active", "agent", {"model_S-1": {"status": "infrastructure_error"}}
    )
    (folder / "state.json").write_text(
        json.dumps(
            {
                "status": run_status,
                "cycle": run_cycle,
                "tasks": {
                    "model_S-1": {"status": task_status, "cycle": task_cycle, "phase": phase}
                },
            }
        )
    )
    (row,) = report.collect([folder])
    assert row["status"] == expected and row["score"] is None
    assert (
        json.loads((folder / "result.json").read_text())["tasks"]["model_S-1"]["status"]
        == "infrastructure_error"
    )


@pytest.mark.parametrize("has_finished_result", [True, False])
def test_planned_tasks_appear_before_any_native_trial_finishes(tmp_path, has_finished_result):
    folder = run_folder(
        tmp_path,
        "planned",
        "agent",
        {"model_S-1": {"status": "complete", "score": 50}},
    )
    (folder / "prepared.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {"task_id": "model_S-1"},
                    {"task_id": "model_A-1"},
                ]
            }
        )
    )
    (folder / "state.json").write_text(json.dumps({"status": "running", "cycle": 1, "tasks": {}}))
    if not has_finished_result:
        (folder / "result.json").unlink()
    rows = {row["task"]: row for row in report.collect([tmp_path, folder])}
    assert set(rows) == {"model_S-1", "model_A-1"}
    assert rows["model_A-1"]["status"] == "queued"
    assert rows["model_A-1"]["score"] is None
    assert rows["model_S-1"]["score"] == (50 if has_finished_result else None)
