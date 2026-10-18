"""Summarize finished runs: SOTA gains, cost against performance and SLO tiers.

Usage: ``python -m eval.report RUN_OR_PARENT... [--output FOLDER]``. Each run
folder contributes the tasks in its result.json. Results group by machine: the
fingerprint verification recorded, or the configured device and resources for
runs that predate it. The output is report.json plus a self-contained report.html.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

from .bench import slo
from .bench.metrics import gains as sota_gains
from .bench.metrics import reward_metrics, score_gains, summarize
from .costs import read_ledger

WIDTH = 760
COLORS = ("#3366cc", "#dc7633", "#2e8b57", "#8e44ad", "#c0392b", "#16a085", "#7f8c8d")


def _runs(paths: list[Path]) -> list[Path]:
    found = []
    for path in paths:
        if (path / "agents").is_dir():
            found.extend(
                sorted(
                    {
                        item.parent
                        for name in ("result.json", "prepared.json")
                        for item in path.glob(f"agents/*/{name}")
                    }
                )
            )
        elif any((path / name).is_file() for name in ("result.json", "prepared.json")):
            found.append(path)
        else:
            found.extend(
                sorted(
                    {
                        item.parent
                        for name in ("result.json", "prepared.json")
                        for item in path.glob(f"*/{name}")
                    }
                )
            )
    return list(dict.fromkeys(found))


def _machine(run: Path, settings: dict, entry: dict) -> tuple[str, str]:
    """A recorded fingerprint, else the configured device and resources."""
    recorded = entry.get("machine")
    if recorded:
        gpus = ", ".join(gpu["name"] for gpu in recorded["gpus"]) or "no GPU"
        label = (
            f"{recorded['category']}: {gpus}, {recorded.get('cpu_group', recorded.get('cpu'))}, "
            f"{recorded['cpus']} cores, {recorded['memory_mb']} MiB RAM, "
            f"{recorded.get('storage_mb', 'unknown')} MiB disk"
        )
        return recorded["fingerprint"], label
    environment = settings["environment"]
    kwargs = environment.get("kwargs", {})
    devices = kwargs.get("gpu_types") or kwargs.get("gpu_ids") or []
    configured = {
        "category": environment["device_category"],
        "gpus": [str(device) for device in devices],
        "count": kwargs.get("gpus", len(devices)),
        "cpus": environment["cpus"],
        "memory_mb": environment["memory_mb"],
    }
    identity = hashlib.sha256(json.dumps(configured, sort_keys=True).encode()).hexdigest()[:16]
    gpus = ", ".join(configured["gpus"]) or "GPUs unspecified"
    label = (
        f"{configured['category']}: {gpus}, {configured['cpus']} CPU, "
        f"{configured['memory_mb']} MB (configured)"
    )
    return identity, label


def _rescore_slo(run: Path, task_id: str, workload: str, report: dict, settings: dict) -> dict:
    """Apply current tiers to saved records and the original complete workload."""
    from datasets import load_from_disk

    prepared = json.loads((run / "prepared.json").read_text())
    task = next(task for task in prepared["tasks"] if task["task_id"] == task_id)
    source = prepared["inputs"][task["model"]][workload]["private"]["source"]
    rows = load_from_disk(source)
    runs = report["bench"]["candidate"]
    records = [
        {**record, "repeat": repeat}
        for repeat, saved in enumerate(runs)
        for record in saved["records"]
    ]
    concurrency = settings["evaluation"].get("concurrency")
    if isinstance(concurrency, dict):
        concurrency = concurrency.get(workload)
    return slo.score(workload, records, rows=rows, repeats=len(runs), concurrency=concurrency)


def _performance(
    run: Path, task_id: str, entry: dict, workload: str, kind: str | None, settings: dict
) -> dict:
    """Use the final scalar reward, or rescore saved evidence from earlier runs."""
    values = entry.get("gains") if kind == "SOTA" else None
    measured = None
    placed = entry.get("slo")
    score = entry.get("score", (entry.get("rewards") or {}).get("score"))
    if score is None and kind and entry.get("verification_result"):
        stage = "measure" if kind == "SOTA" else "slo"
        path = Path(entry["verification_result"]).parent / f"steps/{stage}/verifier/report.json"
        if path.is_file():
            report = json.loads(path.read_text())
            if report.get("outcome") == "passed":
                score = report.get("score", score)
                placed = report.get("slo", placed)
                bench = report.get("bench", {})
                if kind == "SLO" and score is None and "candidate" in bench:
                    placed = _rescore_slo(run, task_id, workload, report, settings)
                    score = placed["score"]
                if (
                    kind == "SOTA"
                    and "candidate" in bench
                    and (
                        workload not in ("S-N", "A-N")
                        or all("records" in bench[name] for name in ("reference", "candidate"))
                    )
                ):
                    measured = {}
                    for name in ("reference", "candidate"):
                        saved = bench[name]
                        measured[name] = (
                            summarize(workload, saved["records"]) if "records" in saved else saved
                        )
                    if all(item.get("valid", True) for item in measured.values()):
                        values = sota_gains(workload, measured["reference"], measured["candidate"])
                        score = score_gains(values)
                    else:
                        values, score = None, None
    if entry["status"] == "candidate_failed":
        score = 0.0
    elif entry["status"] != "complete":
        score = None
    elif score is None and values and workload not in ("S-N", "A-N"):
        if set(reward_metrics(workload)) <= values.keys():
            score = score_gains({key: values[key] for key in reward_metrics(workload)})
    return {"score": score, "gains": values, "slo": placed, "measurements": measured}


def collect(paths: list[Path]) -> list[dict]:
    """One row per task attempt with its agent, machine, type, scores and coding cost."""
    rows = []
    for run in _runs(paths):
        settings = yaml.safe_load((run / "run.yaml").read_text())
        agent = settings.get("agent") or {}
        label = "/".join(str(part) for part in (agent.get("name"), agent.get("model")) if part)
        costs = defaultdict(float)
        unknown = set()
        for attempt in read_ledger(run / "costs.json"):
            if attempt.role != "implementation":
                continue
            task = attempt.task_name.rsplit("/", 1)[-1]
            if attempt.usage.cost_usd is None:
                unknown.add(task)
            else:
                costs[task] += attempt.usage.cost_usd
        result_path = run / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {"tasks": {}}
        prepared_path = run / "prepared.json"
        if prepared_path.exists():
            for task in json.loads(prepared_path.read_text())["tasks"]:
                result["tasks"].setdefault(task["task_id"], {"status": "pending"})
        state_path = run / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        audit_path = run / "zero-audits.json"
        audits = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        for task_id, entry in result["tasks"].items():
            if entry["status"] in {"infrastructure_error", "pending"}:
                current = state.get("tasks", {}).get(task_id, {})
                status = current.get("status")
                if not current:
                    status = "queued" if state.get("status") == "running" else state.get("status")
                if status == "complete" and current.get("phase") != "verify":
                    status = "incomplete"
                elif status == "running":
                    status = state.get("status")
                    if status == "running" and current.get("cycle") != state.get("cycle"):
                        status = "queued"
                if status in {
                    "running",
                    "queued",
                    "waiting",
                    "incomplete",
                    "interrupted",
                    "excluded",
                }:
                    entry = {**entry, "status": status}
            model, _, workload = task_id.rpartition("_")
            hardware = entry.get("machine")
            if hardware is None:
                for field in ("verification_result", "coding_result", "search_result"):
                    record = (
                        Path(entry[field]).parent / "hardware.json" if entry.get(field) else None
                    )
                    if record and record.exists():
                        hardware = json.loads(record.read_text())
                        break
            fingerprint, machine = _machine(run, settings, {**entry, "machine": hardware})
            kind = entry.get("type")
            if kind is None and entry["status"] == "complete" and entry.get("gains"):
                kind = "SOTA"  # Completed historical results recorded gains without a type.
            performance = _performance(run, task_id, entry, workload, kind, settings)
            rows.append(
                {
                    "run": str(run.resolve()),
                    "task": task_id,
                    "model": model,
                    "workload": workload,
                    "agent": label or "unknown",
                    "machine": fingerprint,
                    "machine_label": machine,
                    "hardware": hardware,
                    "gpu": "; ".join(gpu["name"] for gpu in (hardware or {}).get("gpus", [])),
                    "vram_mib": "; ".join(
                        str(gpu["memory_mib"]) for gpu in (hardware or {}).get("gpus", [])
                    ),
                    "cpu": (hardware or {}).get("cpu"),
                    "cpu_group": (hardware or {}).get("cpu_group"),
                    "cores": (hardware or {}).get("cpus"),
                    "ram_mib": (hardware or {}).get("memory_mb"),
                    "disk_mib": (hardware or {}).get("storage_mb"),
                    "pcie": "; ".join(
                        f"Gen{gpu.get('pcie_generation')} x{gpu.get('pcie_width')}"
                        for gpu in (hardware or {}).get("gpus", [])
                    ),
                    "type": kind,
                    "candidate_checkpoint": entry.get("candidate_checkpoint"),
                    "reference_checkpoint": entry.get("reference_checkpoint"),
                    "dataset": entry.get("evaluation_protocol", {}).get("dataset", "legacy"),
                    "status": entry["status"],
                    "rewards": {"score": performance["score"]}
                    if performance["score"] is not None
                    else None,
                    **performance,
                    "reason": entry.get("reason")
                    or entry.get("error")
                    or (
                        audits.get(task_id, {}).get("reason")
                        if entry["status"] == "candidate_failed" or performance["score"] == 0
                        else None
                    ),
                    "cost_usd": None if task_id in unknown else costs.get(task_id),
                    "timings": entry.get("timings"),
                }
            )
    return rows


def _geomean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _svg(height: int, body: list[str]) -> str:
    return (
        f'<svg viewBox="0 0 {WIDTH} {height}" width="100%" role="img" '
        f'xmlns="http://www.w3.org/2000/svg">{"".join(body)}</svg>'
    )


def sota_chart(rows: list[dict], agents: list[str]) -> str:
    """Bars of gain G on a log axis per task and agent; G = 1 is reference parity."""
    grouped = defaultdict(list)
    for row in rows:
        if row["type"] == "SOTA" and row["gains"]:
            for metric, gain in row["gains"].items():
                if gain > 0 and math.isfinite(gain):
                    grouped[(row["task"], metric, row["agent"])].append(gain)
    if not grouped:
        return "<p>No completed SOTA tasks.</p>"
    series = sorted({(task, metric) for task, metric, _ in grouped})
    values = [_geomean(gains) for gains in grouped.values()]
    low = math.log10(min(min(values), 1) / 1.5)
    high = math.log10(max(max(values), 1) * 1.5)
    left, top, bar = 220, 24, 14
    body = []

    def x(value: float) -> float:
        return left + (math.log10(value) - low) / (high - low) * (WIDTH - left - 20)

    y = top
    for task, metric in series:
        body.append(
            f'<text x="{left - 8}" y="{y + bar}" class="label" text-anchor="end">'
            f"{html.escape(task)} {metric}</text>"
        )
        for index, agent in enumerate(agents):
            gains = grouped.get((task, metric, agent))
            if not gains:
                continue
            value = _geomean(gains)
            start, end = sorted((x(1), x(value)))
            body.append(
                f'<rect x="{start:.1f}" y="{y}" width="{max(end - start, 1):.1f}" '
                f'height="{bar - 2}" fill="{COLORS[index % len(COLORS)]}">'
                f"<title>{html.escape(agent)}: G = {value:.3g} (n = {len(gains)})</title></rect>"
            )
            body.append(
                f'<text x="{x(value) + 4:.1f}" y="{y + bar - 3}" class="value">{value:.2f}</text>'
            )
            y += bar
        y += 10
    for tick in range(math.floor(low), math.ceil(high) + 1):
        for factor in (1, 2, 5):
            value = factor * 10**tick
            if low <= math.log10(value) <= high:
                body.append(
                    f'<line x1="{x(value):.1f}" x2="{x(value):.1f}" y1="{top - 6}" y2="{y}" '
                    f'class="{"parity" if value == 1 else "grid"}"/>'
                    f'<text x="{x(value):.1f}" y="{top - 10}" class="tick" '
                    f'text-anchor="middle">{value:g}</text>'
                )
    return _svg(y + 10, body)


def cost_chart(rows: list[dict], agents: list[str]) -> str:
    """Coding cost against the unified task score, including rejected zeros."""
    points = [row for row in rows if row["score"] is not None and row["cost_usd"]]
    if not points:
        return "<p>No scored tasks with known coding cost.</p>"
    height, left, bottom = 320, 60, 40
    costs = [row["cost_usd"] for row in points]
    low, high = math.log10(min(costs) / 1.5), math.log10(max(costs) * 1.5)

    def x(cost: float) -> float:
        return left + (math.log10(cost) - low) / (high - low) * (WIDTH - left - 20)

    def y(score: float) -> float:
        return height - bottom - score / 100 * (height - bottom - 20)

    body = [
        f'<line x1="{left}" x2="{WIDTH - 20}" y1="{y(50):.1f}" y2="{y(50):.1f}" class="parity"/>',
        f'<text x="{left - 6}" y="{y(50) + 4:.1f}" class="tick" text-anchor="end">50</text>',
        f'<text x="{left - 6}" y="{y(100) + 4:.1f}" class="tick" text-anchor="end">100</text>',
        f'<text x="{left - 6}" y="{y(0) + 4:.1f}" class="tick" text-anchor="end">0</text>',
        f'<text x="{WIDTH / 2}" y="{height - 6}" class="tick" text-anchor="middle">'
        "coding cost (USD, log scale); task score 0–100</text>",
    ]
    for tick in range(math.floor(low), math.ceil(high) + 1):
        value = 10**tick
        if low <= tick <= high:
            body.append(
                f'<line x1="{x(value):.1f}" x2="{x(value):.1f}" y1="20" '
                f'y2="{height - bottom}" class="grid"/>'
                f'<text x="{x(value):.1f}" y="{height - bottom + 16}" class="tick" '
                f'text-anchor="middle">${value:g}</text>'
            )
    for row in points:
        score = row["score"]
        color = COLORS[agents.index(row["agent"]) % len(COLORS)]
        body.append(
            f'<circle cx="{x(row["cost_usd"]):.1f}" cy="{y(score):.1f}" r="5" fill="{color}">'
            f"<title>{html.escape(row['agent'])} {html.escape(row['task'])}: "
            f"${row['cost_usd']:.2f}, score {score:.1f}</title></circle>"
        )
    return _svg(height, body)


def slo_table(rows: list[dict], agents: list[str]) -> str:
    """Tier placement and continuous score per task and agent."""
    placed = defaultdict(list)
    for row in rows:
        if row["type"] == "SLO":
            placed[(row["task"], row["agent"])].append(row)
    if not placed:
        return "<p>No SLO tasks.</p>"
    value = slo.table()
    tasks = sorted({task for task, _ in placed})
    header = "".join(f"<th>{html.escape(agent)}</th>" for agent in agents)
    lines = [f"<table><tr><th>Task</th><th>Ladder</th>{header}</tr>"]
    for task in tasks:
        workload = task.rpartition("_")[2]
        tiers = slo.ladder(workload, value)[1]
        cells = []
        for agent in agents:
            attempts = placed.get((task, agent), [])
            scored = [row for row in attempts if row["score"] is not None]
            if not scored:
                status = ", ".join(sorted({row["status"] for row in attempts})) or ""
                cells.append(f"<td class='empty'>{html.escape(status)}</td>")
                continue
            tier = max(
                (
                    row["slo"]["tier_met"]
                    for row in scored
                    if row["slo"] and row["status"] == "complete"
                ),
                default=0,
            )
            name = tiers[tier - 1]["id"] if tier else "below " + tiers[0]["id"]
            score = statistics.fmean(row["score"] for row in scored)
            cells.append(
                f"<td><b>{html.escape(name)}</b> "
                f"{score:.2f}/100<br><small>n = {len(scored)}</small></td>"
            )
        ladder_text = " &lt; ".join(tier["id"] for tier in tiers)
        lines.append(f"<tr><td>{html.escape(task)}</td><td>{ladder_text}</td>{''.join(cells)}</tr>")
    lines.append("</table>")
    return "".join(lines)


def averages(rows: list[dict], *, tasks: set[str] | None = None) -> list[dict]:
    """Equal weight per task cell; pending or missing cells prevent a final mean."""
    tasks = {row["task"] for row in rows} if tasks is None else tasks
    result = []
    for agent in sorted({row["agent"] for row in rows}):
        cells = defaultdict(list)
        for row in rows:
            if row["agent"] == agent:
                cells[row["task"]].append(row)
        scored = {
            task: statistics.fmean(row["score"] for row in attempts)
            for task, attempts in cells.items()
            if all(row["score"] is not None for row in attempts)
        }
        types = {task: attempts[0]["type"] for task, attempts in cells.items()}
        result.append(
            {
                "agent": agent,
                "scored": len(scored),
                "tasks": len(tasks),
                "score": statistics.fmean(scored.values())
                if tasks and scored.keys() == tasks
                else None,
                **{
                    kind: statistics.fmean(scored[task] for task in selected)
                    if selected and selected <= scored.keys()
                    else None
                    for kind in ("SOTA", "SLO")
                    for selected in [{task for task, value in types.items() if value == kind}]
                },
                "rejected": sum(
                    row["status"] == "candidate_failed"
                    for attempts in cells.values()
                    for row in attempts
                ),
            }
        )
    return result


def score_table(rows: list[dict], *, tasks: set[str] | None = None) -> str:
    """One visible score per attempt and final averages with completion coverage."""

    def display(value):
        return "pending" if value is None else f"{value:.2f}"

    lines = [
        "<table><tr><th>Agent</th><th>Scored tasks</th><th>Overall /100</th>"
        "<th>SOTA /100</th><th>SLO /100</th><th>Rejected attempts</th></tr>"
    ]
    for item in averages(rows, tasks=tasks):
        lines.append(
            f"<tr><td>{html.escape(item['agent'])}</td><td>{item['scored']}/{item['tasks']}</td>"
            f"<td>{display(item['score'])}</td><td>{display(item['SOTA'])}</td>"
            f"<td>{display(item['SLO'])}</td><td>{item['rejected']}</td></tr>"
        )
    lines.append(
        "</table><p>Final averages require every task score. Rejected attempts count as zero; "
        "repeated attempts are averaged within each task first.</p>"
    )
    lines.append(
        "<table><tr><th>Task</th><th>Agent</th><th>Type</th><th>Status</th>"
        "<th>Score /100</th><th>TTFT speedup</th><th>Throughput speedup</th><th>Reason</th></tr>"
    )
    for row in sorted(rows, key=lambda row: (row["task"], row["agent"], row["run"])):
        gains = row["gains"] or {}
        lines.append(
            f"<tr><td>{html.escape(row['task'])}</td><td>{html.escape(row['agent'])}</td>"
            f"<td>{row['type'] or 'pending'}</td><td>{html.escape(row['status'])}</td>"
            f"<td>{display(row['score'])}</td><td>{display(gains.get('ttft'))}</td>"
            f"<td>{display(gains.get('throughput'))}</td><td>{html.escape(row.get('reason') or '')}</td></tr>"
        )
    return "".join(lines) + "</table>"


STYLE = """
:root { --fg: #1d1d1f; --bg: #ffffff; --muted: #6e6e73; --grid: #d2d2d7; --accent: #c0392b; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #f5f5f7; --bg: #1c1c1e; --muted: #a1a1a6; --grid: #3a3a3c; --accent: #ff6b5b; }
}
body { font: 14px/1.5 system-ui, sans-serif; color: var(--fg); background: var(--bg);
  margin: 0 auto; padding: 16px; max-width: 960px; }
h1, h2, h3 { line-height: 1.2; }
.label, .value, .tick { fill: var(--fg); font-size: 11px; }
.tick { fill: var(--muted); }
.grid { stroke: var(--grid); stroke-width: 1; }
.parity { stroke: var(--accent); stroke-width: 1.5; stroke-dasharray: 4 3; }
table { border-collapse: collapse; width: 100%; overflow-x: auto; display: block; }
th, td { border: 1px solid var(--grid); padding: 4px 8px; text-align: left; vertical-align: top; }
td.empty { color: var(--muted); }
.legend span { display: inline-block; margin-right: 12px; }
.legend i { display: inline-block; width: 10px; height: 10px; margin-right: 4px; }
"""


def render(rows: list[dict]) -> str:
    machines = {}
    for row in rows:
        key = (row["machine"], row.get("dataset", "legacy"))
        machines.setdefault(key, row["machine_label"])
    sections = []
    for (fingerprint, dataset), label in machines.items():
        selected = [
            row
            for row in rows
            if (row["machine"], row.get("dataset", "legacy")) == (fingerprint, dataset)
        ]
        runs = {row["run"] for row in selected}
        expected = {row["task"] for row in rows if row["run"] in runs}
        agents = sorted({row["agent"] for row in selected})
        legend = "".join(
            f'<span><i style="background:{COLORS[index % len(COLORS)]}"></i>'
            f"{html.escape(agent)}</span>"
            for index, agent in enumerate(agents)
        )
        sections.append(
            f"<h2>{html.escape(label)}</h2>"
            f"<p>Dataset: {html.escape(dataset)}</p>"
            f"<p class='legend'>{legend}</p>"
            "<h3>Task scores and completion</h3>"
            f"{score_table(selected, tasks=expected)}"
            "<h3>SOTA: gain over the searched reference (log scale; dashed line is parity)</h3>"
            f"{sota_chart(selected, agents)}"
            "<h3>Coding cost against task score</h3>"
            f"{cost_chart(selected, agents)}"
            "<h3>SLO: highest tier met and continuous score</h3>"
            f"{slo_table(selected, agents)}"
        )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Local eval results</title><style>{STYLE}</style></head><body>"
        f"<h1>Local eval results</h1><p>{len(rows)} task attempts across "
        f"{len(machines)} machine(s).</p>{''.join(sections)}</body></html>\n"
    )


def build(paths: list[Path], output: Path) -> list[dict]:
    rows = collect(paths)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "index.html").write_text(render(rows))
    groups = defaultdict(list)
    for row in rows:
        device = f"{row['gpu']}-{row['vram_mib']}MiB-{row['cpu_group']}-{row['cores']}cores-{row['ram_mib']}MiBram-{row['disk_mib']}MiBdisk-{row['pcie']}-{row['machine']}"
        key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{device}_{row['agent']}")
        groups[key].append(row)
    for key, values in groups.items():
        with (output / f"{key}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(
                {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
                for row in values
            )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run folders or their parents")
    parser.add_argument("--output", type=Path, help="default: the first run's results folder")
    args = parser.parse_args(argv)
    output = args.output or args.runs[0] / "results"
    rows = build(args.runs, output)
    print(f"{len(rows)} task attempts: {(output / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
