# Candidate serving and verification

Keep source in `/workspace/submission`, installed dependencies in
`/workspace/runtime`, and derived weights in `/workspace/weights`. Submit
`install.sh` and `launch.sh`; optionally submit `convert_weight.sh` to reproduce
conversion. Submit self-contained source without external code links. Our wrapper
transfers only this directory into the verification task. Its fresh container uses
the same category image and runs prepare, review and measure in order.
Coding-session weights, environments, caches, processes and user homes do not transfer.

The root verifier gives conversion and installation the source checkpoint. Both run
as root with internet and the configured installation deadline. Optional conversion
runs first, then install.sh reproduces dependencies and compilation. The image's
trusted preparation script protects the resulting source/runtime/weights from writes
by the judging agent; root retains write access. Quantization
and serving optimizations are allowed; there is no source-size cap or tuning list.

Create a runnable launcher with its final weights, environment and optimized
parameters. It must remain in the foreground and serve
**http://127.0.0.1:8000**, advertising **served**. Complete scripts run unchanged;
no candidate checkpoint is assumed. A copied template with an unfilled
`$CHECKPOINT` or `${CHECKPOINT:?…}` receives the selected reference checkpoint.
Filled assignments and nonempty defaults already supply weights. PORT is never
injected. Wrong port or alias, early exit or an occupied endpoint fails startup.

Extra public-development tuning uses `{"args": [...], "env": {...}}` through
`--params JSON_FILE`. Protected model/host/port/served-name flags and CHECKPOINT,
PORT, BASH_ENV, ENV cannot be overridden. Absolute /bin/bash executes the launcher.
Final settings should be in launch.sh so the fresh verifier can reproduce them.

The complete verification flow is:

```text
search task: chosen reference parameters → protected metadata handoff
  → fresh coding task: implementation → root format gate → collect submission only
  → fresh verification task: fresh container from the same image
  → prepare/root: checks → optional convert_weight.sh → install.sh → readable artifacts
  → review/judge: Harbor Rewardkit rules review
  → measure/root: reference/candidate S-quality replay → paired quality gate
  → measure/root: fresh reference/candidate timed replay → reports and rewards
```

Only the reviewer runs as `judge`, with its own writable home, cache and review
output. It can read the candidate's code, dependencies and weights but cannot
change them or access the private evaluator inputs beneath /root/eval. The native
verification task TOML selects `root`, `judge`, `root` for the three verifier
steps. Harbor executes those users directly; there is no local user-switching
launcher. The image's `/opt/eval/images/prepare.sh` sets artifact ownership and
prepares the judge's writable directories during the root prepare step. Server
launch, quality, timing, scoring and reporting remain root operations. Serving is loopback-only in
IPv4/IPv6 after online preparation and review. Cleanup stops owned process groups
before egress rules are removed. Candidate failures
produce zero; reference, reviewer or environment failures produce no score.

Quality uses the same saved S subset for every workload, concurrency -1, complete
responses, row-selected accuracy functions and paired comparison. It has no warmup
or timing metrics. A passed gate permits fresh timing with excluded warmups.
Throughput and TTFT rewards are separate: twice as fast earns 50; equal or slower zero.

Public inputs are `/workspace/data/{{ pool }}/rows.jsonl`; public S questions also
exist at `/workspace/data/S/rows.jsonl`. Source `{{ assets.source.source }}` is
mounted read-only at `{{ assets.source.checkpoint }}`. The selected reference is
published at `/workspace/reference.json`; its weights are staged separately at
`/models/reference` (source-equal references reuse /models/source).

Run timing from /workspace after reproducing the candidate installation:

```bash
cd /workspace
/opt/eval/.venv/bin/python -I -m eval.bench.run "{{ task.model }}" "{{ task.workload }}" /workspace/submission/launch.sh
```

Run public accuracy with the same launcher:

```bash
/opt/eval/.venv/bin/python -I -m eval.bench.run "{{ task.model }}" "{{ task.workload }}" /workspace/submission/launch.sh --mode quality --concurrency -1
```

Timing loads the task pool and reports performance. Quality loads S regardless of
the supplied workload and reports accuracy only. Both create bench/<uuid> with
report.json and server.log. `--concurrency N` limits independent rows/sessions;
-1 admits all rows while respecting parents and applicable tool delays.

Installed baseline engines and checkpoint identifiers:
{% for engine, asset in assets.baselines.items() %}
- `{{ engine }}`: `{{ asset.checkpoint }}` (source: `{{ asset.source }}`).
  Launcher: `/opt/eval/images/{{ machine.category }}/baselines/{{ engine }}/launch.sh`.
{% endfor %}

Setup supplies source and mapped baseline weights; search mounts them read-only
and tunes the installed baseline engines. Its root finalizer publishes only
reference settings and checkpoint identity. The wrapper transfers that metadata
to this coding task and mounts the selected baseline at `/models/reference`.
Other baseline weights and search-session state are absent here.

Search, coding and verification each use a separate container from the same
category image. Coding receives public rows and the selected reference; it cannot
read search calibration or private verification rows. Only submitted code/scripts
transfer from coding to verification. The verification task's prepare, review and
measure steps share one fresh container. Harbor's no-op agent makes no model
calls; the review verifier invokes Rewardkit. The wrapper publishes the final
verification rewards.
