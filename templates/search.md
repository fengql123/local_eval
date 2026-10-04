# Tune the reference for {{ task.model }}

This is reference search for {{ task.workload }} on {{ machine.category }}.
Work in `/home/search`. `/home/search/data/S/rows.jsonl` (`data/S/rows.jsonl`
relative to this workspace) is a random sample of this model's
private S recordings, shared across workloads. Use S request policy for calibration;
the later private benchmark uses the task's actual workload. The source
checkpoint `{{ assets.source.source }}` is mounted read-only at
`{{ assets.source.checkpoint }}`. Each supplied baseline checkpoint is also
mounted read-only. Keep parameter files, temporary runtime data and measurements
in `/home/search`; mounted model weights are immutable. Search passes settings,
not environments, caches or weight files, to the next phase.

Tune these installed engines and checkpoints:
{% for engine, asset in assets.baselines.items() %}
- `{{ engine }}`: `{{ asset.checkpoint }}` (source: `{{ asset.source }}`).
  Launcher: `/opt/eval/images/{{ machine.category }}/baselines/{{ engine }}/launch.sh`.
{% endfor %}

Each folder supplies its own install.sh and launch.sh. The shared image already
installed engine software in isolated venv/native environments. Weights are not image assets: source-equal checkpoints reuse `/models/source`;
distinct prepared packages mount under `/models/baselines/<engine>`. Native engine
identifiers stay identifiers. Use these supplied checkpoints, not downloaded
replacement weights.

The measurement runner uses the protected launcher and its defaults. If its
checkpoint input is unfilled, model/workload and baseline/device mapping supply
CHECKPOINT. A completed launcher owns its weights and runs unchanged. Every
server listens on **127.0.0.1:8000**, advertises **served**, and stays in the
foreground. The runner does not inject PORT or choose another port; wrong
port/alias, early exit and occupied endpoints fail startup.

Measure the first installed baseline from this calibration workspace:

```bash
cd /home/search
/opt/eval/.venv/bin/python -I -m eval.bench.run "{{ task.model }}" S-1 /opt/eval/images/{{ machine.category }}/baselines/{{ assets.baselines | first }}/launch.sh
```

The baseline path identifies engine/category. Timing loads this model's S rows,
starts the server, excludes one warmup, replays, stops it and saves report.json
plus server.log in bench/<uuid>. `--concurrency N` limits active independent rows;
`--concurrency -1` admits the full sample. Dependencies and applicable tool delays
are still respected. For accuracy without warmup/timing, add
`--mode quality --concurrency -1`; row-selected Python checkers grade answers and
the report contains final accuracy only.

Tune with `--params JSON_FILE`, where the file contains
`{"args": [...], "env": {...}}`. Args follow the script and remaining environment
is forwarded. The runner protects model/host/port/served identity, CHECKPOINT,
PORT, BASH_ENV, ENV and trusted backend-owned aliases/names. Store selected
settings rather than replacing the protected baseline script. Other server
settings are unrestricted within the fixed model/endpoint interface. Any model
conversion needed by the final candidate belongs to its reproducible coding
submission, not the reference parameter handoff.

Search mode is `{{ config.search.mode }}`. An optional recording proxy is for
diagnostics outside scored timing. Write the selected baseline and adjustable
parameters to `/home/search/selection.json`:

```json
{"baseline": "{{ assets.baselines | first }}", "args": [], "env": {}}
```

Use the baseline name from its measured report and the same args/env passed via
`--params`. The root collection hook reads this selection, checks the protected
launch interface, derives category/checkpoint from trusted mappings and writes
`reference.json`, `public.json` and its gate report under `/root/eval/reference`.
Harbor collects that protected directory before the verifier reads the saved gate.
On success, the wrapper supplies it read-only to a new coding
task. No search runtime or cache transfers. Coding and verification separately
mount source weights and the selected reference weights read-only; their
containers reuse this category image. Search evidence is retained, and
search-agent billing is not recorded.
