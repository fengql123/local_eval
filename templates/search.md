# Tune the reference for {{ task.model }}

This is the reference-search step for {{ task.workload }} on {{ machine.category }}.
Use `/home/search` for your work. The recorded calibration inputs are at
`/home/search/data/S/rows.jsonl`: a random sample of this model's private S rows,
shared across its workloads. Use S request policy for calibration; the task's
actual workload is used for the later private benchmark.
The source checkpoint `{{ assets.source.source }}`
is at `{{ assets.source.checkpoint }}`. The original model packages are read-only;
write derived weights, launch parameter files, and tuning results in `/home/search`.

Tune these baseline engines and their corresponding checkpoints:
{% for engine, asset in assets.baselines.items() %}
- `{{ engine }}`: `{{ asset.checkpoint }}` (source: `{{ asset.source }}`).
  Supplied launcher: `/opt/eval/images/{{ machine.category }}/baselines/{{ engine }}/launch.sh`.
{% endfor %}

Each baseline folder provides `install.sh` and `launch.sh`. The image's
`install.sh` scripts already install the baseline engines. The measurement runner
selects the same protected launcher for every trial. It injects the mapped model
as `CHECKPOINT` and the selected port as `PORT` in the process environment. The
script binds to `127.0.0.1`, exposes model name `served`, and owns the engine
command, installed environment, and default settings. The runner launches the
server, waits for readiness, then warms up and replays calibration requests.

For example, measure the first supplied baseline from `/home/search` with model,
S workload, and its protected launch script:

```bash
cd /home/search
/opt/eval/.venv/bin/python -I -m eval.bench.run "{{ task.model }}" S-1 /opt/eval/images/{{ machine.category }}/baselines/{{ assets.baselines | first }}/launch.sh
```

The launch path identifies the category and engine, so the runner chooses their
request policy and mapped checkpoint automatically. It loads `data/S/rows.jsonl`
from this working directory and selects this model's rows. The server uses a free
localhost port. Each invocation writes its report, server log, and launch metadata
to `bench/<uuid>/` here. Omit tuning parameters to measure the supplied defaults.

Tune with a JSON parameter file containing `{"args": [...], "env": {...}}` and
pass it to the benchmark CLI through `--params JSON_FILE`. Arguments are forwarded
through the launcher's `"$@"`; additional environment values are supplied to its
process. The trusted runner supplies the protected backend flag aliases and
environment names. It rejects protected flags, including `--flag=value`, and changes to
`CHECKPOINT`, `PORT`, or protected environment names. Keep defaults in the supplied
script and tune only its remaining arguments and environment. Store the selected
parameters rather than replacing the baseline script or overriding its checkpoint
or served model.

A checkpoint listed as a model identifier is loaded or acquired by its engine inside this
sandbox, using writable role storage. A baseline matching the source checkpoint
reuses its read-only mounted files. Checkpoint weights are not baked into the image.

The supplied baseline installers manage separate environments in this same
sandbox. Use the shared server and replay tools to measure calibration throughput
with the fixed workload request policy. An optional recording proxy is for
diagnostics outside scored timing. Search mode is
`{{ config.search.mode }}`. You may choose server settings, dependencies, tuning
approaches, and derived checkpoint formats, including quantization.

Leave the selected baseline parameter files and measurement outputs under
`/home/search`. After your session, Harbor runs the root reference-finalization
gate. Coding starts only when that gate passes; the same container is reused.
