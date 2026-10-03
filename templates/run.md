# Candidate server

Create an executable `/workspace/launch.sh` that starts an OpenAI-compatible chat
server. The runner injects the source checkpoint as `CHECKPOINT` and the selected
port as `PORT` in its environment. Bind to `127.0.0.1` and expose model name
`served`. Keep the server in the foreground so the evaluator can start, stop, and
restart it for independent runs.

The script contains the engine command and its defaults. Read `CHECKPOINT` and
`PORT` from the environment and forward extra tunable arguments to the engine
using `"$@"`.

Extra launch parameters have the form `{"args": [...], "env": {...}}`; the runner
passes their arguments to `bash launch.sh` and their environment to the process.
It rejects fixed flags, including `--flag=value` forms, and rejects changes to
`CHECKPOINT`, `PORT`, or protected environment names. The trusted runner supplies
the protected backend flag aliases and environment names. The benchmark CLI
accepts tunable parameters through `--params JSON_FILE`. Defaults belong in the
script.

The evaluator sends recorded messages, tool definitions, sampling settings,
thinking settings, and the workload's output-count policy. It runs the pre-run
quality gate before timing the benchmark. Replay measures timings and stream
counts; answer grading belongs to that gate.

Your public task data is at `/workspace/data/{{ pool }}/rows.jsonl`. The source checkpoint
`{{ assets.source.source }}` is at `{{ assets.source.checkpoint }}`. You may
quantize it or build any derived serving format inside `/workspace`.

Run the public measurement from `/workspace` with three inputs: model, workload,
and launch script.

```bash
cd /workspace
/opt/eval/.venv/bin/python -I -m eval.bench.run "{{ task.model }}" "{{ task.workload }}" /workspace/launch.sh
```

The runner loads `data/{{ pool }}/rows.jsonl`, selects this model's rows, chooses
a free localhost port, starts the server, and measures replay. Each invocation
creates `bench/<uuid>/` under the working directory with `report.json`,
`server.log`, and launch metadata. Add `--params JSON_FILE` to apply saved tuning;
omitting it uses the launch script's defaults.

Available baseline checkpoints:
{% for engine, asset in assets.baselines.items() %}
- `{{ engine }}`: `{{ asset.checkpoint }}` (source: `{{ asset.source }}`).
  Launcher: `/opt/eval/images/{{ machine.category }}/baselines/{{ engine }}/launch.sh`.
{% endfor %}

Each category baseline folder supplies `install.sh` and `launch.sh`. The image's
`install.sh` scripts install the engines. The measurement runner selects a
baseline's protected `launch.sh`, injects its mapped checkpoint and port, applies
tunable parameters, waits for the endpoint to respond, and then starts warmup and
replay. A launch path under `images/<category>/baselines/<engine>/launch.sh`
selects that category and baseline's request policy and mapped checkpoint; other
paths use candidate policy and `/models/source`. The launcher owns backend-specific
flags and its installed venv or native prefix. Distinct baseline checkpoints remain
model identifiers for the runtime to load or acquire inside
this sandbox. A matching source checkpoint reuses `/models/source`. Keep runtime
downloads and derived outputs in writable role storage; weights are not image assets.

The search user has its own calibration view under `/home/search`; its home
directory is inaccessible to the coding user. The root-owned evaluator retains
private benchmark and quality inputs. Evaluation uses the same image and sandbox
as the reference search and your coding session.
