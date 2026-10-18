# Reference search: {{ task.model }} / {{ task.workload }}

Before anything else, comprehensively and rigorously determine whether at least
one mapped baseline can serve its corresponding checkpoint on this machine.
If none can, report unsupported with a reason for every mapped baseline.
Otherwise, select the best baseline and adjustable parameters. Choose your own
search method, experiments, subsets, scheduling and stopping point.
Full replay is mandatory only during verification.

Resources: {{ image.category }}, {{ config.environment.cpus }} CPU cores,
{{ config.environment.memory_mb / 1024 }} GiB RAM, {{ config.environment.storage_mb / 1024 }} GiB disk.
GPU settings: {% if config.environment.kwargs.get("gpus") %}{{ config.environment.kwargs.gpus }} GPUs, {% endif %}`{{ config.environment.kwargs.get("gpu_types", config.environment.kwargs.get("gpu_ids", [])) | tojson }}`.

{% if task.workload in ("S-N", "A-N") %}
Maximize serving throughput, including prefill and queueing.
Official concurrency: **N={{ config.evaluation.replay_concurrency(task.workload) }}**.
{% else %}
Maximize the geometric mean of generation throughput and inverse mean TTFT,
with equal weight.
{% endif %}
Performance data: `{{ inputs.private }}`. Verification uses these same rows.
Answer quality is checked separately.

Mapped baselines and checkpoints:
{% for engine, source in assets.baselines.items() %}
- `{{ engine }}`: `{{ source }}`.
{% endfor %}

Keep authored scripts, settings, notes and compact results under `/workspace/output/`.
Keep these outputs lightweight. Put installations, downloaded weights, builds and
caches under `/workspace/runtime/search/`. Represent source changes as source or patches.

Recipes: `/opt/eval/baselines/{{ image.category }}/<baseline>/`.
Installed launchers: `/workspace/runtime/search/baselines/{{ image.category }}/<baseline>/launch.sh`.
Use `/opt/eval/.venv/bin/python -I` with:
- `-m eval.baselines --baseline <baseline>`
- `-m eval.models <checkpoint> <destination>`
- `-m eval.bench.run` — see `--help` for experiment arguments.

Servers use **127.0.0.1:8000**, alias **served**. Use the replay toolkit for request
dependencies and tool waits. Results go under `/workspace/output/experiments/`.

Write `/workspace/output/selection.json`:

```json
{"baseline": "<baseline>", "args": [], "env": {}}
```

Parameters cannot override the fixed checkpoint, host, port or model alias.
If unsupported, write:

```json
{"supported": false, "attempts": [
  {"baseline": "<baseline>", "stage": "<stage>", "error": "<reason>"}
]}
```

Include every mapped baseline. Stages: precheck, install, acquire, launch, serve.
Transient infrastructure failures do not establish lack of support.
