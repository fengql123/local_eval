# {{ task.model }} / {{ task.workload }}

Build the best possible inference server on this machine.

Resources: {{ image.category }}, {{ config.environment.cpus }} CPU cores,
{{ config.environment.memory_mb / 1024 }} GiB RAM, {{ config.environment.storage_mb / 1024 }} GiB disk.
GPU settings: {% if config.environment.kwargs.get("gpus") %}{{ config.environment.kwargs.gpus }} GPUs, {% endif %}`{{ config.environment.kwargs.get("gpu_types", config.environment.kwargs.get("gpu_ids", [])) | tojson }}`.

{% if task.type == "SLO" %}
Your server is scored against the service-level tiers in `/opt/eval/tables/slo.json`.
{% else %}
Improve on the reference baseline and parameters in `/inputs/reference.json`.
{% endif %}

Read-only inputs:
- Fixed checkpoint: {{ assets.checkpoint }} at `/models/target`.
- Tokenizer and chat template: `/models/tokenizer`.
- Public workload: `/inputs/public`.
- Evaluation toolkit: `/opt/eval`.

Serve the fixed checkpoint at `/models/target`. **Do not adjust its weights in any
form. Any weight modification will be flagged during verification and scored zero.**

Verification inspects the weights used for inference and checks output quality
using a hidden quality set.

## Files and installation

Put source, patches and `install.sh` in `/workspace/output/submission/`.
Keep optional experiment scripts, settings, notes and compact results elsewhere
under `/workspace/output/`. Keep these outputs lightweight.

Put downloads, virtual environments, compiled files, caches and temporary files
under `/workspace/runtime/candidate/`. Changes to third-party code must be
represented by submitted source or patches. Top-level submission names `runtime`,
`output` and `.harbor` are reserved for the suite.

`install.sh` locates source relative to its own directory, populates
`/workspace/runtime/candidate/` and creates executable `/workspace/server.launch`.
For development:

```sh
cd /workspace/output/submission
bash ./install.sh
/workspace/server.launch /models/target
```

Fresh verification copies only your submission into `/workspace` and runs:

```sh
cd /workspace
bash ./install.sh
/workspace/server.launch /models/target
```

`server.launch` accepts the checkpoint as its only argument and starts a foreground
OpenAI-compatible streaming server at **127.0.0.1:8000**, alias **served**.
Keep server configuration inside your implementation.

Honor supplied request fields. Stream accepted generated token IDs in
`choices[0].delta.token_ids` and their original decoded text in
`choices[0].delta.raw_text`, preserving whitespace and control tokens before
reasoning/tool parsing. `usage.completion_tokens` must match accepted generation.

## Experiments

```sh
/opt/eval/.venv/bin/python -I -m eval.bench.run {{ task.model | tojson }} {{ task.workload }} /workspace/server.launch --data /inputs/public{% if task.type == "SLO" %} --mode slo{% endif %}
```

Choose public concurrency with `--concurrency`. Full public replay is optional.
Results go under `/workspace/output/experiments/`.
{% if task.workload in ("S-N", "A-N") %}
Official verification uses **N={{ config.evaluation.replay_concurrency(task.workload) }}**.
S-N submits N requests together; A-N runs N session graphs per batch.
{% endif %}
S uses a 32,768-token ceiling with natural EOS. Agentic replay follows recorded
output counts with ignore_eos and preserves dependencies. Recorded tool waits
apply to A-N and branched A-MA.

## Verification and score

Verification installs, reviews, checks hidden quality and measures private replay.
Installed code is protected after preparation; designated caches and temporary
directories stay writable. Serving is offline. Each timing operation uses one
server and one excluded short warmup. Quality must pass within
**{{ config.evaluation.quality_timeout_seconds | duration }}**.

{% if task.type == "SLO" %}
Four tiers in `/opt/eval/tables/slo.json` define the 0–100 score.
{% if task.workload in ("S-N", "A-N") %}
Each tier earns `min(1, throughput / tier_throughput)`.
{% else %}
Each tier earns `sqrt(min(1, tier_TTFT / mean_TTFT) * min(1, throughput / tier_throughput))`.
{% endif %}
Score is 100 times mean credit across tiers and planned graphs/batches.
Incomplete and unstarted units earn zero. Sustained throughput below the lowest
tier for 120 seconds can stop measurement; completed-unit credit remains.
{% else %}
{% if task.workload in ("S-N", "A-N") %}
G is candidate/reference serving throughput, including prefill and queueing.
{% else %}
G is `sqrt((candidate_generation_throughput / reference_generation_throughput) * (reference_mean_TTFT / candidate_mean_TTFT))`.
{% endif %}
Score is `100 * max(0, 1 - 1/G)`.
Sustained failure to improve the scored objective for 120 seconds of eligible
measurement time can terminate measurement with score zero.
{% endif %}
Candidate gate failures earn zero.
