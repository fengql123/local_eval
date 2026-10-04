# Implement serving for {{ task.model }}

This task measures {{ task.workload }} on {{ machine.category }}. A separate
native Harbor search task completed its reference gate before this coding task. Read `/workspace/run.md`
for the interface and `/workspace/reference.json` for the selected baseline and
checkpoint. Public task inputs are at `/workspace/data/{{ pool }}/rows.jsonl`;
public S questions at `/workspace/data/S/rows.jsonl` support answer-quality checks
on every workload. The original source weights are read-only at `/models/source`.

Build the implementation and its source scripts in `/workspace/submission`.
You may choose any implementation, dependencies, server settings or optimization,
including quantization. Use `/workspace/runtime` for installed dependencies and
`/workspace/weights` for derived weights. Preserve authored requests and scoring.

Submit `install.sh` and `launch.sh`; add `convert_weight.sh` if weights need
conversion. Only `/workspace/submission` transfers to verification. Keep source
code there, and keep weights, environments, caches and compiled runtime outputs
outside it. Submit self-contained source without links to code outside the
submission. Do not rely on state created during coding: our wrapper transfers only
the submission into the verification task using the same image. Its fresh container
runs ordered prepare, review and measure steps. Preparation reruns optional
conversion and installation with read-only source and selected baseline weights;
Harbor Rewardkit inspects the resulting implementation before quality and timing.

`install.sh` reproduces dependencies and compilation. `convert_weight.sh`, when
present, reproduces derived weights. `launch.sh` must start the foreground server
at **127.0.0.1:8000**, advertising **served**. A completed launcher contains its
correct resulting weight path and optimized parameters and runs unchanged. An
unfilled baseline CHECKPOINT template receives the selected reference checkpoint;
there is no source-weight fallback. No PORT is injected; another port fails startup.

Conversion, installation, server launch, quality/timing and reporting run as root
in the fresh verification container. Native task TOML selects `root` for prepare
and measure and `judge` for review. Only the Rewardkit judging agent runs as the
restricted `judge` user. Its private workspace is writable; submission, installed
runtime and weights are readable, and private evaluator data is inaccessible.
Installation and review run online; quality/timed serving uses only loopback.
Candidate failures earn zero rewards; reference, reviewer or environment failures
produce no candidate score.
