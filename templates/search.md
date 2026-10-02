# Tune the reference for {{ task.model }}

This is the reference-search step for {{ task.workload }} on {{ machine.category }}.
Use `/home/search` for your work. The recorded calibration inputs are at
`/home/search/data/calibration`: a random sample of this model's private S rows,
shared across its workloads. Use S request policy for calibration; the task's
actual workload is used for the later private benchmark.
The source checkpoint `{{ assets.source.source }}`
is at `{{ assets.source.checkpoint }}`. The original model packages are read-only;
write derived weights, server launch scripts, and tuning results in `/home/search`.

Tune these baseline engines and their corresponding checkpoints:
{% for engine, asset in assets.baselines.items() %}
- `{{ engine }}`: `{{ asset.checkpoint }}` (source: `{{ asset.source }}`).
{% endfor %}

The image's `install.sh` scripts already install the baseline engines. A checkpoint
listed as a model identifier is loaded or acquired by its engine inside this
sandbox, using writable role storage. A baseline matching the source checkpoint
reuses its read-only mounted files. Checkpoint weights are not baked into the image.

The supplied baseline installers manage separate environments in this same
sandbox. Use the shared server and replay tools to measure calibration throughput
with the fixed workload request policy. An optional recording proxy is for
diagnostics outside scored timing. Search mode is
`{{ config.search.mode }}`. You may choose server settings, dependencies, tuning
approaches, and derived checkpoint formats, including quantization.

Leave the selected baseline launch scripts and measurement outputs under
`/home/search`. After your session, Harbor runs the root reference-finalization
gate. Coding starts only when that gate passes; the same container is reused.
