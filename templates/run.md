# Candidate server

Create an executable `/workspace/launch.sh` that starts an OpenAI-compatible chat
server on `127.0.0.1:8000`, serving the model name `served`. Keep the server in the
foreground so the evaluator can start, stop, and restart it for independent runs.

The evaluator sends recorded messages, tool definitions, sampling settings,
thinking settings, and the workload's output-count policy. It runs the pre-run
quality gate before timing the benchmark. Replay measures timings and stream
counts; answer grading belongs to that gate.

Your public task data is at `/workspace/data/public`. The source checkpoint
`{{ assets.source.source }}` is at `{{ assets.source.checkpoint }}`. You may
quantize it or build any derived serving format inside `/workspace`.

Available baseline checkpoints:
{% for engine, asset in assets.baselines.items() %}
- `{{ engine }}`: `{{ asset.checkpoint }}` (source: `{{ asset.source }}`).
{% endfor %}

The category image's `install.sh` scripts install the engines. Distinct baseline
checkpoints remain model identifiers for the runtime to load or acquire inside
this sandbox. A matching source checkpoint reuses `/models/source`. Keep runtime
downloads and derived outputs in writable role storage; weights are not image assets.

The search user has its own calibration view under `/home/search`; its home
directory is inaccessible to the coding user. The root-owned evaluator retains
private benchmark and quality inputs. Evaluation uses the same image and sandbox
as the reference search and your coding session.
