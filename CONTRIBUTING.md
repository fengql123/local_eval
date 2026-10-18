# Contributing

## Development

Use Python 3.12+ and uv:

```bash
uv sync --locked --extra host
uv run pytest -q
uv run ruff check eval tests
uv run ruff format --check eval tests
```

The coordinator uses Harbor, Hugging Face datasets and Jinja. Task images install
the protected evaluator runtime and Harbor Rewardkit; native Harbor adapters own
agent installation and authentication. Baseline installers run in the sandbox,
not during image construction. Device Dockerfiles contain general prerequisites.

## Scope and tests

Keep configuration, task generation, providers, replay, grading and coding costs
with their existing owners. Search, public development and private verification
share the same replay toolkit. Docker and Daytona share its scheduling, handoffs,
workspace checkpoints and recovery; provider code handles transport and lifecycle.

Keep tests focused: two common cases and two edge cases for an area are normally
enough. Use small evidence and local HTTP/SSE fixtures. Recording and replay tests
read accepted public rows from untracked `data/dataset/`; `LOCAL_EVAL_DATA` may
point at another complete HF release. Isolate HF caches for generated test datasets.
Do not download checkpoints or start paid agents in the default suite.

Production integration checks are explicit:

```bash
LOCAL_EVAL_VERIFY_IMAGE=<production-verification-image> \
  uv run pytest -q tests/test_verification_access.py
LOCAL_EVAL_HARBOR_TESTS=1 uv run pytest -q tests/test_oracle.py
```

The first checks real Unix permissions and protected evaluator processes. Oracle
checks run native Harbor jobs, acquire models and use the configured reviewer;
they require suitable hardware, disk, network and credentials. Report these checks
separately from HTTP fixtures and actual model performance measurements.

## Data and run files

The canonical HF release has `workloads` and `quality` configurations. Model quality
ranges belong in `tables/quality.csv`; offline evidence belongs to its run. Keep
`requests[].tool` as recorded seconds, even for workloads that ignore it. Changes
to batching, tool waits, reset/warmup or timing arithmetic change the replay
protocol and must not silently resume an old measurement under new semantics.

All execution state, agent histories, handoffs, results and recovery checkpoints
belong under `runs/<uuid>/`. Do not introduce global campaign scripts or generated
artifact directories. Use temporary directories for development probes and remove
them afterward. Credentials, model caches and datasets stay out of Git.

## Review and documentation

Update the README configuration table, working example, benchmark specification
and API reference when their behavior changes. Explain the problem, resulting
behavior, review order, validation and remaining limitations. Keep examples based
on actual supported workflows. Do not commit until the owner approves; an approved
implementation phase is one conventional commit.
