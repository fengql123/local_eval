# local_eval

A benchmark for coding agents that improve model serving on local-class hardware.
The coordinator runs on your machine; tasks run through Harbor in local Docker or
Daytona spot GPU sandboxes. Search selects a reference, a coding agent submits a
server, and a fresh verifier checks its rules, quality and performance.

## Install

Python 3.12+ and `uv` are required on the coordinator. Docker runs also need a
working Docker daemon and NVIDIA Container Toolkit for NVIDIA tasks.

```bash
./install.sh
# Include the cloud provider when needed:
uv sync --locked --extra daytona
```

Set provider and agent credentials in your shell before starting. Harbor's native
adapters handle agent installation and authentication. Credentials do not belong
in the benchmark configuration or dataset. For native Harbor Codex subscription
authentication, set `CODEX_AUTH_JSON_PATH` to your existing private auth file; other
adapters use their documented environment variables.

## Run

Edit [config.example.yaml](config.example.yaml), then run:

```bash
uv run python -m eval.run config.example.yaml
```

The command prints `runs/<uuid>`. Keep that folder to resume the complete run:

```bash
uv run python -m eval.run --resume runs/<uuid>
```

Initialization saves the configuration and coordinator before preparation begins.
If it fails or is interrupted with Ctrl-C, the new run folder is removed; rerun the
same configuration command to start cleanly. Failures after initialization preserve
the run folder for `--resume`.

Finished and rejected tasks are skipped. Unfinished phases resume their workspace,
native conversation and replay/quality checkpoints. `max_retries` controls automatic
recovery attempts after the initial attempt; explicitly resuming starts another
recovery cycle. Infrastructure retries wait 10s, 20s, 40s, then 60s for each
remaining retry. Infrastructure failure in one task does not stop unrelated tasks.
A native agent quota rejection with a reset timestamp puts its unfinished work in
`waiting` without a sandbox or resource slot. It resumes the saved conversation
after that reset; completed coding can continue to verification with another agent.
No commit or background daemon is created by these commands. For a detached
terminal, run the command in your own tmux session.

The [device profiles](docs/devices.md) list the approved Daytona workstation
allocations and their ready-to-run configurations.

## Preparation

Preparation exports the selected workload views and standalone quality questions.
It generates the search tasks without assigning SOTA/SLO. A supported reference
resolves a task as SOTA; an unsupported verdict resolves it as SLO. Empty baseline
mappings resolve without starting a search agent. Infrastructure failures stay
unfinished. Coding and verification definitions are generated after that decision.
A supplied `search.reference` skips the search job and its image build.
Task generation publishes each `task.toml` last; resume regenerates an incomplete
coding/verification pair.

Each generated task owns `environment/Dockerfile` and trusted `environment/setup.sh`.
Harbor builds or reuses that phase environment when its job starts. The provider
uploads approved inputs and runs setup as root before the restricted agent starts.
Setup installs the evaluator runtime, acquires the fixed target where required,
and protects dynamically supplied files. Dockerfiles define static users and
workspaces. Candidate `install.sh` remains separate and runs under its restricted
verification identity.

Quality exports write to `questions.partial/` and publish `questions/` only after
saving succeeds. Interrupted exports are discarded and recreated during preparation.

## Configuration

`concurrency` limits active task sandboxes across search, coding and verification.
Each task advances as soon as its own handoff is ready. `evaluation.concurrency`
sets official replay batch sizes, either one integer or a mapping such as
`{S-N: 32, A-N: 8}`. Each N must be a positive multiple of eight dividing its
private split. Serial workloads remain serial. Public experiments may choose
any positive `--concurrency`, or `-1` to admit all rows; they may finish with a
smaller final batch.

| Field | Default | Meaning |
| --- | --- | --- |
| `dataset` | required for a run | Absolute local HF dataset directory or HF dataset repository ID. |
| `concurrency` | `1` | Maximum active task sandboxes. |
| `max_retries` | `3` | Recovery attempts per interrupted phase, after its initial attempt. |
| `out` | `runs` | Parent directory for UUID runs. Relative to the YAML file. |
| `tasks.models` | `all` | Model names from `tables/models.csv`; canonical checkpoint strings also resolve. |
| `tasks.workloads` | `all` | S-1, S-N, A-1, A-N, A-MA. |
| `environment.provider` | `docker` | `docker` or `daytona`. |
| `environment.device_category` | `nvidia-cuda-x86` | Selects authored Dockerfiles, recipes and checkpoint mappings. |
| `environment.cpus` | `8` | CPU allocation. |
| `environment.memory_mb` | `65536` | RAM allocation in MiB. |
| `environment.storage_mb` | `524288` | Cloud sandbox disk in MiB; Docker uses host disk. |
| `environment.kwargs` | `{}` | Native provider options: local `gpu_ids`; cloud `gpus`, `gpu_types`, `spot`, etc. |
| `environment.import_path` | unset | Custom Harbor environment adapter. |
| `agent` | required unless `agents` is used | Native coding agent configuration. |
| `agents` | unset | A mapping of labels to coding agent configurations, sharing search within this run. |
| `search.agent` | required | Native search agent configuration; budget defaults to 2h. |
| `search.reference` | unset | Explicit JSON selection or directory of `<task-id>.json` files from another run. Copied into this run. |
| `evaluation.agent` | required | Rewardkit reviewer, run as the restricted judge user; budget defaults to 30m. |
| `evaluation.concurrency` | `{S-N: 32, A-N: 8}` | Official N, shared integer or mapping such as `{S-N: 32, A-N: 8}`. |
| `evaluation.install_timeout` | `4h` | Installation deadline. |
| `evaluation.health_timeout` | `1h` | Server startup deadline. |
| `evaluation.quality_timeout` | `5h` | Quality deadline; unfinished answers count as wrong. |
| `evaluation.slo_budget` | `6h` | Active replay budget for SLO timing. |
| `evaluation.slo_request_cap` | `10m` | Cap for an SLO serial request. |
| `evaluation.slo_repeats` | `1` | Independent SLO measurement repetitions. |

Each agent mapping accepts `name`, optional `model`, native `kwargs`, and `budget`
(seconds or strings such as `24h`). Every agent must explicitly set
`kwargs.reasoning_effort` to a non-empty string; missing values fail configuration
loading before any sandbox starts. The native adapter handles supported effort
levels. Custom coding/search agents use Harbor's import
path format, for example `my_package.agent:Agent`, and may omit `model`. Their native
Harbor trajectory supplies coding usage. Search and review usage are not coding costs.

To compare coding agents in one run, replace `agent` with:

```yaml
agents:
  codex:
    name: codex
    model: gpt-6.1-sol
    kwargs: {reasoning_effort: xhigh}
    budget: 24h
  my_agent:
    name: my_package.agent:Agent
    kwargs: {reasoning_effort: xhigh}
    budget: 24h
```

Search is shared only inside this run. Reuse across runs requires an explicit
`search.reference`; no global search cache is consulted.

For local parallel tasks, give disjoint GPU groups:

```yaml
concurrency: 2
environment:
  provider: docker
  kwargs:
    gpu_ids: [[0, 1], [2, 3]]
```

Each inner list belongs to one task sandbox. A flat list such as `[0, 1]` is one
two-GPU allocation, so GPU work is serialized. Cloud tasks use native GPU resource
options; [examples/cloud.daytona.yaml](examples/cloud.daytona.yaml) selects spot
instances. Source Dockerfiles are built at the selected provider. The coordinator
never uploads a prebuilt image. There are no managed persistent volumes.

## Data

The complete release lives in untracked `data/dataset/`. Local folders and HF
repositories expose the same two configurations:

```python
from datasets import load_dataset

source = "/absolute/path/to/local_eval/data/dataset"  # or "owner/benchmark"
records = load_dataset(source, "workloads", split="records")
quality = load_dataset(source, "quality", split="questions")
qwen_an = records.filter(lambda row: row["model"] == "Qwen3.8-Flash-Next"
                        and "A-N" in row["private"])
```

Each recorded graph is stored once. Its five columns are `id`, `model`,
`requests`, `public` and `private`. The final two list the workloads that select
the graph. Both empty means a retained graph outside the official selections.
A graph cannot have both public and private memberships. Every request retains
`id`, `parents`, exact JSON `messages`, optional JSON `tools`, output count `out`
and tool wait `tool` in seconds after the latest parent finishes. The evaluator
does not execute tools or replace recorded prefill histories with new answers.

All 2,179 original graphs are retained, including 857 private and 240 public
selected graphs. S has 32 private / 8 public rows per model. A-1 keeps its approved
selection. A-N has 32 or 40 private graphs; A-MA has 31–48. Public views are about
one-quarter the corresponding private size. Memberships can overlap between
workloads within the same public/private population.

A-N sampling balances total tool wait. A-MA balances tool-only critical-path
wait and keeps linear sessions in a zero-wait tier. Compute bins cover request
count, prefill, generation and graph width. `sampling.json` documents the approved
method, seeds, bins, selected IDs and stress cases; it is not read during evaluation.
Audit memberships with `python -m examples.sample_workloads data/dataset --check`.
The coordinator exports only `id`, `model` and `requests` from the selected view.
Full data and sampling documentation never enter agent sandboxes.

The independent quality configuration has 30 hidden questions. Its offline
32-round bounds live in [tables/quality.csv](tables/quality.csv). Keep the full HF
release private or gated; only the coordinator needs access to it. Standard
`DatasetDict.push_to_hub(..., config_name="workloads")` publishes the `records`
split; `config_name="quality"` publishes `questions` in the same repository.

### Catalogs and checkpoints

| File | Definition |
| --- | --- |
| `tables/models.csv` | One model name and fixed candidate checkpoint per model. |
| `tables/tasks.csv` | Task ID, model and workload. SOTA/SLO is resolved at runtime. |
| `tables/baseline_map.csv` | Device/model/baseline to reference checkpoint. |
| `tables/request_fields.json` | API, thinking and generation policies. |
| `tables/quality.csv` | Published model quality requirements and offline evidence. |
| `tables/slo.json` | Numerical SLO thresholds and scoring settings. |

Coding and candidate verification always use `models.csv`, including its tokenizer
and chat-template files. Search selects a reference engine and tuning parameters;
that engine uses its own mapped checkpoint. For example, selecting Qwen Next's
llama.cpp GGUF reference leaves the candidate fixed to RadixArk NVFP4. Verification
records both checkpoints and reuses the acquired files when they are identical.


## Replay and recording

Search, coding tools and verification call the same replay implementation. S-N
submits N independent requests at once; A-N follows N session graphs. There is a
barrier between batches. Serial workloads run one session at a time. Graph parents
are preserved. A-N replays tool waits while the other sessions keep progressing.
A-MA replays waits throughout sessions with parallel subagent branches; linear
A-MA sessions omit them. S-1, S-N, A-1 and quality omit tool waits. Waits control
request readiness; they are not counted as server decode/busy time. Replay preserves this policy alongside its fixed N batches and server lifetime.

```bash
uv run python -m eval.bench.run Qwen3.6-35B-A3B S-N /path/to/launch.sh \
  --data /path/to/scoped/saved-dataset --concurrency 8
uv run python -m eval.bench.run --resume runs/<operation-uuid>
```

The toolkit prints its operation folder before server startup and continually
journals request progress. A completed batch/session is committed atomically. The
interrupted unit restarts in full, with the same batching. Each replay starts its
owned server once, warms it once, processes all remaining units, then stops it.
Resume starts and warms a new server before the first unfinished whole unit.
Startup and warmup are excluded from timing; caches may persist between units.
Attached development servers remain the caller's responsibility. Protocol 3
timing journals must be rerun; completed quality answers remain reusable.

Quality mode (`--mode quality`) admits independent questions together (`-1`) and
reports accuracy. Every completed answer, including a wrong one, is retained.
Recording (`--mode record`) starts harness workers against the shared proxy,
returns request graphs and calculates no timing/accuracy report. See
[examples/record.servers.yaml](examples/record.servers.yaml).

## Results and files

All execution state belongs to its run:

```text
local_eval/
  eval/ images/ tables/ templates/ tests/ docs/ examples/
  data/dataset/                    # Untracked canonical HF dataset release.
  runs/<uuid>/
    run.yaml                       # Frozen config; resume needs only this folder.
    state.json                     # Phase status, current native attempts, retry cycle.
    prepared.json                  # Unclassified task selection, scoped inputs and task paths.
    input/                         # Approved HF dataset views and private quality input.
    host/                          # Frozen coordinator used by --resume.
    payload/                       # Frozen sandbox evaluator and baseline recipes.
    references/ provided-reference/
    tasks/{search,agent,evaluation}/<task-id>/
    jobs/<phase-task>/...          # Harbor logs, results, conversation and checkpoint.
    resume/                        # Immutable continuation task definitions.
    agents/<label>/                # Only when multiple coding agents are selected.
    costs.json cost-summary.json
    result.json results/           # HTML/JSON/CSV, including rejected zero scores.
```

New runs create no top-level `artifacts/`, `results/`, `record/` or checkpoint caches.
One-off development probes use temporary directories and remove them afterward.

The verifier fixes the candidate checkpoint and tokenizer from `models.csv`. A candidate submits code
and `install.sh` in `/workspace`. Fresh verification runs `bash ./install.sh`
there, then `/workspace/server.launch /models/target`. The project layout is otherwise
unrestricted; installation writes stay under `/workspace`. Submitted weights are not transferred. Review runs
as `judge`, with its own writable home and read-only candidate artifacts. Serving
users cannot read private questions, trusted evaluator code or reviewer credentials.
See [the execution and access reference](docs/sandboxing.md).

Every task produces one score from 0 to 100. S-1/A-1/A-MA combine throughput
and inverse mean TTFT with equal geometric weight. S-N/A-N score serving
throughput, including prefill and queuing. SOTA scores improvement over the
reference: 0 at parity, 50 at 2×, 75 at 4×. SLO scores attainment of four fixed
tiers; incomplete units earn zero. See [the benchmark specification](docs/benchmark.md)
and [SLO targets](docs/slo.md).

Final averages give every task equal weight and include rejected zeros. They
remain pending until every task has a score. Reports retain component speedups,
completion coverage, hardware and coding-only costs. Mean completion time is not
scored; infrastructure interruptions remain retryable.

## Development

```bash
uv run pytest
uv run ruff check eval tests
```

The tests use local HTTP servers and prepared request graphs. Production Harbor
oracle and permission checks are opt-in because they start real containers, load
models and may use paid reviewers. [docs/quality.md](docs/quality.md) explains offline
quality preparation; [docs/reference.md](docs/reference.md) gives the reading order.
