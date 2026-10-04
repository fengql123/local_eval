# local_eval

Evaluate how well a coding agent implements a local model-serving system. The
finished benchmark compares a candidate against a tuned reference using recorded
requests, a pre-run quality gate, and timing measurements. Harbor owns the task
sandbox and the agent lifecycle through its existing adapters.

**Status: Phases 1–3 committed; Phase 4 implemented and awaiting review.**
Final verification uses a fresh container from the same category image. It checks
the submission, reproduces optional weight conversion and installation as root,
and reviews rules and answer quality before timing. Only the Rewardkit judging
agent runs with restricted access. Native Harbor steps select root/judge/root;
the image's preparation script sets artifact ownership before review.
It reuses the shared server, streaming client and replay modules. The wrapper runs
three native Harbor jobs: search, coding and verification. Each uses a separate
container from the same device image and selects its agent directly. Production device
images and engine scripts await owner supply in Phase 5; Phase 6 validates measured
search/coding and the complete evaluation. See [Phase 4's review packet](docs/phase-04.md).

## Quickstart

Use Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
From this directory:

```bash
./install.sh
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run python examples/inspect_foundations.py config.example.yaml
uv run python -m eval.run --help
```

The default tests need only a CPU, local files and loopback HTTP. They do not download
models, build images, start Harbor jobs, or make paid agent calls. Four opt-in tests
exercise actual Harbor and Docker; see verification below. Phase 3–4 tests
read the supplied public recordings from the sibling
`edge-harness-v2/local_eval_data/`; set `LOCAL_EVAL_DATA` to use another local copy.
Renderer tests
also require the Docker Compose CLI, without a running daemon. The inspector reads
the real example configuration and catalogs, selecting Qwen3.6-35B-A3B / S-1 and
showing its original model checkpoint, FreeToken checkpoint, inferred installer
path, and request fields. It does not install or start that engine.

## Run and resume

The implemented entry points are:

```bash
uv run python -m eval.run run.yaml
uv run python -m eval.run --resume runs/<run-id>
```

For a new run, copy `config.example.yaml` to `run.yaml` and supply your dataset and
coding/search adapters. Set their models when needed by the selected integration.
The wrapper checks
Docker/Compose/Harbor and selected GPUs, checks the supplied category image
assets, prepares inputs/checkpoints, and awaits three Harbor jobs in the current
process. It prints the UUID folder as soon as it is
created. There is no foreground flag or background mode. Trials execute serially
because they share the selected devices.

Resume takes only that folder: its `run.yaml` holds the resolved settings, and
`prepared.json` identifies reusable inputs/assets/tasks. Interrupted preparation
recreates generated input/task files while retaining checkpoint directories. The
normal Hugging Face CLI resumes or reuses downloads through its own local metadata.
Cancelled trial costs are saved before cleanup; completed trials, including a
zero reward, remain available to Harbor. Resume assumes the same dataset version
and preserves the prepared data without downloading it again.

These entry points are implemented and exercised at their Harbor boundary in
tests. A complete benchmark additionally needs the supplied image/scripts and
working engine launchers and real weight shards; this checkout does not yet run a production
evaluation. See [Phase 2's review packet](docs/phase-02.md) for executed examples
and the precise validation boundary.

## Replay prepared recordings

The CLI takes model, workload and launch script:

```bash
python -m eval.bench.run MODEL WORKLOAD LAUNCH_SCRIPT
```

Every managed server must listen at **`http://127.0.0.1:8000`**, advertise model
alias **`served`**, and remain in the foreground. A complete launcher contains
its final weight path and optimized settings; the runner executes it unchanged.
A launcher that leaves its weights open through `$CHECKPOINT` or
`${CHECKPOINT:?…}` is a baseline template. The runner resolves that input from
the selected model/workload context and engine/device mapping. A filled local
assignment or nonempty checkpoint default counts as a complete script. No port
is supplied to the script.

A path under `images/<category>/baselines/<engine>/launch.sh` identifies the
engine and category. A copied template uses the selected reference's public
`reference.json` in the current working directory. A complete candidate needs
neither reference metadata nor a checkpoint mapping. Source-equal mappings use
`/models/source`; other mapped checkpoints remain repositories or native tags.

Run from the role workspace. Timing loads `data/<S|A|A-MA>/rows.jsonl` for the
selected workload and model. Quality always loads `data/S/rows.jsonl`, including
for A and A-MA tasks. Setup mounts public S alongside the task's public pool;
search uses its separate sampled S calibration view. Private verification calls
the same executor with protected rows.

| Argument | Default | Purpose |
| --- | --- | --- |
| `model` | Required | Select recorded rows and request policy. |
| `workload` | Required | `S-1`, `S-N`, `A-1`, `A-N` or `A-MA`. |
| `launch` | Required | Complete candidate or baseline template script. |
| `--params JSON_FILE` | Omitted | External tuning `args`/`env`; omission keeps script defaults. |
| `--mode` | `timing` | `timing` includes excluded warmup and performance metrics; `quality` reports accuracy only. |
| `--concurrency` | Workload default | Positive number of active rows/sessions, or `-1` to admit all independent rows. Quality defaults to `-1`. |

Concurrency changes row/session admission. Parent dependencies and the workload's
recorded tool delays remain intact; `-1` does not send a dependent request before
its parents finish. Quality requires a complete response, without timing or
output-usage accounting. Each row's `evaluation.method` selects its Python
accuracy checker. Internal quality results keep per-row correctness for the
paired gate; the CLI's `report.json` contains only `accuracy`.

Each invocation creates `bench/<uuid>/server.log` and `report.json`. Startup
rejects an occupied endpoint, an exited foreground process, a wrong port or a
wrong model alias. Owned process groups stop after success, failure or cancellation.

The actual CLI regression exports two supplied public C2 S rows into a temporary
workspace and runs a scripted HTTP server from an Ollama-folder template:

```bash
python -m eval.bench.run C2 S-1 \
  <test-workspace>/images/nvidia-cuda-x86/baselines/ollama/launch.sh \
  --params <test-workspace>/params.json
```

Its parameter file is `{"args": ["--workers", "2"], "env": {"TEST_PARALLEL": "2"}}`.
Those are test-peer controls, not verified Ollama flags. The test checks the mapped
native tag `qwen3.6:35b-a3b-mtp-q4_K_M`, port 8000, unchanged recorded bodies,
one excluded warmup, two measured requests, six scripted output tokens and cleanup.
It then runs both that template and a complete candidate in quality mode:

```bash
python -m eval.bench.run C2 A-N <test-workspace>/launch.sh \
  --mode quality --concurrency -1 --params <test-workspace>/params.json
```

This loads S rather than A data. The fixture deliberately returns non-gold text,
so the accuracy report is `{"accuracy": 0.0}`. The complete candidate keeps its
embedded `/workspace/weights/optimized` and `--workers 4`; it receives no
checkpoint substitution. Reproduce these executed cases with:

```bash
uv run pytest -q tests/test_bench_run.py::test_cli_launches_mapped_baseline_and_reports_public_rows
```

See [Phase 3](docs/phase-03.md) for the shared runtime and its Phase 4 revisions.

## Verify a candidate

Submit code, `install.sh`, `launch.sh`, and optional `convert_weight.sh` in
`/workspace/submission`. Keep it self-contained, without external code links.
The wrapper transfers only this directory into the third native Harbor task with
a fresh container from the same category image. Development checkpoints, environments,
caches, homes and background processes do not move with it. The root collection
hook stops leftover role processes and empties Harbor's implicit artifact directory.

Source weights and the selected baseline's weights mount read-only at
`/models/source` and `/models/reference`. Conversion receives `CHECKPOINT` pointing
to the source weights and writes new candidate weights into `WEIGHTS` (`/workspace/weights`).
It runs **before** installation. Both scripts can write installed dependencies
and compiled output into `RUNTIME` (`/workspace/runtime`). These are benchmark
script inputs; they are not provider or backend environment settings.
The final launcher should normally contain the resulting weights and optimized
settings. A copied baseline template may instead retain the reserved checkpoint
input; its selected reference supplies the mapping. The verifier never silently
substitutes source weights for a completed candidate.

The generated verification task has three ordered steps. Their test scripts
invoke the same evaluator entrypoint with different stages:

```bash
# prepare/tests/test.sh — Harbor runs this as root
/opt/eval/.venv/bin/python -I -m eval.verify.run --stage prepare --task /root/eval/task.json
# review/tests/test.sh — Harbor runs this as judge
/opt/eval/.venv/bin/python -I -m eval.verify.run --stage review
# measure/tests/test.sh — Harbor runs this as root
/opt/eval/.venv/bin/python -I -m eval.verify.run --stage measure --task /root/eval/task.json
```

Review reads the prepared rubric rather than root-only task metadata. The flow is:

```text
search task: search.agent → root selection gate → selected reference/parameters
  → coding task: agent → root format gate → collect submission
  → verification task: new container, native nop agent
  → prepare/root: checks → optional conversion → installation → readable artifacts
  → review/judge: Rewardkit rules review
  → measure/root: S quality → paired gate → fresh timed replay → final rewards
```

All three verification steps share the new container. `min_reward = 1` on prepare
and review stops later steps when a gate fails. Harbor's built-in `nop` agent does
nothing before those verifiers; only Rewardkit makes reviewer model calls. The
wrapper takes the benchmark score from `jobs/verify`. `jobs/search` and
`jobs/coding` contain the two handoff gates, not benchmark scores.

Rules review uses Harbor's `harbor-rewardkit` agent judge. Select the verification
agent independently of coding and search:

```yaml
evaluation:
  agent:
    name: claude-code  # claude-code, codex or fx
    model: null       # use the reviewer's default model
    auth: login
    env: []
    budget: 30m
    kwargs: {}        # native Rewardkit version/mcp_servers options
```

The six binary rules use batched review and all-pass aggregation. The reviewer
runs as `judge` with its own writable home, cache and review output. It reads the
installed submission, dependencies and resulting checkpoint. The image's shared
`images/prepare.sh` script makes completed artifacts root-owned and readable
without granting judge write access; root retains write access. Private evaluator
inputs remain inaccessible to judge. Conversion, installation, servers, quality,
timing, scoring and reports all run as root, selected by the native task TOML.
[Rewardkit integration](https://docs.harborframework.com/core-concepts/rewardkit/quick-start).

Rules live in `eval/verify/rules.toml`; `prompt.txt` defines the inspection context.
`judge.py` checks submission format, writes the native Rewardkit rubric/prompt,
interprets review decisions and compares paired quality. It controls neither
access nor processes. `verify/run.py` implements the step procedures and invokes
native Rewardkit directly during review. Harbor supplies its Unix identity. The rubric's native `JudgeConfig` selects the backend and optional model;
Rewardkit owns authentication and agent execution. There are no custom reviewer settings.

For `auth: login`, provide the reviewer's native credential variable on the host:
`CLAUDE_CODE_OAUTH_TOKEN` for Claude, `CODEX_AUTH_JSON` for Codex or
`VERCEL_OIDC_TOKEN` for Fx. Claude's `claude setup-token` command can supply its
token; Codex expects its authentication JSON through the native environment input.
The wrapper forwards these names through Harbor's review-step verifier environment.
It does not read or mount host login files, and an existing host CLI login alone
does not supply verifier credentials. Search/coding adapters keep Harbor's own
authentication behavior.
For `auth: api_key`, the default key is `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or
`AI_GATEWAY_API_KEY`. `env` can supply explicit variable names through Harbor's
verifier environment. Values stay outside saved YAML, task.json and image layers.
For Codex, add `REWARDKIT_FORCE_SUBSCRIPTION` to `env` alongside `CODEX_AUTH_JSON`
when that native flag is needed; it is forwarded only when provided.
The selected reviewer CLI must be installed in the supplied category image.

Claude uses the native backend's headless inspection behavior. Root-owned files,
read-only mounts and role permissions enforce the artifact boundary, including
writes attempted through shell tools. There is no custom reviewer settings file.

Quality uses the saved private S sample for every task, concurrency `-1`, no
warmup and no timing metrics. Dedicated accuracy functions grade complete
answers; an exact paired test rejects significant degradation at 5%. Candidate
script/conversion/install/rule/startup/quality/protocol failures skip timing and earn zero.
A wrong-port candidate is reported at startup. Reference, reviewer or environment
failure produces no score. Installation and review are online; root serving is
restricted to loopback in IPv4 and IPv6 afterward. Cleanup stops owned process
groups before restoring egress; it never terminates processes by the root UID.

Each verifier writes to its native `/logs/verifier`; Harbor archives and resets
that directory between steps. Final measure reports are `report.json`, `report.md`
and, for scored outcomes, `reward.json`. The host wrapper publishes their rewards
at `runs/<uuid>/result.json`, alongside paths to search, coding and verification trials. Throughput and TTFT each earn
`100 * max(0, 1 - 1/G)`: twice as fast earns 50, equal or slower earns zero.
Stale rewards are removed before evaluation. For a known-solution sanity check,
set `agent.name: oracle`. The job uses Harbor's native OracleAgent with no model;
Harbor executes search and coding's task-level `solution/solve.sh`. The coding solution creates an
ordinary submission from the selected reference. It passes the same static,
conversion, installation, review, quality and timing gates as every candidate.
Reviewer credentials are still required when using a live reviewer. There is no
verifier oracle flag or special rule exemption.

The CPU regressions reuse supplied public questions and scripted HTTP responses.
Rewardkit integration exercises its real CLI with a substituted paid backend;
runner/firewall unit tests use explicit command doubles. Run:

```bash
uv run pytest -q tests/test_accuracy.py tests/test_quality.py tests/test_judge.py tests/test_verify.py tests/test_oracle.py
```

The four opt-in reviewer acceptance cases use Harbor's native oracle, an existing
production device image and actual Claude Code with native credentials. They exercise real
source inspection, subscription authentication and Unix role boundaries in
the fresh verifier. They do not build a test image or substitute the reviewer.
A trusted final test hook records reviewer acceptance without model serving;
paired quality and timing acceptance need real weight shards. To run them with Docker and the reviewer credential
environment available:

```bash
LOCAL_EVAL_HARBOR_TESTS=1 uv run pytest -q -s tests/test_oracle.py
```

Validation for the current three-job design is recorded in [Phase 4](docs/phase-04.md),
with earlier two-job and root-controller results labeled as historical evidence.
Source-review acceptance does not establish model inference, quality or timing results.

[Phase 4](docs/phase-04.md) gives the complete review order and validation boundary.

## Configuration

[config.example.yaml](config.example.yaml) includes every supported setting.
Settings map directly into dataclasses before acquisition; there is no separate
schema audit or error-recovery layer. Native parsing, file, and subprocess errors
propagate to the caller. Search and verifier settings are carried into generated
task metadata; Phase 4 consumes the evaluation settings.
Coding/search settings become native Harbor `AgentConfig` objects; reviewer
settings become Rewardkit's native `JudgeConfig`. See the
[mapping reference](docs/reference.md#mapping-to-harbor-and-rewardkit) for the
generated fields and the boundary between OS-user selection and file permissions.

| Setting | Default | Accepted values and purpose |
| --- | --- | --- |
| `dataset` | Omitted | Hugging Face dataset source string; required for a run, optional for foundation inspection. |
| `gpus` | `all` | All visible GPUs, or a list of GPU indices/UUIDs such as `[0, 1]`. |
| `tasks.models` | `all` | All catalog models, or keys, names, slugs, or HF model strings. |
| `tasks.workloads` | `all` | `S-1`, `S-N`, `A-1`, `A-N`, `A-MA`, or `all`. |
| `tasks.types` | `[SOTA]` | A list containing `SOTA` and/or `NEW`, or `all`. |
| `agent` | Omitted | Required for a run, optional for foundation inspection; `name` is required when present. |
| `agent.name` | Required in `agent` | Harbor built-in name or custom `module.path:ClassName`; the module must be importable by Harbor. |
| `agent.model` | Omitted | Optional coding-agent model string, forwarded as Harbor's `model_name` only when supplied. The selected adapter decides whether it needs a model. |
| `agent.kwargs` | `{}` | Adapter-specific options passed to Harbor. |
| `agent.auth` | `login` | For built-in `codex`, select host login or `api_key`. Other built-in/custom adapters manage authentication through their own kwargs/environment. |
| `agent.env` | `[]` | Host environment-variable names to forward; values/secrets do not belong in YAML. No tuning allowlist is imposed. |
| `agent.budget` | `2h` | Coding-agent time budget. |
| `search.mode` | `combined` | `combined` or `single`; reference-search session mode. |
| `search.agent.name` | `claude-code` | Harbor built-in name or custom import path for reference search. |
| `search.agent.model` | Omitted | Optional search-agent model; the chosen adapter owns its model/default requirements. |
| `search.agent.kwargs` | `{}` | Native adapter options for search. |
| `search.agent.auth` | `login` | Same built-in Codex login/API-key behavior as `agent.auth`; other adapters own their authentication. |
| `search.agent.env` | `[]` | Host environment-variable names scoped to search; values are not written to YAML. |
| `search.agent.budget` | `2h` | Search-trial budget, independent of the coding budget. |
| `evaluation.install_timeout` | `4h` | Harbor image-build timeout and deadline for each candidate conversion/installation script. |
| `evaluation.health_timeout` | `1h` | Sandbox readiness and each evaluation server's readiness deadline. |
| `evaluation.agent.name` | `claude-code` | Rewardkit verification agent: `claude-code`, `codex` or `fx`. |
| `evaluation.agent.model` | Omitted | Optional reviewer model; omission/null uses its default. |
| `evaluation.agent.auth` | `login` | Forward native credential environment input or use `api_key`; independently scoped to verification. A host CLI login alone is insufficient. |
| `evaluation.agent.env` | `[]` | Explicit credential variable names; otherwise auth mode selects the defaults described above. |
| `evaluation.agent.budget` | `30m` | Reviewer timeout and process deadline. |
| `evaluation.agent.kwargs` | `{}` | Native Rewardkit options `version` and `mcp_servers`; workspace, isolation and batched review belong to the evaluator. |
| `out` | `runs` | Output root; relative paths resolve beside the YAML file. |

Durations use integer seconds or integer strings ending in `s`, `m`,
or `h`, such as `90`, `30m`, or `2h`. Task selection is the intersection of model,
workload, and type filters. There is no separate task-ID selector. For example,
adding `A-N` to the example's workloads selects its S-1 and A-N tasks.

Agent selection follows Harbor's built-in-name/custom-import-path interface.
The search job translates `search.agent` to its native Harbor agent config;
the coding job translates `agent`. Verification selects native `nop`, with
`evaluation.agent` choosing its independent Rewardkit reviewer. There is no
coordinator agent or nested delegate configuration. Omit a role's `model` (or use
`null`) when its adapter uses a default or manages models internally.
Authentication and model requirements remain the selected adapter's responsibility.
For example, Harbor 0.23's built-in Codex adapter requires `agent.model`;
Claude Code and a custom harness may use their own defaults.
Harbor's Claude adapter reads exported OAuth/API credential variables; it does
not copy a host `claude auth login` session. For subscription authentication,
export `CLAUDE_CODE_OAUTH_TOKEN` and `CLAUDE_FORCE_OAUTH=1`, then name both in
the role's `env` list. The wrapper stores variable names and native placeholders.

`agent.name: oracle` directly selects Harbor's native known-solution agent instead.
It selects native oracle for both search and coding. Their authored solutions use the ordinary
verifiers. It makes no search/coding model calls and adds no coding token/cost
attempts; verification still uses the configured reviewer.

The runner's resume regression uses Harbor's real no-op class to exercise custom
import-path resolution without an agent call:

```yaml
agent:
  name: harbor.agents.nop:NopAgent
  kwargs: {}
  env: [CUSTOM_CREDENTIAL]
```

That test inspects the native coding configuration, forwards the environment-variable
name, leaves its `model_name` unset, and resolves the class with Harbor's actual
`AgentFactory`. It validates
the integration boundary; a no-op agent does not implement a serving system.
For your own harness, use its importable class path and adapter-specific options.

Benchmark defaults are defined in `eval/config.py`: seed `20260921`, S output
allowance `32768`, A-N concurrency `8` sessions, and calibration and quality
subsets of up to `24` private S rows each per model. Calibration samples with the
fixed seed; quality samples independently with seed + 1. They may overlap.
Both views are shared across S, A and MA tasks; no dataset calibration flag is
required. These are benchmark policy, not additional YAML options. Quality evaluation
submits the subset as an offline workload and lets the serving backend batch it;
there is no quality-batch setting. The shared executor uses concurrency `-1` for the gate.

## Device images and baselines

The supplied source layout is:

```text
images/
  prepare.sh
  <device-category>/
    Dockerfile
    baselines/<engine>/
      install.sh
      launch.sh
```

[images/README.md](images/README.md) lists the five device categories. Dockerfiles
and script contents are supplied by the owner when ready. They are source
files, not setup output. Setup checks the selected Dockerfile, installers and launchers
before downloads. Harbor builds them; the wrapper never invents missing assets.

The three authored catalogs are:

| Catalog | What it supplies |
| --- | --- |
| `tables/tasks.csv` | Task identity, model, workload, and type. |
| `tables/baseline_map.csv` | Device category, model, engine, original `source_model_checkpoint`, and baseline `checkpoint`. |
| `tables/request_fields.json` | Shared model sampling/thinking settings and request/output-count fields. |

Installers own their repository, source ref, install prefix, and venv/native
isolation. Launchers own engine arguments and invoke the matching installation.
Their paths follow the fixed folder structure; neither is a table column.
There is no separate device or installer/launch catalog. Launch uses `/bin/bash`,
conditionally fills a mapped `CHECKPOINT` and reserves external overrides of
`CHECKPOINT`, `PORT` and shell startup variables `BASH_ENV`/`ENV`. It never injects `PORT`.
The served name is `served`. Extra arguments/environment are tuning. Scripts will be designed
to this interface when supplied; they keep the server in the foreground.

Setup uses ordinary `hf download <checkpoint> --local-dir <directory>` for the
source and mapped HF baseline checkpoints. The Hugging Face CLI owns acquisition
and reuse; equal source/baseline identifiers reuse the source files. Search mounts
these files read-only. After the selection gate, a run-owned symlink points coding
and verification's read-only `/models/reference` mount at the selected host files;
no weight copy or container cache crosses the handoff. Native engine tags remain
identifiers: fresh offline verification cannot pull an uncached tag. A supplied
launcher must resolve it from available assets; tag-cache transfer is not implemented. Baseline software
is installed in the shared image by each engine's `install.sh`, independently of
run-specific weights. Weights are not image layers or submission artifacts.

Each task exposes the original checkpoint read-only at `/models/source`. A
baseline using that identifier reuses the source mount. `run.md` lists this path
and each mapped baseline weight location. The agent may develop quantization or other formats, then submit
`convert_weight.sh` to reproduce them from the read-only source during verification.
Engine-required tokenizer files remain ordinary checkpoint package contents; the
harness does not load or separately acquire a tokenizer.

The device image installs shared agent-action tools, the evaluator runtime,
the selected Rewardkit reviewer CLI, isolated baselines and the shared trusted
`prepare.sh` permission script. It provides `search`, `agent` and `judge` users plus
`iptables` and `ip6tables` for controlled evaluation. Root runs the entire fresh
prepare and measure steps; Harbor runs the review verifier as judge. Host acquisition/task-generation dependencies belong to
the `host` extra. Harbor adapters install their selected agent CLIs; external
adapters keep their loops on the host. Reviewer authentication is supplied at
runtime, never baked into the image.

Harbor builds one selected device image. **Search → coding → verification** are
three jobs and three separate containers. Search publishes only the selected
reference/parameters; coding transfers only source submission. **Prepare → review
→ measure** share the verification container. Trusted weights/input mounts are
supplied separately. Native step TOML declares the root/judge/root verifier users.
Startup only readies the environment. Search runs as `search` against its private
calibration workspace; coding runs as `agent`; their trusted selection/format gates run as root.
Search and coding each use a native single-step task. The wrapper publishes the verification
task's final measure reward, so search and coding handoff gates do not contribute
to the benchmark score. Native Harbor adapters own setup, execution and authentication.
There is no Dockerfile template,
inline installer generation, or separate image/container per baseline.
`inspect_machine(...).dockerfile` returns the category's source Dockerfile path;
the module does not choose CUDA versions or generate build recipes.

Agents may tune arbitrary arguments and environment variables. Evaluator requests
use the same authored policy for the candidate and engines defined in
`request_fields.json`. Those engines use `/v1/chat/completions`, with backend fields
including the authored Ollama overrides.
There is no native Ollama request conversion or launch environment default in the
harness.

## Repository guide

| Location | Responsibility |
| --- | --- |
| `eval/config.py` | Parse settings and select tasks. |
| `eval/request_policy.py` | Construct baseline/candidate requests from the authored policy and read server output counters. |
| `eval/launch.py` | Detect an unfilled checkpoint input, compose argv/env and protect launch identity. |
| `eval/baselines.py` | Select source/baseline checkpoint mappings, derive launcher paths and deduplicate engine names. |
| `eval/device.py` | Inspect selected GPUs/resources and resolve the source Dockerfile path. |
| `eval/models.py` | Run normal HF checkpoint downloads; retain native engine identifiers. |
| `eval/setup.py` | Select trusted HF rows directly and prepare public/calibration/private/quality JSONL views. |
| `eval/tasks.py`, `templates/` | Render separate search, coding and verification tasks, mounts, budgets and native known solutions. |
| `eval/search/run.py` | Validate mapped reference selection and adjustable parameters; publish the trusted handoff. |
| `eval/run.py` | Save configs, select the native oracle or planned coordinator, resume by folder, and retain coding-only costs. |
| `eval/bench/serve.py` | Prepare Launch argv/environment, await readiness, and clean up owned processes; optionally attach. |
| `eval/bench/clients.py` | Send recorded OpenAI requests; collect streamed content, counts and monotonic timings. |
| `eval/bench/replay.py` | Schedule recorded sessions, dependencies and tool delays for all five workloads. |
| `eval/bench/run.py` | Resolve optional template weights/policy, load timing or S-quality rows and save a UUID report through the shared executor. |
| `eval/bench/metrics.py` | Compute throughput and TTFT from timing records. |
| `eval/accuracy/` | Dispatch each trusted row's mathematical or letter accuracy function. |
| `eval/verify/judge.py`, `rules.toml`, `prompt.txt` | Check submission format, write the native rubric/prompt, interpret review decisions and compare paired quality; no access or process control. |
| `eval/verify/run.py` | Implement handoff, root preparation, native judge review and root quality/timing; own processes, failure gates and reports. |
| `templates/oracle-search.sh`, `oracle-coding.sh` | Supply known solutions to Harbor's native oracle; create an ordinary submission. |
| `eval/costs.py` | Account for coding-agent/harness tokens and costs across attempts. |
| `tables/` | Only tasks, baseline checkpoint mappings, and request policy. |
| `images/` | Root artifact-permission script and runtime image contract; category assets are added when supplied. |
| `tests/` | Focused areas, each with two common cases and two edge cases. |

See [the foundation reference](docs/reference.md) for what each API does and
working usage, [the benchmark specification](docs/benchmark.md) for evaluation
lifecycle and trusted columns, and [the Phase 1 review packet](docs/phase-01.md)
for scope and validation. [CONTRIBUTING.md](CONTRIBUTING.md) describes development
and the phase review process. Each approved phase becomes one commit. Phase 3 is
committed as `efd3650`; Phase 4 remains uncommitted until its review is complete.
