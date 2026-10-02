# local_eval

Evaluate how well a coding agent implements a local model-serving system. The
finished benchmark compares a candidate against a tuned reference using recorded
requests, a pre-run quality gate, and timing measurements. Harbor owns the task
sandbox and the agent lifecycle through its existing adapters.

**Status: Phases 1 and 2 approved.** The
foundations now compose into host acquisition, generated Harbor tasks, and a
foreground run/resume wrapper. Tasks now declare native Harbor search/coding
steps; their coordinator and search runtime arrive in Phase 5. Production
Dockerfiles/installers await owner supply; servers, replay and judging arrive
in their own later phases.

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

The tests need only a CPU and local files. They do not use networking, download
models, build images, start Harbor jobs, or make paid agent calls. Renderer tests
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
assets, prepares inputs/checkpoints, and awaits one Harbor job in the current
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
tests. A complete benchmark additionally needs the supplied image/installers and
the later coordinator/search/verifier modules; this checkout does not yet run a production
evaluation. See [Phase 2's review packet](docs/phase-02.md) for executed examples
and the precise validation boundary.

## Configuration

[config.example.yaml](config.example.yaml) includes every supported setting.
Settings map directly into dataclasses before acquisition; there is no separate
schema audit or error-recovery layer. Native parsing, file, and subprocess errors
propagate to the caller. Search and verifier settings are carried
into generated task metadata for their later runtime implementations.

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
| `search.agent.budget` | `2h` | Search-step budget, independent of the coding budget. |
| `evaluation.install_timeout` | `4h` | Harbor image-build timeout; also carried into the later candidate installation runtime. |
| `evaluation.health_timeout` | `1h` | Sandbox readiness period; also carried into the later serving-system readiness runtime. |
| `out` | `runs` | Output root; relative paths resolve beside the YAML file. |

Durations use integer seconds or integer strings ending in `s`, `m`,
or `h`, such as `90`, `30m`, or `2h`. Task selection is the intersection of model,
workload, and type filters. There is no separate task-ID selector. For example,
adding `A-N` to the example's workloads selects its S-1 and A-N tasks.

Agent selection follows Harbor's built-in-name/custom-import-path interface. The
generated job selects one planned `eval.agents:PhasedAgent` coordinator, which
will delegate both chosen integrations. Harbor's native multi-step trial keeps
one top-level agent instance; putting both adapters in `JobConfig.agents` would
create separate trials. Omit a role's `model` (or use `null`) when its adapter uses
a default or manages models internally. Authentication
and model requirements remain the
selected adapter's responsibility.

The runner's resume regression uses Harbor's real no-op class to exercise custom
import-path resolution without an agent call:

```yaml
agent:
  name: harbor.agents.nop:NopAgent
  kwargs: {}
  env: [CUSTOM_CREDENTIAL]
```

That test inspects the coding child configuration, forwards the environment-variable
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
there is no harness quality-batch setting or concurrency cap.

## Device images and baselines

The supplied source layout is:

```text
images/<device-category>/
  Dockerfile
  baselines/<engine>/install.sh
```

[images/README.md](images/README.md) lists the five device categories. Dockerfiles
and installer contents are supplied by the owner when ready. They are source
files, not setup output. Setup checks the selected Dockerfile and installers
before downloads. Harbor builds them; the wrapper never invents missing assets.

The three authored catalogs are:

| Catalog | What it supplies |
| --- | --- |
| `tables/tasks.csv` | Task identity, model, workload, and type. |
| `tables/baseline_map.csv` | Device category, model, engine, original `source_model_checkpoint`, and baseline `checkpoint`. |
| `tables/request_fields.json` | Shared model sampling/thinking settings and request/output-count fields. |

Installers own their repository, source ref, install prefix, and venv/native
isolation. Their path follows the fixed folder structure; it is not a table
column. There is no separate device or installer/launch catalog.

Setup uses ordinary `hf download <source-checkpoint> --local-dir <directory>`
only for the source model. The Hugging Face CLI owns acquisition and reuse;
the harness has no snapshot-cache copy or publication layer. Baseline engines
are installed inside the shared image by their `install.sh` scripts. Distinct
baseline checkpoints stay model strings for those engines to load or acquire
inside the sandbox; weights are not baked into the image.

Each task exposes the original checkpoint read-only at `/models/source`. A
baseline using that identifier reuses the source mount. `run.md` lists this path
and each distinct baseline model identifier. Runtime downloads use writable
role storage. The agent may quantize the original or produce other formats in `/workspace`.
Engine-required tokenizer files remain ordinary checkpoint package contents; the
harness does not load or separately acquire a tokenizer.

The device image installs shared agent-action tools, the evaluator runtime, and
isolated baselines. Host acquisition/task-generation dependencies belong to the
project's `host` extra. The selected Harbor adapter installs its agent-specific
CLI during setup; an external adapter keeps its agent loop on the host.

Harbor will build the selected device image and share one task sandbox across
**search step → reference gate → coding step → pre-run gate → timed evaluation**.
Startup only readies the environment. Search runs as `search` against its private
calibration workspace; coding runs as `agent`; both verification hooks run as root.
Generated tasks use the last step's reward, so the search gate does not contribute
to the benchmark score. The adapter coordinator/search implementation is Phase 5
work; Phase 2 implements its configuration and generated task contract. There is no Dockerfile template,
inline installer generation, or separate image/container per baseline.
`inspect_machine(...).dockerfile` returns the category's source Dockerfile path;
the module does not choose CUDA versions or generate build recipes.

Agents may tune arbitrary arguments and environment variables. Evaluator requests
use the same authored policy for the candidate and engines defined in
`request_fields.json`. Those engines use `/v1/chat/completions`, with backend fields
including the authored Ollama overrides. The four `lm-studio` mappings belong to
the future Apple Silicon category; no LM Studio request policy is supplied yet.
There is no native Ollama request conversion or launch environment default in the
harness.

## Repository guide

| Location | Responsibility |
| --- | --- |
| `eval/config.py` | Parse settings and select tasks. |
| `eval/request_policy.py` | Construct baseline/candidate requests from the authored policy and read server output counters. |
| `eval/baselines.py` | Select source/baseline checkpoint mappings and deduplicate engine names. |
| `eval/device.py` | Inspect selected GPUs/resources and resolve the source Dockerfile path. |
| `eval/models.py` | Run normal HF checkpoint downloads; retain native engine identifiers. |
| `eval/setup.py` | Select trusted HF rows directly and prepare public/calibration/private/quality JSONL views. |
| `eval/tasks.py`, `templates/` | Render one sandbox with ordered search/coding steps, per-role budgets and shared root gates. |
| `eval/run.py` | Save role configs, delegate one planned coordinator, resume by folder, and retain coding-only costs. |
| `eval/bench/metrics.py` | Compute throughput and TTFT from timing records. |
| `eval/costs.py` | Account for coding-agent/harness tokens and costs across attempts. |
| `tables/` | Only tasks, baseline checkpoint mappings, and request policy. |
| `images/` | Shared runtime-only image contract; category assets are added when supplied. |
| `tests/` | Ten areas, each with two common cases and two edge cases. |

See [the foundation reference](docs/reference.md) for what each API does and
working usage, [the benchmark specification](docs/benchmark.md) for the planned
lifecycle and trusted columns, and [the Phase 1 review packet](docs/phase-01.md)
for scope and validation. [CONTRIBUTING.md](CONTRIBUTING.md) describes development
and the phase review process. Each approved phase becomes one commit. Phase 2 is approved; subsequent phases
remain uncommitted until their review is complete.
