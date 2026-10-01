# local_eval

Evaluate how well a coding agent implements a local model-serving system. The
finished benchmark compares a candidate against a tuned reference using recorded
requests, a pre-run quality gate, and timing measurements. Harbor owns the task
sandbox and the coding-agent lifecycle.

**Status: Phase 1 foundations, uncommitted and awaiting review.** Configuration,
canonical catalogs, baseline metadata, device inspection, evaluator request
formatting, timing arithmetic, and implementation-cost accounting are implemented.
Setup, servers, replay, judging, and execution arrive in later phases.

## Quickstart

Use Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
From this directory:

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run python examples/inspect_foundations.py config.example.yaml
```

The tests need only a CPU and local files. They do not use networking, download
models, build images, run Harbor, or make paid agent calls. The inspector reads
the real example configuration and catalogs, selecting Qwen3.6-35B-A3B / S-1 and
showing its FreeToken source ref, installer path, launch command, and request
fields. It does not install or start that engine.

## Configuration

[config.example.yaml](config.example.yaml) includes every supported setting.
Phase 1 parses these values; later phases perform the described runtime actions.

| Setting | Default | Accepted values and purpose |
| --- | --- | --- |
| `dataset` | Omitted | Hugging Face dataset source string; the owner supplies its actual location when needed. |
| `gpus` | `all` | All visible GPUs, or a nonempty list of GPU indices/UUIDs such as `[0, 1]`. |
| `tasks.models` | `all` | All catalog models, or keys, names, slugs, or HF model strings. |
| `tasks.workloads` | `all` | `S-1`, `S-N`, `A-1`, `A-N`, `A-MA`, or `all`. |
| `tasks.types` | `[SOTA]` | A list containing `SOTA` and/or `NEW`, or `all`. |
| `agent` | Omitted | Optional during foundation inspection; when present, `name` and `model` are required. |
| `agent.name` | Required in `agent` | Harbor agent adapter name. |
| `agent.model` | Required in `agent` | Model string for the coding agent. |
| `agent.kwargs` | `{}` | Adapter-specific options; any mapping with string keys. |
| `agent.auth` | `login` | `login` or `api_key`; authentication choice for the later agent adapter. |
| `agent.env` | `[]` | Host environment-variable names to forward; values/secrets do not belong in YAML. No tuning allowlist is imposed. |
| `agent.budget` | `2h` | Coding-agent time budget. |
| `search.mode` | `combined` | `combined` or `single`; reference-search session mode. |
| `search.auth` | `login` | `login` or `api_key` for reference search. |
| `evaluation.install_timeout` | `4h` | Candidate installation timeout. |
| `evaluation.health_timeout` | `1h` | Serving-system readiness timeout. |
| `out` | `runs` | Output root; relative paths resolve beside the YAML file. |

Durations accept positive integer seconds or integer strings ending in `s`, `m`,
or `h`, such as `90`, `30m`, or `2h`. Task selection is the intersection of model,
workload, and type filters. There is no separate task-ID selector. For example,
adding `A-N` to the example's workloads selects its S-1 and A-N tasks.

Benchmark defaults are defined in `eval/config.py`: seed `20260921`, S output
allowance `32768`, A-N concurrency `8` sessions, and a quality subset of `24` S
rows. These are benchmark policy, not additional YAML options. Quality evaluation
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
files, not setup output. Phase 1 records their paths; it does not invent missing
assets or build images.

`tables/baselines.csv` maps each category/engine to its repository, `code_ref`,
installation `prefix`, `install_script`, and launch argv/env. For example,
FreeToken's NVIDIA x86 row points to
`images/nvidia-cuda-x86/baselines/freetoken/install.sh`, ref `v0.1.3`, and
`/opt/baselines/freetoken`. The script owns clone/checkout/installation. Python
engines use separate venvs; native engines use separate build prefixes.
`tables/baseline_map.csv` maps model/category/engine to its checkpoint format.

Harbor will build the selected device image and share one task sandbox across
**reference search → coding agent → evaluation**. There is no Dockerfile template,
inline installer generation, or separate image/container per baseline.
`inspect_machine(...).dockerfile` returns the category's source Dockerfile path;
the module does not choose CUDA versions or generate build recipes.

Agents may tune arbitrary arguments and environment variables. No launch-contract
allowlist exists. The only catalog environment defaults are `OLLAMA_HOST`
(bind address) and `OLLAMA_MODELS` (model directory), documented in the
[official Ollama FAQ](https://docs.ollama.com/faq); callers can override both.
Evaluator request formatting remains shared so comparison requests are consistent.

## Repository guide

| Location | Responsibility |
| --- | --- |
| `eval/config.py` | Parse settings and select tasks. |
| `eval/tables.py` | Format evaluator requests and read server output counters. |
| `eval/baselines.py` | Resolve baseline source, installer, checkpoint, and launch metadata. |
| `eval/device.py` | Inspect selected GPUs/resources and resolve the source Dockerfile path. |
| `eval/bench/metrics.py` | Compute throughput and TTFT from timing records. |
| `eval/costs.py` | Account for coding-agent/harness tokens and costs across attempts. |
| `tables/` | Authored task, device, engine, model-mapping, and request-policy catalogs. |
| `images/` | Source-image contract; category assets are added when supplied. |
| `tests/` | Six areas, each with two common cases and two edge cases. |

See [the foundation reference](docs/reference.md) for what each API does and
working usage, [the benchmark specification](docs/benchmark.md) for the planned
lifecycle and trusted columns, and [the Phase 1 review packet](docs/phase-01.md)
for scope and validation. [CONTRIBUTING.md](CONTRIBUTING.md) describes development
and the phase review process. Each approved phase becomes one commit; Phase 1
remains uncommitted until the owner approves it.
