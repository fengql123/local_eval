# Foundation reference

The six foundation areas read local settings/catalogs or process explicit
execution evidence. They need no dataset download, image build, server, or agent
call. Phase 2 composes them into setup, generated Harbor tasks, and run/resume.
The examples below were executed against this checkout and its checked-in test
fixtures. They inspect policy or arithmetic; they are not measured GPU results.

## Configuration — `eval/config.py`

`load_config(path)` reads YAML; `parse_config(mapping, base=...)` maps the same
settings into dataclasses. Model aliases become canonical names, and relative
output paths resolve beside the YAML file. `select_tasks()` intersects model,
workload, and type filters in task-table order. An empty task intersection is an
error. Inputs are trusted: native parsing/dataclass errors propagate without a
separate schema audit. The README lists all fields and defaults.

```python
from eval.config import load_config, select_tasks

config = load_config("config.example.yaml")
selected = select_tasks(config.tasks)
assert [task.task_id for task in selected] == ["qwen36-35b-a3b_S-1"]
```

This is the actual example configuration and task catalog. Selection does not
read the dataset. Both `agent.name` and `search.agent.name` use Harbor's built-in
names or importable `module.path:ClassName` paths. A role's omitted/null model
means no model override; its selected adapter owns model and authentication
requirements. The generated task has separate search and coding steps.

## Baseline checkpoints — `eval/baselines.py`

`mappings(category, model)` selects rows from `baseline_map.csv`.
`mapping(category, model, engine)` selects one engine's row. `needed()` returns
engine names once, in selected-model/table order. This is the minimum join needed
to prepare checkpoint packages and select fixed-path installers.

```python
from eval.baselines import mapping, needed

asset = mapping("nvidia-cuda-x86", "C2", "freetoken")
assert asset.source_model_checkpoint == "Qwen/Qwen3.6-35B-A3B"
assert asset.checkpoint == "nvidia/Qwen3.6-35B-A3B-NVFP4"
assert needed("nvidia-cuda-x86", ("C2", "C1")) == ("freetoken", "llama.cpp", "ollama")
```

These are the inspector's real mapping and the baseline tests' shared-engine
selection. Installer paths are inferred as
`images/<category>/baselines/<engine>/install.sh`. Scripts own repository/ref,
prefix, and venv/native isolation. There is no launch recipe or metadata audit.
An unmatched device/model selection yields no mappings; a missing individual
mapping raises Python's normal `StopIteration`.

## Device inspection — `eval/device.py`

`inspect_machine(gpu_ids)` reads selected NVIDIA GPU identities/memory and host
CPU/RAM. The category directly determines `machine.dockerfile`; no Dockerfile is
generated. `check_environment()` performs bounded Docker daemon, Compose, and
Harbor version probes. Command failures propagate from the subprocess API;
unsupported hosts remain an explicit capability error.

```python
from eval.device import Machine, parse_nvidia, select_gpus

gpus = parse_nvidia("0, GPU-fixture, Fixture GPU, 81920")
selected = select_gpus(gpus, ("GPU-fixture",))
machine = Machine("Linux", "x86_64", 8, 32 * 2**30, selected)
assert str(machine.dockerfile) == "images/nvidia-cuda-x86/Dockerfile"
assert machine.gpus[0].memory_gib == 80
```

This explicitly supplied probe fixture demonstrates the path mapping without
claiming a physical GPU inspection. CPU tests also inject environment-command
outputs. Actual inspection currently supports Linux NVIDIA x86; other category
probes and acceptance tests are future work.

## Request policy — `eval/request_policy.py`

`request_fields(model, engine)` joins model thinking/sampling policy and backend
fields from `request_fields.json`. `policy.request()` builds a comparison request
without changing recorded messages or tools. `policy.count(event)` reads
`usage.completion_tokens` and returns `None` when a stream event has no usage.
Every backend defined in `request_fields.json` uses `/v1/chat/completions`;
Ollama's extra fields come from that policy. The four LM Studio mappings belong to
the future Apple Silicon category and have no request policy yet.

```python
from eval.request_policy import request_fields

policy = request_fields("C2", "ollama")
body = policy.request([], workload="A-N", output_limit=12, served_model="baseline")
assert policy.endpoint == "/v1/chat/completions"
assert body["max_tokens"] == 12
assert body["ignore_eos"] is True
assert body["think"] is True
assert policy.count({"usage": {"completion_tokens": 12}}) == 12
```

These assertions are also used in request-policy tests. S requests permit early
stopping; agentic requests apply the recorded output limit and the authored
forced-length fields. This module constructs requests, not server launch commands
or an agent tuning allowlist. Engine support is verified when servers are added.

## Timing — `eval/bench/metrics.py`

`numbers(workload, records)` computes generation throughput, busy throughput,
mean TTFT, and score validity. Overlapping intervals count once. It consumes
monotonic timestamps and server-reported counts directly, without an extra
shape/type validation layer. Failed requests or zero timing duration invalidate
the run's score. Replay owns stream checks; the judge owns answer grading.

```python
from eval.bench.metrics import numbers

report = numbers(
    "S-N",
    [
        dict(ok=True, t_send=0, t_first=1, t_last=4, t_end=6, output_tokens=10),
        dict(ok=True, t_send=2, t_first=3, t_last=5, t_end=7, output_tokens=20),
    ],
)
assert report["decode_s"] == 4
assert report["throughput"] == 7.5
assert report["ttft_mean"] == 1
```

This is the executed overlapping-request arithmetic fixture. The decode union is
`[1, 5]`, so 30 tokens across four seconds means 7.5 tokens/s. Startup, quality,
warmup, and idle tool delays are outside the supplied timing records.

## Implementation costs — `eval/costs.py`

`collect_trial()` reads finalized Harbor evidence. `from_harbor_result()` accepts
it in memory. `merge_attempts()` deduplicates the same immutable attempt while
retaining retries; `update_ledger()` persists the attempt ledger.
`summarize()` separates reported, explicitly estimated, and unknown costs.
Serving-output tokens do not enter this ledger. Cached input is a subset of
input; an explicit rate table is needed for a local estimate.

```python
from eval.costs import collect_trial, merge_attempts, summarize

attempt = collect_trial("tests/fixtures/harbor_trial")
report = summarize(merge_attempts((), [attempt, attempt]))
assert report["attempts"] == 1
assert report["total_tokens"] == 110
assert report["cost_usd"] is None
```

The checked-in synthetic Harbor fixture reports input 100, cache 40, output 10:
110 total tokens. Missing cost stays unknown. A finalized attempt ID cannot be
reused with changed evidence. The foreground runner writes the ledger before
cancelled-trial cleanup and during finalization. Malformed files/structures fail
at their native reader. Search-agent billing is not recorded; only coding-step
usage enters the implementation ledger.

## Phase 2 composition

The [Phase 2 review packet](phase-02.md) explains `models.py`, `tasks.py`,
`setup.py`, and `run.py`. Normal `hf download --local-dir` commands acquire
the source checkpoint. Setup serializes trusted prepared rows unchanged and
samples one private-S calibration view per model for every workload. A baseline
matching the original uses `/models/source`; distinct baseline checkpoints remain
model identifiers for container runtime acquisition. Generated `run.md` lists
the source path and baseline identifiers. Task rendering defines one
shared sandbox; the foreground wrapper saves configuration and delegates to
Harbor. Native search/coding task configuration is implemented; the small
`eval.agents:PhasedAgent` coordinator and search operations are Phase 5 work.
Servers, replay and judging remain their own later implementation phases.
