# Foundation reference

Phase 1 exposes six small areas. These functions read local settings/catalogs or
process explicit evidence; they do not download datasets, build images, start
servers, or run agents. `examples/inspect_foundations.py` exercises the real
configuration → task → baseline metadata → request-policy path.

## Configuration — `eval/config.py`

`load_config(path)` parses YAML into dataclasses. `parse_config(mapping, base=...)`
validates the same input in memory. `select_tasks(config.tasks)` intersects model,
workload, and type filters in catalog order. Model aliases become canonical names;
relative output paths resolve beside the YAML file. Unknown settings and empty
task selections fail clearly. The complete setting/default list is in the README.

```python
from eval.config import load_config, select_tasks

config = load_config("config.example.yaml")
tasks = select_tasks(config.tasks)
assert [task.task_id for task in tasks] == ["qwen36-35b-a3b_S-1"]
```

This reads the checked-in example and actual task catalog. Phase 2 will use the
same task selection to generate Harbor tasks. No dataset is accessed here.

## Baseline metadata — `eval/baselines.py`

`recipe(engine, category)` resolves repository, source ref, prefix, installer path,
and launch metadata. `mapping(category, model, engine)` selects checkpoint format;
`needed(category, models)` returns the deduplicated engine union. `validate_catalog()`
checks our authored table links; it does not audit the owner's dataset.

Installation logic belongs to `images/<category>/baselines/<engine>/install.sh`.
The category Dockerfile includes the evaluator and separate baseline venvs/native
prefixes. Phase 1 resolves metadata without requiring the pending source assets
to exist. Setup will check those inputs when a real build needs them.

```python
from pathlib import Path
from eval.baselines import recipe

engine = recipe("freetoken", "nvidia-cuda-x86")
assert engine.code_ref == "v0.1.3"
assert engine.install_script == Path("images/nvidia-cuda-x86/baselines/freetoken/install.sh")
launch = engine.launch("/models/checkpoint", 8000, args=("--max-model-len", "4096"))
assert launch.argv[-2:] == ("--max-model-len", "4096")
```

This uses the same catalog row as the executable inspector and baseline test.
`launch()` composes argv without executing a shell; arguments and environment
overrides are unrestricted. The defaults are a starting point for tuning.
Engine refs identify source code; serving checkpoints are acquired later using
model strings without added model revision pins. `pending-assets` means launch
metadata exists but image/installer contents are pending; `planned` means the
launch metadata itself is not implemented.

## Device inspection — `eval/device.py`

`inspect_machine(gpu_ids)` measures CPU/RAM and selected NVIDIA GPU identities and
memory. Indices/UUIDs are passed to the probe before parsing, so an unrelated bad
GPU does not block a selected healthy one. `machine.dockerfile` maps its category
directly to `images/<category>/Dockerfile`. There is no CUDA toolchain policy or
Dockerfile-generation layer. `check_environment()` probes Docker daemon access,
Compose, and Harbor versions without starting a task.

On an actual supported GPU host:

```python
from eval.config import load_config
from eval.device import inspect_machine

config = load_config("config.example.yaml")
machine = inspect_machine(config.gpus)
print(machine.dockerfile, [gpu.uuid for gpu in machine.gpus])
```

This requires the NVIDIA driver. Phase 1's CPU tests inject saved probe outputs;
they do not claim a real GPU/container check. Inspection currently supports Linux
NVIDIA x86. Other category image paths remain part of the source design; device
probes and accelerator acceptance are added when those categories are implemented.

## Evaluator request formatting — `eval/tables.py`

`request_fields(model, engine)` reads the shared API/model policy. `policy.request()`
formats recorded messages/tools with streaming, sampling, thinking, and output
fields without mutating the caller's data. `policy.count(event)` reads the final
server-reported output counter. This is evaluator request construction, not a
restriction on the agent's serving-system implementation or tuning.

```python
from eval.config import OUTPUT_ALLOWANCE
from eval.tables import request_fields

policy = request_fields("C2", "freetoken")
body = policy.request(
    [{"role": "user", "content": "Explain caching."}],
    workload="S-1",
    output_limit=OUTPUT_ALLOWANCE,
)
assert body["stream"] is True
assert body["max_tokens"] == 32768
```

The inspector uses this policy for its selected task; later clients provide the
actual recorded messages and tools. This example constructs a body without making
a request. Native Ollama uses `/api/chat`, `think`, and nested `options`; its current
policy cannot enforce agentic exact output counts and reports that limitation
explicitly. `contract.py`, captured-help parsing, and tuning/env allowlists are gone.

## Timing — `eval/bench/metrics.py`

`numbers(workload, records)` calculates throughput, busy throughput, mean TTFT,
and explicit validity. It merges overlapping intervals so concurrent streams share
one elapsed-time denominator. Records use monotonic timestamps and server-reported
output counts. Startup, warmup, and quality inputs stay outside timed records.
Malformed/failed evidence invalidates the whole score. This module never grades
answers and needs no HTTP client or replay runner.

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

These are the hand-calculated inputs from the overlapping-request test. They
exercise the implemented arithmetic, not a measured serving performance claim.

## Implementation costs — `eval/costs.py`

`collect_trial(directory)` reads finalized Harbor usage. `from_harbor_result()`
accepts the same evidence in memory. `merge_attempts()` deduplicates repeated
collection while retaining distinct retries; `update_ledger()` saves them atomically.
`summarize()` reports coding-agent/harness tokens and known/unknown costs.
Serving-output counts never enter implementation costs. Cache is a subset of
input, so input 100/cache 40/output 10 means 110 total tokens. Estimates require
explicit `TokenPrices`; no pricing service or paid call happens here.

```python
from eval.costs import collect_trial, merge_attempts, summarize

attempt = collect_trial("tests/fixtures/harbor_trial")
report = summarize(merge_attempts((), [attempt, attempt]))
assert report["attempts"] == 1
assert report["total_tokens"] == 110
assert report["cost_usd"] is None
```

The checked-in synthetic Harbor fixture is consumed by the cost test too. Missing
USD cost stays unknown. Later foreground orchestration owns ledger writes and
collects finalized attempts before deleting disposable trial folders. Search and
judge usage remain separate roles; none of that orchestration is implemented here.
