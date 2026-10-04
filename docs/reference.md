# API reference

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
requirements. Harbor 0.23's built-in Codex requires a model; Claude Code can use
its default. Each logical task generates separate native search and coding tasks.

## Mapping to Harbor and Rewardkit

The wrapper keeps our YAML format and creates three native Harbor `JobConfig`
objects. Each logical task has separate `search/<id>`, `coding/<id>` and
`verification/<id>` folders. Each starts its own container from the same device
image; only the search task defines the image build.

| Wrapper setting / role | Native configuration |
| --- | --- |
| `search.agent` | Search job's single `AgentConfig`: name/custom import path, optional model, kwargs, env and timeout. |
| `agent` | Coding job's single `AgentConfig`, with the same mapping. |
| Agent context | Native agent env supplies HOME/USER/LOGNAME for the selected role; task environment supplies its workdir. |
| Search task | Native `[agent].user = "search"`; shared root verifier finalizes reference selection. |
| Coding task | Native `[agent].user = "agent"`; shared root verifier checks submission format and collects only source. |
| Verification trial agent | Native `AgentConfig(name="nop")`; no coding/search calls. |
| Verification preparation / measurement | Native `steps.verifier.user = "root"`. |
| Verification review | Native `steps.verifier.user = "judge"`. |
| Verification persistence | All three `steps.verifier.environment_mode = "shared"` within the fresh verification container. |
| Preparation / review gates | Native `min_reward = 1`; final measure step supplies benchmark rewards. |
| Reviewer credential names | Review `steps.verifier.env` uses native `${NAME}` placeholders. |
| `evaluation.agent.name`, `model` | Rewardkit `[judge].judge` and optional `[judge].model`. |
| Reviewer budget / kwargs | Native Rewardkit timeout, supported version/mcp_servers and review step deadline. |

Harbor resolves and executes each search/coding integration directly. No delegate
class or nested adapter configuration is required. `agent.name: oracle` selects
native OracleAgent for search and coding; verification still uses nop and the
configured reviewer. An explicit `search.agent.name: oracle` is also supported
with ordinary coding.

The image defines Unix accounts. Harbor selects their identities from TOML.
Root calls `images/prepare.sh` after conversion/installation to protect completed
artifacts and prepare writable judge directories. Judge can read candidate,
runtime and weights, write its own home/cache/output, and cannot read private
questions under `/root/eval`. Verification Compose uses
`no-new-privileges:true`. `judge.py` implements judging behaviour only.

Search's root gate publishes category, baseline, canonical checkpoint and args/env.
The wrapper copies those two metadata files into coding's protected reference
directory and points a host symlink at trusted prepared weight files. Coding and
verification bind them read-only; no weight or container-state copy is required.
Only the declared submission moves from coding into verification's prepare workdir.

Harbor saves configs in `jobs/search/config.json`, `jobs/coding/config.json` and
`jobs/verify/config.json`. Resume reloads those native configs and completed
trials. The wrapper restores its two nonsecret Codex mode controls from frozen
`run.yaml`, because Harbor masks them in saved native JSON. Credential placeholders
remain unchanged. `runs/<uuid>/result.json` links all three trial results and uses only the
verification score. The costs ledger reads only `jobs/coding`; search and reviewer
usage remain outside the implementation-cost ledger.

Reviewer login inputs are `CLAUDE_CODE_OAUTH_TOKEN`, `CODEX_AUTH_JSON` or
`VERCEL_OIDC_TOKEN`; API-key mode selects provider key names. Explicit
`evaluation.agent.env` replaces those defaults. No host login files or custom
reviewer settings are copied. Harbor handles search/coding adapter authentication.

## Baseline checkpoints — `eval/baselines.py`

`mappings(category, model)` selects rows from `baseline_map.csv`.
`mapping(category, model, engine)` selects one engine's row. `needed()` returns
engine names once, in selected-model/table order. This join selects each model's
baseline checkpoints and fixed-path installation and launch scripts.

```python
from eval.baselines import mapping, needed

asset = mapping("nvidia-cuda-x86", "C2", "freetoken")
assert asset.source_model_checkpoint == "Qwen/Qwen3.6-35B-A3B"
assert asset.checkpoint == "nvidia/Qwen3.6-35B-A3B-NVFP4"
assert needed("nvidia-cuda-x86", ("C2", "C1")) == ("freetoken", "llama.cpp", "ollama")
```

These are the inspector's real mapping and the baseline tests' shared-engine
selection. Each `images/<category>/baselines/<engine>/` contains `install.sh` and
`launch.sh`. Installation owns repository/ref, prefix and venv/native isolation;
launching supplies engine flags around the mapped checkpoint. No launcher table
or metadata audit is needed.
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
Ollama's extra fields come from that policy.

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
or an agent tuning allowlist. Actual engine compatibility is verified with
supplied images and installers in Phase 6.

## Timing — `eval/bench/metrics.py`

`numbers(workload, records)` computes generation throughput, busy throughput,
mean TTFT, and score validity. Overlapping intervals count once. It consumes
monotonic timestamps and server-reported counts directly, without an extra
shape/type validation layer. Failed requests or zero timing duration invalidate
the run's score. The HTTP client checks streams and replay rejects failed
records; the accuracy package owns answer grading.

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
at their native reader. Only coding-job usage enters the implementation ledger;
Harbor may retain ordinary usage records for the other native jobs.

## Phase 2 composition

`eval.models.prepare_asset()` acquires HF checkpoint packages with the ordinary
CLI; native engine tags remain identifiers. `eval.setup.prepare()` maps trusted
HF columns into public/private/calibration/quality views and prepares source plus
mapped baseline weights. Equal source/baseline identifiers reuse the same files.
`eval.tasks.render_task()` generates all three native task folders and their
read-only mounts. `eval.run.run()` awaits search, seeds coding's selected reference,
awaits coding, seeds verification's source workdir, and awaits verification.

The wrapper does not implement an agent loop, build an image on the host, alter
Dockerfiles, audit the supplied dataset or copy development environments.

## Launch contract — `eval/launch.py`

`Launch(script, checkpoint=None, args=(), env={}, fixed_options=..., fixed_env=...)`
executes a supplied script without rewriting it. `needs_checkpoint` detects the
reserved `$CHECKPOINT`/`${CHECKPOINT}` input, excluding full-line comments,
filled local assignments and nonempty defaults such as
`${CHECKPOINT:-/weights/optimized}`. This is the supported input convention,
not general Bash analysis. `prepare()` validates tuning and returns absolute
`/bin/bash` argv plus external environment. Only an unfilled template receives
`CHECKPOINT`; missing mapped weights fail before startup. No `PORT` is injected.

Complete candidate launchers own their final weight path and optimized defaults.
All managed servers use `ENDPOINT = http://127.0.0.1:8000` and alias `served`.
External tuning cannot set model/host/port/served-name flags or the reserved
CHECKPOINT, PORT, BASH_ENV and ENV variables. Trusted backend callers may extend
protected aliases/environment names; parameter files only supply args/env.

The launch tests execute both template and complete-script behaviours. A C2
Ollama template receives the native checkpoint tag unchanged; a local filled
assignment keeps its optimized weights even when a caller supplies another
checkpoint. Tests also reject protected overrides before process/log creation.

## Template mapping — `eval/baselines.py`

`resolve_launch(script, model, workload, task=None, reference=None)` returns
request-policy engine and optional serving checkpoint within the selected task.
Baseline folders identify engine/category. Copied templates read the current
workspace's public `reference.json`, or use the explicitly supplied protected
reference/task metadata during verification. Complete candidate launchers return
candidate policy and no checkpoint; there is no unconditional source fallback.

A public reference contains `category`, `baseline` and `checkpoint`, describing
that model/workload's selected baseline. The trusted reference finalizer publishes
write `/workspace/reference.json` alongside the protected
`/root/eval/reference/reference.json` selection; it contains
no private questions/golds. Known baseline templates map through
`baseline_map.csv`: C2 FreeToken uses `nvidia/Qwen3.6-35B-A3B-NVFP4`, while C2
Ollama uses `qwen3.6:35b-a3b-mtp-q4_K_M`. A source-equal mapping reuses
`/models/source`. Workload identifies the task/reference selection; baseline and
device distinguish different weight representations for the same model.

## Server ownership — `eval/bench/serve.py`

`Server(launch, endpoint=ENDPOINT, cwd=..., log=..., ...)` is an async
context manager. It validates Launch, rejects a preoccupied endpoint, starts an
owned process group, and requires `/v1/models` to advertise `served` while the
foreground process is live. `StartupError` includes expected endpoint and log.
Wrong port/alias, early exit and readiness timeout fail before replay.

Owned groups stop on normal exit, startup failure and cancellation; attached
`Server(None, endpoint)` instances leave their server alive. Diagnostic API tests
may use ephemeral endpoints; production replay/verification use port 8000.
An explicit base `environment` supplies runtime settings without reviewer
credentials. The server inherits its caller's OS user; fresh verification calls
it as root. Verification owns the offline serving interval outside Server and
stops owned process groups before restoring egress. Real CPU process tests cover template/complete launch,
children, attachment, wrong endpoint/alias, occupied ports and cancellation.

## Streaming — `eval/bench/clients.py`

`ChatClient(endpoint, policy, served_model=..., timeout=..., mode="timing")`
pools HTTP connections. `request(record, workload)` parses authored message/tool
JSON, builds the comparison body and reads SSE independently of packet boundaries.
Timing mode retains monotonic timings and genuine output counts; quality mode
keeps final response evidence without timing/count requirements. Both require a
valid, complete response and enforce a total deadline. No tokenizer is loaded.

## Scheduling — `eval/bench/replay.py`

`await replay(rows, workload, client, concurrency=None)` applies workload defaults
or a positive active-row/session limit. `-1` admits all rows immediately. Within
rows, every request still waits for its parents and the applicable recorded tool
delay. Generated answers never replace recorded conversations. Results retain
dataset order; task groups cancel unfinished siblings on failure/cancellation.
`ReplayError.record` identifies the failed row/request. Barrier-based tests check
concurrency without relying on wall-clock speed assertions.

## Execution — `eval/bench/run.py`

`execute(rows, workload, policy, server=..., mode="timing", concurrency=None)`
owns startup, client, replay and cleanup. `measure()` is its timing convenience
entry point: one excluded warmup followed by timed records and interval-union
metrics. `quality()` selects S policy, defaults to concurrency `-1`, sends no
warmup and invokes accuracy functions after cleanup on the main thread. Quality
returns accuracy, per-row correctness and complete response evidence, without
performance metrics.

The CLI takes MODEL WORKLOAD LAUNCH_SCRIPT with optional `--params`, `--mode` and
`--concurrency`. Timing loads cwd/data/<workload-pool>/rows.jsonl; quality loads
cwd/data/S/rows.jsonl even for an A task. It resolves only unfilled weights,
requires port 8000 and saves bench/<uuid>/server.log plus report.json. The quality
CLI report is only `{"accuracy": value}`; timing also retains launch identity,
parameters, records and separate warmup.

The supplied-public-row HTTP regression uses the actual API:

```python
from eval.bench.run import measure
from eval.bench.serve import Server
from eval.request_policy import request_fields

# selected is the supplied public C2 view; endpoint belongs to the CPU SSE fixture.
report = await measure(
    selected,
    "S-1",
    request_fields("C2"),
    server=Server(None, endpoint, health_timeout=2),
)
assert report["requests"] == len(selected)
assert report["warmup"]["row_id"] == selected[0]["id"]
```

It covers all five workloads with original messages/tools and scripted output
counts. The quality fixture admits all selected rows before replying and omits
usage fields; it still achieves accuracy 1 with no warmup or timing evidence.
See [Phase 3](phase-03.md) and [Phase 4](phase-04.md) for reproducing these tests.

## Accuracy — `eval/accuracy/`

`grade(encoded_evaluation, prediction)` resolves the trusted row's `method` to
`math_verify.check(prediction, answer)` or `letter.check(prediction, answer)`.
It excludes reasoning before the final `</think>`. Checkers own answer extraction
and dependencies; the judge owns the decision to gate timing, not answer parsing.
There is no dataset audit or replay-token verification.

The accuracy tests use the public C2 answer formats:

```python
import json
from eval.accuracy import grade

assert grade(json.dumps({"method": "math_verify", "answer": "16"}), r"\boxed{16}")
assert grade(json.dumps({"method": "letter", "answer": "C"}), "Final answer: C.")
assert not grade(
    json.dumps({"method": "math_verify", "answer": "16"}), "The calculation uses 16 numbers."
)
```

The math checker uses Math-Verify with anchored extraction and
`verify(gold, prediction)`. Letter checking rejects ambiguous alternatives.

## Judging behaviours — `eval/verify/judge.py`

The module has four functions and no access, authentication, process or network
control. `static_check(submission)` checks the submitted scripts. `write_review()`
builds a native Rewardkit rubric and inspection prompt from the protected task
and configured verification agent. `rules_result()` interprets all six binary
decisions: unusable or errored evidence is infrastructure failure, while a cited
rule violation fails the candidate. `paired()` compares complete quality results.

Root preparation invokes `/bin/bash /opt/eval/images/prepare.sh SUBMISSION
RUBRIC_DIR OUTPUT_DIR` only to set permissions. Harbor subsequently executes the
review verifier as `judge`; it invokes native Rewardkit directly. The rubric chooses the backend, optional model, timeout and
supported native kwargs. Rewardkit owns its headless inspection and native
credential inputs; no provider-specific settings or host credential files are
managed by the evaluator.

`paired(rows, reference, candidate)` requires exactly the expected row IDs and
runs the one-sided exact McNemar test. The actual threshold fixture is:

```python
from eval.verify.judge import paired

rows = [{"id": f"question-{number}"} for number in range(20)]
reference = {row["id"]: True for row in rows}
candidate = {key: number >= 5 for number, key in enumerate(reference)}
result = paired(rows, reference, candidate)
assert result["mcnemar"]["p"] == 1 / 32
assert result["ok"] is False
```

Four losses instead of five gives `1/16` and passes the 5% threshold.

Raw reviewer results persist at `/workspace/review-output`, and the review step
copies them to its native rules log directory. Final measurement rewards belong
to the measure step; the wrapper publishes those separately from gate rewards. The reviewer can inspect source, installed
dependencies and created checkpoint metadata but cannot change those artifacts.
Only its own home/cache/output is writable; private inputs are denied. Root
verification remains writable. These boundaries belong to the supplied image
and native task step users, not to judging behaviour.

## Verification — `eval/verify/run.py`

`Backend(name, policy, server)` pairs request policy with a factory for a fresh
owned server at each quality/timing stage. `prepare_submission()` performs format,
optional conversion, installation, rubric generation and image-owned permission
preparation. `review_rules()` runs native Rewardkit under Harbor's selected judge
identity and consumes no private task metadata. `pre_run()` owns only paired
quality. `probe()` delegates to `bench.run.quality(..., concurrency=-1)` without duplicating
replay scheduling. `evaluate()` consumes persisted preparation/review decisions,
then runs quality and fresh reference/candidate measurements. Candidate gate or
protocol failure writes zero rewards; reference/reviewer/environment failure
writes no reward. A wrong-port candidate is classified at startup. Stale rewards
are removed first. Results include report.json/report.md and scored reward.json.

The arithmetic fixture demonstrates independent rewards:

```python
from eval.verify.run import rewards

assert rewards(
    {"throughput": 10, "ttft_mean": 2},
    {"throughput": 20, "ttft_mean": 1},
) == {"throughput": 50, "ttft": 50}
```

Harbor invokes the protected interpreter with `-I` and an explicit `--stage`:
`handoff`, `prepare`, `review` or `measure`. Root stages use the default protected
task at `/root/eval/task.json`; review uses only the prepared public rubric. The task's `reference` field names the protected selected
reference JSON, normally `/root/eval/reference/reference.json`. The fresh verifier
mounts only source and selected reference weights read-only at `/models/source`
and `/models/reference`. An explicit reference checkpoint path is used before
any catalog mapping; complete candidate launchers own their generated weight path.
`--task PATH` changes the prepared task location. Reports go under `/logs/verifier`,
and infrastructure failure exits nonzero. The hidden `--collect-submission` option
runs only the native artifact collection hook, before task loading.

Conversion and installation run as root with a clean environment and the fixed
inputs `CHECKPOINT=/models/source`, `WEIGHTS=/workspace/weights` and
`RUNTIME=/workspace/runtime`. The optional conversion runs before installation.
The review step receives only its selected native credential inputs through
Harbor's scoped verifier environment. Submitted scripts and servers do not inherit
those credentials. The image preparation script launches no reviewer.
After online preparation/review, quality and timed serving are loopback-only in
IPv4/IPv6. Owned process groups stop before network restoration, including on
failure/cancellation. Cleanup never targets every root process.

`collect_submission()` stops original agent/search work and clears children of
Harbor's implicit `/logs/artifacts`, preserving the mounted parent. The only
declared transferred artifact is `/workspace/submission`. This keeps code and
scripts separate from installed dependencies, generated weights and coding caches.

## Native Harbor oracle

`agent.name: oracle` makes the foreground wrapper select Harbor's built-in
OracleAgent without a coding model override. Task generation writes
`search/<id>/solution/solve.sh` and `coding/<id>/solution/solve.sh`, which the
native agent executes for the corresponding jobs. The coding solution supplies baseline-backed
code, install/launch scripts and optional conversion in `/workspace/submission`.
It supplies an implementation, not precomputed benchmark answers.

The coding verifier performs the ordinary format gate; its source-only
artifact collection and wrapper transfer feed the new verification task. Native
root preparation, judge Rewardkit review and root quality/timing run unchanged for
an oracle submission. The baseline-backed install
script may reuse image-installed engine software. No local oracle module,
`trusted_control` parameter or verifier `--oracle` flag remains.

The native acceptance command is:

```bash
LOCAL_EVAL_HARBOR_TESTS=1 uv run pytest -q -s tests/test_oracle.py
```

It reuses an existing production device image, actual Claude Code with native credentials
and Harbor's generated three-job/native oracle lifecycle. A trusted test-only
verification task uses native root/judge/root steps and real Rewardkit review.
Its trusted final test hook records acceptance without starting model inference.
No test image or synthetic reviewer is used. Full quality/timing acceptance still
requires real weight shards; these opt-in cases prove the reviewer integration.
