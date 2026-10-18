# Reading order

Start with the configured task's lifecycle, then read the mechanics it calls.

| Order | File | Responsibility |
| --- | --- | --- |
| 1 | `config.example.yaml`, `eval/config.py`, `tables/models.csv` | Parse task filters, native agents, resources, task concurrency, replay N and retries. |
| 2 | `eval/run.py` | Lease task slots, share search within a run, transfer handoffs, select current attempts and resume unfinished work. |
| 3 | `eval/setup.py` | Freeze source inputs and export approved views from the single HF dataset release. |
| 4 | `eval/tasks.py`, `templates/` | Generate task-owned Dockerfiles/setup scripts, search tasks and classified coding/verification tasks. |
| 5 | `eval/providers/{docker,daytona}.py` | Delegate task image builds to Harbor and implement sandbox lifecycle/transport. |
| 6 | `templates/setup.sh`, `eval/providers/transfer.py` | Upload approved inputs and execute trusted root setup before the agent starts. |
| 7 | `eval/providers/checkpoint.py`, `eval/history.py` | Lightweight authored-output/evidence capture and native chat restoration. |
| 8 | `eval/search/run.py`, `eval/launch.py`, `eval/bench/serve.py` | Validate the selected reference and manage one server's launch/readiness/cleanup. |
| 9 | `eval/bench/run.py`, `replay.py`, `progress.py` | Shared public/private executor, graph scheduling, fixed N batches and durable recovery. |
| 10 | `eval/bench/clients.py`, `metrics.py`, `slo.py`, `watchdog.py` | Stream/token evidence, timing arithmetic, tier scoring and the active-serving cutoff. |
| 11 | `eval/verify/{run,judge,review,access}.py` | Installation, restricted native reviewer, private quality gate, reference and candidate measurements. |
| 12 | `eval/quality.py`, `eval/verify/profile.py`, `tables/quality.csv` | Offline 32-round preparation and the published per-model quality range. |
| 13 | `eval/costs.py`, `eval/timings.py`, `eval/report.py` | Coding usage, recorded phase/provider durations, rejected zeros, and results under the run. |
| 14 | `eval/bench/{live,proxy,protocols,harnesses,sources}.py` | Optional live recording; uses the same server/proxy path without evaluation metrics. |

Public and private replay use the same code. Providers do not implement separate
schedulers, quality checks or retry policies. The root coordinator owns those
policies and the run folder owns their recoverable state.

## Recorded time

Each task in `result.json` has `timings.search`, `timings.coding` and
`timings.verification`. `recorded_seconds` separates native trial elapsed time,
environment setup, agent setup, agent execution, and verifier steps (`prepare`,
`review`, `quality`, `reference`, `measure` or `slo`). These are time spent across
recorded attempts, including retries; they are not estimates of an uninterrupted
run. Pauses between attempts are excluded. `incomplete_attempts` identifies
missing end times, so recorded totals can be lower bounds. Unknown values are
`null`. Reused search results have `{"reused": true}`; their original search time
is unknown.

Each trial also journals provider operations in `timings.jsonl`, separating
checkpoint indexing, packing, download and ingestion; restore packing, upload and
extraction; and payload/input upload and setup. `provider_seconds` summarizes
finished operations. An interrupted operation has a start record and no invented
end time. Checkpoint capture runs alongside agent/replay work; its total also
includes its recorded suboperations. These component times overlap and must not
be added to native durations or each other. Reports preserve
the timing property in JSON and CSV. Historical provider breakdowns stay unknown.
