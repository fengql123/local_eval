# Benchmark specification

The benchmark evaluates a coding agent's serving implementation for a model,
workload and declared device category. Agent implementation costs are separate
from the server's generated tokens. A submission must pass review and private
quality before receiving a performance score.

## Inputs

`tables/models.csv` defines the fixed candidate checkpoint. `tables/tasks.csv`
contains only task ID, model and workload. `tables/baseline_map.csv` lists reference
engines and their own checkpoints for each device category. Search determines
which reference works and selects its parameters; it never changes the candidate
checkpoint. With a usable reference the task is SOTA; without one it is SLO.
Task type is an output of search, not a configuration filter.

One HF release has `workloads/records` and `quality/questions`. Workload records
contain `id`, `model`, `requests`, `public` and `private`. Membership lists select
the exact model/workload graphs. All graphs are stored once; empty memberships
retain unused full-corpus rows. The coordinator exports only approved rows and
removes membership columns before staging. Search uses private performance rows,
coding uses public rows, and final measurement uses the same private selection.
There is no calibration split. Quality questions and answers never enter search
or coding. Dataset construction provenance is in `sampling.json`; evaluation
does not run the sampler.

## Native Harbor phases

Search, coding and verification are separate native jobs and fresh sandboxes.
Search transfers only its selected reference and parameters. Coding transfers
only its source submission and required `install.sh`. The installer must create
`/workspace/server.launch`; verification runs exactly
`server.launch /models/target` at localhost port 8000, alias `served`.

The canonical target and its tokenizer are read-only. Target substitution,
fine-tuning and further weight quantization are forbidden, including in-memory
target re-quantization. Activation quantization is allowed when hidden quality passes. Lossless layouts, normal dequantization, compute optimizations
and target-verified speculative decoding are allowed. Installation runs from `/workspace`. Authored code and `install.sh` are submitted
from `/workspace/output/submission`; generated dependencies/builds belong under
`/workspace/runtime/candidate`. The installer locates its source relative to its
own directory and creates `/workspace/server.launch`.

Verification runs prepare → review → quality → reference preparation → reference → measure.
Reference preparation installs and acquires assets online; both measurements run offline.
SLO runs bounded candidate timing after quality and has no reference steps. Root coordinates procedures;
installers and serving processes use unprivileged identities. The reviewer has
read-only access to installed artifacts and its own writable home/output. Serving
and quality measurement run without network access. Both checkpoints are recorded:
the reference may use a different format from the canonical candidate.

## Replay

S-1 and A-1 run sessions serially. A-MA runs sessions serially while honoring each
graph's concurrent branches. S-N submits N independent prompts together; A-N
tracks N session graphs. A complete N batch finishes before the next starts.
Official N is a positive multiple of eight dividing the private selection.
Public experiments may choose their own concurrency.

Every request uses its exact recorded messages and tool definitions. Parent
dependencies determine readiness. A-N retains recorded tool waits. A-MA retains
waits throughout branched sessions; linear sessions omit them. S and A-1 omit
waits. No real tools execute during replay. Tool waits affect scheduling without
being counted as decode time.

`tables/request_fields.json` fixes generation policy. S allows natural EOS up to
32,768 tokens. Agentic timing uses each request's recorded output count and
`ignore_eos`; a mismatched count fails the replay. Hidden quality uses natural EOS.

The excluded warmup is one standard short prompt with at most 16 output tokens
and a five-minute initialization deadline after server readiness, allowing cold
kernel compilation. It is identical for reference
and candidate and does not replay the first workload request. A warmup timeout
fails the operation.

Each timing operation owns one server and one excluded warmup across all its
sessions/batches. On interruption it starts and warms a new server, keeps completed
units and reruns the interrupted whole unit. Quality commits and grades each completed
answer, including wrong answers. Its default active-serving deadline is five hours
(`evaluation.quality_timeout`). Extending this deadline resumes only budget-cut
answers, preserves finished answers (including wrong ones), and counts active
time already spent against the new total. Once the remaining questions cannot meet the
frozen minimum correct count, evaluation stops with a failed quality gate. Resume reads frozen code, inputs and progress;
there is no separate replay version or version selection.

## Timing and SOTA scores

The HTTP client records monotonic send, first genuine output, last output and
response-end times. Role metadata, tool IDs, finish markers, usage and keepalives
cannot start TTFT. Genuine reasoning, tool arguments, content and accepted token
IDs can. Only timed requests enter metrics; startup and warmup are excluded.

For S-1/A-1/A-MA, generation throughput is accepted output tokens divided by the
sum of each unit's union of first-to-last intervals. Overlapping intervals count
once. S-N/A-N score serving throughput instead: tokens divided by the union of
send-to-end intervals per fixed N batch. This includes prefill and queuing and
excludes tool-only idle time. `decode_throughput` and `busy_throughput` retain both
denominators as diagnostics. Mean and p95 TTFT, stream gaps and mean TPOT are
reported where applicable. Mean completion time is not a benchmark metric.

| Workloads | TTFT weight | Throughput weight |
| --- | --- | --- |
| S-1, A-1, A-MA | 50% | 50% |
| S-N, A-N | 0% | 100% |

Calculate throughput gain as candidate/reference throughput and latency gain as
reference/candidate mean TTFT. G is their geometric mean for balanced workloads,
or throughput gain alone for N workloads. Each task has one performance score:

```text
score = 100 * max(0, 1 - 1/G)
```

Parity and regression earn zero improvement points; G=2 earns 50 and G=4 earns
75. Component speedups remain in the report. A valid zero-improvement submission
is distinguished from a rejected submission by its outcome.

Reviewer, quality or candidate-serving rejection scores zero. Infrastructure and
reference failures remain incomplete. Raw accepted token IDs must decode to the
supplied raw output with the fixed tokenizer, and usage must match accepted
counts. The reviewer checks that those claims follow inference and streaming is
honest.

## Performance limits

Official candidate timing has a 120-second active-serving window. SOTA N
workloads compare serving throughput to the measured reference; balanced
workloads compare the geometric TTFT/throughput gain. The window uses accepted
token evidence across active requests, not stream-event counts. Tool-only waits,
startup, warmup and sandbox downtime do not consume it. Balanced workloads also
have a first-token allowance of at least 120 seconds (or twice reference mean
TTFT, whichever is longer); SLO uses 120 seconds. N includes prefill and queueing
in its serving-throughput window.

A sustained scored gain at or below parity ends SOTA measurement with score zero
only when both the recent window and the overall operation are at or below parity.
This avoids rejecting a faster N batch when its final few requests drain slowly.
The result is reported as `performance_cutoff`, an execution-budget policy rather
than a projection of a complete measurement. Public timing experiments have no
reference cutoff. SLO applies the same recent-and-overall check against its lowest
throughput tier and preserves completed-unit
credit, keeping unfinished/unstarted units and repeats at zero. Reference and
infrastructure failures remain retryable; candidate performance cutoffs are
terminal. Progress journals include active requests and accepted-token counts
at five-second intervals where streams expose token evidence.

## SLO scores

`tables/slo.json` contains four fixed tiers. S-1/A-1/A-MA calculate each tier's
TTFT and throughput attainment, cap both at one, and combine them with equal
geometric weight. S-N/A-N use throughput attainment alone. The task score is 100
times the mean credit across tiers and planned units. See [slo.md](slo.md) for
targets, formulas and examples.

Each whole prompt/session graph contributes equally; each whole fixed N batch
contributes equally. A unit must contain all expected requests. Failed, capped,
budget-cut and unstarted units receive zero. No partial completion is projected
and no unfinished work disappears from the denominator. A tier requires 90%
whole-unit attainment; confidence intervals are diagnostic. S prompts cycle
through doubling ranges of prefill payload bytes during bounded timing.

## Results

Both SOTA and SLO publish one final Harbor performance reward, `{"score": value}`,
from 0 to 100. Preparation, review and quality retain their own pass/fail gates.
Final averages weight each selected model/workload task equally and include
rejected zeros. Reports show overall, SOTA and SLO averages, component speedups,
SLO tier attainment and completion coverage. A final overall mean is pending
until every selected task has a score. Repeated attempts are averaged within a
task before averaging tasks. Complete saved SOTA timestamps can be rescored
without launching a server; legacy scores without sufficient evidence remain
unscored rather than being mixed into the new range.

## Quality, costs and oracle

The standalone quality suite contains 30 questions, with one shared model bound
from 32 offline repetitions. The published question identity, checkpoint and
policy must match. Candidate verification requires at least the observed minimum
correct; higher accuracy passes. Accuracy functions grade answers independently
of the semantic LLM reviewer. See [quality.md](quality.md).

Only coding attempts contribute to costs. Native Harbor usage and trajectories
provide the evidence, including interrupted attempts. Missing usage stays unknown.
Result tables retain rejected zeros and group the measured hardware and agent.

Native OracleAgent supplies known submissions and runs the same complete
verification. Its coding recipe must support the canonical candidate checkpoint;
a selected reference with a different format does not change that target.
Recording is a separate host mode: native harnesses send requests through a
capturing proxy using the shared HTTP client and server lifecycle. It does not
calculate evaluation metrics, grade answers or run Harbor tasks.
