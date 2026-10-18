# SLO scoring

A task is SLO when search finds no supported reference on its allocation. The
candidate must pass installation, integrity review and hidden quality before
performance is scored. Thresholds are fixed across models, devices and agents.

## Targets

`tables/slo.json` defines four tiers. These are benchmark policy targets, informed
by the distinction between interactive serving and batch capacity. They are not
claims that one published standard prescribes these exact thresholds.

| Tier | S-1 / A-1 / A-MA TTFT | S-1 / A-1 / A-MA generation throughput | S-N serving throughput | A-N serving throughput |
| --- | --- | --- | --- | --- |
| T1 Basic | ≤10 s | ≥5 tok/s | ≥10 tok/s | ≥10 tok/s |
| T2 Steady | ≤2 s | ≥10 tok/s | ≥50 tok/s | ≥40 tok/s |
| T3 Responsive | ≤1 s | ≥25 tok/s | ≥200 tok/s | ≥160 tok/s |
| T4 Fast | ≤0.5 s | ≥50 tok/s | ≥1,000 tok/s | ≥640 tok/s |

Official replay defaults to S-N=32 and A-N=8. Its configured N must divide the
private selection. Public experiments may choose another N. Comparing systems
requires the same N, dataset and hardware policy.

## Units and credit

One S-1 prompt or one A-1/A-MA session graph is a scoring unit. Compute mean TTFT
within that unit. Generation throughput is accepted output tokens divided by the
union of first-to-last-output intervals. Concurrent branches count overlapping
time once. Long and short graphs each contribute one unit.

For each tier, calculate:

```text
latency_credit = min(1, target_ttft / actual_mean_ttft)
throughput_credit = min(1, actual_throughput / target_throughput)
tier_credit = sqrt(latency_credit * throughput_credit)
```

TTFT and throughput have equal geometric weight. For example, 0.8 latency credit
and 0.45 throughput credit produce 0.6 tier credit. Exceeding a target gives no
extra credit to compensate for missing the other target.

One whole fixed N batch is the scoring unit for S-N/A-N. Throughput includes
prefill and queuing: divide accepted output tokens by the union of send-to-end
intervals. Intervals when all sessions are waiting only on tools are excluded.
Tier credit is just `min(1, actual_throughput / target_throughput)`.

The task score is `100 * mean(tier_credit)` across all planned units and all four
tiers, including every configured repetition. The final Harbor reward contains
only `{"score": <0–100>}`. Tier credits and components remain diagnostic fields.
A tier badge requires at least 90% of planned units to meet every applicable
limit fully. Its Wilson interval describes whole-unit attainment and does not
change the score or gate the badge.

## Incomplete work and resume

A complete unit contains every expected request ID, with successful accounting
and no capped or budget-cut streams. An incomplete graph or N batch earns zero;
completed prefixes are not extrapolated. Unstarted units also earn zero. Reports
show both started and completed coverage. Infrastructure interruptions remain
retryable and do not become performance results.

Replay keeps completed units on resume and restarts the interrupted whole unit.
It starts one server and performs one excluded warmup for each replay operation.
The same arithmetic applies to public experiments and private verification.

## Related work

[MLPerf](https://mlcommons.org/2025/09/small-llm-inference-5-1/) distinguishes
interactive latency constraints from offline capacity.
[DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
measures useful throughput under latency requirements.
[Tempo](https://arxiv.org/html/2504.20068v1) motivates workload-specific objectives
and declining utility when a target is missed. Our four-tier geometric score is
an explicit benchmark policy rather than one of their official metrics.
