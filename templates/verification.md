# {{ stage | capitalize }}: {{ task.model }} / {{ task.workload }}

This is an automated Harbor step using its no-op agent.
{% if stage == "prepare" %}
The trusted controller runs `/workspace/install.sh`, which creates
`/workspace/server.launch` and installs generated dependencies under
`/workspace/runtime/candidate/`. The target and tokenizer are read-only.
Stop installation processes, protect source and installed code, and clear
designated temporary state before review.
{% elif stage == "review" %}
Run the configured reviewer as `judge` using the supplied rubric. It can read
candidate source, installed dependencies and model assets. Its output is under
`/workspace/output/review/`, with scratch under `/workspace/runtime/judge/`.
Private evaluation evidence remains inaccessible. Review must pass before quality.
{% elif stage == "quality" %}
Start the candidate offline and submit the standalone hidden questions at
concurrency -1. Grade against the frozen offline requirement within
{{ config.evaluation.quality_timeout_seconds | duration }}. Stop when passing
becomes impossible. Quality must pass before performance measurement.
{% elif stage == "reference-prepare" %}
After candidate quality passes, install the selected trusted baseline and acquire
its mapped checkpoint online under `/workspace/runtime/reference/`. Protect the
installation and stop its processes before the offline reference step.
Reuse an already completed reference measurement without reinstalling it.
{% elif stage == "reference" %}
Start the prepared reference server offline, perform the standard excluded short warmup and replay
the private workload. Save completed measurement units and stop the server.
Reference performance supplies the SOTA comparison.
{% elif stage == "measure" %}
After review and quality pass, start one candidate server offline, perform the
standard excluded short warmup and replay the private workload. Enforce the
performance cutoff using the workload's scored objective. The trusted controller
owns token checks, timing, comparison and the final 0–100 score.
{% elif stage == "slo" %}
After quality passes, start one candidate server offline, perform the standard
excluded short warmup and replay the private workload within its configured
limits. Enforce the minimum-tier performance cutoff. Completed graphs/batches
retain credit; unfinished and unstarted units earn zero. Write the 0–100 score.
{% endif %}
