# {{ stage | capitalize }} the submitted system

This is a trusted verification step for {{ task.model }} / {{ task.workload }}.
Harbor uses its no-op agent; the step's verifier performs the work.
{% if stage == "prepare" %}
Root validates the source-only submission, converts weights if requested, installs
dependencies and prepares readable artifacts and the review rubric.
{% elif stage == "review" %}
The restricted judge user runs Harbor Rewardkit to inspect the prepared system.
The submission, runtime and weights are readable; private evaluation data is not.
{% else %}
Root runs the quality gate and timed replay, then writes the final reports and rewards.
{% endif %}
