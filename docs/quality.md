# Standalone quality gate

Quality is a hidden configuration of the main dataset release. It does not replace
S timing prompts and is never supplied to search or coding agents. The current
suite has 30 questions: eight math, eight science, six code and eight long context.
Each row selects its checker in `evaluation`; trusted code dispatches the checker.

For each model, offline preparation runs those same questions 32 times with the
benchmark's fixed sampling policy and the canonical checkpoint from
`tables/models.csv`. All questions can queue into the serving engine together;
continuous GPU batching belongs to that engine. The current gate accepts at least
the minimum number correct observed across those rounds. The maximum and all 32
round scores remain in `tables/quality.csv`. The candidate uses that same fixed checkpoint regardless of the selected
reference format. This is an empirical bound on this suite, not a guarantee for every
possible prompt or a formal confidence interval.

## Publish a range

`eval.verify.profile` runs a trusted profiling server and stores answers, round
results, tokenizer/checkpoint provenance and a complete `profile.json` in its run
folder. It resumes per completed answer and round. After reviewing the result:

```bash
uv run python -m eval.quality publish runs/<uuid>/profile.json \
  --dataset /absolute/path/to/local_eval/data/dataset
```

Publication validates all 32 rounds against the selected questions and request
policy, then replaces only that model's CSV row. The table records model,
checkpoint provenance, question count/identity, minimum/maximum correct, round
scores and sampling policy. Raw answer evidence stays in the profiling run.
There is no separate quality dataset or `profiles/<hash>.json` lookup at run time.

## Verification

The coordinator checks that the frozen table matches the model, question suite and
sampling policy before allocating verification. Verification starts the installed
candidate on the fixed checkpoint and admits the hidden questions together. It
does not measure timing in this mode. Completed answers are committed immediately;
wrong answers remain wrong after a resume. Only unfinished answers are resent. An
answer unfinished at the quality deadline counts as wrong.

The quality gate runs after source review and before performance measurement.
Failure produces a rejected zero; infrastructure failure produces no quality
verdict and triggers bounded recovery. A stored successful gate skips repeated
inference. A failed gate's saved replay result also prevents retrying until it gets
a more favourable random answer.

## Maintaining questions

Maintainers can use `python -m eval.quality prepare --help` to publish explicitly
selected question IDs into the release's `quality` configuration. It checks the
intended category mix, distinct prompts and separation from workload prompts.
`python -m eval.s_dataset --help` writes the length-stratified S workload
configuration. These are dataset construction tools, not runtime schema audits.
Changing the questions or fixed sampling policy requires new offline ranges.
