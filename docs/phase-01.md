# Phase 1 review packet

## Status and scope

**Revised after owner feedback; uncommitted and awaiting review.** Phase 1 now
contains six foundation areas. It performs no downloads, engine/image builds,
server launches, agent calls, replay, judging, or run orchestration.

| Read order | File | Responsibility |
| --- | --- | --- |
| 1 | `README.md`, `config.example.yaml` | All config fields/defaults, working inspection, and new image structure |
| 2 | `eval/config.py`, `tables/tasks.csv` | Validated settings and model/workload/type task selection |
| 3 | `images/README.md`, `tables/categories.csv`, `tables/baselines.csv`, `tables/baseline_map.csv` | Supplied category images/installers and source/checkpoint/launch metadata |
| 4 | `eval/baselines.py` | Installer paths and unrestricted argv/env composition; no install generation |
| 5 | `eval/device.py` | Selected GPU/resource evidence and direct category Dockerfile path |
| 6 | `tables/request_fields.json`, `eval/tables.py` | Evaluator request formatting and output counters; no agent tuning policy |
| 7 | `eval/bench/metrics.py` | Timing interval unions, throughput, TTFT and invalid evidence |
| 8 | `eval/costs.py` | Coding-agent/harness usage and attempt ledger |
| 9 | `examples/inspect_foundations.py`, `tests/` | Actual catalog inspection and four cases per foundation area |

## Feedback addressed

1. Removed `QUALITY_BATCH`. The future quality gate submits its S subset as an
   offline workload; the serving backend owns batching/scheduling.
2. The README lists every accepted config field, its default, and its purpose;
   the example YAML includes all sections.
3. Baselines reference `images/<category>/baselines/<engine>/install.sh` with
   catalog repository/ref/prefix. Inline installation, clone-command generation,
   and toolkit recipes are removed.
4. Device inspection exposes `machine.dockerfile`. Generated-image parameters,
   CUDA toolkit selection, compile architectures, and driver-floor policy are removed.
5. Deleted `eval/search/contract.py` and its allowlists/help parsing. Baseline
   argv/env overrides are unrestricted; only officially documented Ollama bind
   and model-directory defaults remain. Evaluator request formatting is retained.
6. Deleted shared HTTP test infrastructure and future-only package markers.
   Tests cover implemented Phase 1 code only: two common and two edge cases
   per area, with no parameter matrices.

## Focused checks

| Area | Common case 1 | Common case 2 | Edge case 1 | Edge case 2 |
| --- | --- | --- | --- | --- |
| Config | Task-filter intersection | Complete YAML with relative output | Unknown field | Empty task intersection |
| Request tables | OpenAI fields/recorded-input preservation | Native Ollama fields/counter | Unsupported exact agentic output | Invalid output limit |
| Baselines | Catalog links and engine union | Unrestricted argv/env composition | Missing model mapping | Planned launch metadata |
| Device | Selected GPU and Dockerfile path | Environment availability probes | Unknown GPU | Unsupported host |
| Metrics | Overlapping timings | Serial timings without idle gaps | Partial request failure | Zero decode window |
| Costs | Saved usage/cache subset | Explicit estimate/ledger round trip | Duplicate collection and retry | Invalid cache count |

Run from the checkout:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run python examples/inspect_foundations.py config.example.yaml
uv build
```

Validated with Python 3.12.14:

* **24 tests passed:** six areas with exactly four tests each, no parameter matrices.
* Ruff lint and format checks passed.
* The real example configuration/catalog inspection ran successfully and printed
  the selected task, category Dockerfile/installer paths, launch argv and request fields.
* The source distribution and wheel built successfully offline.

These checks exercise implemented foundations with local evidence. No network,
dataset, GPU build, Harbor task, or model provider was used.

## Working inspection

The inspector selects the actual `qwen36-35b-a3b_S-1` catalog task. For its
explicit NVIDIA x86 / FreeToken entry it reports source ref `v0.1.3`, prefix
`/opt/baselines/freetoken`, category Dockerfile path, category installer path,
launch argv, and evaluator request fields. This is metadata inspection, not a
claim that an engine or Dockerfile has been built.

## Limits and next phase

Linux NVIDIA x86 is the implemented device-inspection path. Other category names
are source-layout entries; their probes and accelerator acceptance remain later
work. Production Dockerfile/installer contents are supplied by the owner. There
is no generated substitute. Native Ollama's current policy cannot enforce exact
agentic output counts. DeepSeek-V4.1-Flash has no usable NVIDIA model mapping.

Phase 2 composes these APIs with supplied source images into host acquisition,
generated Harbor tasks, UUID folders, and foreground folder-only resume. Its code
and renderer tests use acquisition/source-asset fixtures. Real rows first arrive
in Phase 3; the owner's dataset quality is trusted. HTTP/server test support is
introduced alongside the actual HTTP/server code, not in Phase 1.
