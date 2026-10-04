# Phase 1 review packet

## Status and scope

**Phase 1 was reviewed and committed as `af52a39`.** It introduced six foundation
areas without dataset/model acquisition, image builds, server launches, agent
calls, replay, judging, or run orchestration. The current checkout also contains
approved Phase 2 and the owner's simplification feedback. This packet describes
the current foundation APIs; the original commit remains unchanged.

| Read order | Files | Responsibility |
| --- | --- | --- |
| 1 | `README.md`, `config.example.yaml` | Configuration, commands, defaults, and review status |
| 2 | `eval/config.py`, `tables/tasks.csv` | Settings and model/workload/type selection |
| 3 | `images/README.md`, `tables/baseline_map.csv` | Fixed image/installer paths and source/baseline checkpoints |
| 4 | `eval/baselines.py` | Checkpoint selection and engine-name deduplication |
| 5 | `eval/device.py` | GPU/resource evidence and direct category Dockerfile path |
| 6 | `tables/request_fields.json`, `eval/request_policy.py` | Comparison requests and server output counters |
| 7 | `eval/bench/metrics.py` | Timing unions, throughput, TTFT, and score validity |
| 8 | `eval/costs.py` | Implementation usage and immutable attempt ledger |
| 9 | `examples/inspect_foundations.py`, matching tests | Executed inspection and four cases per foundation area |

## Current architecture

- Only `tasks.csv`, `baseline_map.csv`, and `request_fields.json` are catalogs.
  Install scripts own cloning, refs, install prefixes, and isolation; their paths
  follow `images/<category>/baselines/<engine>/install.sh`.
- `baseline_map.csv` supplies the original checkpoint plus each engine's baseline
  checkpoint. Setup exposes both to the agent, which may quantize the source.
  Matching identifiers share files; generated `run.md` supplies actual paths.
- `request_policy.py` formats requests for the candidate and engines defined in
  `request_fields.json` using the authored OpenAI endpoint policy. No agent tuning
  or launch environment rules are imposed.
- Configuration, catalog, file, and evidence parsing use their ordinary native
  errors. There is no duplicate schema audit or defensive recovery layer.
- Quality uses a fixed S subset as an offline workload. The backend owns batching;
  there is no quality batch-size setting.
- Tests cover implemented behavior only, with two common and two edge cases per
  area. Shared HTTP infrastructure arrives with the later HTTP implementation.

## Focused foundation checks

| Area | Common case 1 | Common case 2 | Edge case 1 | Edge case 2 |
| --- | --- | --- | --- | --- |
| Config | Task-filter intersection | Complete YAML and relative output | Unknown field raises native TypeError | Empty task intersection |
| Request policy | Candidate OpenAI policy | Ollama backend fields | Absent/zero stream usage and agentic sampling | Recorded inputs remain unchanged |
| Baselines | Original/baseline checkpoint mapping | Shared-engine deduplication | Category without rows | Model without category mapping |
| Device | Selected GPU and Dockerfile path | Environment probes | Unknown GPU | Unsupported host |
| Metrics | Overlapping timings | Serial timings | Failed request invalidates run | Zero decode duration |
| Costs | Cache counted once and unknown cost | Explicit estimate and ledger round trip | Duplicate collection and retry | Conflicting immutable attempt evidence |

The current foundation suites contain 24 tests. Phase 2 adds four more areas,
bringing the checkout to 40 tests. Run from the checkout:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run python examples/inspect_foundations.py config.example.yaml
```

All 40 tests passed after the simplification. Model/dataset acquisition and agent
execution are substituted at their external boundary. Task-renderer checks use
the real Harbor schema and Docker Compose parser, without an image build or
running daemon. No paid agent call or physical GPU result is claimed.

## Executed inspection

```bash
uv run python examples/inspect_foundations.py config.example.yaml
```

This reads the real example configuration and catalogs. It selected
`qwen36-35b-a3b_S-1` and reported:

```text
Dockerfile: images/nvidia-cuda-x86/Dockerfile
Installer:  images/nvidia-cuda-x86/baselines/freetoken/install.sh
Source:     Qwen/Qwen3.6-35B-A3B
Baseline:   nvidia/Qwen3.6-35B-A3B-NVFP4
Request:    stream=true, include_usage=true, max_tokens=32768
```

These values are an excerpt from the inspector's JSON output. The category and
engine are explicit example choices; this does not inspect the actual host or
claim that the pending image/installer exists. The [foundation reference](reference.md)
contains executed API examples and explains each module's job.

## Limits and Phase 2

Host inspection currently supports Linux NVIDIA x86. Other category paths specify
the source layout; their probes and acceptance remain future work. Production
Dockerfiles/installers await owner supply. DeepSeek-V4.1-Flash has no NVIDIA mapping
in the current table.

Phase 2 composes these foundations into host acquisition, native Harbor task
tasks, UUID run folders, and foreground folder-only resume. Its code remains
approved as one phase commit. Real owner-supplied rows are first needed for Phase 3;
the dataset's quality is trusted. See [the Phase 2 packet](phase-02.md).
