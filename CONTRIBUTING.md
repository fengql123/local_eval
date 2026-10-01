# Contributing

## Development

Use Python 3.12+ and uv:

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The lockfile fixes development dependencies. Current runtime dependencies contain
only the YAML parser; add Harbor, dataset, serving, and grading dependencies when
their implemented modules actually need them.

## Scope and tests

Implement only the current phase. Keep configuration, catalog metadata, device
observation, timing, and agent-cost accounting with their owners. Installation
belongs to supplied device Dockerfiles and category baseline scripts. There is
no generated-image layer or tuning allowlist.

Each Phase 1 test area has **two common cases and two edge cases**. Use explicit,
small evidence: hand-calculated timing windows, saved hardware replies, and the
local Harbor usage fixture. Do not add large parameter matrices or shared test
infrastructure for later phases. HTTP stream fixtures arrive with HTTP clients;
Harbor integration tests arrive with Harbor orchestration.

Update the README's config table, API reference, and working example when an
interface changes. Separate actual check results from planned acceptance. Keep
credentials, caches, models, and run outputs outside source control; `images/`
contains authored source assets and must remain tracked.

## Phase review process

1. Implement the current phase and its focused behaviour tests.
2. Present the changed files, API responsibilities, execution example, check
   results, and limitations.
3. Stop for owner review. Do not begin the next phase or commit before approval.
4. After approval, create one commit for that phase using a conventional message,
   such as `feat: add independent evaluation foundations`.

| Phase | Deliverable | Inputs needed |
| --- | --- | --- |
| 1 | Config, catalogs, baseline metadata, device, metrics, costs | No dataset or image builds |
| 2 | Setup, model assets, generated Harbor tasks, foreground run/resume | Acquisition/rendering fixtures; supplied images/scripts for production builds |
| 3 | Servers, HTTP clients, recorded replay, measurement | Owner supplies recorded rows/access |
| 4 | Offline pre-run quality/static gate and timed verifier | Reuse supplied rows |
| 5 | Reference search and startup readiness | Reuse calibration views |
| 6 | Harbor and selected device/engine acceptance | Supplied images/scripts and actual target hardware |

The separate recording package is added when its implementation starts, after its
shared server/protocol prerequisites are available.

## Documentation

The README covers purpose, prerequisites, working commands, and all config fields.
The benchmark specification defines workloads, trusted columns, and metrics; the
reference explains implemented APIs with actual examples. The review packet records
what is ready for review and what remains for later phases. Label future interfaces
clearly instead of offering commands that are not implemented.

Useful examples of this organization include
[Terminal-Bench](https://github.com/harbor-framework/terminal-bench),
[its contributor guide](https://github.com/harbor-framework/terminal-bench/blob/main/CONTRIBUTING.md),
and [SWE-bench's quickstart](https://github.com/SWE-bench/SWE-bench/blob/main/docs/guides/quickstart.md).
Harbor's lifecycle/task format is described in [its documentation](https://docs.harborframework.com/).
