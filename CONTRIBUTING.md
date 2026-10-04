# Contributing

## Development

Use Python 3.12+ and uv:

```bash
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The lockfile fixes runtime, host and development dependencies. Phase 2 puts
Harbor, Hugging Face datasets/downloads and Jinja rendering in the `host` extra.
Development includes that extra; the task image installs only the base runtime
(PyYAML, HTTPX, Math-Verify with its ANTLR runtime, and Harbor Rewardkit).
Inference extras are not needed for answer grading. The selected Rewardkit
agent-judge CLI must be available in the supplied image; search/coding
integrations remain Harbor adapter responsibilities.

## Scope and tests

Implement only the current phase. Keep configuration, catalog metadata, device
observation, timing, and agent-cost accounting with their owners. Installation and launches
belong to supplied device Dockerfiles and category baseline install.sh/launch.sh scripts. There is
no generated-image layer or tuning allowlist.

Each test area has **two common cases and two edge cases**. Use explicit,
small evidence: hand-calculated timing windows, saved hardware replies, and the
local Harbor usage fixture. Do not add large parameter matrices or shared test
infrastructure for later phases. Phase 3 uses `tests/http_server.py` as a shared
local HTTP/SSE fixture for its implemented client and measurement tests. Its
replay tests read the supplied public parquet rows from the sibling
`edge-harness-v2/local_eval_data/` or `LOCAL_EVAL_DATA` directory. Phase 4 reuses
that HTTP fixture for shared quality replay and verifier tests. Its native
Rewardkit integration is checked with a substituted paid backend; role commands
and firewall operations use explicit doubles without modifying the host firewall.
Four opt-in `tests/test_oracle.py` cases also start real Harbor trials in an
existing production device image with actual Claude Code review. Run them with
`LOCAL_EVAL_HARBOR_TESTS=1`; they test the lifecycle, permissions and source
review, rather than model inference or production GPU performance. Native Harbor
oracle submissions pass the ordinary verification gates.
Phase 2 checks the actual Harbor task schema and Compose configuration; tests
replace the job execution boundary rather than starting paid jobs or GPU builds.

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
| 2 | Setup, model assets, native search/coding task contract, foreground run/resume | Acquisition/rendering fixtures; supplied images/scripts for production builds |
| 3 | Launch, servers, HTTP clients, recorded replay, measurement (committed `efd3650`) | Supplied local public recordings; CPU and loopback HTTP |
| 4 | Accuracy, fresh verification, conversion/install/freeze, Rewardkit, quality/timing and native Harbor oracle (approved) | Supplied rows, CPU regressions and four opt-in Harbor/Docker trials |
| 5 | Supplied category Dockerfiles, baseline install/launch scripts and read-only/offline weight assets | Owner-supplied image/script sources and target hardware |
| 6 | Real native search/coding/verification acceptance: measured tuning, final quality/timing, resume and coding costs | Accepted Phase 5 assets, recorded rows, weight shards and agent credentials |

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
