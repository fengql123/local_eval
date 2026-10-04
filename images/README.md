# Device image sources

Each device category owns one supplied Dockerfile and its baseline installers
and launchers. Harbor builds that image once. Search, coding and verification
are three native jobs, each with its own container from that image. Search runs
as search and coding as agent, through their configured Harbor adapters. Only
selected reference settings pass from search to coding; only source submission
passes to verification. Its prepare/review/measure steps share one fresh container:
root performs conversion, installation, serving and replay; judge performs rules
inspection. External agent loops can remain on the host.

| Category | Source Dockerfile |
| --- | --- |
| NVIDIA CUDA x86 | `nvidia-cuda-x86/Dockerfile` |
| NVIDIA CUDA ARM with unified memory | `nvidia-cuda-arm-unified/Dockerfile` |
| AMD ROCm x86 | `amd-rocm-x86/Dockerfile` |
| AMD APU with unified memory | `amd-apu-unified/Dockerfile` |
| Apple Silicon | `apple-silicon/Dockerfile` |

```text
images/
  prepare.sh
  <category>/
    Dockerfile
    baselines/<engine>/
      install.sh
      launch.sh
```

Installers own repository/ref, prefix and isolated venv/native build commands.
The Dockerfile receives space-separated `ARG BASELINES` and invokes the selected
installers. There is no duplicate install catalog or generated script recipe.
Source assets await owner supply; setup reads them before acquiring data/weights
and does not invent placeholders. Host inspection currently supports Linux
NVIDIA x86; other categories and real accelerator acceptance are later work.

## Dockerfile contract

Keep the shared action/evaluation tools inside the image. Harbor adapters own
installation of their selected search/coding CLIs. Verification independently
needs the configured Rewardkit agent-judge CLI (`claude-code`, `codex` or `fx`).
The image must provide:

- Python 3.12+, bash, sleep, git, curl, certificates and device/build tools.
  Include procps (`pkill`) for cleanup of coding/search and reviewer processes.
- Selected baseline installations and root-owned, readable launchers under
  `/opt/eval/images/<category>/baselines/<engine>/launch.sh`.
- A noneditable protected evaluator at `/opt/eval/.venv`, installed with
  `uv sync --locked --no-dev --no-editable`, without host/development extras.
  Runtime includes HTTPX, PyYAML, Math-Verify with ANTLR and Harbor Rewardkit
  0.2.1. Answer grading needs no inference extra.
- The reviewer CLI selected by `evaluation.agent.name` on the trusted PATH.
  Native login inputs (`CLAUDE_CODE_OAUTH_TOKEN`, `CODEX_AUTH_JSON` or
  `VERCEL_OIDC_TOKEN`) and API keys arrive through Harbor's scoped verifier
  review-step environment. Host login files are not mounted or copied.
  Credentials never enter image layers; reviewer selection is independent of search/coding.
- Separate `search`, `agent` and `judge` users and writable private homes,
  including `/home/search` and `/home/judge` with mode `0700`.
  `/workspace` initially belongs to `agent` with mode `0755` for coding. In the
  fresh verifier, conversion/installation run as root and create root-owned
  `/workspace/weights` and `/workspace/runtime`. Before review, the image's
  prepare script gives root ownership to the submission, completed weights/runtime
  and `/workspace`, removes group/other writes and enables reading/traversal. Root
  retains write access for serving and evaluation. Judge has its own writable
  `0700` home, `/home/judge/tmp`, `/home/judge/.cache` and
  `/workspace/review-output` for persistent review output.
  Evaluator/baseline installations remain root-owned and unmodifiable by agents.
  No agent requires Docker access or unrestricted sudo.
- Copy the common `images/prepare.sh` to `/opt/eval/images/prepare.sh`, owned by
  root with mode `0755`. It only sets ownership and permissions after installation.
  Native `task.toml` selects `root`, `judge`, `root` for the prepare, review and
  measure verifiers. Harbor executes those identities directly; there is no
  user-switching launcher, provider-specific settings or authentication bridge.
- Verification Compose declares `security_opt: ["no-new-privileges:true"]`.
  Root procedures remain root; judge cannot elevate through setuid execution.
  Original search/coding Compose does not require this additional setting.
- Working `iptables` and `ip6tables`. Compose grants root `NET_ADMIN`; root
  verification restricts the container to loopback after installation and online
  review, before quality/timing. Process cleanup targets owned groups, never all
  root processes.
- Mount destinations `/opt/eval/start.sh`, `/workspace/run.md`,
  `/workspace/data/<pool>`, `/workspace/data/S`, `/home/search/data/S`,
  `/root/eval` and `/models`. Public task/S and calibration directories contain
  read-only `rows.jsonl` in their respective search or coding container. Mapped HF baseline weights are mounted read-only at
  `/models/baselines/<engine>` in search. Coding receives only selected reference weights. The fresh verifier omits
  public/calibration/coding-state mounts and receives read-only `/models/source`,
  `/models/reference`, protected task/private/quality views and
  `/root/eval/reference/reference.json`. Harbor creates its usual log/artifact
  destinations; the coding collection hook empties implicit artifact children.

The root interpreter runs verification with `-I`, keeping imports independent of
candidate workspace files. Rewardkit runs under the separate Unix `judge`
identity with its own HOME inside the fresh verifier. Native Harbor provides
each task's container lifecycle and step users; Unix permissions protect completed
candidate artifacts from the judging agent inside it while root remains writable. Rewardkit
overlay isolation is not requested. Claude's native
headless backend handles inspection; Unix permissions protect the completed
source, runtime and weights, including against shell writes.

`eval.verify.judge` checks submission format, writes the native rubric/prompt,
interprets review decisions and compares paired quality results. It contains no
access control or process launching. `eval.verify.run` implements each step's
behavior. The task TOML scopes native reviewer environment inputs to review, and root preparation calls the image's
permission script. Native Rewardkit runs directly inside the judge verifier.
The image supplies the reviewer CLI and protected evaluator package, including
`rules.toml` and `prompt.txt`. Authentication
comes from the verifier environment at runtime; a host CLI login alone does not
configure the fresh reviewer. No custom reviewer settings file is packaged.

Harbor's build context is the repository root. `.dockerignore` excludes run data,
credentials, downloaded models and development outputs, while including the
trusted evaluator prompt/rubric. Host setup uses the `host` extra for Harbor
orchestration, HF acquisition and Jinja generation; `./install.sh` selects it.

## Weight and launch contract

The host acquires source and mapped HF baseline weights through the ordinary HF
CLI; matching identifiers reuse source files. Search receives read-only source
and baseline mounts. A successful selection handoff sets the host path used by
coding and verification's read-only `/models/reference` mount, without copying
weights. Native engine tags remain identifiers. Fresh offline verification cannot pull
an uncached tag; its supplied launcher must resolve it from available assets.
Native cache transfer is not implemented. Engine installation belongs to
the image build, and run-specific weights never belong in image layers.

Coding writes its implementation under `/workspace/submission` and development
weights/runtime beside it. Only the submission migrates: source, `install.sh`,
`launch.sh` and optional `convert_weight.sh`. Fresh verification runs conversion
before installation, then protects the results before judge inspection. Both
scripts receive `CHECKPOINT=/models/source`, `WEIGHTS=/workspace/weights` and
`RUNTIME=/workspace/runtime`. Conversion must provide prerequisites unavailable
in the image because install.sh has not run yet. Candidate weights are regenerated
there and referenced by the candidate's launch.sh; no coding checkpoint is copied.

Every launcher starts a foreground OpenAI-compatible server at
**`http://127.0.0.1:8000`**, with model alias **`served`**. No port is allocated or
injected. A complete candidate script contains its final weight path, installed
environment and optimized parameters and runs unchanged. An unfilled baseline
input uses `$CHECKPOINT`, `${CHECKPOINT}` or a required form such as
`${CHECKPOINT:?checkpoint required}`. Launch supplies mapped weights only then.
A filled local assignment or a nonempty fallback such as
`${CHECKPOINT:-/workspace/weights/optimized}` counts as complete.

`Launch(script, checkpoint=None, args=(), env={})` prepares absolute `/bin/bash`
argv and optional checkpoint environment. It does not rewrite the script or
require a template header, Template class or generated Dockerfile. External
args/env cannot override model/host/port/served-name identity, CHECKPOINT, PORT,
BASH_ENV or ENV. Trusted backend callers can extend protected aliases/names.
Remaining tuning is unrestricted; script defaults apply when arguments are empty.

The baseline folder identifies engine/category. Its checkpoint is resolved from
the selected model/workload task and existing baseline map. A copied template
uses the public `/workspace/reference.json` selection containing category,
baseline and checkpoint. The trusted reference finalizer publishes
that mirror with its protected selection; private questions/golds stay private.
A complete candidate requires no candidate-weight mapping or source fallback.

Server checks port/alias and live foreground ownership before replay. It rejects
an occupied endpoint, wrong port/alias, early exit or readiness timeout and
cleans up on success, failure and cancellation. Diagnostic API attachment is
available; the public CLI always owns its launcher.

## Replay and runtime order

The CLI uses three inputs plus optional tuning, mode and concurrency:

```text
python -m eval.bench.run MODEL WORKLOAD LAUNCH_SCRIPT
```

Timing loads cwd/data/<task-pool>/rows.jsonl, sends one excluded warmup and
measures replay. Quality loads cwd/data/S/rows.jsonl even for A/MA tasks, sends no
warmup, runs the row's accuracy checker and reports accuracy only. Both share
scheduling: positive concurrency limits independent rows/sessions; `-1` admits
all rows, preserving parent dependencies and applicable tool delays. Quality
defaults to `-1`. Reports and server logs go under bench/<uuid>.

Startup only marks readiness and keeps a container alive. Three native Harbor
jobs execute search, coding and verification in separate containers from one
image. Search runs its configured adapter as search, then root publishes the
selected reference. Coding runs its adapter as agent, then a root format/collection
gate exposes only submission source. Only settings move from search to coding;
only source moves from coding to verification. Required weights are shared by
read-only binds, not copied out of a development container.

Verification uploads submission into its prepare workdir and uses native nop:

```text
prepare/root: checks → optional conversion → install → prepare artifact access
review/judge: native Rewardkit inspection
measure/root: paired S quality → fresh timed replay → reports/rewards
```

Prepare/review gates require reward 1. All three verifiers use shared mode in
this fresh verification container; native TOML user fields select root/judge/root.
Root retains write access to completed artifacts; judge only writes its private
home/cache/output. The wrapper takes final rewards from verification and coding
usage from the coding job.

Native oracle supplies one `solution/solve.sh` in each search/coding task. The
known baseline submission follows all ordinary verification gates. There is no
local agent coordinator or oracle bypass. Host/external adapters continue to use
Harbor sandbox tools; installed CLIs act in their task's respective container.

Opt-in `tests/test_oracle.py` reuses an existing production device image and the
actual Claude Code with native credentials in the fresh verification task. Its
native root/judge/root steps check submission transfer, preparation, protection and
real rules review; the trusted final test step records acceptance without inference. It
builds no test image and reports no model-performance result. Full GPU acceptance
still needs supplied category assets and real model weight shards.
