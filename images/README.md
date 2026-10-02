# Device image sources

Each device category owns one supplied Dockerfile and its baseline installers.
Harbor builds the image and shares one task sandbox across reference search,
coding-agent execution, and evaluation. Search acts as `search`, coding and
candidates as `agent`, and trusted verification as root. Harbor adapters control
agent execution; an external agent loop can stay on the host.

| Category | Source Dockerfile |
| --- | --- |
| NVIDIA CUDA x86 | `nvidia-cuda-x86/Dockerfile` |
| NVIDIA CUDA ARM with unified memory | `nvidia-cuda-arm-unified/Dockerfile` |
| AMD ROCm x86 | `amd-rocm-x86/Dockerfile` |
| AMD APU with unified memory | `amd-apu-unified/Dockerfile` |
| Apple Silicon | `apple-silicon/Dockerfile` |

Each baseline installer lives at
`<category>/baselines/<engine>/install.sh`. The script contains its own repository,
source ref, installation prefix, and installation commands. Python engines use
independent venvs; native engines use independent build prefixes. Those details
belong to the script, not a duplicate CSV recipe or generated shell command.

The Dockerfile receives `ARG BASELINES`: selected engine names separated by
spaces. It invokes their installers at the fixed paths above. The argument
convention belongs to the supplied Dockerfile and scripts.

Category Dockerfiles and installers await owner supply. Setup reads the selected
assets before acquiring any dataset or checkpoint. Missing files raise their
normal file error. Listed categories specify source locations; host inspection
currently supports Linux NVIDIA x86. The Apple Silicon LM Studio mappings are
future entries without a request policy. Image builds and accelerator acceptance
remain later work. No placeholder production image or installer is provided.

## Dockerfile contract

The device image contains the tools needed to act in the sandbox and run the
complete serving/evaluation pipeline. Keep shared dependencies in this image;
Harbor's selected adapter installs an agent-specific CLI during agent setup.
The Dockerfile does not bake in a fixed Claude Code/Codex harness, host-only provider integrations,
credentials, dataset acquisition tools, checkpoints, or the host Harbor runner.

| In the shared device image | On the host or installed by its owner |
| --- | --- |
| Python 3.12+, bash, sleep, git, curl and certificates; build tools needed by the selected engines and candidate | Harbor orchestration and host/external agent loops |
| Device runtime/toolchain and selected baseline venvs/native prefixes | HF dataset/source-checkpoint acquisition and task generation |
| Protected evaluator installation and its runtime dependencies | Selected agent CLI, installed by its Harbor adapter during setup |
| Search/agent users, writable role workspaces and protected evaluator paths | Provider credentials, supplied only to the selected adapter/role |

Harbor's build context is the repository root. `.dockerignore` excludes run data,
credentials, downloaded checkpoints, and development outputs. The supplied image
must:

- Consume `ARG BASELINES` and invoke the selected category `install.sh` scripts.
  Each script owns its repository/ref and isolated venv/native installation.
- Install this project non-editably into the root-owned `/opt/eval/.venv`, without
  its `host` extra or development group. With the copied project and lockfile,
  `uv sync --locked --no-dev --no-editable` installs the runtime dependency set.
  Today it contains PyYAML; HTTP, serving and any judge-model dependencies are
  added when those modules are implemented in Phases 3 and 4.
- Create separate `search` and `agent` users. Provide a writable `/home/search`
  owned by `search` with mode `0700`, and `/workspace` owned by `agent`.
  Keep `/root` and evaluator/baseline installations inaccessible for modification
  by either agent. Neither user needs Docker access or unrestricted sudo.
- Provide mount destinations `/opt/eval/start.sh`, `/workspace/run.md`,
  `/workspace/data/public`, `/home/search/data/calibration`, `/root/eval`, and
  `/models`. Search calibration and public data are mounted read-only.
- Keep `/opt/eval/.venv/bin/python` and its installed evaluator root-owned.
  Verification hooks use that absolute interpreter with `-I`, so an agent's
  `/workspace/eval` cannot replace the trusted evaluator through Python imports.

The project's `host` extra installs Harbor, datasets, the HF CLI and Jinja on the
host. `./install.sh` selects this extra. Development also selects it for the
Phase 2 schema/runner tests. Runtime images do not select either.

The host downloads the original checkpoint for the read-only `/models/source`
mount. A baseline with the same checkpoint reuses that mount; distinct baseline
checkpoints remain model identifiers in the task metadata and `/workspace/run.md`.
Their engines load or acquire them inside the sandbox using writable role
storage. The per-category `install.sh` installs engine software, not run-specific
weights in the image. Quantization and other derived outputs also belong in the
role's writable workspace.

## Runtime order and adapter responsibilities

`start.sh` marks the environment ready and keeps the container alive. It does not
run reference search. Generated `task.toml` declares `search` then `coding`, with
separate users and budgets; search's root verifier gate must pass before coding.
The coding step's shared root verifier runs `pre_run` and the actual benchmark.

Phase 2 supplies this task/job contract. Phase 5 implements the small host
`eval.agents:PhasedAgent` coordinator that delegates both existing Harbor
adapters and the search reference finalization. External adapters execute
sandbox tools through Harbor's environment API; installed adapters run their CLI
inside this same container. The coordinator must scope each child's environment,
HOME, workspace and log/context collection. Changing Docker's exec user alone
is not enough to change its inherited HOME or working directory.

Search-agent billing is not recorded. The coordinator preserves search operation
and measurement logs but leaves search usage/cost fields unset and exports no
search billing trajectory or report. Coding-step evidence alone feeds `costs.py`.

Private benchmark rows, the quality subset and `task.json` remain beneath
`/root/eval`; search receives only its calibration view and mapped checkpoints.
The authoritative reference is finalized by trusted root operations before the
permitted handoff reaches coding. The current root gate and final verifier hooks
refer to later-phase modules; no production image or GPU execution is claimed.
