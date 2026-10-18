# Device assets

```text
images/<device-category>/
  Dockerfile                  # Generic toolchain only.
  search.Dockerfile           # Search account and workspace.
  coding.Dockerfile           # Coding account and workspace.
  verification.Dockerfile     # Root controller, runtime users and judge.
  baselines/<engine>/
    install.sh                # Installs this engine into an isolated workspace.
    launch.sh                 # Foreground server; explicit CHECKPOINT, port8000.
    *.patch                   # Required upstream fixes, when present.
  record/vllm/
    install.sh                # Recording runtime recipe, separate from eval.
    launch.sh
```

Task generation combines the generic Dockerfile with the phase tail into each
task's `environment/Dockerfile`. Harbor builds or reuses it at the selected provider
when that task starts. Images contain CUDA/development tools, Python and phase accounts. They do not install baselines, agents, recording vLLM or weights.
The coordinator never uploads a locally built image to a cloud registry.

Evaluator code/tables and baseline recipes are frozen into a small runtime payload
and uploaded before the trusted task `setup.sh` runs. Recipes appear at
`/opt/eval/baselines/<category>/<engine>`. Search decides which to install in its
writable `/workspace/runtime/search/baselines/<category>/<engine>` workspace. Verification installs
only the chosen reference under `/workspace/runtime/reference/baselines` and deletes that
phase runtime after reference measurements.

Baseline scripts own repository/commit/dependency choices. Launch receives the
acquired checkpoint explicitly and accepts adjustable parameters. Fixed port,
bind/model-alias arguments are declared by the trusted Python launch contract.
The script must keep its foreground process alive, so the shared Server can stop
it and its children reliably. Candidate installation creates `/workspace/server.launch`,
which accepts the suite's fixed checkpoint argument; tuning lives in submitted code.

NVIDIA CUDA x86 uses one CUDA13 toolchain. Installers detect/build for their target
hardware; setup does not inject CUDA architecture lists. Recording recipes may run
on the host or a separately provisioned recording sandbox; they are never part of
Harbor evaluation images or verification jobs.

Unix users and protected parent directories are declared in phase Dockerfiles.
Trusted task `setup.sh` assigns permissions to uploaded files and acquired targets.
It runs as root after container creation and before the restricted agent. After
candidate installation, verification protects the resulting review files through
`eval/verify/access.py`. Workspace/conversation recovery remains separate from setup;
no shared volume or FUSE view is required.
