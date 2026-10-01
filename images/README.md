# Device image sources

Each device category owns one supplied Dockerfile and its baseline installers.
Harbor builds that image and shares the task sandbox across reference search,
coding-agent execution, and evaluation.

| Category | Source Dockerfile |
| --- | --- |
| NVIDIA CUDA x86 | `nvidia-cuda-x86/Dockerfile` |
| NVIDIA CUDA ARM with unified memory | `nvidia-cuda-arm-unified/Dockerfile` |
| AMD ROCm x86 | `amd-rocm-x86/Dockerfile` |
| AMD APU with unified memory | `amd-apu-unified/Dockerfile` |
| Apple Silicon | `apple-silicon/Dockerfile` |

For each category, baseline scripts live at
`<category>/baselines/<engine>/install.sh`. The catalog supplies repository,
source ref, and install prefix; the supplied script owns checkout and installation.
Python engines install into separate venvs and native engines into separate
build prefixes. Exact script invocation/build arguments will be integrated with
the supplied assets in Phase 2.

The category Dockerfiles and scripts are pending owner supply; Phase 1 includes
only their catalog paths and this contract. Listed categories are source locations,
not a claim of validated accelerator support. Actual host inspection currently
supports Linux NVIDIA x86. Image builds and engine acceptance occur in later
phases. No generated Dockerfile, stub installer, or per-baseline image is shipped.
