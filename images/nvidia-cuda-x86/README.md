# NVIDIA CUDA x86

The common toolchain uses CUDA 13.0.2 development packages on Ubuntu 24.04,
Python 3.12, uv, Go and general shell/build tools. One CUDA toolkit is supplied.
Phase tails add the required search, coding or verification identities. No
baseline, evaluator dependencies, agent CLI or recorder is installed during builds.
Their default `USER root` serves trusted startup; Harbor's native task TOML runs
agents and submitted scripts under their declared nonroot identities.

Harbor builds the task-owned common-plus-phase Dockerfile from source at startup. Baseline
installation happens inside an allocated GPU sandbox, so native builds detect
its visible devices. The evaluator does not probe GPUs or inject architecture
values into the Dockerfile.

## Runtime baseline recipes

| Baseline | Checkpoint input | Installation |
| --- | --- | --- |
| FreeToken | Mapped Hugging Face model directory, including DeepSeek-V4.1-Flash. | Shared source pin `74a387777708b9650a027494bcb37d2ab9cf2443`, isolated Python venv and upstream CUDA13 dependencies. |
| llama.cpp | Mapped GGUF file/directory. | Pinned source, native CUDA build using the visible GPU architecture. |
| Ollama | Mapped native tag or local GGUF directory. | Pinned fork and native CUDA13 build. |
| KTransformers | Original GLM-5.3-Flash directory. | Pinned source and isolated Python venv using the category toolkit. |
| KTransformers DSV4 | Original DeepSeek-V4-Flash-0731 directory. | Pinned source and isolated Python venv using the category toolkit. |

FreeToken uses one install/launch recipe for all mapped models. The DeepSeek V4.1
entry selects `deepseek-ai/DeepSeek-V4.1-Flash`; keep the complete checkpoint
layout, including `inference/`. Its Engram tables stream from local checkpoint
shards, so place the checkpoint on sandbox disk. No model-specific launch branch
or additional download is needed in the launcher. See the pinned
[upstream model documentation](https://github.com/FlashML-org/FreeToken/blob/74a387777708b9650a027494bcb37d2ab9cf2443/docs/models.md).
The output-length fix is included upstream; only the sequential-read patch remains.

Every launcher serves **127.0.0.1:8000**, model alias **served**, in the foreground.
Extra arguments follow the fixed defaults. `Launch` rejects overrides of suite
identity/checkpoint inputs. Candidate installations produce `bin/server.launch`,
which receives only `/models/target`; their tuning lives inside submitted code.

GGUF launch chooses the first shard of one mapped quantization and excludes
projection files. Ambiguous quantizations fail rather than selecting arbitrary
weights. Ollama creates a temporary `served` manifest and links existing blobs;
it never edits the checkpoint or its protected installation. For mapped GGUF input,
it registers read-only links to the shards and uses the pinned fork's native
`/api/create` split-GGUF importer. No merged file, second weight copy, or cloud
inference is used. Native model-store inputs continue to use their saved manifest.

MiniMax-M3 maps FreeToken to `nvidia/MiniMax-M3-NVFP4`, and both llama.cpp and
Ollama to `AesSedai/MiniMax-M3-GGUF` (Q4_K_M). Search compares all three within
the same allocation. Coding and verification use the canonical checkpoint from
`models.csv`, independently of the selected reference. Approved baseline formats
share the model-level quality bound.

The two KTransformers recipes use the single category CUDA toolkit; engine/GPU
compatibility still requires runtime acceptance. Their presence in the repository
does not establish compatibility with every model or device.
They invoke upstream's `sglang` and `kt-kernel` build subcommands without its root
system-package installation. CMake's `native` target selects visible GPUs during
installation; no numeric architecture is supplied by the coordinator.

## Host recording

`record/vllm` is a host-only recipe. It is never copied into phase images or run
by a Harbor job. Recording uses the shared server, proxy and live harness runner.

## Validation

Launcher tests exercise fixed endpoint/identity, adjustable options and GGUF
selection. Runtime installation tests exercise isolated workspace copies, reuse,
interrupted installs and failure propagation. These lightweight checks do not
establish full engine inference, performance or provider image-size acceptance.
