# Device profiles

All profiles use NVIDIA CUDA x86 and Daytona spot GPUs. D1, D2, W1 and W2
require the AMD EPYC Zen 4 comparison group; W3 retains AMD EPYC Zen 5.
The CPU column lists allowed models. Exact assigned CPU/GPU, PCIe and resource
observations remain in each task result. A vCPU quota is an allocation, not a
claim about a physical desktop's memory bandwidth.

| Profile | GPUs | VRAM per GPU | CPU models allowed | vCPUs | RAM | Disk | Configuration |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D1 | 1× RTX 4090 | 24 GB | any EPYC Genoa (9004, Zen 4) | 8 | 64 GiB | 512 GiB | [daytona.d1.yaml](../examples/daytona.d1.yaml) |
| D2 | 1× RTX 5090 | 32 GB | any EPYC Genoa (9004, Zen 4) | 8 | 128 GiB | 512 GiB | [daytona.d2.yaml](../examples/daytona.d2.yaml) |
| W1 | 2× RTX 4090 | 24 GB | any EPYC Genoa (9004, Zen 4) | 12 | 192 GiB | 1 TiB | [daytona.w1.yaml](../examples/daytona.w1.yaml) |
| W2 | 2× RTX 5090 | 32 GB | any EPYC Genoa (9004, Zen 4) | 16 | 256 GiB | 1 TiB | [daytona.w2.yaml](../examples/daytona.w2.yaml) |
| W3 | 1× RTX PRO 6000 | 96 GB | EPYC 9335 / 9355 / 9455 / 9555 / 9B45 (Zen 5) | 16 | 128 GiB | 512 GiB | [run.codex.yaml](../run.codex.yaml) |

W4 has been removed. Each new D1–W2 run selects all 35 model/workload tasks
and allows 35 concurrent task pipelines, up to 140 across the four profiles.
Each profile searches independently. Search and verification use Codex
GPT-6.1-sol at medium effort; coding uses Codex GPT-6.1-sol at extra-high effort.
Sandboxes build from the task's Dockerfile on Daytona. Source-built snapshots
cache these builds; no prebuilt image is uploaded.

Start a profile from the repository root after exporting native credentials:

```sh
python -m eval.run examples/daytona.d1.yaml
```

Resume with the UUID run folder printed by that command:

```sh
python -m eval.run --resume runs/<run-id>
```

The declared allocation applies to all three phases. Mismatched CPU groups are
rejected before model downloads. Only declared spot GPUs are accepted;
unavailable matching capacity waits rather than switching GPU or pricing mode.

## Current runs

| Profile | Run folder | tmux session |
| --- | --- | --- |
| D1 | [3e694eff-a67f-4510-9499-59a56aa7d741](../runs/3e694eff-a67f-4510-9499-59a56aa7d741) | `local-eval-d1-codex-3e694eff` |
| D2 | [9a49c45c-0de9-4506-98f0-b359f2d9b11d](../runs/9a49c45c-0de9-4506-98f0-b359f2d9b11d) | `local-eval-d2-codex-9a49c45c` |
| W1 | [191287b8-cba1-4cb0-98e6-124e2e30e637](../runs/191287b8-cba1-4cb0-98e6-124e2e30e637) | `local-eval-w1-codex-191287b8` |
| W2 | [11487707-53b1-426c-ad8f-107bce663669](../runs/11487707-53b1-426c-ad8f-107bce663669) | `local-eval-w2-codex-11487707` |
| W3 Codex | [cbfbefb5-fde6-4d6e-beae-434b24fbb9b0](../runs/cbfbefb5-fde6-4d6e-beae-434b24fbb9b0) | `local-eval-codex-cbfbefb5` |
| W3 Opus | [19136f8c-323c-4f97-9503-31b2200965e9](../runs/19136f8c-323c-4f97-9503-31b2200965e9) | `local-eval-opus-19136f8c` |
