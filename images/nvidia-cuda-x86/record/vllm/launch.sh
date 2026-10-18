#!/usr/bin/env bash
# CHECKPOINT is the source model directory; extra args configure vLLM serving.
set -euo pipefail
: "${CHECKPOINT:?source model directory required}"
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_HOME=/usr/local/cuda-13.0 VIRTUAL_ENV=$PWD/.venv
export PATH=$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH
# Torch CPU operations should not oversubscribe a sandbox's CPU quota.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
exec vllm serve "$CHECKPOINT" --host 127.0.0.1 --port 8000 --served-model-name served "$@"
