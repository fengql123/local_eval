#!/usr/bin/env bash
# GLM-5.3-Flash's documented FP8 default; later arguments may tune it.
set -euo pipefail
: "${CHECKPOINT:?model directory required}"
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_HOME=/usr/local/cuda VIRTUAL_ENV=$PWD/.venv
export PATH=$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH
exec python -m sglang.launch_server --model-path "$CHECKPOINT" --kt-weight-path "$CHECKPOINT" \
    --host 127.0.0.1 --port 8000 --served-model-name served --kt-method FP8 "$@"
