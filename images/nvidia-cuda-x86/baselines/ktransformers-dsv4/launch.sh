#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731 documented defaults; args/env may tune execution.
set -euo pipefail
: "${CHECKPOINT:?model directory required}"
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_HOME=/usr/local/cuda VIRTUAL_ENV=$PWD/.venv
export PATH=$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH
export SGLANG_DSV4_MODE=${SGLANG_DSV4_MODE:-2604} SGLANG_DSV4_2604_SUBMODE=${SGLANG_DSV4_2604_SUBMODE:-2604B}
exec python -m sglang.launch_server --model "$CHECKPOINT" --kt-weight-path "$CHECKPOINT" \
    --host 127.0.0.1 --port 8000 --served-model-name served --kt-method MXFP4 "$@"
