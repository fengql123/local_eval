#!/usr/bin/env bash
# CHECKPOINT is the complete mapped model directory, including its inference/ files.
# FreeToken detects the architecture (including DeepSeek V4.1) from that checkpoint.
# Additional arguments are ft serve options.
set -euo pipefail
: "${CHECKPOINT:?model directory required}"
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_HOME=/usr/local/cuda-13.0 VIRTUAL_ENV=$PWD/.venv
export PATH=$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH
exec "$VIRTUAL_ENV/bin/ft" serve --model "$CHECKPOINT" --host 127.0.0.1 --port 8000 --served-model-name served "$@"
