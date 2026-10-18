#!/usr/bin/env bash
# FreeToken source installation, docs/install.md at 74a3877 (DeepSeek V4.1 support).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
git init -q src
git -C src remote add origin https://github.com/FlashML-org/FreeToken.git
git -C src fetch -q --depth 1 origin 74a387777708b9650a027494bcb37d2ab9cf2443
git -C src checkout -q FETCH_HEAD
git -C src apply ../sequential-read.patch
git -C src apply ../token-evidence.patch
git -C src apply ../raw-tool-arguments.patch
git -C src apply ../private-ipc.patch
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e 'src[accel]'
test -x .venv/bin/ft
