#!/usr/bin/env bash
# DeepSeek-V4-Flash.md prerequisites at 0dce4c9; use matched FlashInfer packages.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
git init -q src
git -C src remote add origin https://github.com/kvcache-ai/ktransformers.git
git -C src fetch -q --depth 1 origin 0dce4c9b097710951d1b3e232f50b973ff0a5505
git -C src checkout -q FETCH_HEAD
git -C src submodule update --init --recursive
export CUDA_HOME=/usr/local/cuda PIP_NO_CACHE_DIR=1
export PATH=$CUDA_HOME/bin:$PATH CPUINFER_CUDA_ARCHS=native
uv venv --seed --python 3.12 .venv
source .venv/bin/activate
# System prerequisites come from the device image; build subcommands need no root.
(
    cd src
    ./install.sh sglang
    ./install.sh kt-kernel
)
pip install 'flashinfer-python==0.6.15.post1'
pip install --index-url https://flashinfer.ai/whl --no-deps 'flashinfer-cubin==0.6.15.post1'
pip install 'transformers==4.57.1' tilelang 'apache-tvm-ffi<0.1.12'
python -c 'import kt_kernel, flashinfer, tilelang, sglang.launch_server'
