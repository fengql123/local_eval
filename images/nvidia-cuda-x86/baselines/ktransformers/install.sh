#!/usr/bin/env bash
# kt-kernel README: source install + one-click SGLang integration at 5632f71.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
git init -q src
git -C src remote add origin https://github.com/kvcache-ai/ktransformers.git
git -C src fetch -q --depth 1 origin 5632f7197e7a9898f0723755c2fc187a946a85a6
git -C src checkout -q FETCH_HEAD
git -C src submodule update --init --recursive
export CUDA_HOME=/usr/local/cuda PIP_NO_CACHE_DIR=1 UV_PYTHON_INSTALL_DIR=$PWD/.python
export PATH=$CUDA_HOME/bin:$PATH CPUINFER_CUDA_ARCHS=native
uv venv --seed --python 3.11 .venv
source .venv/bin/activate
# System prerequisites come from the device image; build subcommands need no root.
(
    cd src
    ./install.sh sglang
    ./install.sh kt-kernel
)
# Upstream dependencies install transformers over transformers-kt's module.
version=$(sed -n 's/.*"transformers-kt==\([^"]*\)".*/\1/p' src/third_party/sglang/python/pyproject.toml)
pip uninstall -y transformers
pip install --force-reinstall --no-deps "transformers-kt==$version"
python -c 'import transformers, sglang, kt_kernel'
