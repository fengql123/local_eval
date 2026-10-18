#!/usr/bin/env bash
# CUDA build from llama.cpp docs/build.md at 7fe450e; no driver is needed to build.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
git init -q src
git -C src remote add origin https://github.com/ggml-org/llama.cpp.git
git -C src fetch -q --depth 1 origin 7fe450e19305b828c199d602c23a8337aaa1f03b
git -C src checkout -q FETCH_HEAD
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
cmake -S src -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native \
    -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined
cmake --build build --config Release --parallel 8
test -x build/bin/llama-server
