#!/usr/bin/env bash
# Suite Ollama fork: docs/development.md native CUDA build at fec0f14.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
git init -q src
git -C src remote add origin https://github.com/fengql123/ollama.git
git -C src fetch -q --depth 1 origin fec0f14bc2c6eca8d604018642613e516f84209e
git -C src checkout -q FETCH_HEAD
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/go/bin:$CUDA_HOME/bin:$PATH
cd src
# Ollama adds :latest to list IDs; keep the suite's OpenAI alias exactly served.
sed -i '/data = append(data, Model{/i\        if id == "served:latest" { id = "served" }' openai/openai.go
gofmt -w openai/openai.go
cmake -B build . -DOLLAMA_LLAMA_BACKENDS=cuda_v13 -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --parallel 8
test -x ollama
test -f build/lib/ollama/cuda_v13/libggml-cuda.so
