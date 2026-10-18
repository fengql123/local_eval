#!/usr/bin/env bash
# Shared trusted vLLM runtime for live capture and offline quality profiling.
# https://docs.vllm.ai/en/latest/getting_started/installation/gpu/#install-the-latest-code
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --upgrade vllm \
    --extra-index-url https://wheels.vllm.ai/nightly/cu130 --torch-backend=cu130
.venv/bin/python -c 'import vllm; print(vllm.__version__)'
