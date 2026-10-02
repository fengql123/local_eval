#!/usr/bin/env bash
# Install the host harness into .venv; Harbor owns device image builds.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec uv sync --locked --extra host
