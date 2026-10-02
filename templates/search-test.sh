#!/usr/bin/env bash
set -euo pipefail

exec /opt/eval/.venv/bin/python -I -m eval.search.run --task /root/eval/task.json --finalize-reference
