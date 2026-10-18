#!/usr/bin/env bash
set -euo pipefail

exec /opt/eval/.venv/bin/python -I -m eval.search.run --task /var/lib/local-eval/task.json --stage gate
