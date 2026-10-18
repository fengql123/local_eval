#!/usr/bin/env bash
set -euo pipefail

exec /opt/eval/.venv/bin/python -I -m eval.verify.run --stage handoff --task /var/lib/local-eval/task.json
