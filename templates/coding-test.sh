#!/usr/bin/env bash
set -euo pipefail

exec /opt/eval/.venv/bin/python -I -m eval.verify.run --task /root/eval/task.json
