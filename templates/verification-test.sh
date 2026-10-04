#!/usr/bin/env bash
set -euo pipefail

exec /opt/eval/.venv/bin/python -I -m eval.verify.run --stage {{ stage }}{% if stage != "review" %} --task /root/eval/task.json{% endif %}
