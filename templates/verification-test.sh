#!/usr/bin/env bash
set -euo pipefail

# Candidate outputs and the review verdict must be readable by the next user.
umask 0022
exec /opt/eval/.venv/bin/python -I -m eval.verify.run --stage {{ stage }}{% if stage != "review" %} --task /var/lib/local-eval/task.json{% endif %}
