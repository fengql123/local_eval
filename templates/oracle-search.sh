#!/usr/bin/env bash
# Known reference search for Harbor's native OracleAgent; no coding model.
set -euo pipefail
export HOME=/home/search USER=search LOGNAME=search
cd /home/search

/opt/eval/.venv/bin/python -I -m eval.bench.run {{ task.model | tojson }} S-N \
  /opt/eval/images/{{ image.category }}/baselines/{{ assets.baselines | first }}/launch.sh

/opt/eval/.venv/bin/python -I - <<'PY'
import json
from pathlib import Path

report = max(Path('bench').glob('*/report.json'), key=lambda path: path.stat().st_mtime_ns)
result = json.loads(report.read_text())
params = result['parameters']
Path('selection.json').write_text(json.dumps({
    'baseline': result['baseline'],
    'args': params.get('args', []),
    'env': params.get('env', {}),
}, indent=2) + '\n')
PY
