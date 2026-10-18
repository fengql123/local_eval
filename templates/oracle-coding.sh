#!/usr/bin/env bash
# Known reproducible project executed by Harbor's native OracleAgent.
set -euo pipefail
umask 0022
cd /workspace/output/submission

/opt/eval/.venv/bin/python -I - <<'PY'
import json
import shlex
import shutil
from pathlib import Path

reference = {'baseline': {{ (oracle_baseline | tojson) if oracle_baseline else 'None' }}, 'args': [], 'env': {}}
{% if task.type != "SLO" %}
selected = json.loads(Path('/inputs/reference.json').read_text())
if selected['baseline'] == reference['baseline']:
    reference = selected
{% endif %}
baseline = reference['baseline']
if baseline is None:
    Path('install.sh').write_text('#!/bin/bash\necho "no known implementation for this device" >&2\nexit 1\n')
    raise SystemExit(0)
shutil.copytree(Path('/opt/eval/baselines/{{ image.category }}') / baseline, 'baseline', dirs_exist_ok=True)
exports = '\n'.join(f'export {name}={shlex.quote(value)}' for name, value in reference['env'].items())
command = shlex.join(['/bin/bash', '/workspace/runtime/candidate/baseline/launch.sh', *reference['args']])
Path('launcher.sh').write_text(
    '#!/bin/bash\nset -euo pipefail\nexport CHECKPOINT="$1"\n'
    + f'{exports}\nexec {command}\n'
)
Path('install.sh').write_text(
    '#!/bin/bash\nset -euo pipefail\n'
    'source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)\n'
    'runtime=/workspace/runtime/candidate\nmkdir -p "$runtime/baseline"\n'
    'cp -a "$source_dir/baseline/." "$runtime/baseline/"\n'
    'cp "$source_dir/launcher.sh" /workspace/server.launch\n'
    'chmod +x /workspace/server.launch\n'
    'exec bash "$runtime/baseline/install.sh"\n'
)
Path('install.sh').chmod(0o755)
Path('README.md').write_text('Known baseline project, reproduced and checked through normal verification.\n')
PY
