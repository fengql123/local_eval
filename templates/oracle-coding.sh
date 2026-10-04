#!/usr/bin/env bash
# Harbor executes this known submission under the ordinary coding user.
set -euo pipefail
export HOME=/home/agent USER=agent LOGNAME=agent
cd /workspace

/opt/eval/.venv/bin/python -I - <<'PY'
import json
import shlex
from pathlib import Path

from eval.launch import Launch

reference = json.loads(Path('reference.json').read_text())
script = Path('/opt/eval/images') / reference['category'] / 'baselines' / reference['baseline'] / 'launch.sh'
launch = Launch(script, reference['checkpoint'], tuple(reference.get('args', ())), dict(reference.get('env', {})))
launch.prepare()

submission = Path('submission')
submission.mkdir(exist_ok=True)
exports = '\n'.join(f'export {name}={shlex.quote(value)}' for name, value in launch.env.items())
command = shlex.join(['/bin/bash', str(script), *launch.args])
(submission / 'launch.sh').write_text(
    '#!/usr/bin/env bash\nset -euo pipefail\n'
    ': "${CHECKPOINT:?mapped reference checkpoint is required}"\n'
    f'{exports}\nexec {command} "$@"\n'
)
(submission / 'install.sh').write_text(
    '#!/usr/bin/env bash\nset -euo pipefail\n'
    '# This known solution reuses the baseline already installed in the category image.\n'
)
(submission / 'reference.json').write_text(json.dumps(reference, indent=2) + '\n')
(submission / 'README.md').write_text(
    'Known baseline solution for the normal verification pipeline.\n\n'
    f'Runtime: {script}\n'
    'The category image supplies its dependencies. The verifier mounts the selected\n'
    'baseline weights read-only and supplies the open CHECKPOINT input.\n'
    'No development weights, environments or caches are submitted.\n'
)
for name in ('install.sh', 'launch.sh'):
    (submission / name).chmod(0o755)
PY
