#!/usr/bin/env bash
# Trusted task startup, executed as root after Harbor creates the sandbox.
set -euo pipefail
runtime=${1:?runtime manifest is required}
export PATH=/opt/python/bin:/usr/local/go/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# Native Harbor package setup runs as root; /tmp stays closed to workload users.
printf 'APT::Sandbox::User "root";\n' > /etc/apt/apt.conf.d/99-local-eval
export UV_PROJECT_ENVIRONMENT=/opt/python
uv sync --locked --no-dev --no-install-project --python 3.12 --project /opt/eval
/opt/python/bin/python -I - <<'PY'
import sysconfig
from pathlib import Path
Path(sysconfig.get_path('purelib'), 'local_eval.pth').write_text('/opt/eval\n')
link = Path('/opt/eval/.venv')
link.unlink(missing_ok=True)
link.symlink_to('/opt/python', target_is_directory=True)
PY

/opt/python/bin/python -I - "$runtime" <<'PY'
import json
import sys
from pathlib import Path
from eval.providers.checkpoint import restore_session_paths
from eval.verify.access import prepare_workspace, project_writable, readonly, seal_submission, writable
from eval.verify.machine import check_cuda, machine
runtime = json.loads(Path(sys.argv[1]).read_text())
check_cuda(runtime)
task = json.loads(Path('/var/lib/local-eval/task.json').read_text())
observed = machine(task)
if runtime.get('expected_hardware') and observed['fingerprint'] != runtime['expected_hardware']:
    raise RuntimeError('replacement hardware differs from the saved phase; retry allocation')
Path('/root/local-eval-hardware.json').write_text(json.dumps(observed))
prepare_workspace({{ phase | tojson }})
actor = {{ {'search': 'search', 'coding': 'agent', 'verification': 'verifier'}[phase] | tojson }}
for item in runtime['files']:
    path = Path(item['target'])
    if item.get('read_only', True):
        readonly(path, private=path.is_relative_to('/var/lib/local-eval'),
                 account={{ ('root' if phase == 'verification' else ('search' if phase == 'search' else 'agent')) | tojson }})
    elif path == Path('/workspace'):
        project_writable(path, item.get('owner', actor))
    else:
        writable(path, item.get('owner', actor))
{% if phase == 'verification' %}
# Preparation results describe prior work, not an installed fresh environment.
for name in ('prepare.json', 'install.json'):
    (Path('/var/lib/local-eval/verification') / name).unlink(missing_ok=True)
{% endif %}
if runtime.get('resume'):
    restore_session_paths(runtime['resume'], actor)
readonly(Path('/opt/eval'))
readonly(Path('/opt/python'))
PY

{% if phase == 'verification' %}
chown root:judge /logs/verifier
chmod 0770 /logs/verifier
{% else %}
chown root:root /logs/verifier
chmod 0700 /logs/verifier
{% endif %}
source=$(jq -r '.target.source // empty' "$runtime")
if [[ -n "$source" ]]; then
    if [[ ! -f /var/lib/local-eval/target.json ]]; then
        /opt/python/bin/python -I -m eval.models "$source" /models/target \
            --tokenizer /models/tokenizer --output /var/lib/local-eval/target.json
    fi
    /opt/python/bin/python -I - <<'PY'
from pathlib import Path
from eval.verify.access import readonly
for path in ('/models/target', '/models/tokenizer', '/var/lib/local-eval/target.json'):
    readonly(Path(path))
PY
fi
ready={% if phase == "search" %}/workspace/inputs-ready.json{% else %}/workspace/.harbor/inputs-ready.json{% endif %}
printf '{"phase":"{{ phase }}","ready":true}\n' > "$ready"
chmod 0444 "$ready"
touch /run/eval-ready
