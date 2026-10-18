#!/usr/bin/env bash
# Known reference search executed by Harbor's native OracleAgent. It selects the
# first mapped baseline that serves the private rows with default parameters;
# when none can, it writes the unsupported verdict that makes this an SLO task.
set -euo pipefail
umask 0022
cd /workspace

exec /opt/eval/.venv/bin/python -I - <<'PY'
import json
import os
import subprocess
from pathlib import Path

python = '/opt/eval/.venv/bin/python'
home = Path('/workspace/output')
logs = home / 'logs'
logs.mkdir(parents=True, exist_ok=True)
baselines = {{ assets.baselines | tojson }}
attempts = []


def run(stage, baseline, command, **options):
    log = logs / f'{baseline}-{stage}.log'
    started = set(Path('/workspace/output/experiments').glob('*'))
    with log.open('wb') as output:
        code = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, **options).returncode
    if code:
        lines = [line for line in log.read_text(errors='replace').splitlines() if line.strip()]
        if stage == 'serve' and not any(
            (folder / 'failure.json').is_file() for folder in set(Path('/workspace/output/experiments').glob('*')) - started
        ):
            stage = 'launch'
        attempts.append({
            'baseline': baseline,
            'stage': stage,
            'error': lines[-1][-500:] if lines else f'exited {code}',
            'log': str(log),
        })
    return code == 0


selection = None
for baseline, source in baselines.items():
    asset = home / f'{baseline}-asset.json'
    if not run('install', baseline, [python, '-I', '-m', 'eval.baselines', '--baseline', baseline]):
        continue
    weights = str(Path('/workspace/runtime/search/weights') / baseline)
    if not run('acquire', baseline, [python, '-I', '-m', 'eval.models', source, weights, '--output', str(asset)]):
        continue
    checkpoint = json.loads(asset.read_text())['checkpoint']
    launch = f'/workspace/runtime/search/baselines/{{ image.category }}/{baseline}/launch.sh'
    command = [python, '-I', '-m', 'eval.bench.run', {{ task.model | tojson }}, {{ task.workload | tojson }}, launch, '--data', {{ inputs.private | tojson }}]
{% if task.workload in ("S-N", "A-N") %}
    command += ['--concurrency', '{{ config.evaluation.replay_concurrency(task.workload) }}']
{% endif %}
    if not run('serve', baseline, command, env={**os.environ, 'CHECKPOINT': checkpoint}):
        continue
    report = max(Path('/workspace/output/experiments').glob('*/report.json'), key=lambda path: path.stat().st_mtime_ns)
    result = json.loads(report.read_text())
    params = result['parameters']
    selection = {'baseline': result['baseline'], 'args': params.get('args', []), 'env': params.get('env', {})}
    break
if selection is None:
    selection = {'supported': False, 'attempts': attempts}
(home / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
PY
