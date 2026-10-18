#!/usr/bin/env bash
# CHECKPOINT is a local Ollama store or mapped GGUF directory. Extra args: --param NAME=VALUE.
set -euo pipefail
: "${CHECKPOINT:?local Ollama checkpoint directory required}"
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
exec python3 - "$CHECKPOINT" "$PWD/src/ollama" "$@" <<'PYTHON'
import argparse
import hashlib
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint')
parser.add_argument('binary')
parser.add_argument('--param', action='append', default=[])
args = parser.parse_args()
store = Path(args.checkpoint).resolve()
registry = 'registry.ollama.ai'
home = Path(tempfile.mkdtemp(prefix='local-eval-ollama-'))
(home / 'blobs').mkdir()
environment = {**os.environ, 'OLLAMA_MODELS': str(home), 'OLLAMA_HOST': '127.0.0.1:8000',
               'OLLAMA_NOPRUNE': '1', 'OLLAMA_NO_CLOUD': '1'}
if (store / 'manifests').is_dir():
    [manifest_path] = list((store / 'manifests' / registry).glob('*/*/*'))
else:
    # The pinned fork imports split GGUF directly. Register links in its native blob
    # store, then let /api/create read metadata and assemble the model manifest.
    # Nothing is merged, re-quantized or copied into a second weight directory.
    files = {}
    for file in sorted(store.rglob('*.gguf')):
        if 'mmproj' in file.name:
            continue
        with file.open('rb') as handle:
            checksum = hashlib.sha256()
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                checksum.update(block)
            digest = 'sha256:' + checksum.hexdigest()
        (home / 'blobs' / digest.replace(':', '-')).symlink_to(file)
        files[str(file.relative_to(store))] = digest
    if not files:
        raise ValueError(f'no GGUF model shards in {store}')
    # Keep the import daemon off the benchmark endpoint until the manifest and
    # adjustable parameters are ready. Readiness must only observe the final server.
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        address = f'127.0.0.1:{listener.getsockname()[1]}'
    bootstrap = {**environment, 'OLLAMA_HOST': address}
    daemon = subprocess.Popen([args.binary, 'serve'], env=bootstrap, stdout=os.sys.stderr)
    for _ in range(600):
        probe = subprocess.run([args.binary, 'list'], env=bootstrap,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if probe.returncode == 0:
            break
        time.sleep(0.1)
    else:
        raise TimeoutError('Ollama import server did not become ready')
    request = urllib.request.Request(f'http://{address}/api/create',
        data=json.dumps({'model': 'served', 'files': files, 'stream': False}).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        created = json.load(response)
    if created.get('status') != 'success':
        raise RuntimeError(f'Ollama GGUF import failed: {created}')
    daemon.terminate()
    daemon.wait(timeout=30)
    manifest_path = home / 'manifests' / registry / 'library' / 'served' / 'latest'
manifest = json.loads(manifest_path.read_text())
for layer in (manifest['config'], *manifest['layers']):
    blob = layer['digest'].replace(':', '-')
    target = home / 'blobs' / blob
    if not target.exists():
        target.symlink_to(store / 'blobs' / blob)
media_type = 'application/vnd.ollama.image.params'
if args.param:
    old = next((layer for layer in manifest['layers'] if layer['mediaType'] == media_type), None)
    params = json.loads((home / 'blobs' / old['digest'].replace(':', '-')).read_text()) if old else {}
    for item in args.param:
        key, value = item.split('=', 1)
        # JSON gives numbers/booleans their native type; other values remain strings.
        if value[:1] in '0123456789-"[' or value in ('true', 'false', 'null'):
            value = json.loads(value)
        if key == 'stop':
            params.setdefault(key, []).append(value)
        else:
            params[key] = value
    body = json.dumps(params).encode()
    digest = 'sha256:' + hashlib.sha256(body).hexdigest()
    (home / 'blobs' / digest.replace(':', '-')).write_bytes(body)
    manifest['layers'] = [layer for layer in manifest['layers'] if layer['mediaType'] != media_type]
    manifest['layers'].append({'mediaType': media_type, 'digest': digest, 'size': len(body)})
output = home / 'manifests' / registry / 'library' / 'served' / 'latest'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(manifest))
os.execve(args.binary, [args.binary, 'serve'], environment)
PYTHON
