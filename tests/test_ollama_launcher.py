"""Native-store and split-GGUF launches, including failed imports."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from eval.launch import Launch

SOURCE = Path(__file__).resolve().parents[1] / "images/nvidia-cuda-x86/baselines/ollama/launch.sh"


ENGINE = r"""
import json, os, sys, urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
root = Path(os.environ['OLLAMA_MODELS'])
address = os.environ['OLLAMA_HOST']
manifest = root / 'manifests/registry.ollama.ai/library/served/latest'
if sys.argv[1] == 'list':
    try:
        urllib.request.urlopen('http://' + address + '/api/tags', timeout=1).read()
    except OSError:
        sys.exit(1)
    sys.exit(0)
if address != '127.0.0.1:8000':
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            links = [root / 'blobs' / digest.replace(':', '-') for digest in request['files'].values()]
            Path(os.environ['AUDIT']).write_text(json.dumps({'request': request,
                'links': [str(p.resolve()) for p in links], 'symlinks': all(p.is_symlink() for p in links),
                'import_address': address, 'cloud': os.environ['OLLAMA_NO_CLOUD']}))
            if os.environ.get('IMPORT_FAIL'):
                self.send_response(500); self.end_headers(); return
            (root / 'blobs/sha256-config').write_text('{}')
            layers = [{'mediaType': 'application/vnd.ollama.image.model', 'digest': d}
                      for d in request['files'].values()]
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({'config': {'digest': 'sha256:config'}, 'layers': layers}))
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"success"}')
    host, port = address.split(':')
    HTTPServer((host, int(port)), Handler).serve_forever()
else:
    doc = json.loads(manifest.read_text())
    layer = next((x for x in doc['layers'] if x['mediaType'] == 'application/vnd.ollama.image.params'), None)
    params = json.loads((root / 'blobs' / layer['digest'].replace(':', '-')).read_text()) if layer else {}
    print(json.dumps({'host': address, 'cloud': os.environ['OLLAMA_NO_CLOUD'], 'params': params,
                      'model_links': [str((root / 'blobs' / x['digest'].replace(':', '-')).resolve())
                                      for x in doc['layers'] if x['mediaType'] == 'application/vnd.ollama.image.model']}))
"""


def launch(tmp_path, checkpoint, *, fail=False):
    folder = tmp_path / "images/nvidia-cuda-x86/baselines/ollama"
    (folder / "src").mkdir(parents=True)
    script = folder / "launch.sh"
    shutil.copyfile(SOURCE, script)
    binary = folder / "src/ollama"
    binary.write_text(f"#!{sys.executable}\n" + ENGINE)
    binary.chmod(0o755)
    command, env = Launch(script, str(checkpoint), ("--param", "num_ctx=65536")).prepare()
    env = {**os.environ, **env, "AUDIT": str(tmp_path / "audit.json")}
    if fail:
        env["IMPORT_FAIL"] = "1"
    with (tmp_path / "output").open("w") as out, (tmp_path / "errors").open("w") as err:
        process = subprocess.Popen(command, env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            code = process.wait(timeout=15)
        finally:
            subprocess.run(["kill", "-TERM", "--", f"-{process.pid}"], capture_output=True)
    return code, (tmp_path / "output").read_text(), (tmp_path / "errors").read_text()


def native_store(path):
    (path / "blobs").mkdir(parents=True)
    (path / "blobs/sha256-config").write_text("{}")
    (path / "blobs/sha256-model").write_bytes(b"original weights")
    manifest = path / "manifests/registry.ollama.ai/library/original/latest"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "config": {"digest": "sha256:config"},
                "layers": [
                    {"digest": "sha256:model", "mediaType": "application/vnd.ollama.image.model"}
                ],
            }
        )
    )
    return manifest


def test_native_store_remains_unchanged_and_parameters_are_local(tmp_path):
    weights = tmp_path / "weights"
    manifest = native_store(weights)
    original = manifest.read_bytes()
    code, output, _ = launch(tmp_path, weights)
    result = json.loads(output)
    assert code == 0 and result["host"] == "127.0.0.1:8000"
    assert result["params"] == {"num_ctx": 65536} and result["cloud"] == "1"
    assert manifest.read_bytes() == original
    assert result["model_links"] == [str(weights / "blobs/sha256-model")]
    assert not (tmp_path / "audit.json").exists()


def test_split_gguf_uses_native_import_without_copying_or_early_readiness(tmp_path):
    weights = tmp_path / "weights/Q4_K_M"
    weights.mkdir(parents=True)
    shards = [weights / f"MiniMax-M3-Q4_K_M-{i:05}-of-00002.gguf" for i in (1, 2)]
    for i, shard in enumerate(shards):
        shard.write_bytes(b"GGUF" + bytes([i]))
    (weights / "mmproj.gguf").write_bytes(b"excluded")
    before = {p.name: p.read_bytes() for p in weights.iterdir()}
    code, output, errors = launch(tmp_path, weights.parent)
    assert code == 0, errors
    result = json.loads(output)
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["symlinks"] and audit["links"] == [str(p) for p in shards]
    assert audit["import_address"] != result["host"] == "127.0.0.1:8000"
    assert audit["request"]["stream"] is False and audit["request"]["model"] == "served"
    assert result["params"] == {"num_ctx": 65536}
    assert {p.name: p.read_bytes() for p in weights.iterdir()} == before


def test_missing_model_shards_fail_before_starting_import_server(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "mmproj.gguf").touch()
    code, _, errors = launch(tmp_path, weights)
    assert code != 0 and "no GGUF model shards" in errors
    assert not (tmp_path / "audit.json").exists()


def test_native_import_failure_never_executes_final_server(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model-Q4_K_M.gguf").write_bytes(b"GGUF")
    code, output, errors = launch(tmp_path, weights, fail=True)
    assert code != 0 and not output and "HTTP Error 500" in errors
    assert json.loads((tmp_path / "audit.json").read_text())["import_address"] != "127.0.0.1:8000"
