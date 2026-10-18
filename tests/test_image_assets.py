"""Exercise the authored shell contracts without downloading engines or weights."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from eval.launch import Launch

IMAGES = Path(__file__).resolve().parents[1] / "images" / "nvidia-cuda-x86"


def capture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'checkpoint': os.environ['CHECKPOINT']}))\n"
    )
    path.chmod(0o755)


def run(script: Path, checkpoint: str, *args: str) -> dict:
    command, environment = Launch(script, checkpoint, args).prepare()
    output = subprocess.run(
        command, env={**os.environ, **environment}, capture_output=True, text=True, check=True
    )
    return json.loads(output.stdout)


def test_python_engine_launchers_fix_identity_and_forward_tuning(tmp_path):
    scripts = (
        ("baselines/freetoken", "ft", ("serve", "--model")),
        ("baselines/ktransformers", "python", ("-m", "sglang.launch_server", "--model-path")),
        ("baselines/ktransformers-dsv4", "python", ("-m", "sglang.launch_server", "--model")),
        ("record/vllm", "vllm", ("serve",)),
    )
    for folder, binary, prefix in scripts:
        destination = tmp_path / folder
        destination.mkdir(parents=True)
        script = destination / "launch.sh"
        shutil.copyfile(IMAGES / folder / "launch.sh", script)
        capture(destination / ".venv" / "bin" / binary)
        result = run(script, "/models/source", "--tensor-parallel-size", "2")
        argv = result["argv"]
        assert argv[: len(prefix) + 1] == [*prefix, "/models/source"]
        assert argv[argv.index("--host") + 1] == "127.0.0.1"
        assert argv[argv.index("--port") + 1] == "8000"
        assert argv[argv.index("--served-model-name") + 1] == "served"
        assert argv[-2:] == ["--tensor-parallel-size", "2"]


def test_gguf_launcher_selects_first_shard_and_ignores_projection(tmp_path):
    script = tmp_path / "launch.sh"
    shutil.copyfile(IMAGES / "baselines/llama.cpp/launch.sh", script)
    capture(tmp_path / "build/bin/llama-server")
    checkpoint = tmp_path / "weights"
    checkpoint.mkdir()
    for name in (
        "model-Q4_K_M-00001-of-00002.gguf",
        "model-Q4_K_M-00002-of-00002.gguf",
        "mmproj.gguf",
        "dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf",
        "DSpark-DeepSeek-V4-Flash-0731-BF16.gguf",
        "model-mtp-Q4_K_M.gguf",
        "model-dflash-Q4_K_M.gguf",
    ):
        (checkpoint / name).touch()
    result = run(script, str(checkpoint), "--ctx-size", "8192")
    argv = result["argv"]
    assert argv[argv.index("--model") + 1] == str(checkpoint / "model-Q4_K_M-00001-of-00002.gguf")
    assert argv[argv.index("--alias") + 1] == "served"
    assert argv[argv.index("--port") + 1] == "8000"
    assert argv[-2:] == ["--ctx-size", "8192"]
    assert run(script, str(checkpoint / "model-Q4_K_M-00001-of-00002.gguf"))["argv"][1] == argv[1]


def test_gguf_launcher_rejects_ambiguous_quantizations(tmp_path):
    script = tmp_path / "launch.sh"
    shutil.copyfile(IMAGES / "baselines/llama.cpp/launch.sh", script)
    (tmp_path / "model-Q4_K_M.gguf").touch()
    (tmp_path / "model-Q8_0.gguf").touch()
    command, environment = Launch(script, str(tmp_path)).prepare()
    result = subprocess.run(
        command, env={**os.environ, **environment}, capture_output=True, text=True
    )
    assert result.returncode == 65
    assert "found 2" in result.stderr


def test_missing_inputs_and_failed_installers_stop_before_next_engine(tmp_path):
    for source in IMAGES.rglob("launch.sh"):
        environment = dict(os.environ)
        environment.pop("CHECKPOINT", None)
        result = subprocess.run(
            ["/bin/bash", str(source)],
            env=environment,
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert result.returncode != 0
        assert "required" in result.stderr

    common = (IMAGES / "Dockerfile").read_text()
    assert "13.0.2-devel" in common
    assert "cuda-nvcc-12" not in common
    assert "CUDA_ARCHS" not in common and "TORCH_CUDA_ARCH_LIST" not in common
    assert "BASELINES" not in common and "RECORD_VLLM" not in common
    assert "COPY eval" not in common and "install.sh" not in common
    for phase, owner in (("search", "search"), ("coding", "agent"), ("verification", "verifier")):
        tail = (IMAGES / f"{phase}.Dockerfile").read_text()
        assert f"FROM toolchain AS {phase}" in tail
        assert "USER root" in tail and f"--gid {owner}" in tail
        assert "-o root -g root -m 0755 /workspace" in tail
        if phase != "verification":
            assert "useradd --uid 1003" not in tail
            assert "useradd --uid 1004" not in tail
    verify = (IMAGES / "verification.Dockerfile").read_text()
    assert "-o root -g root -m 0700 /var/lib/local-eval" in verify
    assert "-o judge -g root -m 2750 /workspace/output/review" in verify
    assert "useradd --uid 1005" in verify and "/workspace/.harbor/scratch/reference" in verify
    assert "chmod 1775 /workspace" in verify
    assert not (IMAGES / "setup.Dockerfile").exists()
    assert "bindfs" not in common and "fuse3" not in common
    assert common.index("COPY --from=certificates") < common.index("apt-get update")
    assert "sed -i 's|http://|https://|g' /etc/apt/sources.list.d/ubuntu.sources" in common
