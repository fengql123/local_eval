"""Two normal local downloads and two interrupted acquisition cases."""

import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from eval import models
from eval.models import ModelAsset, prepare_asset, prepare_tokenizer


def test_hf_download_uses_local_destination_and_mapped_gguf_variant(tmp_path, monkeypatch):
    commands = []

    def download(argv, **kwargs):
        commands.append(argv)
        assert kwargs["check"] and kwargs["umask"] == 0o022
        (Path(argv[4]) / "weights").write_bytes(b"weights")

    monkeypatch.setattr("eval.models.subprocess.run", download)
    destination = tmp_path / "weights with spaces"
    source = "ggml-org/Qwen3.8-Flash-Next-GGUF"
    assert prepare_asset(source, destination) == ModelAsset(source, destination, str(destination))
    assert commands[0][1:5] == ["download", source, "--local-dir", str(destination)]
    assert commands[0][5:] == [
        "--include",
        "*Q8_0*.gguf",
        "--exclude",
        "*mmproj*",
        "--exclude",
        "*mtp-*",
        "--exclude",
        "*dflash-*",
        "--exclude",
        "*dspark*",
    ]
    prepare_asset("owner/fp8", tmp_path / "fp8")
    assert len(commands[1]) == 5

    # Exercise the installed HF parser: repeated excludes must not turn into
    # positional filenames, which silently disables pattern selection.
    from click.testing import CliRunner
    from huggingface_hub.cli.hf import app

    selected = []

    def snapshot(repo_id, **kwargs):
        selected.append(kwargs)
        return str(destination)

    monkeypatch.setattr("huggingface_hub.cli.download.snapshot_download", snapshot)
    monkeypatch.setattr(
        "huggingface_hub.cli.download.hf_hub_download",
        lambda *a, **k: pytest.fail("patterns became filenames"),
    )
    result = CliRunner().invoke(app, commands[0][1:])
    assert result.exit_code == 0, result.output
    assert selected[0]["allow_patterns"] == ["*Q8_0*.gguf"]
    assert selected[0]["ignore_patterns"] == ["*mmproj*", "*mtp-*", "*dflash-*", "*dspark*"]
    prepare_tokenizer("owner/tokenizer", tmp_path / "tokenizer")
    result = CliRunner().invoke(app, commands[-1][1:])
    assert result.exit_code == 0, result.output
    assert selected[-1]["allow_patterns"] == [
        "tokenizer*",
        "vocab*",
        "merges.txt",
        "special_tokens_map.json",
        "chat_template*",
    ]
    assert selected[-1]["ignore_patterns"] is None


def ollama_http(monkeypatch, handler):
    client = httpx.Client
    monkeypatch.setattr(
        "eval.models.httpx.Client",
        lambda **kw: client(transport=httpx.MockTransport(handler), **kw),
    )


def test_ollama_downloads_one_native_store_and_reuses_complete_blobs(tmp_path, monkeypatch):
    document = {
        "config": {"digest": "sha256:cfg", "size": 2},
        "layers": [{"digest": "sha256:model", "size": 7}],
    }
    calls = []

    def request(req):
        calls.append(req.url.path)
        if "/manifests/" in req.url.path:
            return httpx.Response(200, json=document)
        return httpx.Response(200, content=b"{}" if req.url.path.endswith("cfg") else b"weights")

    ollama_http(monkeypatch, request)
    asset = prepare_asset("qwen:tag", tmp_path)
    prepare_asset("qwen:tag", tmp_path)
    assert asset == ModelAsset("qwen:tag", tmp_path, str(tmp_path))
    assert (tmp_path / "blobs/sha256-model").read_bytes() == b"weights"
    assert sum("/blobs/" in p for p in calls) == 2


def test_failed_hf_download_propagates_without_success_metadata(tmp_path, monkeypatch):
    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr("eval.models.subprocess.run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        prepare_asset("owner/model", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_incomplete_ollama_blob_resumes_before_publishing_manifest(tmp_path, monkeypatch):
    document = {"config": {"digest": "sha256:weights", "size": 7}, "layers": []}
    ranges = []

    def request(req):
        if "/manifests/" in req.url.path:
            return httpx.Response(200, json=document)
        ranges.append(req.headers.get("range"))
        return httpx.Response(206 if ranges[-1] else 200, content=b"ghts" if ranges[-1] else b"wei")

    ollama_http(monkeypatch, request)
    with pytest.raises(ValueError, match="incomplete Ollama blob"):
        prepare_asset("qwen:tag", tmp_path)
    manifest = tmp_path / "manifests/registry.ollama.ai/library/qwen/tag"
    assert not manifest.exists()
    prepare_asset("qwen:tag", tmp_path)
    assert ranges == [None, "bytes=3-"]
    assert manifest.is_file() and (tmp_path / "blobs/sha256-weights").read_bytes() == b"weights"


@pytest.mark.parametrize("fail", [False, True])
def test_setup_acquisition_publishes_metadata_only_after_tokenizer(tmp_path, monkeypatch, fail):
    target, tokenizer, output = (tmp_path / name for name in ("target", "tokenizer", "target.json"))

    def acquire(source, destination):
        destination.mkdir()
        return ModelAsset(source, destination, str(destination))

    def acquire_tokenizer(source, destination):
        destination.mkdir()
        (destination / "tokenizer.json").write_text("{}")
        if fail:
            raise RuntimeError("tokenizer download interrupted")
        return "tokenizer-digest"

    monkeypatch.setattr(models, "prepare_asset", acquire)
    monkeypatch.setattr(models, "prepare_tokenizer", acquire_tokenizer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "models",
            "owner/model",
            str(target),
            "--tokenizer",
            str(tokenizer),
            "--output",
            str(output),
        ],
    )
    if fail:
        with pytest.raises(RuntimeError, match="tokenizer download interrupted"):
            models.main()
        assert not output.exists()
    else:
        models.main()
        assert json.loads(output.read_text())["tokenizer_identity"] == "tokenizer-digest"
        assert (target / "tokenizer.json").read_text() == "{}"
    assert not output.with_suffix(".partial").exists()
