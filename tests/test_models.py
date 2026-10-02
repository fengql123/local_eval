"""Checkpoint downloads: two common cases and two CLI boundary edges."""

import os
import subprocess
from pathlib import Path

import pytest

from eval.models import ModelAsset, prepare_asset


def test_hf_checkpoint_downloads_directly_into_the_run(tmp_path, monkeypatch):
    calls = []
    cli = tmp_path / "hf"
    cli.write_text('#!/bin/sh\nmkdir -p "$4"\nprintf "{}" > "$4/config.json"\n')
    cli.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    run = subprocess.run

    def download(argv, **kwargs):
        calls.append((argv, kwargs))
        return run(argv, **kwargs)

    monkeypatch.setattr("eval.models.subprocess.run", download)
    destination = tmp_path / "source"
    previous = os.umask(0o077)
    try:
        asset = prepare_asset("Qwen/Qwen3.6-35B-A3B", destination)
    finally:
        os.umask(previous)
    assert calls == [
        (
            ["hf", "download", asset.source, "--local-dir", str(destination)],
            {"check": True, "umask": 0o022},
        )
    ]
    assert asset == ModelAsset("Qwen/Qwen3.6-35B-A3B", destination, str(destination))
    assert destination.stat().st_mode & 0o777 == 0o755
    assert (destination / "config.json").stat().st_mode & 0o777 == 0o644


def test_native_engine_tags_do_not_download_on_the_host(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "eval.models.subprocess.run", lambda *args, **kwargs: pytest.fail("unexpected download")
    )
    asset = prepare_asset("qwen3.6:35b-a3b-mtp-q4_K_M", tmp_path / "ollama")
    assert asset.path is None
    assert asset.checkpoint == asset.source == "qwen3.6:35b-a3b-mtp-q4_K_M"
    assert not (tmp_path / "ollama").exists()


def test_download_path_with_spaces_is_one_command_argument(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr("eval.models.subprocess.run", lambda argv, **kwargs: commands.append(argv))
    destination = tmp_path / "run folder" / "baseline checkpoint"
    asset = prepare_asset("ggml-org/Qwen3.6-35B-A3B-GGUF", destination)
    assert commands[0][-1] == str(destination)
    assert len(commands[0]) == 5
    assert Path(asset.checkpoint) == destination


def test_download_failure_propagates_from_the_cli(tmp_path, monkeypatch):
    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr("eval.models.subprocess.run", fail)
    with pytest.raises(subprocess.CalledProcessError) as error:
        prepare_asset("Qwen/Qwen3.6-35B-A3B", tmp_path / "source")
    assert error.value.returncode == 1
