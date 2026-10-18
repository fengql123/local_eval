"""Acquire one checkpoint into the caller's local workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx


@dataclass(frozen=True)
class ModelAsset:
    source: str
    path: Path
    checkpoint: str


def checkpoint_revision(destination: Path) -> str | None:
    """Read the acquired HF commit, including releases with only split GGUF weights."""
    metadata = destination / ".cache/huggingface/download"
    revisions = {path.read_text().splitlines()[0] for path in metadata.rglob("*.metadata")}
    if len(revisions) > 1:
        raise ValueError("checkpoint contains files acquired from different HF revisions")
    return next(iter(revisions), None)


def _patterns(checkpoint: str) -> tuple[str, tuple[str, ...]]:
    quant = (
        "MXFP4" if "DeepSeek" in checkpoint else "Q8_0" if "Flash-Next" in checkpoint else "Q4_K_M"
    )
    return f"*{quant}*.gguf", ("*mmproj*", "*mtp-*", "*dflash-*", "*dspark*")


def prepare_asset(checkpoint: str, destination: Path) -> ModelAsset:
    """Download a mapped HF release or Ollama store; never mount or reshard weights."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if ":" not in checkpoint:
        command = [
            str(Path(sys.executable).with_name("hf")),
            "download",
            checkpoint,
            "--local-dir",
            str(destination),
        ]
        if checkpoint.endswith("-GGUF"):
            include, exclude = _patterns(checkpoint)
            command.extend(["--include", include])
            for pattern in exclude:
                command.extend(["--exclude", pattern])
        subprocess.run(command, check=True, umask=0o022, stdout=sys.stderr)
    else:
        name, tag = checkpoint.rsplit(":", 1)
        name = name if "/" in name else f"library/{name}"
        registry = "registry.ollama.ai"
        manifest = destination / "manifests" / registry / name / tag
        manifest.parent.mkdir(parents=True, exist_ok=True)
        (destination / "blobs").mkdir(exist_ok=True)
        with httpx.Client(follow_redirects=True, timeout=None) as client:
            response = client.get(f"https://{registry}/v2/{name}/manifests/{tag}")
            response.raise_for_status()
            document = response.json()
            for layer in (document["config"], *document["layers"]):
                blob = destination / "blobs" / layer["digest"].replace(":", "-")
                if blob.exists() and blob.stat().st_size == layer["size"]:
                    continue
                partial = blob.with_suffix(".partial")
                offset = partial.stat().st_size if partial.exists() else 0
                with client.stream(
                    "GET",
                    f"https://{registry}/v2/{name}/blobs/{layer['digest']}",
                    headers={"Range": f"bytes={offset}-"} if offset else {},
                ) as download:
                    download.raise_for_status()
                    with partial.open(
                        "ab" if offset and download.status_code == 206 else "wb"
                    ) as handle:
                        for chunk in download.iter_bytes(8 * 1024 * 1024):
                            handle.write(chunk)
                if partial.stat().st_size != layer["size"]:
                    raise ValueError(f"incomplete Ollama blob: {layer['digest']}")
                partial.replace(blob)
            manifest.write_text(response.text + "\n")
    return ModelAsset(checkpoint, destination, str(destination))


def prepare_tokenizer(source: str, destination: Path) -> str:
    """Acquire only the fixed tokenizer and chat-template files, never source weights."""
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(Path(sys.executable).with_name("hf")),
            "download",
            source,
            "--local-dir",
            str(destination),
            *(
                argument
                for pattern in (
                    "tokenizer*",
                    "vocab*",
                    "merges.txt",
                    "special_tokens_map.json",
                    "chat_template*",
                )
                for argument in ("--include", pattern)
            ),
        ],
        check=True,
        umask=0o022,
        stdout=sys.stderr,
    )
    digest = hashlib.sha256()
    for path in sorted(destination.iterdir()):
        if path.is_file():
            digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--tokenizer", type=Path, help="also acquire the fixed tokenizer")
    parser.add_argument("--output", type=Path, help="write acquired checkpoint metadata")
    args = parser.parse_args()
    asset = prepare_asset(args.source, args.destination)
    metadata = asdict(asset)
    if args.tokenizer is not None:
        metadata["tokenizer_identity"] = prepare_tokenizer(args.source, args.tokenizer)
        metadata["checkpoint_revision"] = checkpoint_revision(asset.path)
        for path in args.tokenizer.iterdir():
            if path.is_file():
                shutil.copyfile(path, asset.path / path.name)
    value = json.dumps(metadata, default=str) + "\n"
    if args.output:
        partial = args.output.with_suffix(".partial")
        partial.write_text(value)
        partial.replace(args.output)
    else:
        print(value, end="")


if __name__ == "__main__":
    main()
