"""Download checkpoint packages with the Hugging Face CLI."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelAsset:
    source: str
    path: Path | None
    checkpoint: str


def prepare_asset(checkpoint: str, destination: Path) -> ModelAsset:
    """Download HF packages into the run; native engine tags stay identifiers."""
    if ":" in checkpoint:
        return ModelAsset(checkpoint, None, checkpoint)
    destination = destination.resolve()
    subprocess.run(
        ["hf", "download", checkpoint, "--local-dir", str(destination)], check=True, umask=0o022
    )
    return ModelAsset(checkpoint, destination, str(destination))
