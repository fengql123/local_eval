"""Check local Docker and Harbor tools; device selection comes from configuration."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Environment:
    docker_version: str
    compose_version: str
    harbor_version: str


def run_command(argv: Sequence[str]) -> str:
    return subprocess.run(
        list(argv), capture_output=True, text=True, timeout=15, check=True
    ).stdout.strip()


def check_environment(*, runner: Callable = run_command) -> Environment:
    """Check tools used by native Harbor Docker, without inspecting hardware."""
    return Environment(
        runner(("docker", "info", "--format", "{{.ServerVersion}}")),
        runner(("docker", "compose", "version", "--short")),
        runner(("harbor", "--version")),
    )
