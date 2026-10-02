"""Read-only host inspection and selection of an authored device Dockerfile.

The first implementation supports Linux NVIDIA x86 hosts. Device selection is
resolved to UUIDs, so an unknown index never silently falls back to another GPU.
The selected category maps directly to ``images/<category>/Dockerfile``.
This module does not generate Dockerfiles or choose a CUDA toolkit.
"""

from __future__ import annotations

import csv
import os
import platform
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

Runner = Callable[[Sequence[str]], str]
NVIDIA_QUERY = (
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.total",
    "--format=csv,noheader,nounits",
)


class DeviceError(RuntimeError):
    """A required host capability could not be established."""


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    name: str
    memory_mib: float

    @property
    def memory_gib(self) -> float:
        return self.memory_mib / 1024


@dataclass(frozen=True)
class Machine:
    system: str
    architecture: str
    cpu_cores: int
    memory_bytes: int
    gpus: tuple[GPU, ...]
    category: str = "nvidia-cuda-x86"

    @property
    def memory_gib(self) -> float:
        return self.memory_bytes / 2**30

    @property
    def dockerfile(self) -> Path:
        """Repository-relative source path; setup later checks the supplied file."""
        return Path("images") / self.category / "Dockerfile"


@dataclass(frozen=True)
class Environment:
    docker_version: str
    compose_version: str
    harbor_version: str


def run_command(argv: Sequence[str]) -> str:
    """Run a bounded read-only probe; report failure instead of returning no data."""
    return subprocess.run(
        list(argv), capture_output=True, text=True, timeout=15, check=True
    ).stdout.strip()


def parse_nvidia(text: str) -> tuple[GPU, ...]:
    """Read the identities and memory returned by the NVIDIA query."""
    return tuple(
        GPU(int(index), uuid.strip(), name.strip(), float(memory))
        for index, uuid, name, memory in csv.reader(text.splitlines())
    )


def select_gpus(gpus: Sequence[GPU], ids: Sequence[str | int] | None = None) -> tuple[GPU, ...]:
    """Resolve selected identities without substituting another GPU."""
    by_id = {key: gpu for gpu in gpus for key in (str(gpu.index), gpu.uuid)}
    return tuple(gpus) if ids is None else tuple(by_id[str(value)] for value in ids)


def inspect_machine(
    gpu_ids: Sequence[str | int] | None = None,
    *,
    runner: Runner = run_command,
    system: str | None = None,
    architecture: str | None = None,
    cpu_cores: int | None = None,
    memory_bytes: int | None = None,
) -> Machine:
    """Inspect a supported host. Injectable probes let tests run without a GPU."""
    system = system or platform.system()
    architecture = architecture or platform.machine()
    if system != "Linux" or architecture not in {"x86_64", "amd64"}:
        raise DeviceError(
            f"unsupported host {system}/{architecture}; currently supported: Linux/x86_64 NVIDIA"
        )
    cpu_cores = os.cpu_count() if cpu_cores is None else cpu_cores
    memory_bytes = (
        os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        if memory_bytes is None
        else memory_bytes
    )
    # Probe selected GPUs directly so unrelated broken GPUs do not block a run.
    query = (
        NVIDIA_QUERY if gpu_ids is None else (*NVIDIA_QUERY, f"--id={','.join(map(str, gpu_ids))}")
    )
    gpus = select_gpus(parse_nvidia(runner(query)), gpu_ids)
    return Machine(system, "x86_64", cpu_cores, memory_bytes, gpus)


def check_environment(*, runner: Runner = run_command) -> Environment:
    """Check daemon access, Compose and Harbor CLI availability without running a task."""
    docker = runner(("docker", "info", "--format", "{{.ServerVersion}}"))
    compose = runner(("docker", "compose", "version", "--short"))
    harbor = runner(("harbor", "--version"))
    return Environment(docker.strip(), compose.strip(), harbor.strip())
