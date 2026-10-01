"""Read-only host inspection and selection of an authored device Dockerfile.

The first implementation supports Linux NVIDIA x86 hosts. Device selection is
resolved to UUIDs, so an unknown index never silently falls back to another GPU.
The selected category maps directly to ``images/<category>/Dockerfile``.
This module does not generate Dockerfiles or choose a CUDA toolkit.
"""

from __future__ import annotations

import csv
import math
import os
import platform
import re
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
    try:
        result = subprocess.run(list(argv), capture_output=True, text=True, timeout=15, check=True)
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise DeviceError(f"{argv[0]} probe failed: {str(detail).strip()}") from error
    return result.stdout.strip()


def parse_nvidia(text: str) -> tuple[GPU, ...]:
    """Parse GPU identities and memory from this module's NVIDIA query."""
    found = []
    for line_number, row in enumerate(csv.reader(text.splitlines()), 1):
        if not row or all(not field.strip() for field in row):
            continue
        try:
            index, uuid, name, memory = [field.strip() for field in row]
            if not index.isdecimal() or not uuid.startswith("GPU-") or not name:
                raise ValueError("missing GPU identity")
            memory_mib = float(memory)
            if not math.isfinite(memory_mib) or memory_mib <= 0:
                raise ValueError("invalid GPU memory")
            found.append(GPU(int(index), uuid, name, memory_mib))
        except ValueError as error:
            raise DeviceError(f"invalid nvidia-smi row {line_number}: {error}") from error
    if not found:
        raise DeviceError("nvidia-smi reported no GPUs")
    if len({gpu.index for gpu in found}) != len(found) or len({gpu.uuid for gpu in found}) != len(
        found
    ):
        raise DeviceError("nvidia-smi reported duplicate GPU identities")
    return tuple(found)


def select_gpus(gpus: Sequence[GPU], ids: Sequence[str | int] | None = None) -> tuple[GPU, ...]:
    """Select every requested index/UUID in order, or all GPUs when ids is None."""
    if ids is None:
        if not gpus:
            raise DeviceError("no GPUs available")
        return tuple(gpus)
    if isinstance(ids, (str, bytes)) or not ids:
        raise DeviceError("GPU selection must be a nonempty sequence of indices or UUIDs")
    selected = []
    for value in ids:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise DeviceError(f"invalid GPU selector {value!r}")
        matches = [gpu for gpu in gpus if str(gpu.index) == str(value) or gpu.uuid == value]
        if len(matches) != 1:
            raise DeviceError(f"unknown or ambiguous GPU selector {value!r}")
        if matches[0] in selected:
            raise DeviceError(f"GPU {matches[0].uuid} was selected more than once")
        selected.append(matches[0])
    return tuple(selected)


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
    if cpu_cores is None:
        cpu_cores = os.cpu_count()
    if memory_bytes is None:
        try:
            memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (OSError, ValueError) as error:
            raise DeviceError("host memory could not be measured") from error
    if isinstance(cpu_cores, bool) or not isinstance(cpu_cores, int) or cpu_cores <= 0:
        raise DeviceError("host CPU count could not be measured")
    if isinstance(memory_bytes, bool) or not isinstance(memory_bytes, int) or memory_bytes <= 0:
        raise DeviceError("host memory could not be measured")
    query = NVIDIA_QUERY
    if gpu_ids is not None:
        if isinstance(gpu_ids, (str, bytes)) or not gpu_ids:
            raise DeviceError("GPU selection must be a nonempty sequence of indices or UUIDs")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (str, int))
            or not re.fullmatch(r"(?:\d+|GPU-[A-Za-z0-9-]+)", str(value))
            for value in gpu_ids
        ):
            raise DeviceError("GPU selectors must be indices or NVIDIA GPU UUIDs")
        # Probe selected identities directly; unrelated broken GPUs on a shared
        # host must not make a healthy selected GPU unusable.
        query = (*NVIDIA_QUERY, f"--id={','.join(map(str, gpu_ids))}")
    gpus = select_gpus(parse_nvidia(runner(query)), gpu_ids)
    return Machine(system, "x86_64", cpu_cores, memory_bytes, gpus)


def check_environment(*, runner: Runner = run_command) -> Environment:
    """Check daemon access, Compose and Harbor CLI availability without running a task."""
    docker = runner(("docker", "info", "--format", "{{.ServerVersion}}"))
    compose = runner(("docker", "compose", "version", "--short"))
    harbor = runner(("harbor", "--version"))
    if not all(value.strip() for value in (docker, compose, harbor)):
        raise DeviceError("Docker, Compose and Harbor probes must all return a version")
    return Environment(docker.strip(), compose.strip(), harbor.strip())
