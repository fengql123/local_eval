"""Describe the measured machine so results group by hardware, not by run.

Verification runs this inside the timed sandbox, which sees exactly the GPUs and
CPU/memory limits the candidate had. Every probe is best effort: a missing tool
leaves its field empty instead of failing verification.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

CPU_GROUPS = {
    "AMD EPYC Zen 4": (  # Genoa and Genoa-X
        "9124",
        "9174F",
        "9184X",
        "9224",
        "9254",
        "9274F",
        "9334",
        "9354",
        "9354P",
        "9374F",
        "9384X",
        "9454",
        "9454P",
        "9474F",
        "9534",
        "9554",
        "9554P",
        "9634",
        "9654",
        "9654P",
        "9684X",
    ),
    "AMD EPYC Zen 5": ("9335", "9355", "9455", "9555", "9B45"),
    "Intel Xeon 6 Granite Rapids": ("6776P",),
}


def cpu_group(name: str | None) -> str:
    """Explicit benchmark equivalence; retain exact identity for unlisted CPUs."""
    for group, models in CPU_GROUPS.items():
        if name and any(re.search(r"\b" + model + r"\b", name) for model in models):
            return group
    return name or "unknown CPU"


def _gpus() -> list[dict]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in output.strip().splitlines():
        name, memory, driver, generation, width = (part.strip() for part in line.split(","))
        gpus.append(
            {
                "name": name,
                "memory_mib": int(float(memory)),
                "driver": driver,
                "pcie_generation": int(generation) if generation.isdigit() else None,
                "pcie_width": int(width) if width.isdigit() else None,
            }
        )
    return gpus


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def machine(task: dict, *, gpus=_gpus, root: Path = Path("/")) -> dict:
    """GPUs, CPU model and the sandbox's CPU/memory limits, with a stable fingerprint."""
    cpuinfo = _read(root / "proc/cpuinfo") or ""
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in cpuinfo.splitlines()
            if line.startswith("model name")
        ),
        None,
    )
    environment = task.get("environment", {})
    description = {
        "category": task.get("device_category"),
        "gpus": gpus(),
        "cpu": cpu,
        "cpu_group": cpu_group(cpu),
        "storage_mb": environment.get("storage_mb"),
        "cpus": environment.get("cpus"),
        "memory_mb": environment.get("memory_mb"),
        "cgroup": {
            "cpu_max": _read(root / "sys/fs/cgroup/cpu.max"),
            "memory_max": _read(root / "sys/fs/cgroup/memory.max"),
        },
    }
    comparable = {
        key: description[key]
        for key in ("category", "cpu_group", "cpus", "memory_mb", "storage_mb")
    }
    comparable["gpus"] = [
        {key: gpu.get(key) for key in ("name", "memory_mib", "pcie_generation", "pcie_width")}
        for gpu in description["gpus"]
    ]
    identity = json.dumps(comparable, sort_keys=True).encode()
    return {**description, "fingerprint": hashlib.sha256(identity).hexdigest()[:16]}


def check_cuda(runtime: dict) -> None:
    """Reject an unusable NVIDIA sandbox before downloading its model assets."""
    if not runtime["category"].startswith("nvidia-cuda"):
        return
    import ctypes

    driver, count = ctypes.CDLL("libcuda.so.1"), ctypes.c_int()
    code = driver.cuInit(0) or driver.cuDeviceGetCount(ctypes.byref(count))
    expected = runtime.get("allocation_check", {}).get("gpus")
    if code or not count.value or (expected is not None and count.value != expected):
        raise RuntimeError(
            f"CUDA allocation unusable before downloads: code={code}, devices={count.value}"
        )
