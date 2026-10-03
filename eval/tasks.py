"""Select supplied category sources and render one-container Harbor tasks.

This module neither builds images nor runs search or evaluation. Harbor owns the
image build, shared container, and ordered search/coding steps. The coordinator
and verifier entry points are implemented in later phases.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .config import RunConfig, Task
from .device import Machine
from .request_policy import pool_of

_CHECKOUT_TEMPLATES = Path(__file__).resolve().parents[1] / "templates"
_PACKAGED_TEMPLATES = Path(__file__).resolve().parent / "_templates"
TEMPLATES = _PACKAGED_TEMPLATES if _PACKAGED_TEMPLATES.is_dir() else _CHECKOUT_TEMPLATES


@dataclass(frozen=True)
class ImageSpec:
    category: str
    context: Path
    dockerfile: Path
    identity: str
    tag: str
    build_args: dict[str, str]
    manifest: dict[str, Any]


def select_image(repo_root: Path, machine: Machine, engines: tuple[str, ...]) -> ImageSpec:
    """Require authored assets before downloads; identify the shared category image.

    The Dockerfile receives BASELINES as space-separated engine names and invokes
    images/<category>/baselines/<engine>/install.sh. Each engine also supplies a
    launch.sh that receives its mapped checkpoint when measurement starts.
    """
    repo_root = repo_root.resolve()
    engines = tuple(sorted(engines))
    required = (
        machine.dockerfile,
        *(
            Path("images") / machine.category / "baselines" / engine / script
            for engine in engines
            for script in ("install.sh", "launch.sh")
        ),
    )
    sources = set(required) | {Path(".dockerignore")}
    sources.update(
        path.relative_to(repo_root)
        for pattern in (
            "eval/**/*.py",
            "tables/*.csv",
            "tables/*.json",
            "templates/*.j2",
            "templates/*.sh",
            "templates/*.md",
        )
        for path in repo_root.glob(pattern)
        if path.is_file()
    )
    sources.update(
        Path(name)
        for name in ("pyproject.toml", "uv.lock", "README.md")
        if (repo_root / name).is_file()
    )
    hashes = {
        str(path): hashlib.sha256((repo_root / path).read_bytes()).hexdigest()
        for path in sorted(sources)
    }
    build_args = {"BASELINES": " ".join(engines)}
    definition = {
        "category": machine.category,
        "dockerfile": str(machine.dockerfile),
        "architecture": machine.architecture,
        "build_args": build_args,
        "engines": engines,
        "sources": hashes,
    }
    identity = hashlib.sha256(
        json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    tag = f"local-eval:{identity[:32]}"
    manifest = {
        "schema": 1,
        "identity": identity,
        "tag": tag,
        "context": str(repo_root),
        **definition,
    }
    return ImageSpec(
        machine.category, repo_root, machine.dockerfile, identity, tag, build_args, manifest
    )


def render_task(
    task: Task,
    config: RunConfig,
    machine: Machine,
    image: ImageSpec,
    run_folder: Path,
    inputs: dict[str, str],
    assets: dict[str, Any],
) -> Path:
    """Write a Harbor task using selected public data and protected evaluator inputs."""
    task_folder = run_folder.resolve() / "tasks" / task.task_id
    environment = task_folder / "environment"
    environment.mkdir(parents=True, exist_ok=False)
    for phase in ("search", "coding"):
        (task_folder / "steps" / phase / "tests").mkdir(parents=True)
    pool = pool_of(task.workload)
    input_targets = {
        "public": f"/workspace/data/{pool}",
        "calibration": "/home/search/data/S",
        "private": "/root/eval/private",
        "quality": "/root/eval/quality",
    }
    mounts = [
        {
            "type": "bind",
            "source": str(Path(inputs[key]).resolve()),
            "target": target,
            "read_only": True,
        }
        for key, target in input_targets.items()
    ]
    model_root = Path(assets["root"]).resolve()
    mounts.append(
        {"type": "bind", "source": str(model_root), "target": "/models", "read_only": True}
    )

    def container_asset(value):
        value = dict(value)
        if value["path"] is not None:
            value["path"] = str(
                Path("/models") / Path(value["path"]).resolve().relative_to(model_root)
            )
            # Downloaded checkpoints use host paths; native model tags have no
            # local path and must remain unchanged.
            checkpoint = Path(value["checkpoint"])
            value["checkpoint"] = str(
                Path("/models") / checkpoint.resolve().relative_to(model_root)
            )
        return value

    container_assets = {
        "root": "/models",
        "source": container_asset(assets["source"]),
        "baselines": {key: container_asset(value) for key, value in assets["baselines"].items()},
    }
    metadata = {
        "schema": 1,
        **asdict(task),
        "category": image.category,
        "machine": asdict(machine),
        "image": image.tag,
        "inputs": input_targets,
        "assets": container_assets,
        "baselines": list(assets["baselines"]),
        "agent": asdict(config.agent) if config.agent is not None else None,
        "search": {**asdict(config.search), "calibration_pool": "S"},
        "evaluation": asdict(config.evaluation),
        "candidate": {
            "user": "agent",
            "launch": "/workspace/launch.sh",
            "endpoint": "http://127.0.0.1:8000",
            "served_model": "served",
        },
    }
    (environment / "task.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (environment / "task.json").chmod(0o600)
    for source, target in (
        (environment / "task.json", "/root/eval/task.json"),
        (environment / "start.sh", "/opt/eval/start.sh"),
        (environment / "run.md", "/workspace/run.md"),
    ):
        mounts.append({"type": "bind", "source": str(source), "target": target, "read_only": True})
    renderer = Environment(
        loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined, keep_trailing_newline=True
    )
    values = {
        "task": task,
        "pool": pool,
        "config": config,
        "machine": machine,
        "image": image,
        "assets": container_assets,
        "mounts": mounts,
        "gpu_ids": [gpu.uuid for gpu in machine.gpus],
    }
    for template, destination in (
        ("docker-compose.yaml.j2", environment / "docker-compose.yaml"),
        ("task.toml.j2", task_folder / "task.toml"),
        ("search.md", task_folder / "steps/search/instruction.md"),
        ("coding.md", task_folder / "steps/coding/instruction.md"),
        ("run.md", environment / "run.md"),
        ("start.sh", environment / "start.sh"),
        ("search-test.sh", task_folder / "steps/search/tests/test.sh"),
        ("coding-test.sh", task_folder / "steps/coding/tests/test.sh"),
    ):
        destination.write_text(renderer.get_template(template).render(**values))
        if destination.suffix == ".sh":
            destination.chmod(0o755)
        elif destination.suffix == ".md":
            destination.chmod(0o644)
    return task_folder
