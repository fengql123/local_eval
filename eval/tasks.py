"""Render native search, coding and verification tasks using one category image.

Each phase starts a clean container. Search hands off only protected reference
settings; coding hands off only submitted code. The verification task runs its
root/judge/root steps in one shared container.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .config import RunConfig, Task, reviewer_env
from .device import Machine
from .launch import ENDPOINT, SERVED
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
        Path("images/prepare.sh"),
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
            "eval/**/*.txt",
            "eval/**/*.toml",
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
    """Write three native tasks; return the coding path used by setup and resume."""
    run_folder = run_folder.resolve()
    folders = {
        phase: run_folder / phase / task.task_id for phase in ("search", "coding", "verification")
    }
    for phase, folder in folders.items():
        (folder / "environment").mkdir(parents=True, exist_ok=False)
        if phase != "verification":
            (folder / "tests").mkdir()
            (folder / "solution").mkdir()
        else:
            for stage in ("prepare", "review", "measure"):
                (folder / "steps" / stage / "tests").mkdir(parents=True)
            (folder / "steps/prepare/workdir").mkdir()
    search, coding, verification = (
        folders[phase] for phase in ("search", "coding", "verification")
    )
    pool = pool_of(task.workload)
    targets = {
        "public": f"/workspace/data/{pool}",
        "calibration": "/home/search/data/S",
        "private": "/root/eval/private",
        "quality": "/root/eval/quality",
    }
    if pool != "S":
        targets["public_s"] = "/workspace/data/S"

    def bind(source: Path | str, target: str, *, read_only: bool = True) -> dict:
        # Keep the future selected-weights symlink's name in generated Compose.
        return {
            "type": "bind",
            "source": str(Path(source).absolute()),
            "target": target,
            "read_only": read_only,
        }

    metadata_folders = {}
    for phase in ("search", "coding"):
        folder = folders[phase] / "reference/metadata"
        folder.mkdir(parents=True)
        folder.chmod(0o700)
        metadata_folders[phase] = folder
    public_reference = metadata_folders["coding"] / "public.json"
    public_reference.write_text("{}\n")
    public_reference.chmod(0o644)

    def container_asset(value, target):
        value = dict(value)
        if value["path"] is not None:
            source = Path(value["path"]).resolve()
            checkpoint = Path(value["checkpoint"]).resolve().relative_to(source)
            value.update(path=target, checkpoint=str(Path(target) / checkpoint))
        return value

    source = container_asset(assets["source"], "/models/source")
    baselines = {
        name: container_asset(
            value,
            "/models/source"
            if value["path"] == assets["source"]["path"]
            else f"/models/baselines/{name}",
        )
        for name, value in assets["baselines"].items()
    }
    container_assets = {"root": "/models", "source": source, "baselines": baselines}
    common = []
    if assets["source"]["path"] is not None:
        common.append(bind(assets["source"]["path"], "/models/source"))
    common.extend(
        (
            bind(coding / "environment/task.json", "/root/eval/task.json"),
            bind(coding / "environment/start.sh", "/opt/eval/start.sh"),
        )
    )
    search_mounts = [
        *common,
        bind(inputs["calibration"], targets["calibration"]),
        bind(metadata_folders["search"], "/root/eval/reference", read_only=False),
        *(
            bind(value["path"], baselines[name]["path"])
            for name, value in assets["baselines"].items()
            if value["path"] is not None and value["path"] != assets["source"]["path"]
        ),
    ]
    reference = [
        bind(metadata_folders["coding"], "/root/eval/reference"),
        bind(coding / "reference/weights", "/models/reference"),
    ]
    coding_mounts = [
        *common,
        *reference,
        bind(public_reference, "/workspace/reference.json"),
        bind(coding / "environment/run.md", "/workspace/run.md"),
        *(bind(inputs[key], targets[key]) for key in ("public", "public_s") if key in targets),
    ]
    verification_mounts = [
        *common,
        *reference,
        *(bind(inputs[key], targets[key]) for key in ("private", "quality")),
    ]
    metadata = {
        "schema": 1,
        **asdict(task),
        "category": image.category,
        "machine": asdict(machine),
        "image": image.tag,
        "inputs": targets,
        "assets": container_assets,
        "baselines": list(baselines),
        "agent": asdict(config.agent) if config.agent is not None else None,
        "search": {**asdict(config.search), "calibration_pool": "S"},
        "evaluation": asdict(config.evaluation),
        "reference": "/root/eval/reference/reference.json",
        "public_reference": "/root/eval/reference/public.json",
        "candidate": {
            "submission": "/workspace/submission",
            "launch": "/workspace/submission/launch.sh",
            "weights": "/workspace/weights",
            "runtime": "/workspace/runtime",
            "endpoint": ENDPOINT,
            "served_model": SERVED,
        },
    }
    settings = coding / "environment/task.json"
    settings.write_text(json.dumps(metadata, indent=2) + "\n")
    settings.chmod(0o600)
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
        "verification_mounts": verification_mounts,
        "gpu_ids": [gpu.uuid for gpu in machine.gpus],
        "review_env": {
            "HOME": "/home/judge",
            "USER": "judge",
            "LOGNAME": "judge",
            "TMPDIR": "/home/judge/tmp",
            "XDG_CACHE_HOME": "/home/judge/.cache",
            **{
                name: "${" + name + "}"
                for name in reviewer_env(config.evaluation.agent)
                if name in os.environ
            },
        },
    }

    def render(template, destination, **context):
        destination.write_text(renderer.get_template(template).render(**values, **context))
        if destination.suffix == ".sh":
            destination.chmod(0o755)

    render("start.sh", coding / "environment/start.sh")
    render("run.md", coding / "environment/run.md")
    for phase, mounts in (("search", search_mounts), ("coding", coding_mounts)):
        folder = folders[phase]
        render("task.toml.j2", folder / "task.toml", phase=phase)
        render(
            "docker-compose.yaml.j2",
            folder / "environment/docker-compose.yaml",
            phase=phase,
            mounts=mounts,
        )
        render(f"{phase}.md", folder / "instruction.md")
        render(f"{phase}-test.sh", folder / "tests/test.sh")
        render(f"oracle-{phase}.sh", folder / "solution/solve.sh")
    render("verification.toml.j2", verification / "task.toml")
    render("verification-compose.yaml.j2", verification / "environment/docker-compose.yaml")
    for stage in ("prepare", "review", "measure"):
        step = verification / "steps" / stage
        render("verification.md", step / "instruction.md", stage=stage)
        render("verification-test.sh", step / "tests/test.sh", stage=stage)
    return coding
