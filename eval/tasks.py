"""Render native search, coding and verification tasks using phase-specific images.

Each phase starts a clean container. Search hands off only protected reference
settings; coding hands off only submitted code. The verification task runs its
steps in one shared container: prepare/review/quality/reference-prepare/reference/measure for SOTA,
prepare/review/quality/slo for SLO tasks, which have no reference and no search.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .config import RunConfig, Task, reviewer_env
from .launch import ENDPOINT, SERVED
from .request_policy import pool_of

_CHECKOUT_TEMPLATES = Path(__file__).resolve().parents[1] / "templates"
_PACKAGED_TEMPLATES = Path(__file__).resolve().parent / "_templates"
TEMPLATES = _PACKAGED_TEMPLATES if _PACKAGED_TEMPLATES.is_dir() else _CHECKOUT_TEMPLATES
STAGE_FOLDERS = {"search": "search", "coding": "agent", "verification": "evaluation"}
# Recording runs on the host. Its modules and source tables name the datasets
# behind private rows, so the sandbox runtime payload leaves them out.
HOST_ONLY = frozenset(
    {
        "tables/data.csv",
        "tables/harnesses.json",
        "eval/bench/harnesses.py",
        "eval/bench/live.py",
        "eval/bench/protocols.py",
        "eval/bench/proxy.py",
        "eval/bench/sources.py",
        "eval/s_dataset.py",
    }
)
VERIFICATION_STEPS = {
    "SOTA": ("prepare", "review", "quality", "reference-prepare", "reference", "measure"),
    "SLO": ("prepare", "review", "quality", "slo"),
}


def duration(seconds: int) -> str:
    """Write a budget the way configuration files do, e.g. 1h 30m."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, seconds = divmod(rest, 60)
    parts = [f"{value}{unit}" for value, unit in ((hours, "h"), (minutes, "m"), (seconds, "s"))]
    return " ".join(part for part in parts if not part.startswith("0")) or "0s"


@dataclass(frozen=True)
class ImageSpec:
    category: str
    context: Path
    identity: str
    manifest: dict[str, Any]


def _identity(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def select_image(repo_root: Path, category: str, engines: tuple[str, ...]) -> ImageSpec:
    """Identify authored phase builds and the trusted runtime payload separately."""
    root = repo_root.resolve()
    category_root = Path("images") / category
    image_sources = {
        str(path): hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in (
            category_root / "Dockerfile",
            *(category_root / f"{phase}.Dockerfile" for phase in STAGE_FOLDERS),
        )
    }
    payload_sources = {
        str(path.relative_to(root)): str(path.relative_to(root))
        for pattern in (
            "eval/**/*.py",
            "eval/**/*.txt",
            "eval/**/*.toml",
            "tables/*.csv",
            "tables/*.json",
        )
        for path in root.glob(pattern)
        if path.is_file() and str(path.relative_to(root)) not in HOST_ONLY
    }
    payload_sources.update(
        {name: name for name in ("pyproject.toml", "uv.lock") if (root / name).is_file()}
    )
    for engine in sorted(engines):
        recipe = category_root / "baselines" / engine
        for script in ("install.sh", "launch.sh"):
            source = recipe / script
            # Read now so a missing recipe stops before any acquisition.
            (root / source).read_bytes()
            payload_sources[str(source)] = str(Path("baselines") / category / engine / script)
        for patch in (root / recipe).glob("*.patch"):
            payload_sources[str(patch.relative_to(root))] = str(
                Path("baselines") / category / engine / patch.name
            )
    payload_hashes = {
        target: hashlib.sha256((root / source).read_bytes()).hexdigest()
        for source, target in sorted(payload_sources.items())
    }
    payload_identity = _identity(payload_hashes)
    identity = _identity({"category": category, "sources": image_sources})
    images = {}
    for phase in STAGE_FOLDERS:
        sources = {
            f"{category_root}/{phase}.Dockerfile": image_sources[
                f"{category_root}/{phase}.Dockerfile"
            ]
        }
        sources[f"{category_root}/Dockerfile"] = image_sources[f"{category_root}/Dockerfile"]
        phase_identity = _identity(sources)
        images[phase] = {
            "dockerfile": f"{phase}.Dockerfile",
            "identity": phase_identity,
        }
    manifest = {
        "schema": 5,
        "category": category,
        "context": str(root),
        "identity": identity,
        "sources": image_sources,
        "images": images,
        "payload": {
            "identity": payload_identity,
            "sources": payload_sources,
            "hashes": payload_hashes,
        },
    }
    return ImageSpec(category, root, identity, manifest)


def freeze_image(image: ImageSpec, run_folder: Path) -> ImageSpec:
    """Freeze small source builds and immutable evaluator/recipe files separately."""
    category = image.context / "images" / image.category
    common = (category / "Dockerfile").read_text()
    images = {
        phase: {
            **definition,
            "source": common.rstrip() + "\n\n" + (category / f"{phase}.Dockerfile").read_text(),
        }
        for phase, definition in image.manifest["images"].items()
    }
    payload = run_folder.resolve() / "payload"
    for source, target in image.manifest["payload"]["sources"].items():
        destination = payload / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image.context / source, destination)
    manifest = {
        **image.manifest,
        "images": images,
        "payload": {**image.manifest["payload"], "source": str(payload)},
    }
    return ImageSpec(image.category, image.context, image.identity, manifest)


def render_task(
    task: Task,
    config: RunConfig,
    image: ImageSpec,
    run_folder: Path,
    inputs: dict[str, dict[str, Any]],
    assets: dict[str, Any],
    *,
    include_search: bool = True,
) -> Path:
    """Write portable tasks; providers stage their scoped inputs before agents start.

    SLO tasks have no search phase. All candidate targets come from models.csv;
    search only selects the reference. Coding and verification are rendered
    after that selection determines the task type.
    """
    run_folder = run_folder.resolve()
    slo = task.type == "SLO"
    folders = {
        phase: run_folder / "tasks" / name / task.task_id
        for phase, name in STAGE_FOLDERS.items()
        if (
            (phase == "search" and include_search and bool(assets["baselines"]) and not slo)
            or (phase != "search" and task.type is not None)
        )
    }
    steps = VERIFICATION_STEPS.get(task.type, ())
    for phase, folder in folders.items():
        (folder / "environment").mkdir(parents=True, exist_ok=True)
        if phase == "verification":
            # A fresh render must contain only the resolved task's steps.
            shutil.rmtree(folder / "steps", ignore_errors=True)
            for stage in steps:
                (folder / "steps" / stage / "tests").mkdir(parents=True, exist_ok=True)
        else:
            (folder / "tests").mkdir(exist_ok=True)
            (folder / "tests").chmod(0o700)
            (folder / "solution").mkdir(exist_ok=True)
    coding = run_folder / "tasks/agent" / task.task_id
    pool = pool_of(task.workload)
    targets = {
        "public": "/inputs/public",
        "private": "/var/lib/local-eval/private",
        "quality": "/var/lib/local-eval/quality",
    }
    for phase in (name for name in ("search", "coding") if name in folders and not slo):
        folder = folders[phase] / "reference/metadata"
        folder.mkdir(parents=True, exist_ok=True)
        folder.chmod(0o700)
    target = {"source": assets["checkpoint"]}

    metadata = {
        "schema": 6,
        **{key: value for key, value in asdict(task).items() if value is not None},
        "device_category": image.category,
        "environment": asdict(config.environment),
        "image": image.identity,
        "runtime_identity": image.manifest["payload"]["identity"],
        "inputs": targets,
        "assets": assets,
        "baselines": list(assets["baselines"]),
        "agent": asdict(config.agent) if config.agent is not None else None,
        "search": {
            "agent": asdict(config.search.agent),
            "pool": pool,
            "dataset_split": "private",
        },
        "evaluation": {
            **asdict(config.evaluation),
            "concurrency": config.evaluation.replay_concurrency(task.workload),
        },
        "reference": None if slo else "/var/lib/local-eval/reference/reference.json",
        "target": target,
        "candidate": {
            "submission": "/workspace",
            "launch": "/workspace/server.launch",
            "endpoint": ENDPOINT,
            "served_model": SERVED,
        },
    }
    renderer = Environment(
        loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined, keep_trailing_newline=True
    )
    renderer.filters["duration"] = duration
    values = {
        "task": task,
        "pool": pool,
        "config": config,
        "image": image,
        "assets": assets,
        "oracle_baseline": next(
            (
                name
                for name, checkpoint in assets["baselines"].items()
                if checkpoint == assets["checkpoint"]
            ),
            None,
        ),
        "review_env": {
            "HOME": "/workspace/.harbor/scratch/judge",
            "USER": "judge",
            "LOGNAME": "judge",
            "TMPDIR": "/workspace/runtime/judge/tmp",
            "XDG_CACHE_HOME": "/workspace/runtime/judge/cache",
            **{name: "${" + name + "}" for name in reviewer_env(config.evaluation.agent)},
        },
    }

    def render(template, destination, **context):
        partial = destination.with_suffix(destination.suffix + ".partial")
        partial.write_text(renderer.get_template(template).render(**values, **context))
        if destination.suffix == ".sh":
            partial.chmod(0o755)
        partial.replace(destination)

    def supplied(source, target):
        return {
            "source": str(Path(source).absolute()),
            "target": target,
            "read_only": True,
        }

    def dataset_input(descriptor, target):
        return {
            **descriptor,
            "target": target,
            "read_only": True,
        }

    for phase, folder in folders.items():
        settings = folder / "environment/task.json"
        required = {
            "search": ("private",),
            "coding": ("public",),
            "verification": ("private", "quality"),
        }[phase]
        phase_targets = {key: targets[key] for key in required}
        settings.write_text(
            json.dumps(
                {
                    **metadata,
                    "inputs": phase_targets,
                    "candidate": {
                        **metadata["candidate"],
                        "submission": "/workspace/output/submission"
                        if phase == "coding"
                        else "/workspace",
                    },
                },
                indent=2,
            )
            + "\n"
        )
        settings.chmod(0o600)
        files = [supplied(settings, "/var/lib/local-eval/task.json")]
        files.extend(dataset_input(inputs[key], phase_targets[key]) for key in required)
        phase_image = image.manifest["images"][phase]
        runtime = {
            "phase": phase,
            "category": image.category,
            "image": {"identity": phase_image["identity"]},
            "payload": image.manifest["payload"],
            "gpu_ids": [
                str(value)
                for value in (
                    config.environment.kwargs.get("gpu_ids", [])[0]
                    if config.environment.kwargs.get("gpu_ids")
                    and isinstance(config.environment.kwargs["gpu_ids"][0], list)
                    else config.environment.kwargs.get("gpu_ids", [])
                )
            ],
            "files": files,
            **(
                {"allocation_check": {"cpu_prefixes": config.environment.kwargs["cpu_prefixes"]}}
                if config.environment.kwargs.get("cpu_prefixes")
                else {}
            ),
            **({"target": target} if phase != "search" else {}),
        }
        (folder / "environment/Dockerfile").write_text(phase_image["source"])
        (folder / "environment/.dockerignore").write_text("*\n!Dockerfile\n!.dockerignore\n")
        render("setup.sh", folder / "environment/setup.sh", phase=phase)
        runtime_path = folder / "environment/runtime.json"
        if runtime_path.is_file():
            previous = json.loads(runtime_path.read_text())
            if "allocation_check" in previous:
                runtime["allocation_check"] = previous["allocation_check"]
        runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
        if phase != "verification":
            instruction = f"{phase}.md"
            render(instruction, folder / "instruction.md", inputs=phase_targets)
            render(f"{phase}-test.sh", folder / "tests/test.sh")
            render(f"oracle-{phase}.sh", folder / "solution/solve.sh", inputs=phase_targets)
            render("task.toml.j2", folder / "task.toml", phase=phase, phase_image=phase_image)
        else:
            for stage in steps:
                step = folder / "steps" / stage
                render("verification.md", step / "instruction.md", stage=stage)
                render("verification-test.sh", step / "tests/test.sh", stage=stage)
            render("verification.toml.j2", folder / "task.toml", phase_image=phase_image)
    return coding
