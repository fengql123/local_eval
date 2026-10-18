"""Freeze sources, map prepared datasets and render three native Harbor tasks.

Workload sources store each graph once with public/private memberships. Quality
is a separate suite. Absolute paths load locally; other strings name HF datasets.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import load_dataset

from .baselines import mappings, needed
from .config import RunConfig, Task, select_tasks
from .config import model as get_model
from .request_policy import matches_workload

if TYPE_CHECKING:
    from .tasks import ImageSpec

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PreparedRun:
    task_paths: tuple[Path, ...]
    image: ImageSpec
    inputs: dict
    assets: dict


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    partial.replace(path)


def _layout(tasks: tuple[Task, ...]) -> dict:
    inputs = {}
    for name in dict.fromkeys(task.model for task in tasks):
        workloads = dict.fromkeys(task.workload for task in tasks if task.model == name)
        inputs[name] = {
            workload: {split: {"source": None} for split in ("public", "private")}
            for workload in workloads
        }
    return inputs


def task_inputs(inputs: dict, task: Task) -> dict:
    """Resolve the model/workload views prepared for this run."""
    return inputs[task.model][task.workload]


def map_inputs(
    dataset: str,
    tasks: tuple[Task, ...],
    destination: Path,
    inputs: dict,
    *,
    cache_dir: Path | None = None,
) -> dict:
    """Select already prepared model/workload splits; do not sample or decode rows.

    Local folders and HF repository IDs expose the same records split, with
    model and public/private workload membership columns.
    Search and verification use the same private view; the full archive and
    memberships for other tasks are never exported to a task.
    """
    inputs = json.loads(json.dumps(inputs))
    destination.mkdir(parents=True, exist_ok=True)
    destination.chmod(0o700)
    records = load_dataset(dataset, "workloads", split="records", cache_dir=cache_dir)
    for name in dict.fromkeys(task.model for task in tasks):
        for workload, descriptors in inputs[name].items():
            for split, descriptor in descriptors.items():
                view = records.filter(
                    lambda row, name=name, workload=workload, split=split: (
                        row["model"] == name and matches_workload(row, workload, split=split)
                    ),
                    load_from_cache_file=False,
                    keep_in_memory=True,
                )
                if split == "private" and workload in ("S-N", "A-N") and len(view) % 8:
                    raise ValueError(f"{name}/{split}/{workload} must contain a multiple of 8 rows")
                view = view.select_columns(["id", "model", "requests"])
                folder = destination / name / split / workload
                view.save_to_disk(str(folder))
                descriptor["source"] = str(folder.resolve())
                descriptor["identity"] = hashlib.sha256(
                    json.dumps(view.to_list(), sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
    return inputs


def plan(config: RunConfig, run_folder: Path, *, repo_root: Path = REPO_ROOT) -> dict:
    """Freeze category, catalog order, runtime payload and source builds."""
    from .tasks import freeze_image, select_image

    run_folder = Path(run_folder).resolve()
    run_folder.mkdir(parents=True, exist_ok=True)
    frozen = run_folder / "plan.json"
    if frozen.exists():
        return json.loads(frozen.read_text())
    category = config.environment.device_category
    selected = select_tasks(config.tasks)
    names = tuple(dict.fromkeys(task.model for task in selected))
    image = freeze_image(select_image(repo_root, category, needed(category, names)), run_folder)
    assets = {}
    for name in names:
        entries = mappings(category, name)
        assets[name] = {
            "checkpoint": get_model(name).checkpoint,
            "baselines": {entry.baseline: entry.checkpoint for entry in entries},
        }
    manifest = {
        "schema": 8,
        "dataset": config.dataset,
        "tasks": [
            {
                **{key: value for key, value in asdict(task).items() if value is not None},
                "device_category": category,
            }
            for task in selected
        ],
        "image": image.manifest,
        "payload": image.manifest["payload"],
        "inputs": _layout(selected),
        "assets": assets,
    }
    _write_json(frozen, manifest)
    return manifest


async def prepare(
    config: RunConfig, run_folder: Path, *, repo_root: Path = REPO_ROOT
) -> PreparedRun:
    """Map data and generate tasks; Harbor builds each environment when it starts."""
    from .quality import map_suite
    from .tasks import ImageSpec, render_task

    run_folder = Path(run_folder).resolve()
    manifest = plan(config, run_folder, repo_root=repo_root)
    mapped = run_folder / "input/inputs.json"
    if mapped.exists():
        manifest["inputs"] = json.loads(mapped.read_text())
    else:
        manifest["inputs"] = map_inputs(
            manifest["dataset"],
            tuple(Task(**task) for task in manifest["tasks"]),
            run_folder / "input/datasets",
            manifest["inputs"],
            cache_dir=run_folder / "input/hf-cache",
        )
        _write_json(mapped, manifest["inputs"])
    for task in manifest["tasks"]:
        if task["workload"] in ("S-N", "A-N"):
            from datasets import load_from_disk

            source = task_inputs(manifest["inputs"], Task(**task))["private"]["source"]
            if len(load_from_disk(source)) % config.evaluation.replay_concurrency(task["workload"]):
                raise ValueError(
                    f"{task['task_id']}/private is not divisible by evaluation.concurrency"
                )
    manifest["quality"] = map_suite(manifest["dataset"], run_folder / "input/quality")
    manifest["dataset_identity"] = hashlib.sha256(
        json.dumps(
            {
                name: {
                    pool: {split: item["identity"] for split, item in splits.items()}
                    for pool, splits in pools.items()
                }
                for name, pools in manifest["inputs"].items()
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    description = manifest["image"]
    image = ImageSpec(
        description["category"], Path(description["context"]), description["identity"], description
    )
    paths = []
    inputs, assets = manifest["inputs"], manifest["assets"]
    for value in manifest["tasks"]:
        task = Task(**value)
        views = {
            **task_inputs(inputs, task),
            "quality": manifest["quality"],
        }
        paths.append(
            render_task(
                task,
                config,
                image,
                run_folder,
                views,
                assets[task.model],
                include_search=config.search.reference is None,
            )
        )
    prepared = PreparedRun(tuple(paths), image, inputs, assets)
    manifest["task_paths"] = [str(path) for path in paths]
    _write_json(run_folder / "prepared.json", manifest)
    return prepared
