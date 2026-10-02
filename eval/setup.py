"""Prepare run-owned HF views and assets, then render one Harbor task per selection.

The supplied dataset is trusted. This module maps its existing columns; it does
not audit records, build images, or launch containers.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import Dataset, load_dataset

from .baselines import mappings, needed
from .config import CALIBRATION_SUBSET, QUALITY_SUBSET, SEED, RunConfig, select_tasks
from .config import model as get_model
from .device import Machine, inspect_machine
from .models import ModelAsset, prepare_asset
from .request_policy import pool_of
from .tasks import ImageSpec, render_task, select_image

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PreparedRun:
    task_paths: tuple[Path, ...]
    machine: Machine
    image: ImageSpec
    inputs: dict
    assets: dict


def _asset(asset: ModelAsset) -> dict:
    metadata = asdict(asset)
    metadata["path"] = str(asset.path) if asset.path is not None else None
    return metadata


def _save_view(view: Dataset, destination: Path, *, readable: bool = False) -> None:
    """Write prepared rows exactly as supplied, without another Arrow conversion."""
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "rows.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in view:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Bind mounts retain host modes. Search calibration is readable behind the
    # search user's private home; private/quality views stay evaluator-only.
    destination.chmod(0o755 if readable else 0o700)
    output.chmod(0o644 if readable else 0o600)


def prepare(
    config: RunConfig,
    run_folder: Path,
    *,
    repo_root: Path = REPO_ROOT,
    machine: Machine | None = None,
) -> PreparedRun:
    """Prepare selected tasks; only a successful complete preparation writes its marker.

    Category Dockerfiles and required installers are checked before HF acquisition.
    A/MA-only selections still acquire private S rows for shared calibration and quality.
    """
    tasks = select_tasks(config.tasks)
    machine = machine or inspect_machine(config.gpus)
    names = tuple(dict.fromkeys(task.model for task in tasks))
    engines = needed(machine.category, names)
    image = select_image(Path(repo_root).resolve(), machine, engines)
    run_folder = Path(run_folder).resolve()
    run_folder.mkdir(parents=True, exist_ok=True)
    if not (run_folder / "prepared.json").exists():
        # A folder-only resume can restart interrupted preparation. Recreate
        # generated views/tasks while keeping downloaded checkpoints.
        for name in ("input", "tasks"):
            destination = run_folder / name
            if destination.exists():
                shutil.rmtree(destination)
    (run_folder / "image.json").write_text(json.dumps(image.manifest, indent=2) + "\n")

    sources: dict[tuple[str, str], Dataset] = {}
    inputs, assets = {}, {}
    for name in names:
        model = get_model(name)
        pools = tuple(dict.fromkeys(pool_of(task.workload) for task in tasks if task.model == name))
        inputs[model.key] = {}
        for pool in dict.fromkeys((*pools, "S")):
            splits = ("public", "private") if pool in pools else ("private",)
            views = {}
            for split in splits:
                source_key = (pool, split)
                if source_key not in sources:
                    sources[source_key] = load_dataset(config.dataset, pool, split=split)
                source = sources[source_key]
                view = source.filter(
                    lambda row, key=model.key, name=model.name: row["model"] in (key, name),
                    load_from_cache_file=False,
                )
                views[split] = view

            saved = {}
            if pool in pools:
                # The timed/public views preserve the supplied memberships.
                for split, view in views.items():
                    destination = run_folder / "input" / model.key / pool / split
                    _save_view(view, destination, readable=split == "public")
                    saved[split] = str(destination)
            if pool == "S":
                # These independent seeded samples may overlap. Both come from
                # unchanged private S rows and are shared across all workloads.
                for split, limit, seed in (
                    ("calibration", CALIBRATION_SUBSET, SEED),
                    ("quality", QUALITY_SUBSET, SEED + 1),
                ):
                    view = views["private"].shuffle(seed=seed)
                    view = view.select(range(min(limit, len(view))))
                    destination = run_folder / "input" / model.key / "S" / split
                    _save_view(view, destination, readable=split == "calibration")
                    saved[split] = str(destination)
            inputs[model.key][pool] = saved

        model_root = run_folder / "models" / model.key
        model_root.mkdir(parents=True, exist_ok=True)
        model_root.chmod(0o755)
        entries = mappings(machine.category, name)
        source = entries[0].source_model_checkpoint
        source_asset = _asset(prepare_asset(source, model_root / "source"))
        baseline_assets = {
            entry.baseline: source_asset
            if entry.checkpoint == source
            else _asset(ModelAsset(entry.checkpoint, None, entry.checkpoint))
            for entry in entries
        }
        assets[model.key] = {
            "root": str(model_root),
            "source": source_asset,
            "baselines": baseline_assets,
        }

    paths = []
    for task in tasks:
        key = get_model(task.model).key
        task_inputs = {
            **inputs[key][pool_of(task.workload)],
            "calibration": inputs[key]["S"]["calibration"],
            "quality": inputs[key]["S"]["quality"],
        }
        paths.append(
            render_task(task, config, machine, image, run_folder, task_inputs, assets[key])
        )
    prepared = PreparedRun(tuple(paths), machine, image, inputs, assets)
    manifest = {
        "task_paths": [str(path) for path in prepared.task_paths],
        "machine": asdict(machine),
        "image": image.manifest,
        "inputs": inputs,
        "assets": assets,
    }
    (run_folder / "prepared.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return prepared
