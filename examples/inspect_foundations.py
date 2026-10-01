"""Inspect the actual Phase 1 config -> task -> baseline metadata -> request-policy path.

Usage: uv run python examples/inspect_foundations.py config.example.yaml
No source acquisition, engine installation, or server launch occurs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from eval.baselines import mapping, recipe, validate_catalog
from eval.config import OUTPUT_ALLOWANCE, load_config, select_tasks
from eval.tables import request_fields


def inspect(config_path: str) -> dict:
    config = load_config(config_path)
    validate_catalog()
    selected = select_tasks(config.tasks)
    task = selected[0]
    # This example inspects the NVIDIA catalog entry; it does not claim that
    # this host has a GPU or that setup has prepared a checkpoint.
    category = "nvidia-cuda-x86"
    engine = "freetoken"
    source = recipe(engine, category)
    asset = mapping(category, task.model, engine)
    launch = source.launch(asset.checkpoint, 8000)
    policy = request_fields(task.model, engine)
    return {
        "dataset_source": config.dataset,
        "output_root": str(config.out),
        "selected_tasks": [asdict(item) for item in selected],
        "baseline_status": source.status,
        "engine_code_ref": source.code_ref,
        "install_prefix": str(source.prefix),
        "dockerfile": f"images/{category}/Dockerfile",
        "install_script": str(source.install_script),
        "asset": asdict(asset),
        "launch": asdict(launch),
        "request_policy_status": policy.status,
        "request_fields": policy.fields(task.workload, OUTPUT_ALLOWANCE),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="YAML settings to inspect without setup")
    args = parser.parse_args()
    print(json.dumps(inspect(args.config), indent=2))


if __name__ == "__main__":
    main()
