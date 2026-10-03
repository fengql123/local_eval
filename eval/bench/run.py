"""Warm up a serving endpoint, replay prepared rows and report timed evidence.

Search, public development checks and the later verifier share this entry point.
Answer grading and the private quality gate belong to judge.pre_run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

from ..baselines import mapping
from ..config import WORKLOADS
from ..config import model as get_model
from ..launch import Launch
from ..request_policy import RequestPolicy, pool_of, request_fields
from .clients import ChatClient
from .metrics import summarize
from .replay import ReplayError, replay
from .serve import Server


async def measure(
    rows: Iterable[dict],
    workload: str,
    policy: RequestPolicy,
    *,
    server: Server,
    request_timeout: float = 3600,
) -> dict:
    """Own or attach to a server; one warmup stays outside the timed replay.

    Supplied rows may be an HF Dataset or an ordinary iterable of row dictionaries.
    Failed warmup, transport or replay raises; no partial successful score is returned.
    Server ownership, launch user, environment and logs belong to the Server argument.
    """
    rows = list(rows)
    async with server:
        async with ChatClient(
            server.endpoint, policy, served_model=server.served_model, timeout=request_timeout
        ) as client:
            warmup = {
                **await client.request(rows[0]["requests"][0], workload),
                "row_id": rows[0]["id"],
            }
            if not warmup["ok"]:
                raise ReplayError(warmup)
            records = await replay(rows, workload, client)
    return {**summarize(workload, records), "records": records, "warmup": warmup}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="benchmark model key or name")
    parser.add_argument("workload", choices=WORKLOADS)
    parser.add_argument("launch", type=Path, help="baseline or candidate launch.sh path")
    parser.add_argument("--params", type=Path, help="JSON file with external launch args and env")
    args = parser.parse_args(argv)
    selected = get_model(args.model)
    source = Path("data") / pool_of(args.workload) / "rows.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows = [row for row in rows if row["model"] in (selected.key, selected.name)]
    script = args.launch.resolve()
    engine, checkpoint = "candidate", "/models/source"
    if script.parent.parent.name == "baselines":
        engine = script.parent.name
        baseline = mapping(script.parents[2].name, selected.key, engine)
        if baseline.checkpoint != baseline.source_model_checkpoint:
            checkpoint = baseline.checkpoint
    params = json.loads(args.params.read_text()) if args.params else {}
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    launch = Launch(
        script, checkpoint, port, tuple(params.get("args", ())), dict(params.get("env", {}))
    )
    folder = Path("bench") / str(uuid4())
    folder.mkdir(parents=True)
    server = Server(
        launch,
        f"http://127.0.0.1:{port}",
        cwd=script.parent,
        log=folder / "server.log",
    )
    report = asyncio.run(
        measure(
            rows,
            args.workload,
            request_fields(selected.key, engine),
            server=server,
        )
    )
    report.update(
        model=selected.name,
        baseline=engine,
        checkpoint=checkpoint,
        launch=str(script),
        parameters=params,
    )
    output = folder / "report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Measurement report: {output.resolve()}")


if __name__ == "__main__":
    main()
