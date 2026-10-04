"""The launch contract; supplied backend scripts implement this interface."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ENDPOINT = "http://127.0.0.1:8000"
SERVED = "served"
FIXED_OPTIONS = frozenset({"--model", "--host", "--port", "--served-model-name"})
FIXED_ENV = frozenset({"CHECKPOINT", "PORT", "BASH_ENV", "ENV"})


@dataclass(frozen=True)
class Launch:
    """Run a completed script, or fill its reserved CHECKPOINT input.

    A script using $CHECKPOINT without assigning its own value is a template.
    Complete scripts own their weights and optimized defaults.
    Trusted callers can extend the protected option/env sets for a backend.
    External parameter files supply only args/env, never these protected sets.
    """

    script: Path
    checkpoint: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    fixed_options: frozenset[str] = frozenset()
    fixed_env: frozenset[str] = frozenset()

    @property
    def needs_checkpoint(self) -> bool:
        """Detect the reserved input, excluding comments and filled assignments."""
        lines = [
            line
            for line in self.script.read_text().splitlines()
            if not line.lstrip().startswith("#")
        ]
        source = "\n".join(lines)
        input_pattern = r"\$(?:CHECKPOINT\b|\{CHECKPOINT(?:\}|[:?+\-=]))"
        source = re.sub(
            r"\$\{CHECKPOINT:?-([^}]+)\}",
            lambda match: match.group(1) if match.group(1).strip("\"'").strip() else match.group(0),
            source,
        )
        if not re.search(input_pattern, source):
            return False
        for line in source.splitlines():
            assignment = re.match(
                r"\s*(?:export\s+)?CHECKPOINT=(?:\"([^\"]*)\"|'([^']*)'|([^\s;#]*))", line
            )
            if assignment:
                value = next(value for value in assignment.groups() if value is not None)
                if value and not re.search(input_pattern, value):
                    return False
        return True

    def prepare(self) -> tuple[list[str], dict[str, str]]:
        """Validate tuning and supply weights only when the script leaves them open.

        Run immediately before startup so changed parameters are checked too.
        Every script serves `served` at ENDPOINT; the port is never substituted.
        """
        for arg in self.args:
            option = arg.split("=", 1)[0]
            if option in FIXED_OPTIONS | self.fixed_options:
                raise ValueError(f"{option} is fixed by the launch")
        conflicts = (FIXED_ENV | self.fixed_env).intersection(self.env)
        if conflicts:
            raise ValueError(f"environment {', '.join(sorted(conflicts))} is fixed by the launch")
        environment = dict(self.env)
        if self.needs_checkpoint:
            if self.checkpoint is None:
                raise ValueError(f"{self.script} requires a mapped CHECKPOINT")
            environment["CHECKPOINT"] = self.checkpoint
        return ["/bin/bash", str(self.script.resolve()), *self.args], environment
