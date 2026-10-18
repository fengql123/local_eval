"""Fixed candidate entrypoint and trusted baseline script inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ENDPOINT = "http://127.0.0.1:8000"
SERVED = "served"
FIXED_OPTIONS = frozenset({"--model", "--host", "--port", "--served-model-name"})
FIXED_ENV = frozenset({"CHECKPOINT", "PORT", "BASH_ENV", "ENV"})


@dataclass(frozen=True)
class Launch:
    """Execute a script with explicit fixed inputs and adjustable parameters.

    Baseline launch.sh receives CHECKPOINT and searched options. The candidate's
    installed server.launch receives only the suite's read-only checkpoint.
    """

    script: Path
    checkpoint: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    fixed_options: frozenset[str] = frozenset()
    fixed_env: frozenset[str] = frozenset()

    def prepare(self) -> tuple[list[str], dict[str, str]]:
        """Validate external tuning, then apply the caller's fixed checkpoint."""
        if self.script.name == "server.launch":
            if self.checkpoint is None or self.args or self.env:
                raise ValueError("server.launch requires one fixed checkpoint and no tuning inputs")
            return [str(self.script.resolve()), self.checkpoint], {}
        for arg in self.args:
            option = arg.split("=", 1)[0]
            if option in FIXED_OPTIONS | self.fixed_options:
                raise ValueError(f"{option} is fixed by the launch")
        conflicts = (FIXED_ENV | self.fixed_env).intersection(self.env)
        if conflicts:
            raise ValueError(f"environment {', '.join(sorted(conflicts))} is fixed by the launch")
        environment = dict(self.env)
        if self.checkpoint is not None:
            environment["CHECKPOINT"] = self.checkpoint
        return ["/bin/bash", str(self.script.resolve()), *self.args], environment
