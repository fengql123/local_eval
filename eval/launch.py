"""The launch contract; supplied backend scripts implement this interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SERVED = "served"
FIXED_OPTIONS = frozenset({"--model", "--host", "--port", "--served-model-name"})


@dataclass(frozen=True)
class Launch:
    """Separate suite inputs from tuning, without inspecting a script template.

    Trusted callers can extend the protected option/env sets for a backend.
    External parameter files supply only args/env, never these protected sets.
    """

    script: Path
    checkpoint: str
    port: int
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    fixed_options: frozenset[str] = frozenset()
    fixed_env: frozenset[str] = frozenset()

    def prepare(self) -> tuple[list[str], dict[str, str]]:
        """Validate tuning and return argv/environment, without reading the script.

        Run immediately before startup so changed parameters are checked too.
        The script receives CHECKPOINT/PORT and exposes model name `served` on
        127.0.0.1; all remaining arguments and environment are backend tuning.
        """
        for arg in self.args:
            option = arg.split("=", 1)[0]
            if option in FIXED_OPTIONS | self.fixed_options:
                raise ValueError(f"{option} is fixed by the launch")
        conflicts = {"CHECKPOINT", "PORT", *self.fixed_env}.intersection(self.env)
        if conflicts:
            raise ValueError(f"environment {', '.join(sorted(conflicts))} is fixed by the launch")
        return ["bash", str(self.script.resolve()), *self.args], {
            **self.env,
            "CHECKPOINT": self.checkpoint,
            "PORT": str(self.port),
        }
