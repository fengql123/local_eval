"""Native Harbor environment extensions for scoped inputs and conversation resume."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harbor.models.trial.config import EnvironmentConfig

    from ..config import RunConfig


def job_environment(config: RunConfig) -> EnvironmentConfig:
    """Keep native provider options and select our input-staging extension."""
    from harbor.models.trial.config import EnvironmentConfig

    environment = config.environment
    path = environment.import_path
    if path is None:
        provider = environment.provider
        module = {"docker": "docker", "daytona": "daytona"}[provider]
        path = f"eval.providers.{module}:{provider.title()}Environment"
    kwargs = dict(environment.kwargs)
    if environment.provider == "daytona":
        kwargs.setdefault("auto_snapshot", True)
    return EnvironmentConfig(
        import_path=path,
        kwargs=kwargs,
    )
