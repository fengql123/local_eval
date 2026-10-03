"""Launch a model-specific serving script or attach to an existing endpoint."""

from __future__ import annotations

import asyncio
import os
import signal
from contextlib import suppress
from pathlib import Path

import httpx

from ..launch import SERVED, Launch


async def wait_ready(
    endpoint: str, timeout: float, process: asyncio.subprocess.Process | None = None
) -> None:
    """Wait for GET /v1/models; startup failures and the timeout reach the caller."""
    async with asyncio.timeout(timeout), httpx.AsyncClient(timeout=1, trust_env=False) as client:
        while True:
            if process is not None and process.returncode is not None:
                raise RuntimeError(f"server exited with status {process.returncode}")
            try:
                response = await client.get(f"{endpoint.rstrip('/')}/v1/models")
                if response.is_success:
                    return
            except (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.RemoteProtocolError,
                httpx.TimeoutException,
            ):
                pass
            await asyncio.sleep(0.05)


class Server:
    """Own a launch script's process group, or leave an attached server running.

    Launch validates external args/env before any process or log is opened.
    Scripts require CHECKPOINT and PORT, fix host/model identity, and own defaults.
    Launch.env can supply HOME when switching users; cwd is used as supplied.
    Startup and cleanup belong outside the measured replay window.
    """

    def __init__(
        self,
        launch: Launch | None,
        endpoint: str,
        *,
        cwd: Path | None = None,
        health_timeout: float = 3600,
        log: Path | None = None,
        user: str | None = None,
    ) -> None:
        self.launch = launch
        self.endpoint = endpoint.rstrip("/")
        self.served_model = SERVED
        self.cwd = cwd
        self.health_timeout = health_timeout
        self.log = log
        self.user = user
        self._process = None
        self._logfile = None

    async def __aenter__(self) -> Server:
        ready = False
        try:
            if self.launch is not None:
                command, environment = self.launch.prepare()
                self._logfile = self.log.open("ab") if self.log is not None else None
                self._process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=self.cwd,
                    env={**os.environ, **environment},
                    user=self.user,
                    stdout=self._logfile,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
            await wait_ready(self.endpoint, self.health_timeout, self._process)
            ready = True
            return self
        finally:
            if not ready:
                await self._stop()

    async def __aexit__(self, *_exc) -> None:
        await self._stop()

    async def _stop(self) -> None:
        if self._process is not None:
            with suppress(ProcessLookupError):
                os.killpg(self._process.pid, signal.SIGTERM)
            await asyncio.sleep(0.25)
            with suppress(ProcessLookupError):
                os.killpg(self._process.pid, signal.SIGKILL)
            await self._process.wait()
        if self._logfile is not None:
            self._logfile.close()
