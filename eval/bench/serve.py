"""Launch a model-specific serving script or attach to an existing endpoint."""

from __future__ import annotations

import asyncio
import os
import signal
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from ..launch import ENDPOINT, SERVED, Launch


class StartupError(RuntimeError):
    """The expected endpoint did not belong to a ready foreground server."""

    def __init__(self, reason: str, endpoint: str, log: Path | None = None) -> None:
        self.endpoint = endpoint
        self.log = log
        super().__init__(f"{reason}; expected {endpoint} ({SERVED}); launch log: {log}")


async def _occupied(endpoint: str) -> bool:
    """Do not accept a leftover listener as the process about to be launched."""
    address = urlsplit(endpoint)
    try:
        _reader, writer = await asyncio.open_connection(address.hostname, address.port)
    except ConnectionRefusedError:
        return False
    writer.close()
    await writer.wait_closed()
    return True


async def wait_ready(
    endpoint: str,
    timeout: float,
    process: asyncio.subprocess.Process | None = None,
    log: Path | None = None,
) -> None:
    """Wait for a live server advertising the fixed model alias."""
    try:
        async with (
            asyncio.timeout(timeout),
            httpx.AsyncClient(timeout=1, trust_env=False) as client,
        ):
            while True:
                if process is not None and process.returncode is not None:
                    raise StartupError(
                        f"server exited with status {process.returncode}", endpoint, log
                    )
                try:
                    response = await client.get(f"{endpoint.rstrip('/')}/v1/models")
                    if response.is_success:
                        try:
                            advertised = [model["id"] for model in response.json()["data"]]
                        except (KeyError, TypeError, ValueError) as error:
                            raise StartupError(
                                "invalid /v1/models response", endpoint, log
                            ) from error
                        if SERVED not in advertised:
                            raise StartupError(f"server advertises {advertised!r}", endpoint, log)
                        if process is not None and process.returncode is not None:
                            raise StartupError(
                                f"server exited with status {process.returncode}", endpoint, log
                            )
                        return
                except (
                    httpx.ConnectError,
                    httpx.ReadError,
                    httpx.RemoteProtocolError,
                    httpx.TimeoutException,
                ):
                    pass
                await asyncio.sleep(0.05)
    except TimeoutError as error:
        raise StartupError(f"server was not ready within {timeout:g}s", endpoint, log) from error


class Server:
    """Own a launch script's process group, or leave an attached server running.

    Launch validates external args/env before any process or log is opened.
    Scripts own their port and completed weights; only templates receive CHECKPOINT.
    The server inherits its caller's identity; cwd is used as supplied.
    Startup and cleanup belong outside the measured replay window.
    """

    def __init__(
        self,
        launch: Launch | None,
        endpoint: str = ENDPOINT,
        *,
        cwd: Path | None = None,
        health_timeout: float = 3600,
        log: Path | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        self.launch = launch
        self.endpoint = endpoint.rstrip("/")
        self.served_model = SERVED
        self.cwd = cwd
        self.health_timeout = health_timeout
        self.log = log
        self.environment = environment
        self._process = None
        self._logfile = None

    async def __aenter__(self) -> Server:
        ready = False
        try:
            prepared = self.launch.prepare() if self.launch is not None else None
            if self.launch is not None:
                if await _occupied(self.endpoint):
                    raise StartupError("endpoint is already occupied", self.endpoint, self.log)
                command, environment = prepared
                self._logfile = self.log.open("ab") if self.log is not None else None
                self._process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=self.cwd,
                    env={
                        **(os.environ if self.environment is None else self.environment),
                        **environment,
                    },
                    stdout=self._logfile,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
            await wait_ready(self.endpoint, self.health_timeout, self._process, self.log)
            ready = True
            return self
        finally:
            if not ready:
                await self._stop()

    async def __aexit__(self, *_exc) -> None:
        await self._stop()

    async def _stop(self) -> None:
        try:
            if self._process is not None:
                with suppress(ProcessLookupError):
                    os.killpg(self._process.pid, signal.SIGTERM)
                await asyncio.sleep(0.25)
                with suppress(ProcessLookupError):
                    os.killpg(self._process.pid, signal.SIGKILL)
                await self._process.wait()
        finally:
            if self._logfile is not None:
                self._logfile.close()
