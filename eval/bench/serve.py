"""Launch a model-specific serving script or attach to an existing endpoint."""

from __future__ import annotations

import asyncio
import os
import select
import signal
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from ..launch import ENDPOINT, SERVED, Launch

_STOP_TIMEOUT = 300


def process_command(
    command: list[str], environment: dict[str, str], user: int | None = None
) -> tuple[list[str], dict[str, str]]:
    """Apply runtime environment only after dropping a trusted caller's identity."""
    if user is None:
        return command, environment
    command = [
        "/usr/bin/setpriv",
        f"--reuid={user}",
        f"--regid={user}",
        "--clear-groups",
        "--no-new-privs",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--bounding-set=-all",
        "--",
        "/usr/bin/env",
        "-i",
        "--",
        *(f"{name}={value}" for name, value in environment.items()),
        *command,
    ]
    # In particular, LD_PRELOAD must not reach the privileged setpriv process.
    return command, {"PATH": "/usr/bin:/bin"}


class StartupError(RuntimeError):
    """The expected endpoint did not belong to a ready foreground server."""

    def __init__(self, reason: str, endpoint: str, log: Path | None = None) -> None:
        self.endpoint = endpoint
        self.log = log
        super().__init__(f"{reason}; expected {endpoint} ({SERVED}); launch log: {log}")


class ShutdownError(RuntimeError):
    """Owned server workers have not exited; the next server must not start."""


def _group_members(pgid: int) -> list[int]:
    """Owned Linux workers whose pidfds have not reported whole-process exit."""
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != pgid:
                continue
            pid = int(path.parent.name)
            descriptor = os.pidfd_open(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        try:
            poller = select.poll()
            poller.register(descriptor, select.POLLIN)
            # A leader can be Z while another thread still releases GPU memory.
            if not poller.poll(0):
                members.append(pid)
        finally:
            os.close(descriptor)
    return members


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


async def _health_ready(client: httpx.AsyncClient, endpoint: str, log: Path | None) -> bool:
    """Use an optional health route; model listings can precede engine readiness."""
    response = await client.get(f"{endpoint.rstrip('/')}/health")
    if response.status_code in (404, 405):
        return True
    if not response.is_success:
        return False
    try:
        health = response.json()
    except ValueError:
        return True  # Some engines return an empty or plain-text successful response.
    if not isinstance(health, dict):
        return True
    if health.get("status") == "error":
        raise StartupError(health.get("message", "server reported a health error"), endpoint, log)
    waiting = {"loading", "rebuilding", "failed"}
    return health.get("status") not in waiting and health.get("maintenance") not in waiting


async def wait_ready(
    endpoint: str,
    timeout: float,
    process: asyncio.subprocess.Process | None = None,
    log: Path | None = None,
) -> None:
    """Wait for a healthy server advertising the fixed model alias."""
    try:
        async with (
            asyncio.timeout(timeout),
            httpx.AsyncClient(timeout=httpx.Timeout(30, connect=1), trust_env=False) as client,
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
                        if await _health_ready(client, endpoint, log):
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
    The default inherits its caller; verification can select a runtime identity.
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
        user: int | None = None,
    ) -> None:
        self.launch = launch
        self.endpoint = endpoint.rstrip("/")
        self.served_model = SERVED
        self.cwd = cwd
        self.health_timeout = health_timeout
        self.log = log
        self.environment = environment
        self.user = user
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
                command, environment = process_command(
                    command,
                    {
                        **(os.environ if self.environment is None else self.environment),
                        **environment,
                    },
                    self.user,
                )
                self._logfile = self.log.open("ab") if self.log is not None else None
                self._process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=self.cwd,
                    env=environment,
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
                try:
                    async with asyncio.timeout(_STOP_TIMEOUT):
                        await self._process.wait()
                        # GPU workers can still be releasing resources after their
                        # API parent exits. A sent signal is not completed teardown.
                        while _group_members(self._process.pid):
                            await asyncio.sleep(0.05)
                except TimeoutError as error:
                    raise ShutdownError(
                        f"server group {self._process.pid} did not stop within {_STOP_TIMEOUT}s; "
                        f"live workers: {_group_members(self._process.pid)}; launch log: {self.log}"
                    ) from error
        finally:
            if self._logfile is not None:
                self._logfile.close()
