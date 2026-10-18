"""Spot allocation: immediate success, queued success, exhaustion and cancellation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest
from daytona.common.errors import DaytonaBadRequestError
from tenacity import stop_after_attempt, wait_none

from eval.providers.daytona import DaytonaEnvironment


@pytest.fixture
def allocation():
    environment = object.__new__(DaytonaEnvironment)
    environment.task_env_config = SimpleNamespace(build_timeout_sec=14400)
    environment._sandbox = None
    environment._sandbox_labels = lambda: {}
    client = SimpleNamespace(create=AsyncMock())
    params = SimpleNamespace(spot=True)
    create = DaytonaEnvironment._create_sandbox.retry_with(
        wait=wait_none(), stop=stop_after_attempt(6)
    )
    return environment, client, params, create


def test_available_spot_runner_creates_one_sandbox(allocation):
    environment, client, params, create = allocation
    sandbox = SimpleNamespace(id="spot-sandbox")
    client.create.return_value = sandbox
    asyncio.run(create(environment, params=params, daytona=client))
    assert environment._sandbox is sandbox
    client.create.assert_awaited_once_with(params=params, timeout=14400)
    assert params.spot is True


def test_capacity_wait_outlasts_native_retries_without_creating_duplicates(allocation):
    environment, client, params, create = allocation
    sandbox = SimpleNamespace(id="spot-sandbox")
    client.create.side_effect = [DaytonaBadRequestError("No available GPU spot runners")] * 4 + [
        sandbox
    ]
    asyncio.run(create(environment, params=params, daytona=client))
    assert environment._sandbox is sandbox
    assert client.create.await_args_list == [call(params=params, timeout=14400)] * 5
    assert params.spot is True


def test_exhausted_allocation_budget_reports_capacity_and_keeps_no_sandbox(allocation):
    environment, client, params, create = allocation
    error = DaytonaBadRequestError("No available GPU spot runners")
    client.create.side_effect = error
    with pytest.raises(DaytonaBadRequestError) as raised:
        asyncio.run(create(environment, params=params, daytona=client))
    assert raised.value is error
    assert client.create.await_count == 6 and environment._sandbox is None


def test_cancellation_stops_allocation_without_retrying(allocation):
    environment, client, params, create = allocation
    client.create.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(create(environment, params=params, daytona=client))
    client.create.assert_awaited_once()
    assert environment._sandbox is None
