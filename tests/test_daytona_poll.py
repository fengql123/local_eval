"""Read-only cloud command recovery: two common cases and two failure boundaries."""

import asyncio
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest
from daytona.common.errors import (
    DaytonaBadGatewayError,
    DaytonaBadRequestError,
    DaytonaNotFoundError,
)
from tenacity import stop_after_attempt, wait_none

from eval.providers.daytona import DaytonaEnvironment


@pytest.fixture(params=("get_session_command", "get_session_command_logs"))
def command_read(request):
    environment = object.__new__(DaytonaEnvironment)
    process = AsyncMock()
    environment._sandbox = SimpleNamespace(process=process)
    read = getattr(DaytonaEnvironment, f"_{request.param}_with_retry").retry_with(
        wait=wait_none(), stop=stop_after_attempt(6)
    )
    return (
        partial(read, environment, "session", "command"),
        getattr(process, request.param),
        process,
    )


def test_returns_a_running_command_without_restarting_it(command_read):
    read, api, process = command_read
    result = SimpleNamespace(exit_code=None, stdout="")
    api.return_value = result
    assert asyncio.run(read()) is result
    api.assert_awaited_once_with("session", "command")
    process.create_session.assert_not_awaited()
    process.exec.assert_not_awaited()


def test_recovers_after_more_gateway_errors_than_native_harbor_allows(command_read):
    read, api, process = command_read
    result = SimpleNamespace(exit_code=0, stdout="complete")
    api.side_effect = (
        [DaytonaBadGatewayError("temporary")] * 4
        + [DaytonaBadRequestError("failed to resolve container IP after 3 attempts")]
        + [result]
    )
    assert asyncio.run(read()) is result
    assert api.await_args_list == [call("session", "command")] * 6
    process.exec.assert_not_awaited()


def test_reports_a_gateway_failure_when_the_retry_budget_is_exhausted(command_read):
    read, api, _ = command_read
    error = DaytonaBadGatewayError("unavailable")
    api.side_effect = error
    with pytest.raises(DaytonaBadGatewayError) as raised:
        asyncio.run(read())
    assert raised.value is error
    assert api.await_count == 6


def test_does_not_retry_a_removed_session(command_read):
    read, api, _ = command_read
    for error in (DaytonaNotFoundError("gone"), DaytonaBadRequestError("invalid command")):
        api.reset_mock()
        api.side_effect = error
        with pytest.raises(type(error)):
            asyncio.run(read())
        api.assert_awaited_once_with("session", "command")
