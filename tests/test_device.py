"""Check required local tools, without probing hardware or starting containers."""

import subprocess

import pytest

from eval.device import check_environment, run_command


def test_native_docker_tools_are_checked_without_gpu_inspection():
    calls = []

    def runner(argv):
        calls.append(argv)
        return {"docker": "docker-ready", "harbor": "harbor-ready"}[argv[0]]

    result = check_environment(runner=runner)
    assert result.docker_version == result.compose_version == "docker-ready"
    assert result.harbor_version == "harbor-ready"
    assert all(argv[0] != "nvidia-smi" for argv in calls)


def test_command_runner_returns_trimmed_output(monkeypatch):
    def run(argv, **kwargs):
        assert kwargs["check"] and kwargs["timeout"] == 15
        return subprocess.CompletedProcess(argv, 0, " version \n", "")

    monkeypatch.setattr(subprocess, "run", run)
    assert run_command(("harbor", "--version")) == "version"


def test_missing_tool_error_propagates():
    def missing(argv):
        raise FileNotFoundError(argv[0])

    with pytest.raises(FileNotFoundError, match="docker"):
        check_environment(runner=missing)


def test_daemon_failure_stops_before_later_checks():
    calls = []

    def broken(argv):
        calls.append(argv)
        raise subprocess.CalledProcessError(1, argv)

    with pytest.raises(subprocess.CalledProcessError):
        check_environment(runner=broken)
    assert len(calls) == 1
