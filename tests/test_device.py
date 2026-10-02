"""Read-only Phase 1 host checks; no GPU or shared server fixtures required."""

from pathlib import Path

import pytest

from eval.device import NVIDIA_QUERY, DeviceError, check_environment, inspect_machine


def test_selected_gpu_resolves_authored_device_dockerfile():
    probes = []

    def probe(argv):
        probes.append(argv)
        return "9, GPU-nine, NVIDIA A100, 81920"

    host = inspect_machine(
        ["9"],
        runner=probe,
        system="Linux",
        architecture="amd64",
        cpu_cores=32,
        memory_bytes=256 * 2**30,
    )
    assert probes == [(*NVIDIA_QUERY, "--id=9")]
    assert host.architecture == "x86_64"
    assert host.memory_gib == 256
    assert [(gpu.index, gpu.uuid, gpu.memory_gib) for gpu in host.gpus] == [(9, "GPU-nine", 80)]
    assert host.dockerfile == Path("images/nvidia-cuda-x86/Dockerfile")


def test_environment_checks_daemon_and_tools_without_starting_containers():
    replies = {
        ("docker", "info", "--format", "{{.ServerVersion}}"): "28.0.0",
        ("docker", "compose", "version", "--short"): "2.39.0",
        ("harbor", "--version"): "harbor 0.23.0",
    }
    probes = []

    def probe(argv):
        probes.append(argv)
        return replies[tuple(argv)]

    environment = check_environment(runner=probe)
    assert environment.docker_version == "28.0.0"
    assert environment.compose_version == "2.39.0"
    assert environment.harbor_version == "harbor 0.23.0"
    assert set(probes) == set(replies)


def test_unknown_gpu_does_not_fall_back_to_an_available_device():
    with pytest.raises(KeyError, match="99"):
        inspect_machine(
            ["99"],
            runner=lambda argv: "0, GPU-zero, NVIDIA L40, 46068",
            system="Linux",
            architecture="x86_64",
            cpu_cores=16,
            memory_bytes=64 * 2**30,
        )


def test_unimplemented_host_fails_before_hardware_probe():
    with pytest.raises(DeviceError, match="unsupported host"):
        inspect_machine(
            runner=lambda argv: pytest.fail("must not probe an unsupported host"),
            system="Darwin",
            architecture="arm64",
        )
