"""Machine descriptions group results by the hardware that was actually timed."""

from eval.verify.machine import cpu_group, machine


def test_fingerprint_covers_hardware_and_limits_but_not_the_run(tmp_path):
    (tmp_path / "proc").mkdir()
    (tmp_path / "proc/cpuinfo").write_text("processor : 0\nmodel name : Fixture CPU 9000\n")
    (tmp_path / "sys/fs/cgroup").mkdir(parents=True)
    (tmp_path / "sys/fs/cgroup/cpu.max").write_text("800000 100000\n")
    task = {
        "device_category": "nvidia-cuda-x86",
        "environment": {"cpus": 8, "memory_mb": 65536},
        "task_id": "one",
    }
    gpu = [{"name": "RTX 4090", "memory_mib": 24564, "driver": "580.1"}]
    first = machine(task, gpus=lambda: gpu, root=tmp_path)
    assert first["cpu"] == "Fixture CPU 9000" and first["gpus"] == gpu
    assert first["cgroup"] == {"cpu_max": "800000 100000", "memory_max": None}
    assert machine({**task, "task_id": "two"}, gpus=lambda: gpu, root=tmp_path) == first
    other = machine(task, gpus=lambda: [{**gpu[0], "name": "RTX 5090"}], root=tmp_path)
    assert other["fingerprint"] != first["fingerprint"]
    assert machine({}, gpus=list, root=tmp_path / "missing")["cpu"] is None


def test_cpu_groups_match_approved_generations_without_absorbing_unknown_models():
    assert cpu_group("AMD EPYC 9554P 64-Core Processor") == "AMD EPYC Zen 4"
    assert cpu_group("AMD EPYC 9654 96-Core Processor") == "AMD EPYC Zen 4"
    assert cpu_group("AMD EPYC 9354 32-Core Processor") == "AMD EPYC Zen 4"
    assert cpu_group("AMD EPYC 9455 48-Core Processor") == "AMD EPYC Zen 5"
    assert cpu_group("AMD EPYC 75F3 32-Core Processor") == "AMD EPYC 75F3 32-Core Processor"
    assert cpu_group(None) == "unknown CPU"
