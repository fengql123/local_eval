"""Two code-grading successes and two execution boundaries, using real isolation."""

import json
import socket
import subprocess

import pytest

from eval.accuracy import code, grade


def rule(*cases):
    return json.dumps({"method": "code", "answer": {"tests": list(cases)}})


def test_python_solution_passes_all_stdio_tests_with_normalized_whitespace():
    assert grade(
        rule({"input": "2 3\n", "output": "5\n"}, {"input": "-4 9\n", "output": "5"}),
        "<think>reasoning</think>```python\nprint(sum(map(int, input().split())))\n```",
    )


def test_generated_program_cannot_see_private_files_environment_or_host_network(
    tmp_path, monkeypatch
):
    secret = tmp_path / "private-answer.txt"
    secret.write_text("private")
    monkeypatch.setenv("QUALITY_TEST_SECRET", "private")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        program = f"""import os, pathlib, socket
try:
    pathlib.Path({str(secret)!r}).read_text()
except OSError:
    pass
else:
    raise AssertionError('private file was readable')
assert 'QUALITY_TEST_SECRET' not in os.environ
try:
    pathlib.Path('/tmp/tamper').write_text('x')
except OSError:
    pass
else:
    raise AssertionError('writable filesystem')
try:
    socket.socket()
except OSError:
    pass
else:
    raise AssertionError('network sockets allowed')
print('isolated')
"""
        assert grade(rule({"input": "", "output": "isolated"}), program)
    assert secret.read_text() == "private"


def test_wrong_answer_syntax_error_and_unbounded_program_fail():
    evaluation = rule({"input": "", "output": "5"})
    for text in ("print(4)", "this is not Python!", "while True: pass"):
        assert not grade(evaluation, text)


def test_unavailable_sandbox_is_an_infrastructure_error_without_unsafe_fallback(monkeypatch):
    code.available.cache_clear()
    monkeypatch.setattr(
        code.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 1, "", "namespaces disabled"),
    )
    with pytest.raises(OSError, match="isolated code grading is unavailable"):
        grade(rule({"input": "", "output": "5"}), "print(5)")
    code.available.cache_clear()
