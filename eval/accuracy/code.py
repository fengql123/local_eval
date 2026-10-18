"""Grade Python stdio answers in a restricted process, including inside Docker.

Landlock permits reading only Python/system libraries. Seccomp permits computation
and stdio, with no networking, process creation or execution. No privileges or
nested containers are required. Missing kernel support is an infrastructure error.
"""

import ctypes
import errno
import os
import re
import resource
import signal
import struct
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path


def restrict() -> None:
    """Apply irreversible kernel restrictions before evaluating generated Python."""
    libc = ctypes.CDLL(None, use_errno=True)

    def checked(value):
        if value < 0:
            raise OSError(ctypes.get_errno(), "Landlock isolation unavailable")
        return value

    # Landlock syscall numbers are shared by Linux x86-64 and aarch64.
    abi = checked(libc.syscall(444, 0, 0, 1))
    if abi < 3:
        raise OSError("code grading requires Landlock ABI 3 or newer")
    handled = ctypes.create_string_buffer(struct.pack("Q", (1 << 15) - 1))
    rules = checked(libc.syscall(444, handled, 8, 0))
    for path in {sys.base_prefix, "/usr/lib", "/lib", "/lib64"}:
        if Path(path).exists():
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            allowed = ctypes.create_string_buffer(struct.pack("Qi", 4 | 8, fd))
            checked(libc.syscall(445, rules, 1, allowed, 0))
            os.close(fd)
    checked(libc.prctl(38, 1, 0, 0, 0))  # PR_SET_NO_NEW_PRIVS
    checked(libc.syscall(446, rules, 0))
    os.close(rules)
    if os.getuid() == 0:
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
    for limit, value in (
        (resource.RLIMIT_AS, 1024**3),
        (resource.RLIMIT_CPU, 6),
        (resource.RLIMIT_FSIZE, 1024**2),
        (resource.RLIMIT_NOFILE, 64),
        (resource.RLIMIT_CORE, 0),
    ):
        resource.setrlimit(limit, (value, value))
    seccomp = ctypes.CDLL("libseccomp.so.2")
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    context = seccomp.seccomp_init(0x00050000 | errno.EPERM)
    if not context:
        raise OSError("could not allocate the code-grading syscall filter")
    # Files opened below remain constrained by Landlock. Existing stdio handles
    # contain only the test's input and this process's output, never gold answers.
    calls = """read write readv writev close close_range exit exit_group
        fstat newfstatat stat lstat statx lseek pread64 open openat access faccessat faccessat2
        readlink readlinkat getcwd getdents64 mmap mmap2 mprotect munmap mremap madvise brk
        fcntl ioctl rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack restart_syscall
        futex futex_waitv clock_gettime clock_getres clock_nanosleep nanosleep gettimeofday
        time times getrusage getrlimit prlimit64 sched_yield sched_getaffinity
        getpid getppid gettid getuid geteuid getgid getegid uname getrandom""".split()
    for name in calls:
        number = seccomp.seccomp_syscall_resolve_name(name.encode())
        if number >= 0:
            result = seccomp.seccomp_rule_add(context, 0x7FFF0000, number, 0)
            if result < 0:
                raise OSError(-result, f"could not allow grading syscall {name}")
    result = seccomp.seccomp_load(context)
    seccomp.seccomp_release(context)
    if result < 0:
        raise OSError(-result, "could not activate the code-grading syscall filter")


@lru_cache(maxsize=1)
def available() -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-B", __file__], capture_output=True, text=True, timeout=10, env={}
    )
    if result.returncode:
        raise OSError(f"isolated code grading is unavailable: {result.stderr.strip()}")


def check(prediction: str, answer: dict) -> bool:
    """All trusted cases must pass; stdout comparison ignores whitespace only."""
    available()
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", prediction, re.S)
    code = blocks[-1] if blocks else prediction
    if not answer["tests"]:
        raise ValueError("code quality questions require test cases")
    with tempfile.TemporaryDirectory(prefix="quality-code-") as temporary:
        source = Path(temporary) / "solution.py"
        source.write_text(code)
        command = [sys.executable, "-I", "-B", __file__, str(source)]
        for case in answer["tests"]:
            with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=output,
                    stderr=errors,
                    start_new_session=True,
                    env={},
                )
                try:
                    process.communicate(case["input"].encode(), timeout=8)
                except subprocess.TimeoutExpired:
                    return False
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                if process.returncode:
                    return False
                output.seek(0)
                if (
                    output.read().decode("utf-8", errors="replace").split()
                    != case["output"].split()
                ):
                    return False
    return True


if __name__ == "__main__":
    program = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else "pass"
    restrict()
    exec(compile(program, "solution.py", "exec"), {"__name__": "__main__"})
