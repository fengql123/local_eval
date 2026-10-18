"""Native rebuilds preserve progress; executable and constant changes do not."""

import hashlib
import subprocess

from elftools.elf.elffile import ELFFile

from eval.verify.fingerprint import file_digest
from eval.verify.run import installed_identity


def build(folder, code):
    folder.mkdir()
    (folder / "unit.c").write_text(code)
    subprocess.run(
        ["cc", "-shared", "-fPIC", "-g", "-Wl,--build-id", "unit.c", "-o", "unit.so"],
        cwd=folder,
        check=True,
        capture_output=True,
    )
    return folder / "unit.so"


def test_same_native_program_built_in_different_directories_has_same_identity(tmp_path):
    one = build(tmp_path / "one", "int answer(void) { return 42; }")
    two = build(tmp_path / "two", "int answer(void) { return 42; }")
    assert hashlib.sha256(one.read_bytes()).digest() != hashlib.sha256(two.read_bytes()).digest()
    assert file_digest(one) == file_digest(two)


def test_build_log_changes_preserve_installed_identity(tmp_path):
    (tmp_path / "server.py").write_text("print('serving')")
    original = installed_identity(tmp_path)
    (tmp_path / ".ninja_log").write_text("different build timings")
    (tmp_path / ".ninja_deps").write_bytes(b"different compiler dependency timestamps")
    assert installed_identity(tmp_path) == original


def test_native_code_changes_invalidate_identity(tmp_path):
    one = build(tmp_path / "one", "int answer(void) { return 42; }")
    two = build(tmp_path / "two", "int answer(void) { return 43; }")
    assert file_digest(one) != file_digest(two)
    with one.open("rb") as stream:
        elf = ELFFile(stream)
        covered = [
            (0, elf.structs.Elf_Ehdr.sizeof()),
            (elf["e_phoff"], elf["e_phoff"] + elf["e_phnum"] * elf["e_phentsize"]),
        ]
        covered.extend(
            (s["sh_offset"], s["sh_offset"] + s["sh_size"])
            for s in elf.iter_sections()
            if s["sh_type"] != "SHT_NOBITS"
        )
        gaps = [
            offset
            for segment in elf.iter_segments()
            if segment["p_type"] == "PT_LOAD"
            for offset in range(segment["p_offset"], segment["p_offset"] + segment["p_filesz"])
            if not any(left <= offset < right for left, right in covered)
        ]
    assert gaps
    original = file_digest(one)
    with one.open("r+b") as stream:
        stream.seek(gaps[0])
        value = stream.read(1)
        stream.seek(gaps[0])
        stream.write(bytes([value[0] ^ 1]))
    assert file_digest(one) != original


def test_native_readonly_data_changes_invalidate_identity(tmp_path):
    one = build(tmp_path / "one", 'const char *answer(void) { return "original"; }')
    two = build(tmp_path / "two", 'const char *answer(void) { return "modified"; }')
    assert file_digest(one) != file_digest(two)
