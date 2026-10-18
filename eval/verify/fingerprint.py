"""Hash installed programs without compiler debug data or build bookkeeping."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from elftools.elf.elffile import ELFFile

BUILD_METADATA = {".git", "__pycache__", ".ninja_log", ".ninja_deps"}
DEBUG_SECTIONS = {".comment", ".note.gnu.build-id", ".gnu_debuglink", ".gdb_index"}


def file_digest(path: Path) -> bytes:
    """Retain executable contents, constants, relocation data and loader settings.

    Rebuilding a native extension embeds different debug paths and build IDs.
    Neither is part of its serving implementation. Ordinary files retain a full
    content hash; ELF files retain every other section and their load layout.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        if source.read(4) != b"\x7fELF":
            source.seek(0)
            while chunk := source.read(8 * 1024**2):
                digest.update(chunk)
            return digest.digest()
        source.seek(0)
        elf = ELFFile(source)
        header = tuple(
            elf.header[key]
            for key in (
                "e_ident",
                "e_type",
                "e_machine",
                "e_version",
                "e_entry",
                "e_flags",
                "e_phentsize",
            )
        )
        digest.update(repr((elf.elfclass, elf.little_endian, header)).encode())
        for segment in elf.iter_segments():
            layout = tuple(
                segment[key]
                for key in (
                    "p_type",
                    "p_offset",
                    "p_vaddr",
                    "p_paddr",
                    "p_flags",
                    "p_align",
                    "p_memsz",
                    "p_filesz",
                )
            )
            digest.update(repr(layout).encode())
        covered = [
            (0, elf.structs.Elf_Ehdr.sizeof()),
            (elf["e_phoff"], elf["e_phoff"] + elf["e_phnum"] * elf["e_phentsize"]),
        ]
        for section in elf.iter_sections():
            if section["sh_type"] != "SHT_NOBITS":
                covered.append((section["sh_offset"], section["sh_offset"] + section["sh_size"]))
            debug = section.name in DEBUG_SECTIONS or section.name.startswith(
                (".debug", ".zdebug", ".rela.debug", ".rel.debug")
            )
            # Allocated sections are runtime data, even with a debug-looking name.
            if debug and (not section["sh_flags"] & 2 or section.name == ".note.gnu.build-id"):
                continue
            fields = tuple(
                section[key]
                for key in ("sh_type", "sh_flags", "sh_addr", "sh_size", "sh_link", "sh_info")
            )
            digest.update(repr((section.name, fields)).encode())
            if section["sh_type"] == "SHT_NOBITS":
                continue
            source.seek(section["sh_offset"])
            remaining = section["sh_size"]
            while remaining and (chunk := source.read(min(remaining, 8 * 1024**2))):
                digest.update(chunk)
                remaining -= len(chunk)
        # Sectionless bytes in loadable segments can also be executed or read.
        for segment in elf.iter_segments():
            if segment["p_type"] != "PT_LOAD":
                continue
            cursor = segment["p_offset"]
            end = cursor + segment["p_filesz"]
            for left, right in sorted(covered) + [(end, end)]:
                if right <= cursor:
                    continue
                if left > cursor:
                    source.seek(cursor)
                    remaining = min(left, end) - cursor
                    while remaining and (chunk := source.read(min(remaining, 8 * 1024**2))):
                        digest.update(chunk)
                        remaining -= len(chunk)
                cursor = max(cursor, right)
                if cursor >= end:
                    break
    return digest.digest()


def installed_identity(submission: Path) -> str:
    """Bind saved approvals to reproduced source and installed executable contents."""
    digest = hashlib.sha256()
    roots = [
        path for path in submission.iterdir() if path.name not in (".harbor", "runtime", "output")
    ]
    runtime = Path("/workspace/runtime/candidate")
    if runtime.is_dir():
        roots.extend(path for path in runtime.iterdir() if path.name not in ("cache", "tmp"))
    for root in sorted(roots):
        pending = [root]
        while pending:
            path = pending.pop()
            if path.name in BUILD_METADATA:
                continue
            digest.update(str(path).encode() + b"\0")
            if path.is_symlink():
                digest.update(b"L" + os.readlink(path).encode() + b"\0")
            elif path.is_dir():
                digest.update(b"D")
                pending.extend(
                    sorted(
                        path.iterdir(),
                        reverse=True,
                    )
                )
            elif path.is_file():
                digest.update(b"F" + bytes([path.stat().st_mode & 0o111]) + file_digest(path))
    return digest.hexdigest()
