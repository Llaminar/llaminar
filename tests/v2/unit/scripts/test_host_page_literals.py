#!/usr/bin/env python3
"""Forbid hard-coded host page sizes in production sources.

AArch64 hosts run 4, 16 or 64 KiB base pages. A page size written as a
literal (``page_size = 4096``, ``constexpr size_t PAGE_SIZE = 4096``) made
mbind/madvise fail or discard neighbouring bytes on 64 KiB hosts and made
CPU expert admission under-count page-aligned slots. Page-granular owners
read ``hostPageBytes()`` from ``utils/HostPageSize.h`` instead.

The scan looks for a page-named variable, constant or parameter default
initialized with a power-of-two literal of at least 4096. Comments and string
literals are ignored. ``--self-test`` runs the negative controls; every
invocation runs them first.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
import tempfile

SOURCE_SUFFIXES = (".h", ".hpp", ".cpp", ".cu", ".cuh", ".hip", ".inc")
PAGE_LITERAL = re.compile(
    r"\b(?:k?page[_a-z]*|PAGE_SIZE|PAGE_BYTES|kPage[A-Za-z]*)\s*=\s*"
    r"(?:static_cast<[^>]*>\s*\(\s*)?(4096|8192|16384|65536)u?l?\b",
    re.IGNORECASE,
)
COMMENT_OR_STRING = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', re.DOTALL)


def violations(root: pathlib.Path) -> list[str]:
    """Return every hard-coded page-size initialization under ``root/src``."""
    found: list[str] = []
    for path in sorted((root / "src").rglob("*")):
        if not path.is_file() or not path.name.endswith(SOURCE_SUFFIXES):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        code = COMMENT_OR_STRING.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
        for match in PAGE_LITERAL.finditer(code):
            line = code.count("\n", 0, match.start()) + 1
            found.append(f"{path.relative_to(root)}:{line}: hard-coded page size "
                         f"{match.group(1)}; use hostPageBytes()")
    return found


def run_self_test() -> None:
    """Prove the scan finds every literal form and ignores prose and other names."""
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        (root / "src").mkdir()
        (root / "src" / "bad.cpp").write_text(
            "constexpr size_t PAGE_SIZE = 4096;\n"
            "const size_t page_size = 4096;\n"
            "const size_t alignment = static_cast<size_t>(4096);\n"
            "void f(std::size_t page_bytes = 4096u);\n"
            "constexpr std::size_t kPageBytes = 65536u;\n"
            "const auto page_alignment = static_cast<size_t>(16384);\n",
            encoding="utf-8")
        (root / "src" / "good.cpp").write_text(
            "// const size_t page_size = 4096;\n"
            "const char *note = \"page_size = 4096\";\n"
            "size_t transaction_slot_count = 4096;\n"
            "const size_t page_size = hostPageBytes();\n",
            encoding="utf-8")
        lines = sorted(int(item.split(":")[1]) for item in violations(root))
        assert lines == [1, 2, 4, 5, 6], lines


def main() -> int:
    """Run the controls, then scan the repository."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    run_self_test()
    if args.self_test:
        print("host page literal policy self-test: PASS")
        return 0
    if args.repo_root is None:
        parser.error("--repo-root is required unless --self-test is given")
    found = violations(args.repo_root.resolve())
    for item in found:
        print(item, file=sys.stderr)
    print("host page literal policy: " + ("FAIL" if found else "PASS"),
          file=sys.stderr if found else sys.stdout)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
