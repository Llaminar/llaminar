#!/usr/bin/env python3
"""Validate generated NativeVNNI dispatch includes against canonical codebook ids."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from native_vnni_codebooks import CODEBOOK_TO_FORMAT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("include", type=Path, nargs="+", help="Generated C++ include(s)")
    return parser.parse_args()


def _known_aliases(codebook: int) -> set[str]:
    label = CODEBOOK_TO_FORMAT.get(codebook)
    if not label:
        return set()
    return {part.strip() for part in label.split("/") if part.strip()}


def _extract_codebooks(text: str) -> set[int]:
    ids: set[int] = set()
    patterns = [
        r"\bCB\s*==\s*(\d+)\b",
        r"\bCB=(\d+)\b",
        r"\bselectTuning_CB(\d+)\s*\(",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            ids.add(int(match.group(1)))
    return ids


def _validate_labeled_branches(path: Path, text: str) -> None:
    branch_patterns = [
        r"CB\s*==\s*(\d+)\)\s*\{\s*//\s*([A-Za-z0-9_./-]+)",
        r"CB=(\d+)\s*\(([A-Za-z0-9_./-]+)",
        r"CB=(\d+)\s*\(\s*([A-Za-z0-9_./-]+)",
    ]
    for pattern in branch_patterns:
        for match in re.finditer(pattern, text):
            codebook = int(match.group(1))
            label = match.group(2).strip()
            aliases = _known_aliases(codebook)
            if not aliases:
                raise SystemExit(
                    f"{path}: generated dispatch references unknown codebook {codebook} "
                    f"with label {label!r}"
                )
            if label not in aliases:
                raise SystemExit(
                    f"{path}: codebook {codebook} label {label!r} does not match "
                    f"canonical alias set {sorted(aliases)}"
                )


def validate_file(path: Path) -> int:
    if not path.is_file():
        raise SystemExit(f"generated include not found: {path}")

    text = path.read_text()
    codebooks = _extract_codebooks(text)
    if not codebooks:
        raise SystemExit(f"{path}: found no generated codebook dispatch branches")

    known = set(CODEBOOK_TO_FORMAT)
    unknown = sorted(codebooks - known)
    if unknown:
        raise SystemExit(
            f"{path}: generated dispatch references unknown codebook id(s): "
            f"{', '.join(str(value) for value in unknown)}"
        )

    _validate_labeled_branches(path, text)
    return len(codebooks)


def main() -> int:
    args = parse_args()
    total = 0
    for path in args.include:
        total += validate_file(path)
    print(f"validated {total} generated NativeVNNI dispatch codebook reference(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
