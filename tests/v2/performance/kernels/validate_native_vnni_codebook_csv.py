#!/usr/bin/env python3
"""Validate NativeVNNI sweep CSV codebook columns against tensor metadata IDs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from native_vnni_codebooks import FORMAT_TO_CODEBOOK  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="+", help="CSV file(s) with format and codebook columns")
    return parser.parse_args()


def validate_file(path: Path) -> tuple[int, int]:
    checked = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"format", "codebook"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path}: missing required column(s): {', '.join(sorted(missing))}")

        for row_index, row in enumerate(reader, start=2):
            fmt = (row.get("format") or "").strip()
            codebook_raw = (row.get("codebook") or "").strip()
            if not fmt or not codebook_raw:
                raise SystemExit(f"{path}:{row_index}: empty format/codebook column")

            expected = FORMAT_TO_CODEBOOK.get(fmt.upper())
            if expected is None:
                raise SystemExit(f"{path}:{row_index}: unknown NativeVNNI format {fmt!r}")

            try:
                actual = int(codebook_raw, 0)
            except ValueError as exc:
                raise SystemExit(f"{path}:{row_index}: invalid codebook {codebook_raw!r}") from exc

            if actual != expected:
                raise SystemExit(
                    f"{path}:{row_index}: codebook mismatch for {fmt}: "
                    f"expected {expected}, found {actual}"
                )
            checked += 1

    return checked, 1


def main() -> int:
    args = parse_args()
    total_rows = 0
    total_files = 0
    for path in args.csv:
        rows, files = validate_file(path)
        total_rows += rows
        total_files += files
    print(f"validated {total_rows} NativeVNNI sweep row(s) across {total_files} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
