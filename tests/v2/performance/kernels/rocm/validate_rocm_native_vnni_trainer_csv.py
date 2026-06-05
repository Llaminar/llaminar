#!/usr/bin/env python3
"""Validate ROCm NativeVNNI trainer CSV schema and codebook ids."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from native_vnni_codebooks import FORMAT_TO_CODEBOOK  # noqa: E402


COMMON_REQUIRED = {
    "backend",
    "phase",
    "format",
    "codebook",
    "shape",
    "n",
    "k",
    "min_us",
    "correctness_pass",
}

PHASE_REQUIRED = {
    "decode": {"weight_bytes", "eff_bw_gbs", "speedup_vs_int8"},
    "prefill": {"category", "m", "variant", "gflops", "is_best"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-alias-group",
        action="store_true",
        help="Require at least one pair of format aliases sharing a codebook",
    )
    parser.add_argument("csv", type=Path, nargs="+")
    return parser.parse_args()


def _parse_int(path: Path, row_index: int, column: str, value: str) -> int:
    try:
        return int(value, 0)
    except ValueError as exc:
        raise SystemExit(f"{path}:{row_index}: invalid {column}={value!r}") from exc


def validate_file(path: Path, require_alias_group: bool) -> int:
    rows = 0
    phases: set[str] = set()
    formats_by_codebook: dict[int, set[str]] = {}

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_common = COMMON_REQUIRED - fieldnames
        if missing_common:
            raise SystemExit(
                f"{path}: missing common required column(s): "
                f"{', '.join(sorted(missing_common))}"
            )

        for row_index, row in enumerate(reader, start=2):
            backend = (row.get("backend") or "").strip()
            phase = (row.get("phase") or "").strip()
            if backend != "rocm":
                raise SystemExit(f"{path}:{row_index}: expected backend=rocm, found {backend!r}")
            if phase not in PHASE_REQUIRED:
                raise SystemExit(f"{path}:{row_index}: unsupported phase {phase!r}")

            missing_phase = PHASE_REQUIRED[phase] - fieldnames
            if missing_phase:
                raise SystemExit(
                    f"{path}: {phase} CSV missing required column(s): "
                    f"{', '.join(sorted(missing_phase))}"
                )

            fmt = (row.get("format") or "").strip()
            expected = FORMAT_TO_CODEBOOK.get(fmt.upper())
            if expected is None:
                raise SystemExit(f"{path}:{row_index}: unknown format {fmt!r}")

            actual = _parse_int(path, row_index, "codebook", row.get("codebook", ""))
            if actual != expected:
                raise SystemExit(
                    f"{path}:{row_index}: codebook mismatch for {fmt}: "
                    f"expected {expected}, found {actual}"
                )

            correctness = _parse_int(
                path,
                row_index,
                "correctness_pass",
                row.get("correctness_pass", ""),
            )
            if correctness not in (0, 1):
                raise SystemExit(
                    f"{path}:{row_index}: correctness_pass must be 0 or 1, found {correctness}"
                )

            if phase == "prefill":
                m = _parse_int(path, row_index, "m", row.get("m", ""))
                if m <= 1:
                    raise SystemExit(f"{path}:{row_index}: prefill trainer row must have M > 1")
                is_best = _parse_int(path, row_index, "is_best", row.get("is_best", ""))
                if is_best not in (0, 1):
                    raise SystemExit(f"{path}:{row_index}: is_best must be 0 or 1")

            phases.add(phase)
            formats_by_codebook.setdefault(actual, set()).add(fmt.upper())
            rows += 1

    if rows == 0:
        raise SystemExit(f"{path}: no trainer rows found")

    alias_groups = [formats for formats in formats_by_codebook.values() if len(formats) > 1]
    if require_alias_group and not alias_groups:
        raise SystemExit(f"{path}: expected at least one alias group sharing a codebook")

    print(
        f"{path}: validated {rows} ROCm trainer row(s), "
        f"phase(s)={','.join(sorted(phases))}"
    )
    return rows


def main() -> int:
    args = parse_args()
    total = 0
    for path in args.csv:
        total += validate_file(path, args.require_alias_group)
    print(f"validated {total} ROCm NativeVNNI trainer CSV row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
