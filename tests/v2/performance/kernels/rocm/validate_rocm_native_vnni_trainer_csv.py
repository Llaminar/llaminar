#!/usr/bin/env python3
"""Validate ROCm NativeVNNI trainer CSV schema and codebook ids."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from native_vnni_codebooks import FORMAT_TO_CODEBOOK  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_POLICY_HEADER = REPO_ROOT / "src/v2/utils/PrefillGraphBucketDefaults.h"


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
    "decode": {"m", "weight_bytes", "eff_bw_gbs", "speedup_vs_int8", "variant", "kb", "target_waves", "is_best"},
    "batched_decode": {
        "m",
        "projections",
        "total_n",
        "projection_ns",
        "codebooks",
        "variant",
        "kb",
        "target_waves",
        "is_best",
    },
    "prefill": {"category", "m", "variant", "gflops", "is_best"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-alias-group",
        action="store_true",
        help="Require at least one pair of format aliases sharing a codebook",
    )
    parser.add_argument(
        "--require-policy-prefill-m",
        action="store_true",
        help="Require every prefill M row to be part of the canonical NativeVNNI training policy",
    )
    parser.add_argument(
        "--require-prefill-m",
        type=int,
        action="append",
        default=[],
        help="Require at least one prefill row with this M value; may be repeated",
    )
    parser.add_argument(
        "--m-policy-header",
        type=Path,
        default=DEFAULT_POLICY_HEADER,
        help="C++ header containing canonical NativeVNNI MTP/prefill bucket policy",
    )
    parser.add_argument("csv", type=Path, nargs="+")
    return parser.parse_args()


def _parse_int_array_from_header(text: str, symbol: str, path: Path) -> list[int]:
    pattern = rf"{re.escape(symbol)}[^=]*=\s*\{{([^}}]+)\}}"
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if not match:
        raise SystemExit(f"{path}: could not find {symbol}")
    values = [int(value) for value in re.findall(r"-?\d+", match.group(1))]
    if not values:
        raise SystemExit(f"{path}: {symbol} was empty")
    return values


def load_prefill_m_policy(path: Path) -> set[int]:
    text = path.read_text()
    small = _parse_int_array_from_header(text, "kDefaultNativeVNNISmallMRows", path)
    buckets = _parse_int_array_from_header(text, "kDefaultPrefillGraphBucketSizes", path)
    return {value for value in [*small, *buckets] if value > 0}


def _parse_int(path: Path, row_index: int, column: str, value: str) -> int:
    try:
        return int(value, 0)
    except ValueError as exc:
        raise SystemExit(f"{path}:{row_index}: invalid {column}={value!r}") from exc


def validate_file(
    path: Path,
    require_alias_group: bool,
    require_policy_prefill_m: bool,
    m_policy: set[int],
    seen_prefill_m: set[int],
) -> int:
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
                if require_policy_prefill_m and m not in m_policy:
                    raise SystemExit(
                        f"{path}:{row_index}: prefill trainer M={m} is not in the "
                        "canonical NativeVNNI training policy"
                    )
                seen_prefill_m.add(m)
                is_best = _parse_int(path, row_index, "is_best", row.get("is_best", ""))
                if is_best not in (0, 1):
                    raise SystemExit(f"{path}:{row_index}: is_best must be 0 or 1")
            elif phase == "decode":
                m = _parse_int(path, row_index, "m", row.get("m", ""))
                if m <= 0:
                    raise SystemExit(f"{path}:{row_index}: decode M must be > 0")
                kb = _parse_int(path, row_index, "kb", row.get("kb", ""))
                target_waves = _parse_int(path, row_index, "target_waves", row.get("target_waves", ""))
                is_best = _parse_int(path, row_index, "is_best", row.get("is_best", ""))
                if kb < 0:
                    raise SystemExit(f"{path}:{row_index}: decode kb must be >= 0")
                if target_waves < 0:
                    raise SystemExit(f"{path}:{row_index}: decode target_waves must be >= 0")
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
    m_policy = load_prefill_m_policy(args.m_policy_header)
    seen_prefill_m: set[int] = set()
    total = 0
    for path in args.csv:
        total += validate_file(
            path,
            args.require_alias_group,
            args.require_policy_prefill_m,
            m_policy,
            seen_prefill_m,
        )
    missing_m = sorted(set(args.require_prefill_m) - seen_prefill_m)
    if missing_m:
        raise SystemExit(
            "missing required prefill M row(s): "
            + ", ".join(str(value) for value in missing_m)
        )
    print(f"validated {total} ROCm NativeVNNI trainer CSV row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
