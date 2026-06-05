#!/usr/bin/env python3
"""Generate ROCm NativeVNNI prefill tuning artifacts from trainer CSVs.

The ROCm native-VNNI GEMM launcher currently owns its runtime heuristic in the
HIP translation unit. This analyzer gives the ROCm side the same repeatable
"sweep CSV -> generated C++ artifact -> validator" loop used by CUDA, so policy
changes can be trained and reviewed from measured rows instead of hand-copied
overrides.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from native_vnni_codebooks import CODEBOOK_TO_FORMAT, FORMAT_TO_CODEBOOK  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_POLICY_HEADER = REPO_ROOT / "src/v2/utils/PrefillGraphBucketDefaults.h"


@dataclass(frozen=True)
class TuningRow:
    fmt: str
    codebook: int
    shape: str
    category: str
    m: int
    n: int
    k: int
    variant: str
    min_us: float
    mean_us: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="ROCm trainer CSV path(s)")
    parser.add_argument("--output", required=True, help="Generated C++ include/artifact path")
    parser.add_argument("--summary", default=None, help="Human-readable summary path")
    parser.add_argument(
        "--m-policy-header",
        type=Path,
        default=DEFAULT_POLICY_HEADER,
        help="C++ header containing canonical NativeVNNI MTP/prefill bucket policy",
    )
    parser.add_argument(
        "--include-off-policy-m",
        action="store_true",
        help="Include prefill rows whose M value is outside the canonical policy",
    )
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


def load_prefill_m_policy(path: Path) -> list[int]:
    text = path.read_text()
    small = _parse_int_array_from_header(text, "kDefaultNativeVNNISmallMRows", path)
    buckets = _parse_int_array_from_header(text, "kDefaultPrefillGraphBucketSizes", path)
    return sorted({value for value in [*small, *buckets] if value > 0})


def canonical_label(codebook: int) -> str:
    return CODEBOOK_TO_FORMAT.get(codebook, f"CB{codebook}").split("/")[0]


def pack_key(codebook: int, m: int, n: int, k: int) -> int:
    """Pack (codebook, M, N, K) into a stable 64-bit key.

    Layout: [63:56] codebook | [55:40] M | [39:20] K | [19:0] N
    """

    return ((codebook & 0xFF) << 56) | ((m & 0xFFFF) << 40) | ((k & 0xFFFFF) << 20) | (n & 0xFFFFF)


def _to_int(path: Path, row_index: int, column: str, value: str) -> int:
    try:
        return int(value, 0)
    except ValueError as exc:
        raise SystemExit(f"{path}:{row_index}: invalid {column}={value!r}") from exc


def _to_float(path: Path, row_index: int, column: str, value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise SystemExit(f"{path}:{row_index}: invalid {column}={value!r}") from exc


def load_prefill_rows(paths: list[str], m_policy: list[int], include_off_policy_m: bool) -> tuple[list[TuningRow], int]:
    best_by_key: dict[tuple[int, int, int, int], TuningRow] = {}
    skipped_off_policy = 0
    policy_set = set(m_policy)

    for raw_path in paths:
        path = Path(raw_path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader, start=2):
                if (row.get("backend") or "").strip() != "rocm":
                    continue
                if (row.get("phase") or "").strip() != "prefill":
                    continue
                if _to_int(path, row_index, "correctness_pass", row.get("correctness_pass", "")) != 1:
                    continue

                fmt = (row.get("format") or "").strip().upper()
                expected_codebook = FORMAT_TO_CODEBOOK.get(fmt)
                if expected_codebook is None:
                    raise SystemExit(f"{path}:{row_index}: unknown format {fmt!r}")

                codebook = _to_int(path, row_index, "codebook", row.get("codebook", ""))
                if codebook != expected_codebook:
                    raise SystemExit(
                        f"{path}:{row_index}: codebook mismatch for {fmt}: "
                        f"expected {expected_codebook}, found {codebook}"
                    )

                m = _to_int(path, row_index, "m", row.get("m", ""))
                if m <= 1:
                    continue
                if policy_set and not include_off_policy_m and m not in policy_set:
                    skipped_off_policy += 1
                    continue

                is_best = _to_int(path, row_index, "is_best", row.get("is_best", "0"))
                min_us = _to_float(path, row_index, "min_us", row.get("min_us", ""))
                if is_best != 1:
                    continue

                n = _to_int(path, row_index, "n", row.get("n", ""))
                k = _to_int(path, row_index, "k", row.get("k", ""))
                key = (codebook, m, n, k)
                entry = TuningRow(
                    fmt=fmt,
                    codebook=codebook,
                    shape=(row.get("shape") or f"{n}x{k}").strip(),
                    category=(row.get("category") or "").strip(),
                    m=m,
                    n=n,
                    k=k,
                    variant=(row.get("variant") or "").strip(),
                    min_us=min_us,
                    mean_us=_to_float(path, row_index, "mean_us", row.get("mean_us", "0")),
                )

                previous = best_by_key.get(key)
                if previous is None or entry.min_us < previous.min_us:
                    best_by_key[key] = entry

    rows = sorted(best_by_key.values(), key=lambda item: (item.codebook, item.m, item.n, item.k, item.variant))
    return rows, skipped_off_policy


def emit_cpp(rows: list[TuningRow], output: Path) -> None:
    lines: list[str] = []
    lines.append("// Generated by tests/v2/performance/kernels/rocm/analyze_rocm_native_vnni_trainer.py.")
    lines.append("// Do not edit by hand; regenerate from ROCm NativeVNNI trainer CSVs.")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append("namespace llaminar2::rocm::generated")
    lines.append("{")
    lines.append("struct ROCmNativeVNNIPrefillTuningEntry")
    lines.append("{")
    lines.append("    uint64_t key;")
    lines.append("    uint8_t codebook;")
    lines.append("    uint16_t m;")
    lines.append("    uint32_t n;")
    lines.append("    uint32_t k;")
    lines.append("    const char *variant;")
    lines.append("    float min_us;")
    lines.append("};")
    lines.append("")
    lines.append("static constexpr ROCmNativeVNNIPrefillTuningEntry kROCmNativeVNNIPrefillTuning[] =")
    lines.append("{")
    for row in rows:
        key = pack_key(row.codebook, row.m, row.n, row.k)
        label = canonical_label(row.codebook)
        escaped_variant = row.variant.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(
            f"    {{0x{key:016X}ULL, {row.codebook}, {row.m}, {row.n}, {row.k}, "
            f"\"{escaped_variant}\", {row.min_us:.3f}f}}, // CB={row.codebook} ({label}) "
            f"{row.shape}"
        )
    lines.append("};")
    lines.append("")
    lines.append("} // namespace llaminar2::rocm::generated")
    lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def emit_summary(rows: list[TuningRow], skipped_off_policy: int, summary_path: Path | None) -> None:
    lines: list[str] = []
    lines.append(f"ROCm NativeVNNI prefill tuning rows: {len(rows)}")
    if skipped_off_policy:
        lines.append(f"Skipped off-policy M rows: {skipped_off_policy}")
    for row in rows:
        lines.append(
            f"CB={row.codebook} ({canonical_label(row.codebook)}) "
            f"M={row.m} N={row.n} K={row.k} shape={row.shape} "
            f"variant={row.variant} min_us={row.min_us:.3f}"
        )
    text = "\n".join(lines) + "\n"
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(text)
    print(text, end="")


def main() -> int:
    args = parse_args()
    m_policy = load_prefill_m_policy(args.m_policy_header)
    rows, skipped_off_policy = load_prefill_rows(args.input, m_policy, args.include_off_policy_m)
    if not rows:
        raise SystemExit("no ROCm prefill tuning rows found")
    emit_cpp(rows, Path(args.output))
    emit_summary(rows, skipped_off_policy, Path(args.summary) if args.summary else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
