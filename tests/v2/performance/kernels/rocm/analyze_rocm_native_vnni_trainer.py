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
    n_tile: int
    m_tile: int
    min_blocks: int
    unroll: int


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


def pack_shape_key(m: int, n: int, k: int) -> int:
    """Pack (M, N, K) into a stable 64-bit key for one codebook table."""

    return ((m & 0xFFFFFF) << 40) | ((k & 0xFFFFF) << 20) | (n & 0xFFFFF)


def parse_variant_config(path: Path, row_index: int, variant: str) -> tuple[int, int, int, int] | None:
    """Parse benchmark variant strings into launcher template parameters.

    Supported spellings:
      - N64/MT32/MB1/U2
      - N64/MT32/MB1
      - N64_M32_MB1_UG2

    Rows whose best variant is Auto do not need a generated exact-match entry.
    """

    value = variant.strip().upper()
    if value == "AUTO":
        return None

    match = re.fullmatch(
        r"N(?P<n>64|128)[/_]M(?:T)?(?P<m>16|32|64)[/_]MB(?P<mb>1|2|3)(?:[/_]U(?:G)?(?P<ug>0|1|2|4))?",
        value,
    )
    if not match:
        raise SystemExit(f"{path}:{row_index}: unsupported ROCm NativeVNNI variant {variant!r}")

    n_tile = int(match.group("n"))
    m_tile = int(match.group("m"))
    min_blocks = int(match.group("mb"))
    unroll = int(match.group("ug") or "4")

    if min_blocks == 3:
        raise SystemExit(
            f"{path}:{row_index}: invalid ROCm NativeVNNI variant {variant!r}; "
            "MB3 is reserved for hand-written heuristic sites and is not part of "
            "the generated trainer search space"
        )

    if n_tile == 128 and m_tile > 32:
        raise SystemExit(
            f"{path}:{row_index}: invalid ROCm NativeVNNI variant {variant!r}; "
            "N128/MT64 is not a real launcher mode and must not appear in trainer winners"
        )

    return n_tile, m_tile, min_blocks, unroll


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

                variant = (row.get("variant") or "").strip()
                parsed_variant = parse_variant_config(path, row_index, variant)
                if parsed_variant is None:
                    continue

                n = _to_int(path, row_index, "n", row.get("n", ""))
                k = _to_int(path, row_index, "k", row.get("k", ""))
                key = (codebook, m, n, k)
                n_tile, m_tile, min_blocks, unroll = parsed_variant
                entry = TuningRow(
                    fmt=fmt,
                    codebook=codebook,
                    shape=(row.get("shape") or f"{n}x{k}").strip(),
                    category=(row.get("category") or "").strip(),
                    m=m,
                    n=n,
                    k=k,
                    variant=variant,
                    min_us=min_us,
                    mean_us=_to_float(path, row_index, "mean_us", row.get("mean_us", "0")),
                    n_tile=n_tile,
                    m_tile=m_tile,
                    min_blocks=min_blocks,
                    unroll=unroll,
                )

                previous = best_by_key.get(key)
                if previous is None or entry.min_us < previous.min_us:
                    best_by_key[key] = entry

    rows = sorted(best_by_key.values(), key=lambda item: (item.codebook, item.m, item.n, item.k, item.variant))
    return rows, skipped_off_policy


def emit_cpp(rows: list[TuningRow], output: Path, m_policy: list[int]) -> None:
    rows_by_codebook: dict[int, list[TuningRow]] = {}
    for row in rows:
        rows_by_codebook.setdefault(row.codebook, []).append(row)
    for codebook_rows in rows_by_codebook.values():
        codebook_rows.sort(key=lambda item: pack_shape_key(item.m, item.n, item.k))

    lines: list[str] = []
    lines.append("// Generated by tests/v2/performance/kernels/rocm/analyze_rocm_native_vnni_trainer.py.")
    lines.append("// Do not edit by hand; regenerate from ROCm NativeVNNI trainer CSVs.")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <cstddef>")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append("namespace llaminar2::rocm::generated")
    lines.append("{")
    lines.append("struct ROCmNativeVNNIPrefillDispatchConfig")
    lines.append("{")
    lines.append("    uint8_t n_tile = 0;")
    lines.append("    uint8_t m_tile = 0;")
    lines.append("    uint8_t min_blocks = 0;")
    lines.append("    uint8_t unroll = 0;")
    lines.append("};")
    lines.append("")
    lines.append("struct ROCmNativeVNNIPrefillTuningEntry")
    lines.append("{")
    lines.append("    uint64_t key;")
    lines.append("    ROCmNativeVNNIPrefillDispatchConfig config;")
    lines.append("};")
    lines.append("")
    lines.append("inline constexpr uint64_t packROCmNativeVNNIPrefillDispatchKey(int m_key, int n, int k)")
    lines.append("{")
    lines.append("    return (static_cast<uint64_t>(m_key & 0xFFFFFF) << 40) |")
    lines.append("           (static_cast<uint64_t>(k & 0xFFFFF) << 20) |")
    lines.append("           static_cast<uint64_t>(n & 0xFFFFF);")
    lines.append("}")
    lines.append("")
    lines.append("inline int selectROCmNativeVNNIPrefillDispatchMKey(int m)")
    lines.append("{")
    lines.append("    static constexpr int kMPolicy[] = {")
    for index in range(0, len(m_policy), 12):
        chunk = m_policy[index:index + 12]
        lines.append("        " + ", ".join(str(value) for value in chunk) + ",")
    lines.append("    };")
    lines.append("    for (int key : kMPolicy)")
    lines.append("    {")
    lines.append("        if (m <= key)")
    lines.append("            return key;")
    lines.append("    }")
    lines.append("    return m;")
    lines.append("}")
    lines.append("")
    lines.append("template <size_t Count>")
    lines.append("inline bool findROCmNativeVNNIPrefillDispatchEntry(")
    lines.append("    const ROCmNativeVNNIPrefillTuningEntry (&table)[Count],")
    lines.append("    uint64_t key,")
    lines.append("    ROCmNativeVNNIPrefillDispatchConfig &out)")
    lines.append("{")
    lines.append("    size_t lo = 0;")
    lines.append("    size_t hi = Count;")
    lines.append("    while (lo < hi)")
    lines.append("    {")
    lines.append("        const size_t mid = lo + ((hi - lo) / 2);")
    lines.append("        const uint64_t candidate = table[mid].key;")
    lines.append("        if (candidate == key)")
    lines.append("        {")
    lines.append("            out = table[mid].config;")
    lines.append("            return true;")
    lines.append("        }")
    lines.append("        if (candidate < key)")
    lines.append("            lo = mid + 1;")
    lines.append("        else")
    lines.append("            hi = mid;")
    lines.append("    }")
    lines.append("    return false;")
    lines.append("}")
    lines.append("")
    lines.append("template <uint8_t CB>")
    lines.append("inline bool selectROCmNativeVNNIPrefillGenerated(")
    lines.append("    int m, int n, int k, ROCmNativeVNNIPrefillDispatchConfig &out)")
    lines.append("{")
    lines.append("    const int m_key = selectROCmNativeVNNIPrefillDispatchMKey(m);")
    lines.append("    const uint64_t key = packROCmNativeVNNIPrefillDispatchKey(m_key, n, k);")
    for codebook in sorted(rows_by_codebook):
        label = canonical_label(codebook)
        codebook_rows = rows_by_codebook[codebook]
        lines.append(f"    if constexpr (CB == {codebook}) {{ // {label}")
        lines.append("        static constexpr ROCmNativeVNNIPrefillTuningEntry kTable[] = {")
        for row in codebook_rows:
            shape_key = pack_shape_key(row.m, row.n, row.k)
            lines.append(
                f"            {{0x{shape_key:016X}ULL, "
                f"{{{row.n_tile}, {row.m_tile}, {row.min_blocks}, {row.unroll}}}}}, "
                f"// CB={row.codebook} ({label}) M={row.m} {row.n}x{row.k} "
                f"{row.variant} {row.shape} {row.min_us:.3f}us"
            )
        lines.append("        };")
        lines.append("        return findROCmNativeVNNIPrefillDispatchEntry(kTable, key, out);")
        lines.append("    }")
    lines.append("    return false;")
    lines.append("}")
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
    emit_cpp(rows, Path(args.output), m_policy)
    emit_summary(rows, skipped_off_policy, Path(args.summary) if args.summary else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
