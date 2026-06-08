#!/usr/bin/env python3
"""Generate ROCm NativeVNNI decode dispatch artifacts from trainer CSVs.

Run sweeps with LLAMINAR_ROCM_NVNNI_DISABLE_GENERATED=1 when refreshing
checked-in tables so AUTO rows do not benchmark the previous table.
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


@dataclass(frozen=True)
class DecodeRow:
    fmt: str
    codebook: int
    shape: str
    n: int
    k: int
    variant: str
    kb: int
    target_waves: int
    min_us: float
    mean_us: float
    eff_bw_gbs: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="ROCm decode trainer CSV path(s)")
    parser.add_argument("--output", required=True, help="Generated C++ include path")
    parser.add_argument("--summary", default=None, help="Human-readable summary path")
    parser.add_argument(
        "--max-generated-kb",
        type=int,
        default=8,
        help=(
            "Maximum KB split allowed in generated decode dispatch. This must match "
            "the graph-safe native-VNNI small-M verifier cap."
        ),
    )
    return parser.parse_args()


def canonical_label(codebook: int) -> str:
    return CODEBOOK_TO_FORMAT.get(codebook, f"CB{codebook}").split("/")[0]


def pack_shape_key(n: int, k: int) -> int:
    return ((k & 0xFFFFF) << 20) | (n & 0xFFFFF)


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


def parse_variant(
    path: Path,
    row_index: int,
    row: dict[str, str],
    max_generated_kb: int,
) -> tuple[int, int] | None:
    variant = (row.get("variant") or "").strip().upper()
    kb = _to_int(path, row_index, "kb", row.get("kb", ""))
    target_waves = _to_int(path, row_index, "target_waves", row.get("target_waves", ""))
    if variant == "AUTO":
        if kb != 0 or target_waves != 0:
            raise SystemExit(f"{path}:{row_index}: AUTO rows must use kb=0,target_waves=0")
        return None

    match = re.fullmatch(r"KB(?P<kb>\d+)[/_]TW(?P<tw>\d+)", variant)
    if not match:
        raise SystemExit(f"{path}:{row_index}: unsupported decode variant {variant!r}")
    parsed_kb = int(match.group("kb"))
    parsed_tw = int(match.group("tw"))
    if parsed_kb != kb or parsed_tw != target_waves:
        raise SystemExit(
            f"{path}:{row_index}: variant {variant!r} disagrees with "
            f"kb={kb},target_waves={target_waves}"
        )
    if parsed_kb <= 0 or parsed_tw <= 0:
        raise SystemExit(f"{path}:{row_index}: generated decode variants need positive kb and target_waves")
    if parsed_kb > 64:
        raise SystemExit(f"{path}:{row_index}: decode kb={parsed_kb} exceeds kernel cap 64")
    if parsed_kb > max_generated_kb:
        return None
    return parsed_kb, parsed_tw


def load_decode_rows(paths: list[str], max_generated_kb: int) -> list[DecodeRow]:
    if max_generated_kb <= 0:
        raise SystemExit("--max-generated-kb must be positive")

    best_by_key: dict[tuple[int, int, int], DecodeRow] = {}
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader, start=2):
                if (row.get("backend") or "").strip() != "rocm":
                    continue
                if (row.get("phase") or "").strip() != "decode":
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

                parsed_variant = parse_variant(path, row_index, row, max_generated_kb)
                if parsed_variant is None:
                    continue
                kb, target_waves = parsed_variant
                n = _to_int(path, row_index, "n", row.get("n", ""))
                k = _to_int(path, row_index, "k", row.get("k", ""))
                key = (codebook, n, k)
                entry = DecodeRow(
                    fmt=fmt,
                    codebook=codebook,
                    shape=(row.get("shape") or f"{n}x{k}").strip(),
                    n=n,
                    k=k,
                    variant=(row.get("variant") or "").strip().upper(),
                    kb=kb,
                    target_waves=target_waves,
                    min_us=_to_float(path, row_index, "min_us", row.get("min_us", "")),
                    mean_us=_to_float(path, row_index, "mean_us", row.get("mean_us", "0")),
                    eff_bw_gbs=_to_float(path, row_index, "eff_bw_gbs", row.get("eff_bw_gbs", "0")),
                )
                previous = best_by_key.get(key)
                if previous is None or entry.min_us < previous.min_us:
                    best_by_key[key] = entry

    return sorted(best_by_key.values(), key=lambda item: (item.codebook, item.n, item.k, item.variant))


def emit_cpp(rows: list[DecodeRow], output: Path) -> None:
    rows_by_codebook: dict[int, list[DecodeRow]] = {}
    for row in rows:
        rows_by_codebook.setdefault(row.codebook, []).append(row)
    for codebook_rows in rows_by_codebook.values():
        codebook_rows.sort(key=lambda item: pack_shape_key(item.n, item.k))

    lines: list[str] = []
    lines.append("// Generated by tests/v2/performance/kernels/rocm/analyze_rocm_native_vnni_decode_trainer.py.")
    lines.append("// Do not edit by hand; regenerate from ROCm NativeVNNI decode trainer CSVs.")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <cstddef>")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append("namespace llaminar2::rocm::generated")
    lines.append("{")
    lines.append("struct ROCmNativeVNNIDecodeDispatchConfig")
    lines.append("{")
    lines.append("    uint8_t kb = 0;")
    lines.append("    uint8_t target_waves_per_cu = 0;")
    lines.append("};")
    lines.append("")
    lines.append("struct ROCmNativeVNNIDecodeTuningEntry")
    lines.append("{")
    lines.append("    uint64_t key;")
    lines.append("    ROCmNativeVNNIDecodeDispatchConfig config;")
    lines.append("};")
    lines.append("")
    lines.append("inline constexpr uint64_t packROCmNativeVNNIDecodeDispatchKey(int n, int k)")
    lines.append("{")
    lines.append("    return (static_cast<uint64_t>(k & 0xFFFFF) << 20) |")
    lines.append("           static_cast<uint64_t>(n & 0xFFFFF);")
    lines.append("}")
    lines.append("")
    lines.append("template <size_t Count>")
    lines.append("inline bool findROCmNativeVNNIDecodeDispatchEntry(")
    lines.append("    const ROCmNativeVNNIDecodeTuningEntry (&table)[Count],")
    lines.append("    uint64_t key,")
    lines.append("    ROCmNativeVNNIDecodeDispatchConfig &out)")
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
    lines.append("inline bool selectROCmNativeVNNIDecodeGenerated(")
    lines.append("    uint8_t codebook_id, int n, int k, ROCmNativeVNNIDecodeDispatchConfig &out)")
    lines.append("{")
    lines.append("    const uint64_t key = packROCmNativeVNNIDecodeDispatchKey(n, k);")

    for codebook in sorted(rows_by_codebook):
        label = canonical_label(codebook)
        lines.append(f"    if (codebook_id == {codebook}) {{ // CB={codebook} ({label})")
        lines.append("        static constexpr ROCmNativeVNNIDecodeTuningEntry kTable[] = {")
        for row in rows_by_codebook[codebook]:
            lines.append(
                f"            {{0x{pack_shape_key(row.n, row.k):012x}ULL, "
                f"{{{row.kb}, {row.target_waves}}}}}, // CB={codebook} ({label}) "
                f"{row.n}x{row.k} {row.variant} {row.shape} {row.min_us:.3f}us"
            )
        lines.append("        };")
        lines.append("        return findROCmNativeVNNIDecodeDispatchEntry(kTable, key, out);")
        lines.append("    }")

    lines.append("    return false;")
    lines.append("}")
    lines.append("")
    lines.append("} // namespace llaminar2::rocm::generated")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def emit_summary(rows: list[DecodeRow], summary_path: Path) -> None:
    lines = [
        f"ROCm NativeVNNI decode generated entries: {len(rows)}",
        "codebook,format,shape,n,k,variant,kb,target_waves,min_us,eff_bw_gbs",
    ]
    for row in rows:
        lines.append(
            f"{row.codebook},{canonical_label(row.codebook)},{row.shape},"
            f"{row.n},{row.k},{row.variant},{row.kb},{row.target_waves},"
            f"{row.min_us:.3f},{row.eff_bw_gbs:.3f}"
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    rows = load_decode_rows(args.input, args.max_generated_kb)
    if not rows:
        raise SystemExit("no generated ROCm NativeVNNI decode rows; no correct rows fit the generated KB cap")
    output = Path(args.output)
    emit_cpp(rows, output)
    if args.summary:
        emit_summary(rows, Path(args.summary))
    print(f"generated {len(rows)} ROCm NativeVNNI decode dispatch entries -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
