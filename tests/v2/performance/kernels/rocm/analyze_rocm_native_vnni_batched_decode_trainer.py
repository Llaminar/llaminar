#!/usr/bin/env python3
"""Generate ROCm NativeVNNI batched small-M decode dispatch artifacts."""

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


MAX_PROJECTIONS = 8


@dataclass(frozen=True)
class BatchedDecodeRow:
    fmt: str
    codebook: int
    shape: str
    n: int
    k: int
    m: int
    projections: int
    total_n: int
    projection_ns: tuple[int, ...]
    codebooks: tuple[int, ...]
    variant: str
    kb: int
    target_waves: int
    min_us: float
    mean_us: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="ROCm batched decode trainer CSV path(s)")
    parser.add_argument("--output", required=True, help="Generated C++ include path")
    parser.add_argument("--summary", default=None, help="Human-readable summary path")
    parser.add_argument(
        "--max-generated-kb",
        type=int,
        default=8,
        help="Maximum KB split allowed in generated batched decode dispatch.",
    )
    parser.add_argument(
        "--min-generated-m",
        type=int,
        default=3,
        help=(
            "Minimum small-M row count eligible for generated batched dispatch. "
            "M=2 is the fixed-depth-1 verifier path and needs a stricter "
            "model-level equivalence gate before overrides are emitted."
        ),
    )
    return parser.parse_args()


def canonical_label(codebook: int) -> str:
    return CODEBOOK_TO_FORMAT.get(codebook, f"CB{codebook}").split("/")[0]


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


def _parse_plus_ints(path: Path, row_index: int, column: str, value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split("+") if part.strip()]
    if not parts:
        raise SystemExit(f"{path}:{row_index}: empty {column}")
    return tuple(_to_int(path, row_index, column, part) for part in parts)


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
        raise SystemExit(f"{path}:{row_index}: unsupported batched decode variant {variant!r}")
    parsed_kb = int(match.group("kb"))
    parsed_tw = int(match.group("tw"))
    if parsed_kb != kb or parsed_tw != target_waves:
        raise SystemExit(
            f"{path}:{row_index}: variant {variant!r} disagrees with "
            f"kb={kb},target_waves={target_waves}"
        )
    if parsed_kb <= 0 or parsed_tw <= 0:
        raise SystemExit(f"{path}:{row_index}: generated variants need positive kb and target_waves")
    if parsed_kb > max_generated_kb:
        return None
    return parsed_kb, parsed_tw


def load_rows(
    paths: list[str],
    max_generated_kb: int,
    min_generated_m: int,
) -> list[BatchedDecodeRow]:
    if max_generated_kb <= 0:
        raise SystemExit("--max-generated-kb must be positive")
    if min_generated_m < 2 or min_generated_m > 4:
        raise SystemExit("--min-generated-m must be in [2, 4]")

    best_by_key: dict[tuple[int, int, tuple[int, ...], tuple[int, ...]], BatchedDecodeRow | None] = {}
    auto_by_key: dict[tuple[int, int, tuple[int, ...], tuple[int, ...]], float] = {}
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader, start=2):
                if (row.get("backend") or "").strip() != "rocm":
                    continue
                if (row.get("phase") or "").strip() != "batched_decode":
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
                if m < min_generated_m:
                    continue
                k = _to_int(path, row_index, "k", row.get("k", ""))
                projection_ns = _parse_plus_ints(path, row_index, "projection_ns", row.get("projection_ns", ""))
                codebooks = _parse_plus_ints(path, row_index, "codebooks", row.get("codebooks", ""))
                projections = _to_int(path, row_index, "projections", row.get("projections", ""))
                if projections <= 0 or projections > MAX_PROJECTIONS:
                    raise SystemExit(f"{path}:{row_index}: invalid projections={projections}")
                if len(projection_ns) != projections or len(codebooks) != projections:
                    raise SystemExit(
                        f"{path}:{row_index}: projections={projections} but projection_ns/codebooks "
                        f"lengths are {len(projection_ns)}/{len(codebooks)}"
                    )
                if any(item not in CODEBOOK_TO_FORMAT for item in codebooks):
                    raise SystemExit(f"{path}:{row_index}: unknown projection codebook in {codebooks}")

                key = (m, k, projection_ns, codebooks)
                min_us = _to_float(path, row_index, "min_us", row.get("min_us", ""))
                if (row.get("variant") or "").strip().upper() == "AUTO":
                    old_auto = auto_by_key.get(key)
                    if old_auto is None or min_us < old_auto:
                        auto_by_key[key] = min_us
                    best_by_key.setdefault(key, None)
                    continue

                parsed_variant = parse_variant(path, row_index, row, max_generated_kb)
                if parsed_variant is None:
                    continue
                kb, target_waves = parsed_variant
                entry = BatchedDecodeRow(
                    fmt=fmt,
                    codebook=codebook,
                    shape=(row.get("shape") or "").strip(),
                    n=_to_int(path, row_index, "n", row.get("n", "")),
                    k=k,
                    m=m,
                    projections=projections,
                    total_n=_to_int(path, row_index, "total_n", row.get("total_n", "")),
                    projection_ns=projection_ns,
                    codebooks=codebooks,
                    variant=(row.get("variant") or "").strip().upper(),
                    kb=kb,
                    target_waves=target_waves,
                    min_us=min_us,
                    mean_us=_to_float(path, row_index, "mean_us", row.get("mean_us", "0")),
                )
                previous = best_by_key.get(key)
                if previous is None or entry.min_us < previous.min_us:
                    best_by_key[key] = entry

    rows: list[BatchedDecodeRow] = []
    for key, candidate in best_by_key.items():
        if candidate is None:
            continue
        auto_us = auto_by_key.get(key)
        if auto_us is not None and auto_us <= candidate.min_us:
            continue
        rows.append(candidate)
    return sorted(rows, key=lambda item: (item.m, item.k, item.projections, item.projection_ns, item.codebooks))


def _array_literal(values: tuple[int, ...], cast: str | None = None) -> str:
    padded = list(values) + [0] * (MAX_PROJECTIONS - len(values))
    if cast:
        return "{" + ", ".join(f"static_cast<{cast}>({value})" for value in padded) + "}"
    return "{" + ", ".join(str(value) for value in padded) + "}"


def emit_cpp(rows: list[BatchedDecodeRow], output: Path) -> None:
    lines: list[str] = []
    lines.append("// Generated by tests/v2/performance/kernels/rocm/analyze_rocm_native_vnni_batched_decode_trainer.py.")
    lines.append("// Do not edit by hand; regenerate from ROCm NativeVNNI batched decode trainer CSVs.")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <cstddef>")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append("namespace llaminar2::rocm::generated")
    lines.append("{")
    lines.append("inline constexpr int kROCmNativeVNNIBatchedDecodeMaxProjections = 8;")
    lines.append("")
    lines.append("struct ROCmNativeVNNIBatchedDecodeDispatchConfig")
    lines.append("{")
    lines.append("    uint8_t kb = 0;")
    lines.append("    uint8_t target_waves_per_cu = 0;")
    lines.append("};")
    lines.append("")
    lines.append("struct ROCmNativeVNNIBatchedDecodeTuningEntry")
    lines.append("{")
    lines.append("    uint8_t m;")
    lines.append("    uint8_t projections;")
    lines.append("    int k;")
    lines.append("    int total_n;")
    lines.append("    uint8_t codebooks[kROCmNativeVNNIBatchedDecodeMaxProjections];")
    lines.append("    int ns[kROCmNativeVNNIBatchedDecodeMaxProjections];")
    lines.append("    ROCmNativeVNNIBatchedDecodeDispatchConfig config;")
    lines.append("};")
    lines.append("")
    lines.append("inline bool rocmNativeVNNIBatchedDecodeEntryMatches(")
    lines.append("    const ROCmNativeVNNIBatchedDecodeTuningEntry &entry,")
    lines.append("    const uint8_t *codebook_ids,")
    lines.append("    const int *Ns,")
    lines.append("    int projections, int m, int k)")
    lines.append("{")
    lines.append("    if (!codebook_ids || !Ns || projections <= 0 ||")
    lines.append("        projections > kROCmNativeVNNIBatchedDecodeMaxProjections)")
    lines.append("        return false;")
    lines.append("    if (entry.m != static_cast<uint8_t>(m) ||")
    lines.append("        entry.projections != static_cast<uint8_t>(projections) ||")
    lines.append("        entry.k != k)")
    lines.append("        return false;")
    lines.append("    int total_n = 0;")
    lines.append("    for (int i = 0; i < projections; ++i)")
    lines.append("    {")
    lines.append("        total_n += Ns[i];")
    lines.append("        if (entry.ns[i] != Ns[i] || entry.codebooks[i] != codebook_ids[i])")
    lines.append("            return false;")
    lines.append("    }")
    lines.append("    return total_n == entry.total_n;")
    lines.append("}")
    lines.append("")
    lines.append("inline bool selectROCmNativeVNNIBatchedDecodeGenerated(")
    lines.append("    const uint8_t *codebook_ids,")
    lines.append("    const int *Ns,")
    lines.append("    int projections, int m, int k,")
    lines.append("    ROCmNativeVNNIBatchedDecodeDispatchConfig &out)")
    lines.append("{")
    lines.append("    static constexpr ROCmNativeVNNIBatchedDecodeTuningEntry kTable[] = {")
    for row in rows:
        codebook_comment = "+".join(f"CB={codebook} ({canonical_label(codebook)})" for codebook in row.codebooks)
        ns_comment = "+".join(str(value) for value in row.projection_ns)
        lines.append(
            f"        {{{row.m}, {row.projections}, {row.k}, {row.total_n}, "
            f"{_array_literal(row.codebooks, 'uint8_t')}, {_array_literal(row.projection_ns)}, "
            f"{{{row.kb}, {row.target_waves}}}}}, // {codebook_comment} "
            f"Ns={ns_comment} {row.variant} {row.shape} {row.min_us:.3f}us"
        )
    lines.append("    };")
    lines.append("    for (const auto &entry : kTable)")
    lines.append("    {")
    lines.append("        if (rocmNativeVNNIBatchedDecodeEntryMatches(")
    lines.append("                entry, codebook_ids, Ns, projections, m, k))")
    lines.append("        {")
    lines.append("            out = entry.config;")
    lines.append("            return true;")
    lines.append("        }")
    lines.append("    }")
    lines.append("    return false;")
    lines.append("}")
    lines.append("")
    lines.append("} // namespace llaminar2::rocm::generated")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def emit_summary(rows: list[BatchedDecodeRow], summary_path: Path) -> None:
    lines = [
        f"ROCm NativeVNNI batched decode generated entries: {len(rows)}",
        "format,shape,m,k,projections,projection_ns,codebooks,variant,kb,target_waves,min_us",
    ]
    for row in rows:
        lines.append(
            f"{row.fmt},{row.shape},{row.m},{row.k},{row.projections},"
            f"{'+'.join(str(value) for value in row.projection_ns)},"
            f"{'+'.join(str(value) for value in row.codebooks)},"
            f"{row.variant},{row.kb},{row.target_waves},{row.min_us:.3f}"
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    rows = load_rows(args.input, args.max_generated_kb, args.min_generated_m)
    emit_cpp(rows, Path(args.output))
    if args.summary:
        emit_summary(rows, Path(args.summary))
    print(f"Generated {len(rows)} ROCm NativeVNNI batched decode dispatch entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
