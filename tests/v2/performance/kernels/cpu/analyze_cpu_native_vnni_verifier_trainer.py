#!/usr/bin/env python3
"""Generate CPU NativeVNNI verifier-row dispatch policy from trainer CSVs.

The CPU verifier-row perf harness emits strict numerical metrics for grouped
M=2..4 verifier kernels versus serial M=1 decode GEMVs. This analyzer turns
those rows into a generated C++ include so CPU verifier policy can be refreshed
from data in the same way CUDA and ROCm NativeVNNI dispatch tables are.

The generated table is intentionally conservative: a row is eligible only when
cosine, relative L2, and symmetric KL all pass tight decode-equivalence gates.
Correctness alone is not enough: generated policy rows must also beat serial
decode, otherwise the trainer would quietly promote a verifier path that is
mathematically valid but uneconomical for MTP.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from native_vnni_codebooks import CODEBOOK_TO_FORMAT, FORMAT_TO_CODEBOOK  # noqa: E402


@dataclass(frozen=True)
class VerifierRow:
    fmt: str
    codebook: int
    phase: str
    shape: str
    m: int
    n: int
    k: int
    isa: str
    grouped_min_us: float
    serial_min_us: float
    speedup: float
    cosine: float
    relative_l2: float
    symmetric_kl: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="CPU verifier CSV path(s)")
    parser.add_argument("--output", required=True, help="Generated C++ include path")
    parser.add_argument("--summary", default=None, help="Human-readable summary path")
    parser.add_argument("--min-cosine", type=float, default=0.999999)
    parser.add_argument("--max-relative-l2", type=float, default=1.0e-6)
    parser.add_argument("--max-symmetric-kl", type=float, default=1.0e-8)
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.0,
        help="Reject rows at or below this grouped-vs-serial speedup.",
    )
    parser.add_argument(
        "--require-key",
        action="append",
        default=[],
        metavar="FORMAT:M:N:K",
        help=(
            "Require a generated policy row for this exact production case. "
            "Use for qwen36 profile installs so missing rows fail closed instead "
            "of falling back to the heuristic selector."
        ),
    )
    return parser.parse_args()


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


def _canonical_label(codebook: int) -> str:
    return CODEBOOK_TO_FORMAT.get(codebook, f"CB{codebook}").split("/")[0]


def _policy_for_phase(phase: str, m: int) -> str:
    """Map a measured trainer phase onto a generated verifier policy.

    `verifier_rows` is the production single-projection path. For M=2 it is the
    paired-row kernel; for M=3/4 it is whatever the current generated selector
    chooses. Explicit policy rows let the trainer compare pairwise and wider
    row sharing on the same production entrypoint, including long-K k-tiled
    verifier shapes.
    """

    if phase == "verifier_rows_pairwise_policy":
        return "Pairwise"
    if phase == "verifier_rows_wide_policy":
        return "WideRows"
    if phase == "verifier_rows" and m >= 3:
        return "WideRows"
    return "Pairwise"


def _pack_key(codebook: int, m: int, n: int, k: int) -> int:
    return ((codebook & 0xFF) << 56) | ((m & 0xFF) << 48) | ((k & 0xFFFFFF) << 24) | (n & 0xFFFFFF)


def _parse_required_key(raw: str) -> tuple[int, int, int, int, str]:
    parts = raw.split(":")
    if len(parts) != 4:
        raise SystemExit(
            f"invalid --require-key {raw!r}; expected FORMAT:M:N:K"
        )
    fmt = parts[0].strip().upper()
    codebook = FORMAT_TO_CODEBOOK.get(fmt)
    if codebook is None:
        raise SystemExit(f"invalid --require-key {raw!r}; unknown format {fmt!r}")
    try:
        m = int(parts[1], 0)
        n = int(parts[2], 0)
        k = int(parts[3], 0)
    except ValueError as exc:
        raise SystemExit(
            f"invalid --require-key {raw!r}; M, N, and K must be integers"
        ) from exc
    if m < 2 or m > 4 or n <= 0 or k <= 0:
        raise SystemExit(
            f"invalid --require-key {raw!r}; expected 2 <= M <= 4 and positive N/K"
        )
    return codebook, m, n, k, fmt


def validate_required_keys(rows: list[VerifierRow], required_keys: list[str]) -> None:
    if not required_keys:
        return
    present = {(row.codebook, row.m, row.n, row.k) for row in rows}
    missing: list[str] = []
    for raw in required_keys:
        codebook, m, n, k, fmt = _parse_required_key(raw)
        if (codebook, m, n, k) not in present:
            missing.append(f"{fmt}:M{m}:N{n}:K{k}")
    if missing:
        raise SystemExit(
            "missing required CPU verifier policy row(s): " + ", ".join(missing)
        )


def load_rows(
    paths: list[str],
    min_cosine: float,
    max_relative_l2: float,
    max_symmetric_kl: float,
    min_speedup: float,
) -> list[VerifierRow]:
    best_by_key: dict[tuple[int, int, int, int], VerifierRow] = {}
    skipped_for_metrics = 0
    skipped_for_speedup = 0

    for raw_path in paths:
        path = Path(raw_path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader, start=2):
                if (row.get("backend") or "").strip() != "cpu":
                    continue
                phase = (row.get("phase") or "").strip()
                if not phase.startswith("verifier_rows"):
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

                cosine = _to_float(path, row_index, "cosine", row.get("cosine", ""))
                relative_l2 = _to_float(path, row_index, "relative_l2", row.get("relative_l2", ""))
                symmetric_kl = _to_float(path, row_index, "symmetric_kl", row.get("symmetric_kl", ""))
                if (
                    cosine < min_cosine
                    or relative_l2 > max_relative_l2
                    or symmetric_kl > max_symmetric_kl
                ):
                    skipped_for_metrics += 1
                    continue

                m = _to_int(path, row_index, "m", row.get("m", ""))
                if m < 2 or m > 4:
                    continue
                n = _to_int(path, row_index, "n", row.get("n", ""))
                k = _to_int(path, row_index, "k", row.get("k", ""))
                grouped_min_us = _to_float(path, row_index, "grouped_min_us", row.get("grouped_min_us", ""))
                serial_min_us = _to_float(path, row_index, "serial_min_us", row.get("serial_min_us", ""))
                if grouped_min_us <= 0.0 or serial_min_us <= 0.0:
                    raise SystemExit(f"{path}:{row_index}: timing values must be positive")
                speedup = _to_float(path, row_index, "speedup", row.get("speedup", "0"))
                if speedup <= min_speedup:
                    skipped_for_speedup += 1
                    continue

                entry = VerifierRow(
                    fmt=fmt,
                    codebook=codebook,
                    phase=phase,
                    shape=(row.get("shape") or f"{n}x{k}").strip(),
                    m=m,
                    n=n,
                    k=k,
                    isa=(row.get("isa") or "AUTO").strip().upper(),
                    grouped_min_us=grouped_min_us,
                    serial_min_us=serial_min_us,
                    speedup=speedup,
                    cosine=cosine,
                    relative_l2=relative_l2,
                    symmetric_kl=symmetric_kl,
                )
                key = (codebook, m, n, k)
                previous = best_by_key.get(key)
                if previous is None or entry.grouped_min_us < previous.grouped_min_us:
                    best_by_key[key] = entry

    rows = sorted(best_by_key.values(), key=lambda item: (item.codebook, item.m, item.k, item.n))
    if not rows:
        skipped_reasons = []
        if skipped_for_metrics:
            skipped_reasons.append(f"{skipped_for_metrics} row(s) failed metrics")
        if skipped_for_speedup:
            skipped_reasons.append(f"{skipped_for_speedup} row(s) failed speedup")
        raise SystemExit(
            "no CPU verifier rows survived strict metrics"
            + (f" ({', '.join(skipped_reasons)})" if skipped_reasons else "")
        )
    return rows


def emit_cpp(rows: list[VerifierRow], output: Path) -> None:
    lines: list[str] = []
    lines.append("// Generated by tests/v2/performance/kernels/cpu/analyze_cpu_native_vnni_verifier_trainer.py.")
    lines.append("// Do not edit by hand; regenerate from CPU NativeVNNI verifier trainer CSVs.")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append("namespace llaminar2::cpu::native_vnni::generated")
    lines.append("{")
    lines.append("enum class CPUNativeVNNIVerifierRowsPolicy : uint8_t")
    lines.append("{")
    lines.append("    Pairwise = 0,")
    lines.append("    WideRows = 1,")
    lines.append("};")
    lines.append("")
    lines.append("struct CPUNativeVNNIVerifierRowsPolicyEntry")
    lines.append("{")
    lines.append("    uint64_t key;")
    lines.append("    CPUNativeVNNIVerifierRowsPolicy policy;")
    lines.append("    float measured_speedup;")
    lines.append("};")
    lines.append("")
    lines.append("inline constexpr uint64_t packCPUNativeVNNIVerifierRowsPolicyKey(uint8_t codebook, int m, int n, int k)")
    lines.append("{")
    lines.append("    return (static_cast<uint64_t>(codebook) << 56) |")
    lines.append("           (static_cast<uint64_t>(m & 0xFF) << 48) |")
    lines.append("           (static_cast<uint64_t>(k & 0xFFFFFF) << 24) |")
    lines.append("           static_cast<uint64_t>(n & 0xFFFFFF);")
    lines.append("}")
    lines.append("")
    lines.append("inline bool selectCPUNativeVNNIVerifierRowsGeneratedPolicy(")
    lines.append("    uint8_t codebook, int m, int n, int k,")
    lines.append("    CPUNativeVNNIVerifierRowsPolicy &policy, float *measured_speedup = nullptr)")
    lines.append("{")
    lines.append("    const uint64_t key = packCPUNativeVNNIVerifierRowsPolicyKey(codebook, m, n, k);")
    lines.append("    switch (key)")
    lines.append("    {")

    for row in rows:
        policy = _policy_for_phase(row.phase, row.m)
        label = _canonical_label(row.codebook)
        key = _pack_key(row.codebook, row.m, row.n, row.k)
        lines.append(
            f"    case 0x{key:016x}ULL: // CB={row.codebook} ({label}) {row.shape} "
            f"M={row.m} phase={row.phase} isa={row.isa}"
        )
        lines.append(f"        policy = CPUNativeVNNIVerifierRowsPolicy::{policy};")
        lines.append("        if (measured_speedup)")
        lines.append(f"            *measured_speedup = {row.speedup:.6f}f;")
        lines.append("        return true;")

    lines.append("    default:")
    lines.append("        return false;")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    lines.append("} // namespace llaminar2::cpu::native_vnni::generated")
    lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def emit_summary(rows: list[VerifierRow], summary: Path | None) -> None:
    if summary is None:
        return
    summary.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "CPU NativeVNNI verifier generated-policy summary",
        f"rows: {len(rows)}",
        "",
        "codebook format m n k policy speedup phase",
    ]
    for row in rows:
        lines.append(
            f"{row.codebook} {_canonical_label(row.codebook)} {row.m} {row.n} {row.k} "
            f"{_policy_for_phase(row.phase, row.m)} {row.speedup:.3f} {row.phase}"
        )
    summary.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    rows = load_rows(
        args.input,
        min_cosine=args.min_cosine,
        max_relative_l2=args.max_relative_l2,
        max_symmetric_kl=args.max_symmetric_kl,
        min_speedup=args.min_speedup,
    )
    validate_required_keys(rows, args.require_key)
    emit_cpp(rows, Path(args.output))
    emit_summary(rows, Path(args.summary) if args.summary else None)
    print(f"generated CPU NativeVNNI verifier policy rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
