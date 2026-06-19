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


ASPECT_BUCKETS = (
    ("very_wide", 16.0),
    ("wide", 2.0),
    ("balanced", 0.75),
)

ASPECT_ORDER = ("very_wide", "wide", "balanced", "tall")

ASPECT_CONDITIONS = {
    "very_wide": "aspect_ratio >= 16.0f",
    "wide": "aspect_ratio >= 2.0f",
    "balanced": "aspect_ratio >= 0.75f",
    "tall": "true",
}


@dataclass(frozen=True)
class DecodeRow:
    fmt: str
    codebook: int
    shape: str
    m: int
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
        "--base-include",
        default=None,
        help=(
            "Existing generated include to preserve while layering exact "
            "shape winners from the input CSV. Use this for staged profiles "
            "such as qwen36-moe so a partial refresh does not discard the "
            "broader dispatch table."
        ),
    )
    parser.add_argument(
        "--max-generated-kb",
        type=int,
        default=16,
        help=(
            "Maximum KB split allowed in generated decode dispatch. Keep this aligned "
            "with the strict-parity native-VNNI small-M verifier sweep envelope."
        ),
    )
    return parser.parse_args()


def canonical_label(codebook: int) -> str:
    return CODEBOOK_TO_FORMAT.get(codebook, f"CB{codebook}").split("/")[0]


def aspect_bucket(n: int, k: int) -> str:
    ratio = (float(n) / float(k)) if k > 0 else 0.0
    for name, threshold in ASPECT_BUCKETS:
        if ratio >= threshold:
            return name
    return "tall"


def work_items(row: DecodeRow) -> int:
    return row.n * row.k


def dispatch_candidate(row: DecodeRow) -> tuple[int, int]:
    return row.kb, row.target_waves


def pack_shape_key(m: int, n: int, k: int) -> int:
    """Pack the verifier-row count with the logical GEMV shape.

    ROCm decode now has two distinct regimes: classic single-token GEMV (M=1)
    and MTP verifier catch-up rows (M=2..4).  The kernels are templated on M,
    so a table trained for M=1 can pick a poor split for verifier rows.  Packing
    M into the key keeps those dispatch choices independent while preserving the
    compact binary-search table layout used by the generated C++ include.
    """

    return ((m & 0xFF) << 56) | ((k & 0xFFFFFF) << 28) | (n & 0x0FFFFFFF)


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

    best_by_key: dict[tuple[int, int, int, int], DecodeRow] = {}
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
                m = _to_int(path, row_index, "m", row.get("m", "1"))
                if m <= 0:
                    raise SystemExit(f"{path}:{row_index}: decode m must be positive")
                n = _to_int(path, row_index, "n", row.get("n", ""))
                k = _to_int(path, row_index, "k", row.get("k", ""))
                key = (codebook, m, n, k)
                entry = DecodeRow(
                    fmt=fmt,
                    codebook=codebook,
                    shape=(row.get("shape") or f"{n}x{k}").strip(),
                    m=m,
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

    return sorted(best_by_key.values(), key=lambda item: (item.codebook, item.m, item.n, item.k, item.variant))


def _modal_candidate(rows: list[DecodeRow]) -> tuple[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    best_time: dict[tuple[int, int], float] = {}
    for row in rows:
        candidate = dispatch_candidate(row)
        counts[candidate] = counts.get(candidate, 0) + 1
        best_time[candidate] = min(best_time.get(candidate, row.min_us), row.min_us)
    return max(
        counts,
        key=lambda candidate: (
            counts[candidate],
            -best_time[candidate],
            -candidate[0],
            -candidate[1],
        ),
    ), max(counts.values())


def _candidate_cuts(rows: list[DecodeRow]) -> list[int]:
    works = sorted({work_items(row) for row in rows})
    return [((left + right) // 2) for left, right in zip(works, works[1:]) if left != right]


def _split_rows(rows: list[DecodeRow], cuts: list[int]) -> list[list[DecodeRow]]:
    segments: list[list[DecodeRow]] = []
    start = 0
    for cut in cuts:
        end = start
        while end < len(rows) and work_items(rows[end]) <= cut:
            end += 1
        segments.append(rows[start:end])
        start = end
    segments.append(rows[start:])
    return segments


def _best_segments(rows: list[DecodeRow], max_segments: int = 3, min_size: int = 4) -> list[dict]:
    """Train a small work-size segmented policy for one codebook/M/aspect bucket.

    The durable ROCm policy must generalize to future models whose matrix sizes
    are close but not identical to Qwen3.6.  We therefore train on aspect ratio
    and total work size, keeping exact-shape winners only as an optional overlay.
    """

    if not rows:
        return []

    ordered = sorted(rows, key=lambda row: (work_items(row), row.n, row.k, row.shape, row.fmt))
    cuts = _candidate_cuts(ordered)

    def score_segments(segments: list[list[DecodeRow]]) -> tuple[int, list[dict]] | None:
        if any(not segment for segment in segments):
            return None
        if len(ordered) >= min_size * len(segments) and any(len(segment) < min_size for segment in segments):
            return None

        score = 0
        rules: list[dict] = []
        for segment in segments:
            candidate, support = _modal_candidate(segment)
            score += support
            rules.append(
                {
                    "max_work": work_items(segment[-1]),
                    "candidate": candidate,
                    "support": support,
                    "total": len(segment),
                }
            )
        return score, rules

    scored = score_segments([ordered])
    assert scored is not None
    best_score, best_rules = scored
    best_len = 1

    for cut in cuts:
        scored = score_segments(_split_rows(ordered, [cut]))
        if scored is None:
            continue
        score, rules = scored
        if score > best_score or (score == best_score and 2 < best_len):
            best_score, best_rules, best_len = score, rules, 2

    if max_segments >= 3:
        for index, left in enumerate(cuts):
            for right in cuts[index + 1:]:
                scored = score_segments(_split_rows(ordered, [left, right]))
                if scored is None:
                    continue
                score, rules = scored
                if score > best_score or (score == best_score and 3 < best_len):
                    best_score, best_rules, best_len = score, rules, 3

    return best_rules


def build_aspect_fallback_rules(rows: list[DecodeRow]) -> dict[tuple[int, int, str], list[dict]]:
    grouped: dict[tuple[int, int, str], list[DecodeRow]] = {}
    for row in rows:
        grouped.setdefault((row.codebook, row.m, aspect_bucket(row.n, row.k)), []).append(row)
    return {
        key: _best_segments(bucket)
        for key, bucket in sorted(grouped.items(), key=lambda item: item[0])
    }


def _pick_fallback_candidate(
    rules: dict[tuple[int, int, str], list[dict]],
    row: DecodeRow,
) -> tuple[int, int] | None:
    bucket_rules = rules.get((row.codebook, row.m, aspect_bucket(row.n, row.k)))
    if not bucket_rules:
        return None
    row_work = work_items(row)
    for rule in bucket_rules:
        if row_work <= rule["max_work"]:
            return rule["candidate"]
    return bucket_rules[-1]["candidate"]


def compute_aspect_fallback_metrics(rows: list[DecodeRow], rules: dict[tuple[int, int, str], list[dict]]) -> dict:
    exact_hits = 0
    covered = 0
    for row in rows:
        predicted = _pick_fallback_candidate(rules, row)
        if predicted is None:
            continue
        covered += 1
        if predicted == dispatch_candidate(row):
            exact_hits += 1
    total = len(rows)
    return {
        "total": total,
        "covered": covered,
        "coverage_pct": (100.0 * covered / total) if total else 0.0,
        "exact_hits": exact_hits,
        "exact_pct": (100.0 * exact_hits / total) if total else 0.0,
    }


OVERLAY_BEGIN = "    // BEGIN ROCM_NATIVE_VNNI_KNOWN_SHAPE_OVERLAY"
OVERLAY_END = "    // END ROCM_NATIVE_VNNI_KNOWN_SHAPE_OVERLAY"


def emit_overlay_block(rows: list[DecodeRow]) -> str:
    """Return an exact-shape overlay block for a staged ROCm refresh.

    The broad generated ROCm table is intentionally replaced only by full
    qwen36/all refreshes.  Staged profiles add known shape winners before the
    existing binary-search tables, preserving older entries while allowing the
    verifier pipeline to learn newly added production buckets.
    """

    lines: list[str] = [OVERLAY_BEGIN]
    for row in sorted(rows, key=lambda item: (item.codebook, item.m, item.n, item.k)):
        label = canonical_label(row.codebook)
        lines.append(
            f"    if (codebook_id == {row.codebook} && key == "
            f"0x{pack_shape_key(row.m, row.n, row.k):016x}ULL) "
            f"{{ out = ROCmNativeVNNIDecodeDispatchConfig{{{row.kb}, {row.target_waves}}}; "
            f"return true; }} // CB={row.codebook} ({label}) M={row.m} "
            f"{row.n}x{row.k} {row.variant} {row.shape} {row.min_us:.3f}us"
        )
    lines.append(OVERLAY_END)
    return "\n".join(lines)


def strip_existing_overlay(text: str) -> str:
    begin = text.find(OVERLAY_BEGIN)
    if begin < 0:
        return text
    end = text.find(OVERLAY_END, begin)
    if end < 0:
        raise SystemExit("base include contains overlay begin marker without end marker")
    end += len(OVERLAY_END)
    if end < len(text) and text[end] == "\n":
        end += 1
    return text[:begin] + text[end:]


def emit_cpp_overlay(rows: list[DecodeRow], output: Path, base_include: Path) -> None:
    if not base_include.is_file():
        raise SystemExit(f"base include not found: {base_include}")
    text = strip_existing_overlay(base_include.read_text())
    marker = "    const uint64_t key = packROCmNativeVNNIDecodeDispatchKey(m, n, k);\n"
    marker_index = text.find(marker)
    if marker_index < 0:
        raise SystemExit("base include does not contain ROCm NativeVNNI selector key marker")
    insert_at = marker_index + len(marker)
    updated = text[:insert_at] + emit_overlay_block(rows) + "\n" + text[insert_at:]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(updated)


def emit_cpp(rows: list[DecodeRow], output: Path) -> None:
    rows_by_codebook: dict[int, list[DecodeRow]] = {}
    for row in rows:
        rows_by_codebook.setdefault(row.codebook, []).append(row)
    for codebook_rows in rows_by_codebook.values():
        codebook_rows.sort(key=lambda item: pack_shape_key(item.m, item.n, item.k))
    aspect_rules = build_aspect_fallback_rules(rows)
    fallback_metrics = compute_aspect_fallback_metrics(rows, aspect_rules)

    lines: list[str] = []
    lines.append("// Generated by tests/v2/performance/kernels/rocm/analyze_rocm_native_vnni_decode_trainer.py.")
    lines.append("// Do not edit by hand; regenerate from ROCm NativeVNNI decode trainer CSVs.")
    lines.append(
        f"// Aspect fallback exact hit rate: {fallback_metrics['exact_hits']}/{fallback_metrics['total']} "
        f"({fallback_metrics['exact_pct']:.2f}%), coverage {fallback_metrics['covered']}/{fallback_metrics['total']} "
        f"({fallback_metrics['coverage_pct']:.2f}%)."
    )
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
    lines.append("struct ROCmNativeVNNIDecodeAspectRule")
    lines.append("{")
    lines.append("    long long max_work_items;")
    lines.append("    ROCmNativeVNNIDecodeDispatchConfig config;")
    lines.append("};")
    lines.append("")
    lines.append("inline constexpr uint64_t packROCmNativeVNNIDecodeDispatchKey(int m, int n, int k)")
    lines.append("{")
    lines.append("    return (static_cast<uint64_t>(m & 0xFF) << 56) |")
    lines.append("           (static_cast<uint64_t>(k & 0xFFFFFF) << 28) |")
    lines.append("           static_cast<uint64_t>(n & 0x0FFFFFFF);")
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
    lines.append("template <size_t Count>")
    lines.append("inline ROCmNativeVNNIDecodeDispatchConfig pickROCmNativeVNNIDecodeAspectRule(")
    lines.append("    const ROCmNativeVNNIDecodeAspectRule (&rules)[Count],")
    lines.append("    long long work_items)")
    lines.append("{")
    lines.append("    for (size_t i = 0; i < Count; ++i)")
    lines.append("    {")
    lines.append("        if (work_items <= rules[i].max_work_items || i + 1 == Count)")
    lines.append("            return rules[i].config;")
    lines.append("    }")
    lines.append("    return ROCmNativeVNNIDecodeDispatchConfig{};")
    lines.append("}")
    lines.append("")
    lines.append("inline bool selectROCmNativeVNNIDecodeAspectFallback(")
    lines.append("    uint8_t codebook_id, int m, int n, int k, ROCmNativeVNNIDecodeDispatchConfig &out)")
    lines.append("{")
    lines.append("    const float aspect_ratio = (k > 0) ? (static_cast<float>(n) / static_cast<float>(k)) : 0.0f;")
    lines.append("    const long long work_items = static_cast<long long>(n) * static_cast<long long>(k);")
    for codebook in sorted({key[0] for key in aspect_rules}):
        label = canonical_label(codebook)
        lines.append(f"    if (codebook_id == {codebook}) {{ // CB={codebook} ({label})")
        first_m = True
        for m in sorted({key[1] for key in aspect_rules if key[0] == codebook}):
            m_prefix = "if" if first_m else "else if"
            first_m = False
            lines.append(f"        {m_prefix} (m == {m}) {{")
            first_aspect = True
            for aspect in ASPECT_ORDER:
                rules = aspect_rules.get((codebook, m, aspect))
                if not rules:
                    continue
                aspect_prefix = "if" if first_aspect else "else if"
                first_aspect = False
                lines.append(f"            {aspect_prefix} ({ASPECT_CONDITIONS[aspect]}) {{")
                lines.append("                static constexpr ROCmNativeVNNIDecodeAspectRule kRules[] = {")
                for rule in rules:
                    kb, target_waves = rule["candidate"]
                    lines.append(
                        f"                    {{{rule['max_work']}LL, {{{kb}, {target_waves}}}}}, "
                        f"// {rule['support']}/{rule['total']}"
                    )
                lines.append("                };")
                lines.append("                out = pickROCmNativeVNNIDecodeAspectRule(kRules, work_items);")
                lines.append("                return true;")
                lines.append("            }")
            lines.append("        }")
        lines.append("    }")
    lines.append("    return false;")
    lines.append("}")
    lines.append("")
    lines.append("inline bool selectROCmNativeVNNIDecodeGenerated(")
    lines.append("    uint8_t codebook_id, int m, int n, int k, ROCmNativeVNNIDecodeDispatchConfig &out)")
    lines.append("{")
    lines.append("    const uint64_t key = packROCmNativeVNNIDecodeDispatchKey(m, n, k);")

    for codebook in sorted(rows_by_codebook):
        label = canonical_label(codebook)
        lines.append(f"    if (codebook_id == {codebook}) {{ // CB={codebook} ({label})")
        lines.append("        static constexpr ROCmNativeVNNIDecodeTuningEntry kTable[] = {")
        for row in rows_by_codebook[codebook]:
            lines.append(
                f"            {{0x{pack_shape_key(row.m, row.n, row.k):016x}ULL, "
                f"{{{row.kb}, {row.target_waves}}}}}, // CB={codebook} ({label}) "
                f"M={row.m} {row.n}x{row.k} {row.variant} {row.shape} {row.min_us:.3f}us"
            )
        lines.append("        };")
        lines.append("        if (findROCmNativeVNNIDecodeDispatchEntry(kTable, key, out))")
        lines.append("            return true;")
        lines.append("    }")

    lines.append("    return selectROCmNativeVNNIDecodeAspectFallback(codebook_id, m, n, k, out);")
    lines.append("}")
    lines.append("")
    lines.append("} // namespace llaminar2::rocm::generated")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def emit_summary(rows: list[DecodeRow], summary_path: Path) -> None:
    aspect_rules = build_aspect_fallback_rules(rows)
    fallback_metrics = compute_aspect_fallback_metrics(rows, aspect_rules)
    lines = [
        f"ROCm NativeVNNI decode generated entries: {len(rows)}",
        (
            "Aspect fallback exact hit rate: "
            f"{fallback_metrics['exact_hits']}/{fallback_metrics['total']} "
            f"({fallback_metrics['exact_pct']:.2f}%), coverage "
            f"{fallback_metrics['covered']}/{fallback_metrics['total']} "
            f"({fallback_metrics['coverage_pct']:.2f}%)"
        ),
        "codebook,format,shape,m,n,k,variant,kb,target_waves,min_us,eff_bw_gbs",
    ]
    for row in rows:
        lines.append(
            f"{row.codebook},{canonical_label(row.codebook)},{row.shape},"
            f"{row.m},{row.n},{row.k},{row.variant},{row.kb},{row.target_waves},"
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
    if args.base_include:
        emit_cpp_overlay(rows, output, Path(args.base_include))
    else:
        emit_cpp(rows, output)
    if args.summary:
        emit_summary(rows, Path(args.summary))
    print(f"generated {len(rows)} ROCm NativeVNNI decode dispatch entries -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
