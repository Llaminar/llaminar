#!/usr/bin/env python3

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from native_vnni_codebooks import CODEBOOK_TO_FORMAT, FORMAT_TO_CODEBOOK


ASPECT_BUCKETS = (
    ("very_wide", 16.0),
    ("wide", 2.0),
    ("balanced", 0.75),
)

ASPECT_ORDER = ["very_wide", "wide", "balanced", "tall"]
FAMILY_ORDER = ["wide", "kpar", "rowpar", "direct"]
ASPECT_CONDITIONS = {
    "very_wide": "aspect_ratio >= 16.0f",
    "wide": "aspect_ratio >= 2.0f",
    "balanced": "aspect_ratio >= 0.75f",
    "tall": "true",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CUDA native-payload GEMV dispatch heuristics from an existing sweep CSV.")
    parser.add_argument("--input", required=True, help="Input sweep CSV path")
    parser.add_argument("--output", required=True, help="Generated heuristic include path")
    parser.add_argument("--summary", required=True, help="Human-readable summary output path")
    parser.add_argument("--base-include", default=None,
                        help="Existing generated include to preserve as fallback while applying CSV known-shape winners")
    parser.add_argument("--min-overall-family-pct", type=float, default=0.0)
    parser.add_argument("--min-overall-exact-pct", type=float, default=0.0)
    parser.add_argument("--min-fallback-family-pct", type=float, default=0.0)
    parser.add_argument("--min-fallback-exact-pct", type=float, default=0.0)
    return parser.parse_args()


KNOWN_OVERRIDE_BEGIN = "// BEGIN generated known-shape overrides from analyze_cuda_tc_gemv_dispatch.py"
KNOWN_OVERRIDE_END = "// END generated known-shape overrides from analyze_cuda_tc_gemv_dispatch.py"
PUBLIC_API_MARKER = "// Public API consumed by dispatchCodebook<CB>()"


def canonical_branch_label(codebook: int) -> str:
    return CODEBOOK_TO_FORMAT.get(codebook, f"CB{codebook}").split("/")[0]


def aspect_bucket(ratio: float) -> str:
    for name, threshold in ASPECT_BUCKETS:
        if ratio >= threshold:
            return name
    return "tall"


def load_rows(path: Path, winners_only: bool = True):
    rows = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            try:
                is_best = int(raw["is_best"])
            except (KeyError, ValueError):
                continue
            if winners_only and is_best != 1:
                continue

            format_name = raw["format"]
            if format_name not in FORMAT_TO_CODEBOOK:
                raise SystemExit(f"unsupported format in CSV: {format_name}")
            codebook = FORMAT_TO_CODEBOOK[format_name]
            if raw.get("codebook", "").strip():
                csv_codebook = int(raw["codebook"])
                if csv_codebook != codebook:
                    raise SystemExit(
                        f"codebook mismatch in CSV: format {format_name} maps to {codebook}, row has {csv_codebook}")

            m = int(raw.get("m", "1"))
            n = int(raw["n"])
            k = int(raw["k"])
            ratio = float(n) / float(k) if k else 0.0
            work = n * k
            rows.append({
                "format": format_name,
                "codebook": codebook,
                "shape": raw["shape"],
                "m": m,
                "n": n,
                "k": k,
                "ratio": ratio,
                "work": work,
                "aspect": aspect_bucket(ratio),
                "family": raw["family"],
                "candidate": (
                    int(raw["tile_n"]),
                    int(raw["cpt"]),
                    int(raw["target_waves"]),
                    int(raw["mkg"]),
                    int(raw["max_kb"]),
                    int(raw["force_two_phase"]),
                ),
                "bw": float(raw["eff_bw_gbs"]),
                "is_best": is_best == 1,
            })
    return rows


def select_runtime_rows(all_rows):
    """Collapse source-format rows to the dispatch surface used at runtime.

    CUDA NativeVNNI GEMV dispatch is keyed by ``(codebook, M, N, K)``. Some GGUF
    formats are aliases of the same native codebook (for example Q4_1/Q4_K), so
    the runtime cannot choose two different tunings for the same key. Pick one
    aggregate winner per key by maximizing the worst alias-relative bandwidth,
    then the mean alias-relative bandwidth. This keeps validation honest: exact
    hit rates measure what the generated table can actually express.
    """

    grouped = defaultdict(list)
    for row in all_rows:
        grouped[(row["codebook"], row["m"], row["n"], row["k"])].append(row)

    runtime_rows = []
    alias_groups = 0
    conflict_groups = 0

    for key, bucket in grouped.items():
        formats = sorted({row["format"] for row in bucket})
        if len(formats) > 1:
            alias_groups += 1

        oracle_by_format = {}
        for row in bucket:
            oracle_by_format[row["format"]] = max(oracle_by_format.get(row["format"], 0.0), row["bw"])

        candidates = sorted({(row["family"], row["candidate"]) for row in bucket})
        scored = []
        for family, candidate in candidates:
            ratios = []
            total_bw = 0.0
            for fmt in formats:
                fmt_rows = [
                    row for row in bucket
                    if row["format"] == fmt and row["family"] == family and row["candidate"] == candidate
                ]
                best_bw = max((row["bw"] for row in fmt_rows), default=0.0)
                oracle = oracle_by_format.get(fmt, 0.0)
                ratios.append((best_bw / oracle) if oracle > 0.0 else 0.0)
                total_bw += best_bw
            scored.append({
                "family": family,
                "candidate": candidate,
                "min_ratio": min(ratios) if ratios else 0.0,
                "mean_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
                "total_bw": total_bw,
            })

        best = max(
            scored,
            key=lambda item: (
                item["min_ratio"],
                item["mean_ratio"],
                item["total_bw"],
                item["family"],
                item["candidate"],
            ),
        )

        source_winners = {
            (row["family"], row["candidate"])
            for row in bucket
            if row["is_best"]
        }
        if len(source_winners) > 1:
            conflict_groups += 1

        template = max(bucket, key=lambda row: row["bw"]).copy()
        template["format"] = CODEBOOK_TO_FORMAT.get(key[0], template["format"])
        template["family"] = best["family"]
        template["candidate"] = best["candidate"]
        template["alias_formats"] = "/".join(formats)
        template["alias_count"] = len(formats)
        template["alias_min_ratio"] = best["min_ratio"]
        template["alias_mean_ratio"] = best["mean_ratio"]
        runtime_rows.append(template)

    runtime_rows.sort(key=lambda row: (row["codebook"], row["m"], row["k"], row["n"]))
    return runtime_rows, {
        "source_rows": sum(1 for row in all_rows if row["is_best"]),
        "runtime_rows": len(runtime_rows),
        "alias_groups": alias_groups,
        "conflict_groups": conflict_groups,
    }


def modal_value(rows, key_name):
    counts = Counter(row[key_name] for row in rows)
    return max(counts.items(), key=lambda item: (item[1], str(item[0])))


def modal_candidate(rows):
    counts = Counter(row["candidate"] for row in rows)
    return max(counts.items(), key=lambda item: (item[1], item[0]))


def split_rows(rows, cuts):
    segments = []
    start = 0
    for cut in cuts:
        end = start
        while end < len(rows) and rows[end]["work"] <= cut:
            end += 1
        segments.append(rows[start:end])
        start = end
    segments.append(rows[start:])
    return segments


def candidate_cuts(rows):
    works = sorted({row["work"] for row in rows})
    return [((left + right) // 2) for left, right in zip(works, works[1:]) if left != right]


def best_segments(rows, key_name, max_segments=3, min_size=4):
    if not rows:
        return []

    ordered = sorted(rows, key=lambda row: (row["work"], row["n"], row["k"], row["format"], row["shape"]))
    cuts = candidate_cuts(ordered)

    def score_segments(segments):
        if any(not segment for segment in segments):
            return None
        if len(ordered) >= min_size * len(segments):
            if any(len(segment) < min_size for segment in segments):
                return None

        rules = []
        score = 0
        for segment in segments:
            if key_name == "family":
                choice, support = modal_value(segment, key_name)
            else:
                choice, support = modal_candidate(segment)
            score += support
            rules.append({
                "max_work": segment[-1]["work"],
                "choice": choice,
                "support": support,
                "total": len(segment),
            })
        return score, rules

    best_score, best_rules = score_segments([ordered])
    best_len = 1

    for cut in cuts:
        scored = score_segments(split_rows(ordered, [cut]))
        if scored is None:
            continue
        score, rules = scored
        if score > best_score or (score == best_score and 2 < best_len):
            best_score, best_rules, best_len = score, rules, 2

    if max_segments >= 3:
        for index, left in enumerate(cuts):
            for right in cuts[index + 1:]:
                scored = score_segments(split_rows(ordered, [left, right]))
                if scored is None:
                    continue
                score, rules = scored
                if score > best_score or (score == best_score and 3 < best_len):
                    best_score, best_rules, best_len = score, rules, 3

    return best_rules


def build_exact_rules(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["codebook"], row["m"], row["n"], row["k"])].append(row)

    exact_rules = defaultdict(list)
    for (codebook, m, n, k), bucket in grouped.items():
        family, family_support = modal_value(bucket, "family")
        candidate, candidate_support = modal_candidate([row for row in bucket if row["family"] == family])
        exact_rules[codebook].append({
            "m": m,
            "n": n,
            "k": k,
            "family": family,
            "candidate": candidate,
            "family_support": family_support,
            "candidate_support": candidate_support,
            "total": len(bucket),
        })

    for codebook in exact_rules:
        exact_rules[codebook].sort(key=lambda rule: (rule["m"], rule["k"], rule["n"]))
    return exact_rules


def build_family_fallback_rules(rows):
    rules = {}
    for codebook in sorted({row["codebook"] for row in rows}):
        for m in sorted({row["m"] for row in rows if row["codebook"] == codebook}):
            for aspect in ASPECT_ORDER:
                bucket = [
                    row for row in rows
                    if row["codebook"] == codebook and row["m"] == m and row["aspect"] == aspect
                ]
                if bucket:
                    rules[(codebook, m, aspect)] = best_segments(bucket, "family")
    return rules


def build_tuning_fallback_rules(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["codebook"], row["m"], row["aspect"], row["family"])].append(row)

    rules = {}
    for key, bucket in grouped.items():
        rules[key] = best_segments(bucket, "candidate")
    return rules


def pick_exact_rule(exact_rules, row):
    for rule in exact_rules.get(row["codebook"], []):
        if rule["m"] == row["m"] and rule["n"] == row["n"] and rule["k"] == row["k"]:
            return rule
    return None


def pick_family_fallback(family_rules, row):
    rules = family_rules.get((row["codebook"], row["m"], row["aspect"]))
    if not rules:
        return "kpar"
    for rule in rules:
        if row["work"] <= rule["max_work"]:
            return rule["choice"]
    return rules[-1]["choice"]


def pick_candidate_fallback(tuning_rules, row, family):
    rules = tuning_rules.get((row["codebook"], row["m"], row["aspect"], family))
    if not rules:
        return None
    for rule in rules:
        if row["work"] <= rule["max_work"]:
            return rule["choice"]
    return rules[-1]["choice"]


def compute_metrics(rows, exact_rules, family_rules, tuning_rules, use_exact_overrides):
    family_hits = 0
    exact_hits = 0
    exact_override_hits = 0

    for row in rows:
        exact_rule = pick_exact_rule(exact_rules, row) if use_exact_overrides else None
        if exact_rule is not None:
            predicted_family = exact_rule["family"]
            predicted_candidate = exact_rule["candidate"]
            exact_override_hits += 1
        else:
            predicted_family = pick_family_fallback(family_rules, row)
            predicted_candidate = pick_candidate_fallback(tuning_rules, row, predicted_family)

        if predicted_family == row["family"]:
            family_hits += 1
        if predicted_family == row["family"] and predicted_candidate == row["candidate"]:
            exact_hits += 1

    total = len(rows)
    return {
        "total": total,
        "family_hits": family_hits,
        "family_pct": (100.0 * family_hits / total) if total else 0.0,
        "exact_hits": exact_hits,
        "exact_pct": (100.0 * exact_hits / total) if total else 0.0,
        "exact_override_hits": exact_override_hits,
        "exact_override_pct": (100.0 * exact_override_hits / total) if total else 0.0,
    }


def format_candidate(candidate):
    tile_n, cpt, target_waves, mkg, max_kb, force_two_phase = candidate
    return f"{{ {tile_n}, {cpt}, {target_waves}, {mkg}, {max_kb}, {force_two_phase} }}"


def format_packed_key(m, n, k):
    return f"0x{(((m & 0xFF) << 56) | ((k & 0xFFFFFF) << 28) | (n & 0x0FFFFFFF)):016x}ULL"


def write_exact_override_function(lines, exact_rules):
    lines.append("struct GeneratedKnownShapeEntry")
    lines.append("{")
    lines.append("    uint64_t packed_key;")
    lines.append("    NativeGemvShape shape;")
    lines.append("    GeneratedDispatchTuning tuning;")
    lines.append("};")
    lines.append("")
    lines.append("inline constexpr uint64_t packGeneratedDispatchKey(int M, int N, int K)")
    lines.append("{")
    lines.append("    return (static_cast<uint64_t>(M & 0xFF) << 56) |")
    lines.append("           (static_cast<uint64_t>(K & 0xFFFFFF) << 28) |")
    lines.append("           static_cast<uint64_t>(N & 0x0FFFFFFF);")
    lines.append("}")
    lines.append("")
    lines.append("template <size_t Count>")
    lines.append("inline bool findGeneratedKnownShape(const GeneratedKnownShapeEntry (&table)[Count], uint64_t packed_key, NativeGemvShape &out_shape, GeneratedDispatchTuning &out_tuning)")
    lines.append("{")
    lines.append("    size_t lo = 0;")
    lines.append("    size_t hi = Count;")
    lines.append("    while (lo < hi)")
    lines.append("    {")
    lines.append("        const size_t mid = lo + ((hi - lo) / 2);")
    lines.append("        const uint64_t candidate = table[mid].packed_key;")
    lines.append("        if (candidate == packed_key)")
    lines.append("        {")
    lines.append("            out_shape = table[mid].shape;")
    lines.append("            out_tuning = table[mid].tuning;")
    lines.append("            return true;")
    lines.append("        }")
    lines.append("        if (candidate < packed_key)")
    lines.append("            lo = mid + 1;")
    lines.append("        else")
    lines.append("            hi = mid;")
    lines.append("    }")
    lines.append("    return false;")
    lines.append("}")
    lines.append("")
    lines.append("template <uint8_t CB>")
    lines.append("inline bool selectKnownShapeGenerated(int M, int N, int K, NativeGemvShape &out_shape, GeneratedDispatchTuning &out_tuning)")
    lines.append("{")
    lines.append("    const uint64_t packed_key = packGeneratedDispatchKey(M, N, K);")
    first_codebook = True
    for codebook in sorted(exact_rules):
        prefix = "if constexpr" if first_codebook else "else if constexpr"
        first_codebook = False
        format_name = canonical_branch_label(codebook)
        lines.append(f"    {prefix} (CB == {codebook}) {{ // {format_name}")
        lines.append("        static constexpr GeneratedKnownShapeEntry kTable[] = {")
        for rule in exact_rules[codebook]:
            lines.append(
                f"            {{ {format_packed_key(rule['m'], rule['n'], rule['k'])}, NativeGemvShape::{rule['family'].upper()}, {format_candidate(rule['candidate'])} }}, // M={rule['m']} {rule['n']}x{rule['k']} {rule['family_support']}/{rule['total']}")
        lines.append("        };")
        lines.append("        return findGeneratedKnownShape(kTable, packed_key, out_shape, out_tuning);")
        lines.append("    }")
    lines.append("    return false;")
    lines.append("}")
    lines.append("")


def write_family_fallback(lines, family_rules):
    lines.append("template <uint8_t CB>")
    lines.append("inline NativeGemvShape classifyShapeGeneratedFallback(int M, int N, int K)")
    lines.append("{")
    lines.append("    const float aspect_ratio = (K > 0) ? (static_cast<float>(N) / static_cast<float>(K)) : 0.0f;")
    lines.append("    const long long work_items = static_cast<long long>(N) * static_cast<long long>(K);")
    first_codebook = True
    for codebook in sorted({key[0] for key in family_rules}):
        prefix = "    if constexpr" if first_codebook else "    else if constexpr"
        first_codebook = False
        lines.append(f"    {prefix} (CB == {codebook}) {{ // {canonical_branch_label(codebook)}")
        first_m = True
        for m in sorted({key[1] for key in family_rules if key[0] == codebook}):
            m_prefix = "        if" if first_m else "        else if"
            first_m = False
            lines.append(f"        {m_prefix} (M == {m}) {{")
            first_aspect = True
            for aspect in ASPECT_ORDER:
                rules = family_rules.get((codebook, m, aspect))
                if not rules:
                    continue
                aspect_prefix = "            if" if first_aspect else "            else if"
                first_aspect = False
                lines.append(f"            {aspect_prefix} ({ASPECT_CONDITIONS[aspect]}) {{")
                for index, rule in enumerate(rules):
                    choice = f"NativeGemvShape::{rule['choice'].upper()}"
                    if index == len(rules) - 1:
                        lines.append(f"                return {choice}; // {rule['support']}/{rule['total']}")
                    else:
                        keyword = "if" if index == 0 else "else if"
                        lines.append(f"                {keyword} (work_items <= {rule['max_work']}LL) return {choice}; // {rule['support']}/{rule['total']}")
                lines.append("            }")
            lines.append("        }")
        lines.append("        return NativeGemvShape::KPAR;")
        lines.append("    }")
    lines.append("    return NativeGemvShape::KPAR;")
    lines.append("}")
    lines.append("")


def write_tuning_fallback(lines, tuning_rules):
    lines.append("template <uint8_t CB>")
    lines.append("inline GeneratedDispatchTuning selectGeneratedTuningFallback(NativeGemvShape shape, int M, int N, int K)")
    lines.append("{")
    lines.append("    const float aspect_ratio = (K > 0) ? (static_cast<float>(N) / static_cast<float>(K)) : 0.0f;")
    lines.append("    const long long work_items = static_cast<long long>(N) * static_cast<long long>(K);")
    first_codebook = True
    for codebook in sorted({key[0] for key in tuning_rules}):
        prefix = "    if constexpr" if first_codebook else "    else if constexpr"
        first_codebook = False
        lines.append(f"    {prefix} (CB == {codebook}) {{ // {canonical_branch_label(codebook)}")
        first_m = True
        for m in sorted({key[1] for key in tuning_rules if key[0] == codebook}):
            m_prefix = "        if" if first_m else "        else if"
            first_m = False
            lines.append(f"        {m_prefix} (M == {m}) {{")
            lines.append("            switch (shape)")
            lines.append("            {")
            for family in FAMILY_ORDER:
                lines.append(f"            case NativeGemvShape::{family.upper()}:")
                first_aspect = True
                for aspect in ASPECT_ORDER:
                    rules = tuning_rules.get((codebook, m, aspect, family))
                    if not rules:
                        continue
                    aspect_prefix = "                if" if first_aspect else "                else if"
                    first_aspect = False
                    lines.append(f"                {aspect_prefix} ({ASPECT_CONDITIONS[aspect]}) {{")
                    for index, rule in enumerate(rules):
                        candidate = format_candidate(rule['choice'])
                        if index == len(rules) - 1:
                            lines.append(f"                    return {candidate}; // {rule['support']}/{rule['total']}")
                        else:
                            keyword = "if" if index == 0 else "else if"
                            lines.append(f"                    {keyword} (work_items <= {rule['max_work']}LL) return {candidate}; // {rule['support']}/{rule['total']}")
                    lines.append("                }")
                lines.append("                break;")
            lines.append("            }")
            lines.append("        }")
        lines.append("    }")
    lines.append("    return { 128, 1, 0, 0, 0, 0 };")
    lines.append("}")
    lines.append("")


def write_output(path: Path, input_path: Path, exact_rules, family_rules, tuning_rules, overall_metrics, fallback_metrics):
    lines = []
    lines.append("// Auto-generated by analyze_cuda_tc_gemv_dispatch.py")
    lines.append(f"// Source CSV: {input_path}")
    lines.append(f"// Overall family hit rate: {overall_metrics['family_hits']}/{overall_metrics['total']} ({overall_metrics['family_pct']:.2f}%)")
    lines.append(f"// Overall exact hit rate: {overall_metrics['exact_hits']}/{overall_metrics['total']} ({overall_metrics['exact_pct']:.2f}%)")
    lines.append(f"// Known-shape override coverage: {overall_metrics['exact_override_hits']}/{overall_metrics['total']} ({overall_metrics['exact_override_pct']:.2f}%)")
    lines.append(f"// Fallback family hit rate: {fallback_metrics['family_hits']}/{fallback_metrics['total']} ({fallback_metrics['family_pct']:.2f}%)")
    lines.append(f"// Fallback exact hit rate: {fallback_metrics['exact_hits']}/{fallback_metrics['total']} ({fallback_metrics['exact_pct']:.2f}%)")
    lines.append("")
    lines.append("struct GeneratedDispatchTuning")
    lines.append("{")
    lines.append("    int tile_n;")
    lines.append("    int cpt;")
    lines.append("    int target_waves;")
    lines.append("    int mkg;")
    lines.append("    int max_kb;")
    lines.append("    int force_two_phase;")
    lines.append("};")
    lines.append("")
    lines.append("// Known-shape overrides for the swept decode shapes.")
    write_exact_override_function(lines, exact_rules)
    lines.append("template <uint8_t CB>")
    lines.append("inline NativeGemvShape classifyShapeGeneratedFallback(int M, int N, int K);")
    lines.append("")
    lines.append("template <uint8_t CB>")
    lines.append("inline GeneratedDispatchTuning selectGeneratedTuningFallback(NativeGemvShape shape, int M, int N, int K);")
    lines.append("")
    lines.append("template <uint8_t CB>")
    lines.append("inline NativeGemvShape classifyShapeGenerated(int M, int N, int K)")
    lines.append("{")
    lines.append("    GeneratedDispatchTuning tuning{};")
    lines.append("    NativeGemvShape shape = NativeGemvShape::KPAR;")
    lines.append("    if (selectKnownShapeGenerated<CB>(M, N, K, shape, tuning))")
    lines.append("        return shape;")
    lines.append("    return classifyShapeGeneratedFallback<CB>(M, N, K);")
    lines.append("}")
    lines.append("")
    lines.append("template <uint8_t CB>")
    lines.append("inline NativeGemvShape classifyShapeGenerated(int N, int K)")
    lines.append("{")
    lines.append("    return classifyShapeGenerated<CB>(1, N, K);")
    lines.append("}")
    lines.append("")
    write_family_fallback(lines, family_rules)
    lines.append("template <uint8_t CB>")
    lines.append("inline GeneratedDispatchTuning selectGeneratedTuning(int M, int N, int K)")
    lines.append("{")
    lines.append("    GeneratedDispatchTuning tuning{};")
    lines.append("    NativeGemvShape shape = NativeGemvShape::KPAR;")
    lines.append("    if (selectKnownShapeGenerated<CB>(M, N, K, shape, tuning))")
    lines.append("        return tuning;")
    lines.append("    return selectGeneratedTuningFallback<CB>(classifyShapeGeneratedFallback<CB>(M, N, K), M, N, K);")
    lines.append("}")
    lines.append("")
    lines.append("template <uint8_t CB>")
    lines.append("inline GeneratedDispatchTuning selectGeneratedTuning(int N, int K)")
    lines.append("{")
    lines.append("    return selectGeneratedTuning<CB>(1, N, K);")
    lines.append("}")
    lines.append("")
    write_tuning_fallback(lines, tuning_rules)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def render_known_shape_override_block(input_path: Path, exact_rules, overall_metrics) -> str:
    lines = []
    lines.append(KNOWN_OVERRIDE_BEGIN)
    lines.append(f"// Source CSV: {input_path}")
    lines.append(f"// Known-shape override coverage: {overall_metrics['exact_override_hits']}/{overall_metrics['total']} ({overall_metrics['exact_override_pct']:.2f}%)")
    lines.append("// These exact (M,N,K) winners are layered over the existing broad fallback table.")
    lines.append("")
    write_exact_override_function(lines, exact_rules)
    lines.append(KNOWN_OVERRIDE_END)
    return "\n".join(lines)


def render_known_shape_public_api() -> str:
    return "\n".join([
        PUBLIC_API_MARKER,
        "template <uint8_t CB>",
        "inline NativeGemvShape classifyShapeGenerated(int M, int N, int K)",
        "{",
        "    NativeGemvShape shape = NativeGemvShape::KPAR;",
        "    GeneratedDispatchTuning tuning{};",
        "    if (selectKnownShapeGenerated<CB>(M, N, K, shape, tuning))",
        "        return shape;",
        "    selectGeneratedDispatch_<CB>(M, N, K, shape, tuning);",
        "    return shape;",
        "}",
        "",
        "template <uint8_t CB>",
        "inline NativeGemvShape classifyShapeGenerated(int N, int K)",
        "{",
        "    return classifyShapeGenerated<CB>(1, N, K);",
        "}",
        "",
        "template <uint8_t CB>",
        "inline GeneratedDispatchTuning selectGeneratedTuning(int M, int N, int K)",
        "{",
        "    NativeGemvShape shape = NativeGemvShape::KPAR;",
        "    GeneratedDispatchTuning tuning{};",
        "    if (selectKnownShapeGenerated<CB>(M, N, K, shape, tuning))",
        "        return tuning;",
        "    selectGeneratedDispatch_<CB>(M, N, K, shape, tuning);",
        "    return tuning;",
        "}",
        "",
        "template <uint8_t CB>",
        "inline GeneratedDispatchTuning selectGeneratedTuning(int N, int K)",
        "{",
        "    return selectGeneratedTuning<CB>(1, N, K);",
        "}",
    ])


def strip_existing_known_shape_block(text: str) -> str:
    pattern = re.compile(
        rf"\n?{re.escape(KNOWN_OVERRIDE_BEGIN)}.*?{re.escape(KNOWN_OVERRIDE_END)}\n?",
        re.DOTALL,
    )
    return re.sub(pattern, "\n", text)


def replace_public_api(text: str) -> str:
    marker_index = text.find(PUBLIC_API_MARKER)
    if marker_index < 0:
        raise SystemExit("base include does not contain the GEMV generated public API marker")

    return text[:marker_index].rstrip() + "\n" + render_known_shape_public_api()


def write_output_with_base(path: Path, input_path: Path, base_include: Path, exact_rules, overall_metrics):
    if not base_include.is_file():
        raise SystemExit(f"base include not found: {base_include}")

    text = strip_existing_known_shape_block(base_include.read_text())
    public_api_index = text.find(PUBLIC_API_MARKER)
    if public_api_index < 0:
        raise SystemExit(f"base include does not contain public API marker: {base_include}")

    block = render_known_shape_override_block(input_path, exact_rules, overall_metrics)
    text = text[:public_api_index].rstrip() + "\n\n" + block + "\n\n" + text[public_api_index:].lstrip()
    text = replace_public_api(text)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def write_summary(
    path: Path,
    input_path: Path,
    exact_rules,
    family_rules,
    tuning_rules,
    overall_metrics,
    fallback_metrics,
    runtime_stats,
):
    lines = []
    lines.append("CUDA Native-Payload GEMV Dispatch Heuristic Summary")
    lines.append(f"Input CSV: {input_path}")
    lines.append(f"Source winner rows: {runtime_stats['source_rows']}")
    lines.append(f"Runtime dispatch rows: {runtime_stats['runtime_rows']}")
    lines.append(f"Alias runtime keys: {runtime_stats['alias_groups']}")
    lines.append(f"Alias winner-conflict keys: {runtime_stats['conflict_groups']}")
    lines.append(f"Overall family hit rate: {overall_metrics['family_hits']}/{overall_metrics['total']} ({overall_metrics['family_pct']:.2f}%)")
    lines.append(f"Overall exact hit rate: {overall_metrics['exact_hits']}/{overall_metrics['total']} ({overall_metrics['exact_pct']:.2f}%)")
    lines.append(f"Known-shape override coverage: {overall_metrics['exact_override_hits']}/{overall_metrics['total']} ({overall_metrics['exact_override_pct']:.2f}%)")
    lines.append(f"Fallback family hit rate: {fallback_metrics['family_hits']}/{fallback_metrics['total']} ({fallback_metrics['family_pct']:.2f}%)")
    lines.append(f"Fallback exact hit rate: {fallback_metrics['exact_hits']}/{fallback_metrics['total']} ({fallback_metrics['exact_pct']:.2f}%)")
    lines.append("")
    lines.append("Known-shape override counts by codebook:")
    for codebook in sorted(exact_rules):
        lines.append(f"  CB {codebook:2d} ({CODEBOOK_TO_FORMAT[codebook]}): {len(exact_rules[codebook])} exact (M,N,K) overrides")
    lines.append("")
    lines.append("Fallback family routing rules by codebook:")
    for codebook in sorted({key[0] for key in family_rules}):
        lines.append(f"  CB {codebook:2d} ({CODEBOOK_TO_FORMAT[codebook]}):")
        for m in sorted({key[1] for key in family_rules if key[0] == codebook}):
            lines.append(f"    M={m}:")
            for aspect in ASPECT_ORDER:
                rules = family_rules.get((codebook, m, aspect))
                if not rules:
                    continue
                lines.append(f"      {aspect}:")
                for rule in rules:
                    lines.append(f"        work<= {rule['max_work']}: {rule['choice']} ({rule['support']}/{rule['total']})")
    lines.append("")
    lines.append("Fallback tuning rules by codebook:")
    for codebook in sorted({key[0] for key in tuning_rules}):
        lines.append(f"  CB {codebook:2d} ({CODEBOOK_TO_FORMAT[codebook]}):")
        saw_codebook = False
        for m in sorted({key[1] for key in tuning_rules if key[0] == codebook}):
            lines.append(f"    M={m}:")
            for family in FAMILY_ORDER:
                for aspect in ASPECT_ORDER:
                    rules = tuning_rules.get((codebook, m, aspect, family))
                    if not rules:
                        continue
                    saw_codebook = True
                    lines.append(f"      {family}/{aspect}:")
                    for rule in rules:
                        lines.append(f"        work<= {rule['max_work']}: {rule['choice']} ({rule['support']}/{rule['total']})")
        if not saw_codebook:
            lines.append("    no fallback tuning rules generated")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def validate_thresholds(args, overall_metrics, fallback_metrics):
    checks = [
        (overall_metrics["family_pct"], args.min_overall_family_pct, "overall family hit rate"),
        (overall_metrics["exact_pct"], args.min_overall_exact_pct, "overall exact hit rate"),
        (fallback_metrics["family_pct"], args.min_fallback_family_pct, "fallback family hit rate"),
        (fallback_metrics["exact_pct"], args.min_fallback_exact_pct, "fallback exact hit rate"),
    ]
    failures = [
        f"{label} {actual:.2f}% < required {minimum:.2f}%"
        for actual, minimum, label in checks
        if actual + 1e-9 < minimum
    ]
    if failures:
        raise SystemExit("; ".join(failures))


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    if not input_path.is_file():
        raise SystemExit(f"input CSV not found: {input_path}")

    all_rows = load_rows(input_path, winners_only=False)
    rows, runtime_stats = select_runtime_rows(all_rows)
    if not rows:
        raise SystemExit(f"no runtime winner rows found in: {input_path}")

    exact_rules = build_exact_rules(rows)
    family_rules = build_family_fallback_rules(rows)
    tuning_rules = build_tuning_fallback_rules(rows)
    overall_metrics = compute_metrics(rows, exact_rules, family_rules, tuning_rules, use_exact_overrides=True)
    fallback_metrics = compute_metrics(rows, exact_rules, family_rules, tuning_rules, use_exact_overrides=False)

    validate_thresholds(args, overall_metrics, fallback_metrics)
    if args.base_include:
        write_output_with_base(output_path, input_path, Path(args.base_include), exact_rules, overall_metrics)
    else:
        write_output(output_path, input_path, exact_rules, family_rules, tuning_rules, overall_metrics, fallback_metrics)
    write_summary(
        summary_path,
        input_path,
        exact_rules,
        family_rules,
        tuning_rules,
        overall_metrics,
        fallback_metrics,
        runtime_stats,
    )

    print(f"generated heuristic include: {output_path}")
    print(f"generated heuristic summary: {summary_path}")
    print(f"source winner rows: {runtime_stats['source_rows']}")
    print(f"runtime dispatch rows: {runtime_stats['runtime_rows']}")
    print(f"alias runtime keys: {runtime_stats['alias_groups']}")
    print(f"alias winner-conflict keys: {runtime_stats['conflict_groups']}")
    print(f"overall family hit rate: {overall_metrics['family_hits']}/{overall_metrics['total']} ({overall_metrics['family_pct']:.2f}%)")
    print(f"overall exact hit rate: {overall_metrics['exact_hits']}/{overall_metrics['total']} ({overall_metrics['exact_pct']:.2f}%)")
    print(f"known-shape override coverage: {overall_metrics['exact_override_hits']}/{overall_metrics['total']} ({overall_metrics['exact_override_pct']:.2f}%)")
    print(f"fallback family hit rate: {fallback_metrics['family_hits']}/{fallback_metrics['total']} ({fallback_metrics['family_pct']:.2f}%)")
    print(f"fallback exact hit rate: {fallback_metrics['exact_hits']}/{fallback_metrics['total']} ({fallback_metrics['exact_pct']:.2f}%)")


if __name__ == "__main__":
    main()
