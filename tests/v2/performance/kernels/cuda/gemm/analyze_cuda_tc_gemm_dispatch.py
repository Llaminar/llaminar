#!/usr/bin/env python3
"""
Generate CUDA native-VNNI GEMM prefill dispatch heuristics from sweep CSVs.

Reads one or more tile-sweep CSVs (produced by TileSweep_AllStrategies test),
picks the best (tile_id, split_k) per (codebook, M_bin, N, K) combination,
and emits a C++ .inc file with per-codebook binary-search lookup tables.

Usage:
    python3 analyze_cuda_tc_gemm_dispatch.py \
        --input /tmp/sweep_q4_0.csv /tmp/sweep_q4_1.csv ... \
        --output src/v2/kernels/cuda/gemm/CUDANativeVNNIPrefillDispatchGenerated.inc \
        --exceptions src/v2/kernels/cuda/gemm/CUDANativeVNNIPrefillDispatchExceptions.json \
        --summary /tmp/gemm_dispatch_summary.txt

The generated .inc file is included by CUDANativeVNNIPrefillKernels.cu and
provides selectPrefillTileGenerated<CB>() for exact-match dispatch.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

FORMAT_TO_CODEBOOK = {
    "Q4_0": 0,
    "IQ4_NL": 4,
    "Q4_1": 5,
    "Q5_0": 6,
    "Q5_1": 7,
    "Q6_K": 8,
    "Q3_K": 9,
    "Q2_K": 10,
    "IQ3_S": 11,
    "IQ3_XXS": 12,
    "IQ2_S": 13,
    "IQ2_XS": 14,
    "IQ2_XXS": 15,
    "IQ1_S": 16,
    "IQ1_M": 17,
    "Q8_0": 18,
}
CODEBOOK_TO_FORMAT = {v: k for k, v in FORMAT_TO_CODEBOOK.items()}

# M binning: map arbitrary M to nearest swept bucket
M_BINS = [64, 128, 256, 512]


def bin_m(m: int) -> int:
    """Bin M to the nearest sweep bucket."""
    if m <= 96:
        return 64
    if m <= 192:
        return 128
    if m <= 384:
        return 256
    return 512


def pack_key(m_bin: int, n: int, k: int) -> int:
    """Pack (M_bin, N, K) into a 64-bit lookup key.
    Layout: [63:40] M_bin (24 bits) | [39:20] K (20 bits) | [19:0] N (20 bits)
    """
    return (m_bin << 40) | ((k & 0xFFFFF) << 20) | (n & 0xFFFFF)


def format_packed_key(m_bin: int, n: int, k: int) -> str:
    key = pack_key(m_bin, n, k)
    return f"0x{key:016X}ULL"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", required=True,
                   help="Input sweep CSV path(s)")
    p.add_argument("--output", required=True,
                   help="Output .inc file path")
    p.add_argument("--exceptions", default=None,
                   help="JSON file with manual exception overrides")
    p.add_argument("--summary", default=None,
                   help="Human-readable summary output path")
    p.add_argument("--format-name", default=None,
                   help="Override format name (for CSVs without a format column)")
    return p.parse_args()


def load_rows(paths, format_override=None):
    """Load sweep CSV rows, keeping only STD strategy rows with tile_id >= 0."""
    rows = []
    for path in paths:
        p = Path(path)
        with p.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                strategy = raw.get("strategy", "").strip()
                # Only consider STD (standard tile) rows, not AUTO
                if strategy != "STD":
                    continue

                tile_id = int(raw["tile_id"])
                if tile_id < 0:
                    continue

                # Detect format: explicit column, or from filename, or override
                fmt = None
                if "format" in raw and raw["format"].strip():
                    fmt = raw["format"].strip()
                elif format_override:
                    fmt = format_override
                else:
                    # Try to infer from filename: sweep_final_q4_0.csv → Q4_0
                    name = p.stem.upper()
                    for candidate in FORMAT_TO_CODEBOOK:
                        if candidate.replace("_", "").lower() in name.replace("_", "").lower():
                            fmt = candidate
                            break
                    if not fmt:
                        # Try harder: sweep_final_iq4_nl → IQ4_NL
                        for candidate in FORMAT_TO_CODEBOOK:
                            if candidate.lower().replace("_", "") in name.lower().replace("_", ""):
                                fmt = candidate
                                break

                if not fmt:
                    print(f"WARNING: Cannot determine format for {p}, skipping row", file=sys.stderr)
                    continue

                if fmt not in FORMAT_TO_CODEBOOK:
                    print(f"WARNING: Unknown format '{fmt}' in {p}, skipping", file=sys.stderr)
                    continue

                m = int(raw["m"])
                n = int(raw["n"])
                k = int(raw["k"])
                split_k = int(raw.get("split_k", 1))
                min_us = float(raw["min_us"])
                gpu = int(raw.get("gpu", 0))
                tile_name = raw.get("tile", f"tile_{tile_id}")

                rows.append({
                    "format": fmt,
                    "codebook": FORMAT_TO_CODEBOOK[fmt],
                    "shape": raw.get("shape", f"{n}x{k}"),
                    "m": m,
                    "m_bin": bin_m(m),
                    "n": n,
                    "k": k,
                    "tile_id": tile_id,
                    "tile_name": tile_name,
                    "split_k": max(1, split_k),
                    "min_us": min_us,
                    "gpu": gpu,
                })
    return rows


def load_auto_rows(paths, format_override=None):
    """Load AUTO (production heuristic) rows for gap analysis."""
    rows = []
    for path in paths:
        p = Path(path)
        with p.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                strategy = raw.get("strategy", "").strip()
                if strategy != "AUTO":
                    continue

                fmt = None
                if "format" in raw and raw["format"].strip():
                    fmt = raw["format"].strip()
                elif format_override:
                    fmt = format_override
                else:
                    name = p.stem.upper()
                    for candidate in FORMAT_TO_CODEBOOK:
                        if candidate.replace("_", "").lower() in name.replace("_", "").lower():
                            fmt = candidate
                            break

                if not fmt or fmt not in FORMAT_TO_CODEBOOK:
                    continue

                m = int(raw["m"])
                n = int(raw["n"])
                k = int(raw["k"])
                min_us = float(raw["min_us"])

                rows.append({
                    "format": fmt,
                    "codebook": FORMAT_TO_CODEBOOK[fmt],
                    "m": m,
                    "m_bin": bin_m(m),
                    "n": n,
                    "k": k,
                    "min_us": min_us,
                })
    return rows


def pick_best_per_key(rows):
    """For each (codebook, m_bin, n, k), pick the row with lowest min_us.
    When multiple GPUs are present, only compare within same GPU to avoid
    cross-GPU noise."""
    # Group by (codebook, m_bin, n, k, gpu)
    by_key_gpu = defaultdict(list)
    for r in rows:
        key = (r["codebook"], r["m_bin"], r["n"], r["k"], r["gpu"])
        by_key_gpu[key].append(r)

    # For each (codebook, m_bin, n, k), pick global best from across GPUs
    best = {}
    for (cb, m_bin, n, k, gpu), group in by_key_gpu.items():
        group.sort(key=lambda r: r["min_us"])
        winner = group[0]
        dispatch_key = (cb, m_bin, n, k)
        if dispatch_key not in best or winner["min_us"] < best[dispatch_key]["min_us"]:
            best[dispatch_key] = winner

    return best


def load_exceptions(path):
    """Load manual exception overrides from JSON.
    Format: list of {format, m_bin, n, k, tile_id, split_k, comment}
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as f:
        data = json.load(f)

    exceptions = {}
    for entry in data.get("exceptions", []):
        fmt = entry["format"]
        if fmt not in FORMAT_TO_CODEBOOK:
            print(f"WARNING: Unknown format '{fmt}' in exceptions, skipping", file=sys.stderr)
            continue
        cb = FORMAT_TO_CODEBOOK[fmt]
        m_bin = entry["m_bin"]
        n = entry["n"]
        k = entry["k"]
        key = (cb, m_bin, n, k)
        exceptions[key] = {
            "tile_id": entry["tile_id"],
            "split_k": entry.get("split_k", 1),
            "comment": entry.get("comment", "manual override"),
        }
    return exceptions


TILE_NAMES = [
    "T64x64_w2x2",
    "T64x128_w2x2",
    "T64x128_w4x2",
    "T64x128_w2x4",
    "T128x128_w4x2",
    "T128x128_w4x4",
]


def write_inc(out_path, input_paths, best, exceptions, auto_rows):
    """Generate the C++ .inc file with per-codebook dispatch tables."""

    # Merge exceptions into best (exceptions take priority)
    merged = dict(best)  # shallow copy
    exception_count = 0
    for key, exc in exceptions.items():
        if key in merged:
            merged[key] = {
                **merged[key],
                "tile_id": exc["tile_id"],
                "split_k": exc["split_k"],
                "exception": True,
                "comment": exc["comment"],
            }
            exception_count += 1
        else:
            # Exception for a shape not in sweep data — create synthetic entry
            cb, m_bin, n, k = key
            merged[key] = {
                "codebook": cb,
                "format": CODEBOOK_TO_FORMAT.get(cb, f"CB{cb}"),
                "m_bin": m_bin,
                "n": n,
                "k": k,
                "tile_id": exc["tile_id"],
                "split_k": exc["split_k"],
                "tile_name": TILE_NAMES[exc["tile_id"]] if exc["tile_id"] < len(TILE_NAMES) else "?",
                "exception": True,
                "comment": exc["comment"],
            }
            exception_count += 1

    # Group by codebook
    by_cb = defaultdict(list)
    for (cb, m_bin, n, k), entry in merged.items():
        by_cb[cb].append({
            "m_bin": m_bin,
            "n": n,
            "k": k,
            "tile_id": entry["tile_id"],
            "split_k": entry.get("split_k", 1),
            "tile_name": entry.get("tile_name", TILE_NAMES[entry["tile_id"]]),
            "is_exception": entry.get("exception", False),
            "comment": entry.get("comment", ""),
            "packed_key": pack_key(m_bin, n, k),
        })

    # Sort each CB's entries by packed_key for binary search
    for cb in by_cb:
        by_cb[cb].sort(key=lambda e: e["packed_key"])

    # Compute gap statistics vs AUTO
    auto_lookup = {}
    for r in auto_rows:
        key = (r["codebook"], r["m_bin"], r["n"], r["k"])
        auto_lookup[key] = r["min_us"]

    total_entries = sum(len(v) for v in by_cb.values())

    lines = []
    lines.append("// Auto-generated by analyze_cuda_tc_gemm_dispatch.py — DO NOT EDIT")
    lines.append(f"// Source CSVs: {', '.join(str(p) for p in input_paths)}")
    lines.append(f"// Total entries: {total_entries} across {len(by_cb)} codebook(s)")
    lines.append(f"// Manual exceptions applied: {exception_count}")
    lines.append(f"// M bins: {M_BINS}")
    lines.append("")
    lines.append("struct GeneratedPrefillDispatchEntry")
    lines.append("{")
    lines.append("    uint64_t packed_key;  // (M_bin << 40) | (K << 20) | N")
    lines.append("    uint8_t tile_id;      // TileId enum value (0..5)")
    lines.append("    uint8_t split_k;      // split-K factor (1, 2, 4, 8)")
    lines.append("};")
    lines.append("")
    lines.append("inline constexpr uint64_t packPrefillDispatchKey(int M_bin, int N, int K)")
    lines.append("{")
    lines.append("    return (static_cast<uint64_t>(M_bin) << 40) |")
    lines.append("           (static_cast<uint64_t>(K & 0xFFFFF) << 20) |")
    lines.append("           static_cast<uint64_t>(N & 0xFFFFF);")
    lines.append("}")
    lines.append("")
    lines.append("inline constexpr int binPrefillM(int M)")
    lines.append("{")
    lines.append("    if (M <= 96) return 64;")
    lines.append("    if (M <= 192) return 128;")
    lines.append("    if (M <= 384) return 256;")
    lines.append("    return 512;")
    lines.append("}")
    lines.append("")
    lines.append("template <size_t Count>")
    lines.append("inline bool findPrefillDispatchEntry(")
    lines.append("    const GeneratedPrefillDispatchEntry (&table)[Count],")
    lines.append("    uint64_t packed_key,")
    lines.append("    uint8_t &out_tile_id,")
    lines.append("    uint8_t &out_split_k)")
    lines.append("{")
    lines.append("    size_t lo = 0;")
    lines.append("    size_t hi = Count;")
    lines.append("    while (lo < hi)")
    lines.append("    {")
    lines.append("        const size_t mid = lo + ((hi - lo) / 2);")
    lines.append("        const uint64_t candidate = table[mid].packed_key;")
    lines.append("        if (candidate == packed_key)")
    lines.append("        {")
    lines.append("            out_tile_id = table[mid].tile_id;")
    lines.append("            out_split_k = table[mid].split_k;")
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
    lines.append("// Per-codebook dispatch: returns true if a known-shape match was found.")
    lines.append("// On match, out_tile_id and out_split_k are set to the optimal config.")
    lines.append("template <uint8_t CB>")
    lines.append("inline bool selectPrefillTileGenerated(int M, int N, int K,")
    lines.append("                                       uint8_t &out_tile_id, uint8_t &out_split_k)")
    lines.append("{")
    lines.append("    const int m_bin = binPrefillM(M);")
    lines.append("    const uint64_t packed_key = packPrefillDispatchKey(m_bin, N, K);")

    first_cb = True
    for cb in sorted(by_cb):
        prefix = "    if constexpr" if first_cb else "    else if constexpr"
        first_cb = False
        fmt_name = CODEBOOK_TO_FORMAT.get(cb, f"CB{cb}")
        entries = by_cb[cb]
        lines.append(f"    {prefix} (CB == {cb}) {{ // {fmt_name} ({len(entries)} entries)")
        lines.append("        static constexpr GeneratedPrefillDispatchEntry kTable[] = {")
        for e in entries:
            exc_marker = " [EXCEPTION]" if e["is_exception"] else ""
            comment = f" // M={e['m_bin']} {e['n']}x{e['k']} {e['tile_name']} sk={e['split_k']}{exc_marker}"
            if e.get("comment"):
                comment += f" ({e['comment']})"
            lines.append(
                f"            {{ {format_packed_key(e['m_bin'], e['n'], e['k'])}, "
                f"{e['tile_id']}, {e['split_k']} }},{comment}")
        lines.append("        };")
        lines.append("        return findPrefillDispatchEntry(kTable, packed_key, out_tile_id, out_split_k);")
        lines.append("    }")

    lines.append("    return false;")
    lines.append("}")
    lines.append("")

    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path} ({total_entries} entries, {len(by_cb)} codebooks, {exception_count} exceptions)")


def write_summary(summary_path, best, exceptions, auto_rows, input_paths):
    """Write a human-readable summary of the dispatch decisions."""
    if not summary_path:
        return

    auto_lookup = {}
    for r in auto_rows:
        key = (r["codebook"], r["m_bin"], r["n"], r["k"])
        auto_lookup[key] = r["min_us"]

    lines = []
    lines.append("=" * 90)
    lines.append("  GEMM PREFILL DISPATCH SUMMARY")
    lines.append(f"  Source CSVs: {', '.join(str(p) for p in input_paths)}")
    lines.append("=" * 90)
    lines.append("")

    # Group by codebook
    by_cb = defaultdict(list)
    for (cb, m_bin, n, k), entry in best.items():
        auto_us = auto_lookup.get((cb, m_bin, n, k))
        gap_pct = None
        if auto_us and auto_us > 0:
            gap_pct = (entry["min_us"] / auto_us - 1.0) * 100.0
        by_cb[cb].append({
            "m_bin": m_bin, "n": n, "k": k,
            "tile_id": entry["tile_id"],
            "tile_name": entry.get("tile_name", "?"),
            "split_k": entry.get("split_k", 1),
            "min_us": entry["min_us"],
            "auto_us": auto_us,
            "gap_pct": gap_pct,
            "is_exception": (cb, m_bin, n, k) in exceptions,
        })

    for cb in sorted(by_cb):
        fmt = CODEBOOK_TO_FORMAT.get(cb, f"CB{cb}")
        entries = sorted(by_cb[cb], key=lambda e: (e["n"], e["k"], e["m_bin"]))
        lines.append(f"--- {fmt} (CB={cb}) ---")
        lines.append(f"{'Shape':>20s}  {'M':>4s}  {'Tile':>18s}  {'SK':>2s}  {'Best(us)':>9s}  {'Auto(us)':>9s}  {'Gap':>7s}  {'Exc':>3s}")
        for e in entries:
            shape = f"{e['n']}x{e['k']}"
            gap = f"{e['gap_pct']:+6.1f}%" if e['gap_pct'] is not None else "N/A"
            auto = f"{e['auto_us']:.3f}" if e['auto_us'] else "N/A"
            exc = "YES" if e["is_exception"] else ""
            lines.append(
                f"{shape:>20s}  {e['m_bin']:>4d}  {e['tile_name']:>18s}  {e['split_k']:>2d}  "
                f"{e['min_us']:>9.3f}  {auto:>9s}  {gap:>7s}  {exc:>3s}")

        # Summary stats
        gaps = [e["gap_pct"] for e in entries if e["gap_pct"] is not None]
        if gaps:
            max_gap = max(gaps)
            worse_2pct = sum(1 for g in gaps if g > 2.0)
            lines.append(f"  Max gap: {max_gap:+.1f}%  |  Shapes >2%: {worse_2pct}/{len(gaps)}")
        lines.append("")

    Path(summary_path).write_text("\n".join(lines) + "\n")
    print(f"Wrote summary to {summary_path}")


def main():
    args = parse_args()

    rows = load_rows(args.input, format_override=args.format_name)
    auto_rows = load_auto_rows(args.input, format_override=args.format_name)

    if not rows:
        print("ERROR: No STD rows found in input CSVs", file=sys.stderr)
        sys.exit(1)

    # Pick best per (codebook, m_bin, n, k)
    best = pick_best_per_key(rows)
    print(f"Loaded {len(rows)} STD rows → {len(best)} unique dispatch entries")

    # Load manual exceptions
    exceptions = load_exceptions(args.exceptions)
    if exceptions:
        print(f"Loaded {len(exceptions)} manual exceptions")

    # Generate .inc file
    write_inc(args.output, args.input, best, exceptions, auto_rows)

    # Generate summary
    write_summary(args.summary, best, exceptions, auto_rows, args.input)

    print("Done.")


if __name__ == "__main__":
    main()
