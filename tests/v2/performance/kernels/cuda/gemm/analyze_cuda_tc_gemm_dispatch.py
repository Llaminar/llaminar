#!/usr/bin/env python3
"""
Generate CUDA native-VNNI GEMM prefill dispatch heuristics from sweep CSVs.

Reads one or more tile-sweep CSVs (produced by TileSweep_AllStrategies test),
picks the best (tile_id, split_k) per (codebook, M_key, N, K) combination,
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
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from native_vnni_codebooks import CODEBOOK_TO_FORMAT, FORMAT_TO_CODEBOOK, infer_format_from_filename

REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_POLICY_HEADER = REPO_ROOT / "src/v2/utils/PrefillGraphBucketDefaults.h"


def _parse_int_array_from_header(text: str, symbol: str) -> list[int]:
    pattern = rf"{re.escape(symbol)}[^=]*=\s*\{{([^}}]+)\}}"
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if not match:
        raise SystemExit(f"could not find {symbol} in {DEFAULT_POLICY_HEADER}")
    values = [int(value) for value in re.findall(r"-?\d+", match.group(1))]
    if not values:
        raise SystemExit(f"{symbol} in {DEFAULT_POLICY_HEADER} was empty")
    return values


def load_default_m_policy(header_path: Path = DEFAULT_POLICY_HEADER) -> list[int]:
    """Read the runtime MTP/prefill bucket policy from the C++ header."""
    text = header_path.read_text()
    small = _parse_int_array_from_header(text, "kDefaultNativeVNNISmallMRows")
    buckets = _parse_int_array_from_header(text, "kDefaultPrefillGraphBucketSizes")
    policy = sorted({m for m in [*small, *buckets] if m > 0})
    if not policy:
        raise SystemExit("empty NativeVNNI dispatch M policy")
    return policy


def canonical_branch_label(codebook: int) -> str:
    """Return the single alias accepted by generated-dispatch validators."""
    return CODEBOOK_TO_FORMAT.get(codebook, f"CB{codebook}").split("/")[0]


def pack_key(m_key: int, n: int, k: int) -> int:
    """Pack (M_key, N, K) into a 64-bit lookup key.
    Layout: [63:40] M_key (24 bits) | [39:20] K (20 bits) | [19:0] N (20 bits)
    """
    return (m_key << 40) | ((k & 0xFFFFF) << 20) | (n & 0xFFFFF)


def unpack_key(packed_key: int) -> tuple[int, int, int]:
    m_key = packed_key >> 40
    k = (packed_key >> 20) & 0xFFFFF
    n = packed_key & 0xFFFFF
    return m_key, n, k


def format_packed_key(m_key: int, n: int, k: int) -> str:
    key = pack_key(m_key, n, k)
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
    p.add_argument("--base-include", default=None,
                   help="Existing generated include to merge before applying new CSV winners")
    p.add_argument("--m-policy-header", default=str(DEFAULT_POLICY_HEADER),
                   help="C++ header containing canonical NativeVNNI MTP/prefill bucket policy")
    p.add_argument("--include-off-policy-m", action="store_true",
                   help="Include CSV rows whose M is not in the canonical policy list")
    return p.parse_args()


def load_rows(paths, format_override=None, m_policy=None, include_off_policy_m=False):
    """Load sweep CSV rows, keeping only STD strategy rows with tile_id >= 0."""
    rows = []
    skipped_off_policy = 0
    policy_set = set(m_policy or [])
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
                    fmt = infer_format_from_filename(p)

                if not fmt:
                    print(f"WARNING: Cannot determine format for {p}, skipping row", file=sys.stderr)
                    continue

                if fmt not in FORMAT_TO_CODEBOOK:
                    print(f"WARNING: Unknown format '{fmt}' in {p}, skipping", file=sys.stderr)
                    continue

                codebook = FORMAT_TO_CODEBOOK[fmt]
                if "codebook" in raw and raw["codebook"].strip():
                    csv_codebook = int(raw["codebook"])
                    if csv_codebook != codebook:
                        raise SystemExit(
                            f"codebook mismatch in {p}: format {fmt} maps to {codebook}, CSV row has {csv_codebook}")
                    codebook = csv_codebook

                m = int(raw["m"])
                n = int(raw["n"])
                k = int(raw["k"])
                split_k = int(raw.get("split_k", 1))
                min_us = float(raw["min_us"])
                gpu = int(raw.get("gpu", 0))
                tile_name = raw.get("tile", f"tile_{tile_id}")

                if policy_set and not include_off_policy_m and m not in policy_set:
                    skipped_off_policy += 1
                    continue

                rows.append({
                    "format": fmt,
                    "codebook": codebook,
                    "shape": raw.get("shape", f"{n}x{k}"),
                    "m": m,
                    "m_key": m,
                    "n": n,
                    "k": k,
                    "tile_id": tile_id,
                    "tile_name": tile_name,
                    "split_k": max(1, split_k),
                    "min_us": min_us,
                    "gpu": gpu,
                })
    if skipped_off_policy:
        print(f"Skipped {skipped_off_policy} off-policy M row(s); "
              "sweep canonical bucket sizes or pass --include-off-policy-m")
    return rows


def load_auto_rows(paths, format_override=None, m_policy=None, include_off_policy_m=False):
    """Load AUTO (production heuristic) rows for gap analysis."""
    rows = []
    policy_set = set(m_policy or [])
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
                    fmt = infer_format_from_filename(p)

                if not fmt or fmt not in FORMAT_TO_CODEBOOK:
                    continue

                codebook = FORMAT_TO_CODEBOOK[fmt]
                if "codebook" in raw and raw["codebook"].strip():
                    csv_codebook = int(raw["codebook"])
                    if csv_codebook != codebook:
                        raise SystemExit(
                            f"codebook mismatch in {p}: format {fmt} maps to {codebook}, CSV row has {csv_codebook}")
                    codebook = csv_codebook

                m = int(raw["m"])
                n = int(raw["n"])
                k = int(raw["k"])
                min_us = float(raw["min_us"])

                if policy_set and not include_off_policy_m and m not in policy_set:
                    continue

                rows.append({
                    "format": fmt,
                    "codebook": codebook,
                    "m": m,
                    "m_key": m,
                    "n": n,
                    "k": k,
                    "min_us": min_us,
                })
    return rows


def pick_best_per_key(rows):
    """For each (codebook, m_key, n, k), pick the row with lowest min_us.
    When multiple GPUs are present, only compare within same GPU to avoid
    cross-GPU noise."""
    # Group by (codebook, m_key, n, k, gpu)
    by_key_gpu = defaultdict(list)
    for r in rows:
        key = (r["codebook"], r["m_key"], r["n"], r["k"], r["gpu"])
        by_key_gpu[key].append(r)

    # For each (codebook, m_key, n, k), pick global best from across GPUs
    best = {}
    for (cb, m_key, n, k, gpu), group in by_key_gpu.items():
        group.sort(key=lambda r: r["min_us"])
        winner = group[0]
        dispatch_key = (cb, m_key, n, k)
        if dispatch_key not in best or winner["min_us"] < best[dispatch_key]["min_us"]:
            best[dispatch_key] = winner

    return best


def load_exceptions(path):
    """Load manual exception overrides from JSON.
    Format: list of {format, m_key, n, k, tile_id, split_k, comment}
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
        m_key = entry.get("m_key", entry.get("m_bin"))
        if m_key is None:
            raise SystemExit("exception entry must include m_key")
        n = entry["n"]
        k = entry["k"]
        key = (cb, int(m_key), n, k)
        exceptions[key] = {
            "tile_id": entry["tile_id"],
            "split_k": entry.get("split_k", 1),
            "comment": entry.get("comment", "manual override"),
        }
    return exceptions


def load_base_include(path):
    """Parse an existing generated include into dispatch entries.

    This supports incremental training: focused sweeps overlay updated winners
    while historical generated coverage remains intact.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}

    entries = {}
    current_cb = None
    cb_pattern = re.compile(r"CB\s*==\s*(\d+)")
    entry_pattern = re.compile(
        r"\{\s*(0x[0-9A-Fa-f]+)ULL,\s*(\d+),\s*(\d+)\s*\}")

    for line in p.read_text().splitlines():
        cb_match = cb_pattern.search(line)
        if cb_match:
            current_cb = int(cb_match.group(1))
            continue
        if current_cb is None:
            continue
        entry_match = entry_pattern.search(line)
        if not entry_match:
            continue

        packed = int(entry_match.group(1), 16)
        m_key, n, k = unpack_key(packed)
        tile_id = int(entry_match.group(2))
        split_k = int(entry_match.group(3))
        entries[(current_cb, m_key, n, k)] = {
            "codebook": current_cb,
            "format": CODEBOOK_TO_FORMAT.get(current_cb, f"CB{current_cb}"),
            "shape": "base_include",
            "m": m_key,
            "m_key": m_key,
            "n": n,
            "k": k,
            "tile_id": tile_id,
            "tile_name": TILE_NAMES[tile_id] if 0 <= tile_id < len(TILE_NAMES) else "?",
            "split_k": split_k,
            "min_us": 0.0,
            "gpu": 0,
            "base_include": True,
        }
    if entries:
        print(f"Loaded {len(entries)} base dispatch entries from {p}")
    return entries


TILE_NAMES = [
    "T64x64_w2x2",
    "T64x128_w2x2",
    "T64x128_w4x2",
    "T64x128_w2x4",
    "T128x128_w4x2",
    "T128x128_w4x4",
]


def write_inc(out_path, input_paths, best, exceptions, auto_rows, m_policy):
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
            cb, m_key, n, k = key
            merged[key] = {
                "codebook": cb,
                "format": CODEBOOK_TO_FORMAT.get(cb, f"CB{cb}"),
                "m_key": m_key,
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
    for (cb, m_key, n, k), entry in merged.items():
        by_cb[cb].append({
            "m_key": m_key,
            "n": n,
            "k": k,
            "tile_id": entry["tile_id"],
            "split_k": entry.get("split_k", 1),
            "tile_name": entry.get("tile_name", TILE_NAMES[entry["tile_id"]]),
            "is_exception": entry.get("exception", False),
            "comment": entry.get("comment", ""),
            "packed_key": pack_key(m_key, n, k),
        })

    # Sort each CB's entries by packed_key for binary search
    for cb in by_cb:
        by_cb[cb].sort(key=lambda e: e["packed_key"])

    # Compute gap statistics vs AUTO
    auto_lookup = {}
    for r in auto_rows:
        key = (r["codebook"], r["m_key"], r["n"], r["k"])
        auto_lookup[key] = r["min_us"]

    total_entries = sum(len(v) for v in by_cb.values())

    lines = []
    lines.append("// Auto-generated by analyze_cuda_tc_gemm_dispatch.py — DO NOT EDIT")
    lines.append(f"// Source CSVs: {', '.join(str(p) for p in input_paths)}")
    lines.append(f"// Total entries: {total_entries} across {len(by_cb)} codebook(s)")
    lines.append(f"// Manual exceptions applied: {exception_count}")
    lines.append(f"// Dispatch M policy: {m_policy}")
    lines.append("")
    lines.append("struct GeneratedPrefillDispatchEntry")
    lines.append("{")
    lines.append("    uint64_t packed_key;  // (M_key << 40) | (K << 20) | N")
    lines.append("    uint8_t tile_id;      // TileId enum value (0..5)")
    lines.append("    uint8_t split_k;      // split-K factor (1, 2, 4, 8)")
    lines.append("};")
    lines.append("")
    lines.append("inline constexpr uint64_t packPrefillDispatchKey(int M_key, int N, int K)")
    lines.append("{")
    lines.append("    return (static_cast<uint64_t>(M_key) << 40) |")
    lines.append("           (static_cast<uint64_t>(K & 0xFFFFF) << 20) |")
    lines.append("           static_cast<uint64_t>(N & 0xFFFFF);")
    lines.append("}")
    lines.append("")
    lines.append("inline int selectPrefillDispatchMKey(int M)")
    lines.append("{")
    lines.append("    static constexpr int kMPolicy[] = {")
    policy_chunks = [m_policy[i:i + 12] for i in range(0, len(m_policy), 12)]
    for chunk in policy_chunks:
        lines.append("        " + ", ".join(str(v) for v in chunk) + ",")
    lines.append("    };")
    lines.append("    for (int key : kMPolicy)")
    lines.append("    {")
    lines.append("        if (M <= key)")
    lines.append("            return key;")
    lines.append("    }")
    lines.append("    return M;")
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
    lines.append("    const int m_key = selectPrefillDispatchMKey(M);")
    lines.append("    const uint64_t packed_key = packPrefillDispatchKey(m_key, N, K);")

    first_cb = True
    for cb in sorted(by_cb):
        prefix = "if constexpr" if first_cb else "else if constexpr"
        first_cb = False
        fmt_name = canonical_branch_label(cb)
        entries = by_cb[cb]
        lines.append(f"    {prefix} (CB == {cb}) {{ // {fmt_name} ({len(entries)} entries)")
        lines.append("        static constexpr GeneratedPrefillDispatchEntry kTable[] = {")
        for e in entries:
            exc_marker = " [EXCEPTION]" if e["is_exception"] else ""
            comment = f" // M={e['m_key']} {e['n']}x{e['k']} {e['tile_name']} sk={e['split_k']}{exc_marker}"
            if e.get("comment"):
                comment += f" ({e['comment']})"
            lines.append(
                f"            {{ {format_packed_key(e['m_key'], e['n'], e['k'])}, "
                f"{e['tile_id']}, {e['split_k']} }},{comment}")
        lines.append("        };")
        lines.append("        return findPrefillDispatchEntry(kTable, packed_key, out_tile_id, out_split_k);")
        lines.append("    }")

    lines.append("    return false;")
    lines.append("}")

    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path} ({total_entries} entries, {len(by_cb)} codebooks, {exception_count} exceptions)")


def write_summary(summary_path, best, exceptions, auto_rows, input_paths):
    """Write a human-readable summary of the dispatch decisions."""
    if not summary_path:
        return

    auto_lookup = {}
    for r in auto_rows:
        key = (r["codebook"], r["m_key"], r["n"], r["k"])
        auto_lookup[key] = r["min_us"]

    lines = []
    lines.append("=" * 90)
    lines.append("  GEMM PREFILL DISPATCH SUMMARY")
    lines.append(f"  Source CSVs: {', '.join(str(p) for p in input_paths)}")
    lines.append("=" * 90)
    lines.append("")

    # Group by codebook
    by_cb = defaultdict(list)
    for (cb, m_key, n, k), entry in best.items():
        auto_us = auto_lookup.get((cb, m_key, n, k))
        gap_pct = None
        if auto_us and auto_us > 0:
            gap_pct = (entry["min_us"] / auto_us - 1.0) * 100.0
        by_cb[cb].append({
            "m_key": m_key, "n": n, "k": k,
            "tile_id": entry["tile_id"],
            "tile_name": entry.get("tile_name", "?"),
            "split_k": entry.get("split_k", 1),
            "min_us": entry["min_us"],
            "auto_us": auto_us,
            "gap_pct": gap_pct,
            "is_exception": (cb, m_key, n, k) in exceptions,
        })

    for cb in sorted(by_cb):
        fmt = CODEBOOK_TO_FORMAT.get(cb, f"CB{cb}")
        entries = sorted(by_cb[cb], key=lambda e: (e["n"], e["k"], e["m_key"]))
        lines.append(f"--- {fmt} (CB={cb}) ---")
        lines.append(f"{'Shape':>20s}  {'M':>4s}  {'Tile':>18s}  {'SK':>2s}  {'Best(us)':>9s}  {'Auto(us)':>9s}  {'Gap':>7s}  {'Exc':>3s}")
        for e in entries:
            shape = f"{e['n']}x{e['k']}"
            gap = f"{e['gap_pct']:+6.1f}%" if e['gap_pct'] is not None else "N/A"
            auto = f"{e['auto_us']:.3f}" if e['auto_us'] else "N/A"
            exc = "YES" if e["is_exception"] else ""
            lines.append(
                f"{shape:>20s}  {e['m_key']:>4d}  {e['tile_name']:>18s}  {e['split_k']:>2d}  "
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
    m_policy = load_default_m_policy(Path(args.m_policy_header))

    rows = load_rows(
        args.input,
        format_override=args.format_name,
        m_policy=m_policy,
        include_off_policy_m=args.include_off_policy_m)
    auto_rows = load_auto_rows(
        args.input,
        format_override=args.format_name,
        m_policy=m_policy,
        include_off_policy_m=args.include_off_policy_m)

    if not rows:
        print("ERROR: No STD rows found in input CSVs", file=sys.stderr)
        sys.exit(1)

    # Pick best per (codebook, m_key, n, k)
    best = pick_best_per_key(rows)
    print(f"Loaded {len(rows)} STD rows → {len(best)} unique dispatch entries")

    base = load_base_include(args.base_include)
    if base:
        base.update(best)
        best = base
        print(f"Merged dispatch table has {len(best)} unique entries")

    # Load manual exceptions
    exceptions = load_exceptions(args.exceptions)
    if exceptions:
        print(f"Loaded {len(exceptions)} manual exceptions")

    # Generate .inc file
    write_inc(args.output, args.input, best, exceptions, auto_rows, m_policy)

    # Generate summary
    write_summary(args.summary, best, exceptions, auto_rows, args.input)

    print("Done.")


if __name__ == "__main__":
    main()
