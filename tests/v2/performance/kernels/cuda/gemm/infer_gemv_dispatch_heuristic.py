#!/usr/bin/env python3
"""
Infer shape-agnostic GEMV dispatch heuristic from sweep CSV data.

Approach: Per-codebook set-cover + decision tree.

1. For each codebook (format), use greedy set cover to find a minimal set
   of "universal good" configs that cover all shapes within 3% of optimal.
2. Relabel each shape with its best config from the set-cover set.
3. Train a sklearn DecisionTreeClassifier on (log2(N), log2(K)) -> config_id.
4. Export the tree as C++ if/else for the .inc file.

This eliminates per-shape lookup tables while maintaining <3% BW penalty.

Usage:
    python infer_gemv_dispatch_heuristic.py \
        --input /tmp/llaminar_cuda_tc_gemv_sweep_expanded_20260312.csv \
        --output src/v2/kernels/cuda/gemm/CUDANativeVNNIGemvDispatchHeuristicGenerated.inc
"""

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from native_vnni_codebooks import CODEBOOK_PAYLOAD_BYTES, CODEBOOK_TO_FORMAT, FORMAT_TO_CODEBOOK


# --- Codebook metadata ------------------------------------------------
FAMILY_MAP = {"kpar": "KPAR", "wide": "WIDE", "direct": "DIRECT"}


# --- Data types -------------------------------------------------------
@dataclass
class FormatTreeResult:
    fmt: str
    cb: int
    n_shapes: int
    n_raw_configs: int
    n_master_configs: int
    master_configs: list         # config tuples
    tree: DecisionTreeClassifier
    depth: int
    n_leaves: int
    mean_penalty_pct: float
    p95_penalty_pct: float
    max_penalty_pct: float
    shape_labels: dict           # (N,K) -> master_config_idx


# --- Data loading -----------------------------------------------------
def load_sweep_csv(path: Path) -> list[dict]:
    rows = []
    with path.open("r", newline="") as f:
        for raw in csv.DictReader(f):
            fmt = raw.get("format", "")
            if fmt not in FORMAT_TO_CODEBOOK:
                continue
            codebook = FORMAT_TO_CODEBOOK[fmt]
            if raw.get("codebook", "").strip():
                csv_codebook = int(raw["codebook"])
                if csv_codebook != codebook:
                    raise SystemExit(
                        f"codebook mismatch in {path}: format {fmt} maps to {codebook}, row has {csv_codebook}")
            try:
                rows.append({
                    "fmt": fmt, "cb": codebook,
                    "n": int(raw["n"]), "k": int(raw["k"]),
                    "family": raw.get("family", "kpar"),
                    "tile_n": int(raw.get("tile_n", 128)),
                    "cpt": int(raw.get("cpt", 1)),
                    "tw": int(raw.get("target_waves", 8)),
                    "mkg": int(raw.get("mkg", 2)),
                    "max_kb": int(raw.get("max_kb", 0)),
                    "f2p": int(raw.get("force_two_phase", 0)),
                    "bw": float(raw["eff_bw_gbs"]),
                    "is_best": int(raw.get("is_best", 0)) == 1,
                })
            except (KeyError, ValueError):
                continue
    return rows


def cfg_tuple(r: dict) -> tuple:
    return (r["family"], r["tile_n"], r["cpt"], r["tw"], r["mkg"], r["max_kb"], r["f2p"])


# --- Per-format set-cover + tree training ------------------------------
def train_format_tree(fmt: str, all_rows: list[dict],
                      threshold: float = 0.03) -> FormatTreeResult:
    """Train a decision tree for a single format using set-cover relabeling."""

    cb = FORMAT_TO_CODEBOOK[fmt]
    fmt_all = [r for r in all_rows if r["fmt"] == fmt]
    fmt_best = [r for r in fmt_all if r["is_best"]]

    # Oracle BW per (N,K)
    nk_oracle: dict[tuple, float] = {}
    nk_best_cfg: dict[tuple, tuple] = {}
    for r in fmt_best:
        nk = (r["n"], r["k"])
        if nk not in nk_oracle or r["bw"] > nk_oracle[nk]:
            nk_oracle[nk] = r["bw"]
            nk_best_cfg[nk] = cfg_tuple(r)

    shapes = sorted(nk_oracle.keys())
    n_raw = len(set(nk_best_cfg.values()))

    # BW lookup: (N,K,config) -> best BW
    nk_cfg_bw: dict[tuple, float] = defaultdict(float)
    for r in fmt_all:
        key = ((r["n"], r["k"]), cfg_tuple(r))
        nk_cfg_bw[key] = max(nk_cfg_bw[key], r["bw"])

    all_cfgs = sorted(set(cfg_tuple(r) for r in fmt_all))

    # Greedy set cover
    config_coverage = {}
    for cfg in all_cfgs:
        covered = set()
        for nk in shapes:
            bw = nk_cfg_bw.get((nk, cfg), 0)
            if nk_oracle[nk] > 0 and bw >= (1 - threshold) * nk_oracle[nk]:
                covered.add(nk)
        config_coverage[cfg] = covered

    remaining = set(shapes)
    master_configs = []
    while remaining:
        best = max(all_cfgs, key=lambda c: len(config_coverage[c] & remaining))
        newly = config_coverage[best] & remaining
        if not newly:
            break
        master_configs.append(best)
        remaining -= newly

    # Relabel: each (N,K) -> master config with highest BW
    nk_to_master = {}
    for nk in shapes:
        best_idx, best_bw = 0, 0.0
        for i, cfg in enumerate(master_configs):
            bw = nk_cfg_bw.get((nk, cfg), 0)
            if bw > best_bw:
                best_bw = bw
                best_idx = i
        nk_to_master[nk] = best_idx

    # Train tree
    X = np.array([[math.log2(nk[0]), math.log2(nk[1])] for nk in shapes],
                 dtype=np.float32)
    y = np.array([nk_to_master[nk] for nk in shapes], dtype=np.int32)

    tree = DecisionTreeClassifier(min_samples_leaf=1, min_samples_split=2)
    tree.fit(X, y)
    pred = tree.predict(X)

    # Evaluate BW penalty
    penalties = []
    for i, nk in enumerate(shapes):
        oracle_bw = nk_oracle[nk]
        pred_cfg = master_configs[pred[i]]
        pred_bw = nk_cfg_bw.get((nk, pred_cfg), 0)
        if oracle_bw > 0:
            penalties.append(max(0.0, 1.0 - pred_bw / oracle_bw))

    return FormatTreeResult(
        fmt=fmt, cb=cb,
        n_shapes=len(shapes),
        n_raw_configs=n_raw,
        n_master_configs=len(master_configs),
        master_configs=master_configs,
        tree=tree,
        depth=tree.get_depth(),
        n_leaves=tree.get_n_leaves(),
        mean_penalty_pct=100 * np.mean(penalties),
        p95_penalty_pct=100 * np.percentile(penalties, 95) if penalties else 0,
        max_penalty_pct=100 * max(penalties) if penalties else 0,
        shape_labels=nk_to_master,
    )


# --- C++ code generation from sklearn tree ----------------------------
def _tree_to_cpp(tree: DecisionTreeClassifier,
                 feature_names: list[str],
                 decode_fn, indent: int = 1) -> list[str]:
    """Recursively convert a sklearn DecisionTree to nested C++ if/else."""
    t = tree.tree_
    lines = []

    def recurse(node_id: int, depth: int):
        pad = "    " * depth
        if t.children_left[node_id] == t.children_right[node_id]:
            # Leaf
            class_id = int(np.argmax(t.value[node_id][0]))
            lines.append(f"{pad}{decode_fn(class_id)}")
        else:
            feat_idx = t.feature[node_id]
            threshold = t.threshold[node_id]
            feat_name = feature_names[feat_idx]
            thresh_str = f"{threshold:.6f}f"

            lines.append(f"{pad}if ({feat_name} <= {thresh_str})")
            lines.append(f"{pad}{{")
            recurse(t.children_left[node_id], depth + 1)
            lines.append(f"{pad}}}")
            lines.append(f"{pad}else")
            lines.append(f"{pad}{{")
            recurse(t.children_right[node_id], depth + 1)
            lines.append(f"{pad}}}")

    recurse(0, indent)
    return lines


def generate_cpp(results: list[FormatTreeResult], input_path: Path) -> str:
    """Generate the complete C++ .inc file with per-CB tree dispatch.

    Provides the exact API consumed by CUDANativeVNNIGemvTuned.cu:
      template <uint8_t CB> classifyShapeGenerated(int N, int K) -> NativeGemvShape
      template <uint8_t CB> selectGeneratedTuning(int N, int K) -> GeneratedDispatchTuning
    """
    lines = []

    # Compute overall stats
    all_pen = []
    for r in results:
        all_pen.extend([r.mean_penalty_pct] * r.n_shapes)
    overall_mean = np.mean(all_pen) if all_pen else 0
    overall_max = max(r.max_penalty_pct for r in results)

    # Header
    lines.append("// clang-format off")
    lines.append("// Auto-generated by infer_gemv_dispatch_heuristic.py")
    lines.append(f"// Source CSV: {input_path.name}")
    lines.append(f"// Method: per-codebook set-cover (3%) + decision tree")
    lines.append(f"// Overall: mean BW penalty {overall_mean:.2f}%, max {overall_max:.2f}%")
    lines.append(f"// Formats: {len(results)}")
    for r in results:
        lines.append(f"//   CB={r.cb:2d} ({r.fmt:8s}): {r.n_master_configs:2d} configs, "
                      f"depth={r.depth:2d}, leaves={r.n_leaves:2d}, "
                      f"mean={r.mean_penalty_pct:.2f}% max={r.max_penalty_pct:.2f}%")
    lines.append("//")
    lines.append("// Dispatch is shape-agnostic: rules are based on")
    lines.append("//   log2f(N), log2f(K)  (continuous shape features)")
    lines.append("// No per-model shape lookup tables.")
    lines.append("")
    lines.append("#include <cmath>  // log2f")
    lines.append("")

    # Struct definition (same as before — consumed by CUDANativeVNNIGemvTuned.cu)
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

    # ---- Per-CB internal helper functions ----
    for r in results:
        lines.append(f"// --- CB={r.cb} ({r.fmt}) ---")
        lines.append(f"// {r.n_master_configs} master configs, "
                      f"depth={r.depth}, leaves={r.n_leaves}, "
                      f"mean_bw_penalty={r.mean_penalty_pct:.2f}%")
        lines.append(f"inline __host__ void selectTuning_CB{r.cb}(")
        lines.append(f"    float log2_n, float log2_k,")
        lines.append(f"    NativeGemvShape& out_shape, GeneratedDispatchTuning& out_tuning)")
        lines.append("{")

        # Capture r in a local to avoid late-binding closure issue with loop var
        master_cfgs = r.master_configs

        def decode_config(class_id, _cfgs=master_cfgs):
            cfg = _cfgs[class_id]
            family, tile_n, cpt, tw, mkg, max_kb, f2p = cfg
            fam_cpp = FAMILY_MAP.get(family, "KPAR")
            return (f"out_shape = NativeGemvShape::{fam_cpp}; "
                    f"out_tuning = {{ {tile_n}, {cpt}, {tw}, {mkg}, {max_kb}, {f2p} }}; "
                    f"return;")

        tree_lines = _tree_to_cpp(r.tree, ["log2_n", "log2_k"],
                                  decode_config, indent=1)
        lines.extend(tree_lines)
        lines.append("}")
        lines.append("")

    # ---- Internal combined dispatcher ----
    lines.append("// Internal: combined shape + tuning dispatch via per-CB tree")
    lines.append("template <uint8_t CB>")
    lines.append("inline __host__ void selectGeneratedDispatch_(")
    lines.append("    int N, int K,")
    lines.append("    NativeGemvShape& out_shape, GeneratedDispatchTuning& out_tuning)")
    lines.append("{")
    lines.append("    const float log2_n = log2f(static_cast<float>(N > 0 ? N : 1));")
    lines.append("    const float log2_k = log2f(static_cast<float>(K > 0 ? K : 1));")
    lines.append("")

    first = True
    for r in results:
        kw = "if constexpr" if first else "else if constexpr"
        lines.append(f"    {kw} (CB == {r.cb})")
        lines.append(f"    {{")
        lines.append(f"        selectTuning_CB{r.cb}(log2_n, log2_k, out_shape, out_tuning);")
        lines.append(f"    }}")
        first = False

    # Fallback for any unsupported CB
    lines.append(f"    else")
    lines.append(f"    {{")
    lines.append(f"        out_shape = NativeGemvShape::KPAR;")
    lines.append(f"        out_tuning = {{ 128, 1, 16, 2, 0, 2 }};")
    lines.append(f"    }}")
    lines.append("}")
    lines.append("")

    # ---- Public API: classifyShapeGenerated<CB>(N, K) ----
    lines.append("// Public API consumed by dispatchCodebook<CB>()")
    lines.append("template <uint8_t CB>")
    lines.append("inline NativeGemvShape classifyShapeGenerated(int N, int K)")
    lines.append("{")
    lines.append("    NativeGemvShape shape;")
    lines.append("    GeneratedDispatchTuning tuning;")
    lines.append("    selectGeneratedDispatch_<CB>(N, K, shape, tuning);")
    lines.append("    return shape;")
    lines.append("}")
    lines.append("")

    # ---- Public API: selectGeneratedTuning<CB>(N, K) ----
    lines.append("template <uint8_t CB>")
    lines.append("inline GeneratedDispatchTuning selectGeneratedTuning(int N, int K)")
    lines.append("{")
    lines.append("    NativeGemvShape shape;")
    lines.append("    GeneratedDispatchTuning tuning;")
    lines.append("    selectGeneratedDispatch_<CB>(N, K, shape, tuning);")
    lines.append("    return tuning;")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


# --- Summary report --------------------------------------------------
def write_summary(path: Path, results: list[FormatTreeResult], input_path: Path):
    lines = []
    lines.append("=" * 72)
    lines.append("GEMV Dispatch Heuristic - Per-Format Set-Cover + Tree")
    lines.append(f"Source CSV: {input_path.name}")
    lines.append(f"Method: greedy set cover (3% threshold) + DecisionTree")
    lines.append("=" * 72)
    lines.append("")

    lines.append(f"{'Format':12s} {'CB':>3s} {'Shapes':>6s} {'Raw':>5s} "
                 f"{'SC':>4s} {'Depth':>5s} {'Leaves':>6s} "
                 f"{'Mean%':>6s} {'P95%':>6s} {'Max%':>6s}")
    lines.append("-" * 72)

    for r in results:
        lines.append(f"{r.fmt:12s} {r.cb:3d} {r.n_shapes:6d} {r.n_raw_configs:5d} "
                     f"{r.n_master_configs:4d} {r.depth:5d} {r.n_leaves:6d} "
                     f"{r.mean_penalty_pct:5.2f}% {r.p95_penalty_pct:5.2f}% "
                     f"{r.max_penalty_pct:5.2f}%")

    all_pens = [r.mean_penalty_pct for r in results]
    lines.append("-" * 72)
    lines.append(f"{'OVERALL':12s} {'':>3s} {'':>6s} {'':>5s} "
                 f"{'':>4s} {'':>5s} {'':>6s} "
                 f"{np.mean(all_pens):5.2f}% {np.percentile(all_pens,95):5.2f}% "
                 f"{max(all_pens):5.2f}%")
    lines.append("")

    path.write_text("\n".join(lines))
    print(f"Summary written to {path}")


# --- Main -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Infer shape-agnostic GEMV dispatch heuristic.")
    parser.add_argument("--input", required=True,
                        help="Sweep CSV (all rows, with is_best column)")
    parser.add_argument("--output", default=None,
                        help="Output C++ .inc file path")
    parser.add_argument("--summary", default=None,
                        help="Output summary text file path")
    parser.add_argument("--threshold", type=float, default=0.03,
                        help="Set-cover BW threshold (default: 0.03 = 3%%)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input CSV not found: {input_path}")

    print(f"Loading {input_path} ...")
    all_rows = load_sweep_csv(input_path)
    print(f"  Total rows: {len(all_rows)}")

    formats = sorted(set(r["fmt"] for r in all_rows if r["is_best"]))
    print(f"  Formats with best data: {len(formats)} ({', '.join(formats)})")

    # Train per-format
    results = []
    for fmt in formats:
        print(f"\n  Training {fmt} (CB={FORMAT_TO_CODEBOOK[fmt]}) ...")
        r = train_format_tree(fmt, all_rows, threshold=args.threshold)
        results.append(r)
        print(f"    {r.n_master_configs} master configs, "
              f"depth={r.depth}, leaves={r.n_leaves}, "
              f"mean={r.mean_penalty_pct:.2f}%, max={r.max_penalty_pct:.2f}%")

    # Overall
    all_pens = [r.mean_penalty_pct for r in results]
    print(f"\n  Overall mean BW penalty: {np.mean(all_pens):.2f}%")
    print(f"  Overall max BW penalty: {max(r.max_penalty_pct for r in results):.2f}%")

    # Generate C++
    if args.output:
        output_path = Path(args.output)
        print(f"\nGenerating C++ -> {output_path} ...")
        cpp_code = generate_cpp(results, input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cpp_code)
        print(f"  Generated {cpp_code.count(chr(10))} lines")

    # Summary
    if args.summary:
        write_summary(Path(args.summary), results, input_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
