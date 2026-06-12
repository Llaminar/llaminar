#!/usr/bin/env python3
"""Smoke-test CUDA NativeVNNI GEMV dispatch base-include merge mode."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-include", type=Path, required=True)
    parser.add_argument("--dispatch-validator", type=Path, required=True)
    return parser.parse_args()


def run_generator(generator: Path, input_csv: Path, base_include: Path, output: Path, summary: Path) -> None:
    subprocess.run([
        sys.executable,
        str(generator),
        "--input",
        str(input_csv),
        "--base-include",
        str(base_include),
        "--output",
        str(output),
        "--summary",
        str(summary),
    ], check=True)


def run_standalone_generator(generator: Path, input_csv: Path, output: Path, summary: Path) -> None:
    subprocess.run([
        sys.executable,
        str(generator),
        "--input",
        str(input_csv),
        "--output",
        str(output),
        "--summary",
        str(summary),
        "--min-overall-family-pct",
        "99.0",
        "--min-overall-exact-pct",
        "99.0",
        "--min-fallback-family-pct",
        "97.0",
        "--min-fallback-exact-pct",
        "30.0",
    ], check=True)


def validate_output(dispatch_validator: Path, output: Path) -> str:
    subprocess.run([sys.executable, str(dispatch_validator), str(output)], check=True)
    text = output.read_text()
    if text.count("BEGIN generated known-shape overrides from analyze_cuda_tc_gemv_dispatch.py") != 1:
        raise SystemExit("expected exactly one known-shape override block")
    if "selectKnownShapeGenerated<CB>" not in text:
        raise SystemExit("merged include does not route public API through known-shape overrides")
    if "selectGeneratedDispatch_<CB>" not in text:
        raise SystemExit("merged include dropped the base fallback dispatcher")
    if not re.search(r"\bselectTuning_CB\d+\s*\(", text):
        raise SystemExit("merged include dropped base codebook tuning helpers")
    for fragment in (
        "packGeneratedDispatchKey(int M, int N, int K)",
        "selectKnownShapeGenerated(int M, int N, int K",
        "classifyShapeGenerated(int M, int N, int K)",
        "selectGeneratedTuning(int M, int N, int K)",
        "return classifyShapeGenerated<CB>(1, N, K);",
        "return selectGeneratedTuning<CB>(1, N, K);",
    ):
        if fragment not in text:
            raise SystemExit(f"merged include is missing M-aware fragment: {fragment}")
    return text


def write_m_split_fixture(path: Path) -> None:
    """Create a tiny fixture where one (N,K) needs different families by M."""
    path.write_text(
        "\n".join([
            "format,codebook,shape,m,n,k,weight_bytes,family,tile_n,cpt,target_waves,mkg,max_kb,force_two_phase,min_us,eff_bw_gbs,pct_peak,is_best",
            "Q4_0,0,lm_head_m1,1,248320,5120,715161600,wide,128,1,0,0,0,0,812.0,882.0,94.0,1",
            "Q4_0,0,lm_head_m2,2,248320,5120,715161600,kpar,128,1,4,2,0,0,833.0,860.0,92.0,1",
            "Q4_0,0,lm_head_m3,3,248320,5120,715161600,kpar,128,1,4,2,0,0,838.0,856.0,91.0,1",
            "Q4_0,0,lm_head_m4,4,248320,5120,715161600,kpar,128,1,4,2,0,0,902.0,797.0,85.0,1",
            "",
        ])
    )


def write_alias_conflict_fixture(path: Path) -> None:
    """Create aliased formats whose row-wise winners cannot both be encoded.

    Q4_1 and Q4_K share NativeVNNI codebook 5, so the runtime dispatch key is
    only (CB=5, M, N, K). The analyzer must collapse these source-format rows
    to one aggregate runtime winner before enforcing exact-hit thresholds.
    """
    path.write_text(
        "\n".join([
            "format,codebook,shape,m,n,k,weight_bytes,family,tile_n,cpt,target_waves,mkg,max_kb,force_two_phase,min_us,eff_bw_gbs,pct_peak,is_best",
            "Q4_1,5,alias_q4_1,2,248320,5120,715161600,wide,128,1,0,0,0,0,900.0,100.0,90.0,1",
            "Q4_1,5,alias_q4_1,2,248320,5120,715161600,kpar,64,2,4,2,0,0,910.0,98.0,88.0,0",
            "Q4_K,5,alias_q4_k,2,248320,5120,715161600,wide,128,1,0,0,0,0,950.0,80.0,70.0,0",
            "Q4_K,5,alias_q4_k,2,248320,5120,715161600,kpar,64,2,4,2,0,0,900.0,100.0,90.0,1",
            "",
        ])
    )


def main() -> int:
    args = parse_args()
    for path in (args.generator, args.input, args.base_include, args.dispatch_validator):
        if not path.is_file():
            raise SystemExit(f"required file not found: {path}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        first = temp / "generated_first.inc"
        second = temp / "generated_second.inc"
        summary = temp / "summary.txt"

        run_generator(args.generator, args.input, args.base_include, first, summary)
        first_text = validate_output(args.dispatch_validator, first)

        run_generator(args.generator, args.input, first, second, summary)
        second_text = validate_output(args.dispatch_validator, second)

        if first_text != second_text:
            raise SystemExit("base-include merge mode is not idempotent")
        if not summary.read_text().strip():
            raise SystemExit("generator summary was empty")

        m_split_csv = temp / "m_split.csv"
        m_split_output = temp / "m_split_generated.inc"
        m_split_summary = temp / "m_split_summary.txt"
        write_m_split_fixture(m_split_csv)
        run_standalone_generator(args.generator, m_split_csv, m_split_output, m_split_summary)
        subprocess.run([sys.executable, str(args.dispatch_validator), str(m_split_output)], check=True)
        summary_text = m_split_summary.read_text()
        if "Fallback family hit rate: 4/4 (100.00%)" not in summary_text:
            raise SystemExit("M-aware fallback did not classify the split-M fixture perfectly")
        for fragment in ("M=1:", "M=2:", "M=3:", "M=4:"):
            if fragment not in summary_text:
                raise SystemExit(f"M-aware fallback summary missing {fragment}")

        alias_csv = temp / "alias_conflict.csv"
        alias_output = temp / "alias_generated.inc"
        alias_summary = temp / "alias_summary.txt"
        write_alias_conflict_fixture(alias_csv)
        run_standalone_generator(args.generator, alias_csv, alias_output, alias_summary)
        subprocess.run([sys.executable, str(args.dispatch_validator), str(alias_output)], check=True)
        alias_summary_text = alias_summary.read_text()
        for fragment in (
            "Source winner rows: 2",
            "Runtime dispatch rows: 1",
            "Alias runtime keys: 1",
            "Alias winner-conflict keys: 1",
            "Overall exact hit rate: 1/1 (100.00%)",
        ):
            if fragment not in alias_summary_text:
                raise SystemExit(f"alias-collapse summary missing {fragment}")

    print("validated CUDA GEMV dispatch base-include merge mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
