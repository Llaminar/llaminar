#!/usr/bin/env python3
"""Smoke-test the CUDA NativeVNNI GEMV dispatch generator.

The regression this catches is alias-driven duplicate helpers: Q4_1/Q4_K,
Q5_1/Q5_K, and IQ4_NL/IQ4_XS share numeric NativeVNNI codebook ids. The
generated C++ must therefore emit one helper per codebook, not one per format
spelling.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.generator.is_file():
        raise SystemExit(f"generator not found: {args.generator}")
    if not args.input.is_file():
        raise SystemExit(f"input CSV not found: {args.input}")

    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "generated.inc"
        summary = Path(temp_dir) / "summary.txt"
        command = [
            sys.executable,
            str(args.generator),
            "--input",
            str(args.input),
            "--output",
            str(output),
            "--summary",
            str(summary),
        ]
        env = os.environ.copy()
        # This validator trains on a six-row smoke fixture.  Letting NumPy or
        # sklearn fan out to every CPU core makes the full CTest unit sweep
        # oversubscribe badly, which can turn a sub-second logical check into a
        # timeout.  The production sweep/training scripts remain unconstrained;
        # only this tiny regression child process is pinned to one worker.
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            env[name] = "1"
        subprocess.run(command, check=True, env=env)

        text = output.read_text()
        helpers = re.findall(r"inline __host__ void selectTuning_CB(\d+)\(", text)
        duplicates = sorted({cb for cb in helpers if helpers.count(cb) > 1})
        if duplicates:
            raise SystemExit(f"duplicate generated codebook helper(s): {', '.join(duplicates)}")

        expected = {"4", "5", "7"}
        found = set(helpers)
        if found != expected:
            raise SystemExit(f"expected helpers {sorted(expected)}, found {sorted(found)}")

        required_fragments = (
            "int M, int N, int K",
            "const float log2_m",
            "classifyShapeGenerated(int M, int N, int K)",
            "selectGeneratedTuning(int M, int N, int K)",
            "return classifyShapeGenerated<CB>(1, N, K);",
            "return selectGeneratedTuning<CB>(1, N, K);",
        )
        for fragment in required_fragments:
            if fragment not in text:
                raise SystemExit(f"generated CUDA GEMV dispatch is missing M-aware fragment: {fragment}")

        if not summary.read_text().strip():
            raise SystemExit("generator summary was empty")

    print("validated CUDA GEMV dispatch generator codebook alias grouping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
