#!/usr/bin/env python3
"""Smoke-test the CUDA NativeVNNI prefill dispatch generator.

Q4_1/Q4_K, Q5_1/Q5_K, and IQ4_NL/IQ4_XS share NativeVNNI codebook ids.
Q8_0/Q8_1/Q8_K also normalize to GPU execution codebook 19 despite retaining
distinct source codebooks in the evidence CSV. The generated C++ must group by
runtime codebook, not by source-format spelling.
"""

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
        subprocess.run(command, check=True)

        text = output.read_text()
        helpers = re.findall(r"if constexpr \(CB == (\d+)\)", text)
        duplicates = sorted({cb for cb in helpers if helpers.count(cb) > 1})
        if duplicates:
            raise SystemExit(f"duplicate generated codebook branch(es): {', '.join(duplicates)}")

        expected = {"4", "5", "7", "19"}
        found = set(helpers)
        if found != expected:
            raise SystemExit(f"expected codebook branches {sorted(expected)}, found {sorted(found)}")

        # The fixture contains a deliberately faster but byte-incorrect Q4_1
        # candidate. Exact overlays must retain the slower byte-exact tile.
        packed_key = (64 << 40) | (2048 << 20) | 512
        key = f"0x{packed_key:016X}ULL"
        entry = re.search(
            rf"\{{\s*{key},\s*(\d+),\s*(\d+)\s*\}}", text)
        if not entry or entry.groups() != ("1", "1"):
            raise SystemExit(
                "byte-incorrect CUDA prefill candidate entered generated dispatch")

        if not summary.read_text().strip():
            raise SystemExit("generator summary was empty")

    print("validated CUDA prefill dispatch generator codebook alias grouping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
