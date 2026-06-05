#!/usr/bin/env python3
"""Smoke-test the ROCm NativeVNNI trainer analyzer/generator."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--dispatch-validator", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        output = tmp / "ROCmNativeVNNIPrefillTuningGenerated.inc"
        summary = tmp / "summary.txt"

        subprocess.run(
            [
                sys.executable,
                str(args.generator),
                "--input",
                str(args.input),
                "--output",
                str(output),
                "--summary",
                str(summary),
            ],
            check=True,
        )

        text = output.read_text()
        summary_text = summary.read_text()
        required_fragments = [
            "kROCmNativeVNNIPrefillTuning",
            "CB=5 (Q4_1)",
            "CB=7 (Q5_1)",
            "CB=19 (Q8_0)",
            "M=600",
            "qwen36_ffn_down_bucket",
        ]
        for fragment in required_fragments:
            if fragment not in text and fragment not in summary_text:
                raise SystemExit(f"missing expected generated fragment: {fragment}")

        subprocess.run(
            [sys.executable, str(args.dispatch_validator), str(output)],
            check=True,
        )

    print("ROCm NativeVNNI trainer generator smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
