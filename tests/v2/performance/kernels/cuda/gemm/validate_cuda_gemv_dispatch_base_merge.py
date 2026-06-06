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
    return text


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

    print("validated CUDA GEMV dispatch base-include merge mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
