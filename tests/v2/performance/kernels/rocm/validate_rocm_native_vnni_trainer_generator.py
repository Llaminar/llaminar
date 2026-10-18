#!/usr/bin/env python3
"""Smoke-test the ROCm NativeVNNI trainer analyzer/generator."""

from __future__ import annotations

import argparse
import csv
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
            "selectROCmNativeVNNIPrefillGenerated",
            "ROCmNativeVNNIPrefillDispatchConfig",
            "CB=5 (Q4_1)",
            "CB=7 (Q5_1)",
            "CB=19 (Q8_0)",
            "{128, 32, 1, 2}",
            "M=600",
            "qwen36_ffn_down_bucket",
        ]
        required_fragments.extend(["CB=24", "IQ4_XS", "CB=4"])
        for fragment in required_fragments:
            if fragment not in text and fragment not in summary_text:
                raise SystemExit(f"missing expected generated fragment: {fragment}")

        if text.count("CB=19 (Q8_0)") != 1:
            raise SystemExit(
                "Q8_0/Q8_1/Q8_K did not collapse to one ROCm runtime branch"
            )

        subprocess.run(
            [sys.executable, str(args.dispatch_validator), str(output)],
            check=True,
        )

        # Negative fixture retains source4 while claiming retired execution4.
        # The generator must reject it; old measured corpora are never upgraded.
        with args.input.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            stale_rows = list(reader)
        xs_rows = [row for row in stale_rows if row["format"] == "IQ4_XS"]
        if not xs_rows:
            raise SystemExit("synthetic fixture lacks IQ4_XS source identity")
        for row in xs_rows:
            if row["codebook"] != "4" or row["execution_codebook"] != "24":
                raise SystemExit("IQ4_XS synthetic fixture must retain source4/physical24")
            row["execution_codebook"] = "4"
        stale = tmp / "retired-iq4xs.csv"
        with stale.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(stale_rows)
        rejected = subprocess.run(
            [sys.executable, str(args.generator), "--input", str(stale),
             "--output", str(tmp / "must-not-generate.inc")],
            text=True, capture_output=True, check=False,
        )
        if rejected.returncode == 0 or "execution codebook disagrees with registry" not in rejected.stderr:
            raise SystemExit("retired IQ4_XS execution4 was not rejected precisely")

        # A legacy source-only row must also fail, rather than acquire24 merely
        # because today's registry knows the original source label.
        for row in xs_rows:
            row["execution_codebook"] = ""
        with stale.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(stale_rows)
        rejected = subprocess.run(
            [sys.executable, str(args.generator), "--input", str(stale),
             "--output", str(tmp / "must-not-infer.inc")],
            text=True, capture_output=True, check=False,
        )
        if rejected.returncode == 0 or "IQ4_XS requires explicit execution_codebook=24" not in rejected.stderr:
            raise SystemExit("legacy source-only IQ4_XS row was silently relabeled")

        # The dispatch validator independently rejects unknown or mislabeled
        # physical identities, while the actual generated24 branch must pass.
        for filename, branch in (
            ("unknown.inc", "if (CB == 25) { // IQ4_XS\n}\n"),
            ("wrong-label.inc", "if (CB == 24) { // IQ4_NL\n}\n"),
        ):
            invalid = tmp / filename
            invalid.write_text(branch)
            rejected = subprocess.run(
                [sys.executable, str(args.dispatch_validator), str(invalid)],
                text=True, capture_output=True, check=False,
            )
            if rejected.returncode == 0:
                raise SystemExit(f"invalid generated physical identity accepted: {filename}")

    print("ROCm NativeVNNI trainer generator smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
