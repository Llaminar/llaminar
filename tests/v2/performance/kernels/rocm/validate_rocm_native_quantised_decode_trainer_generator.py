#!/usr/bin/env python3
"""Check strong ROCm source identities and supported diagnostic exact emission.

The immutable synthetic fixture is not a complete production timing matrix.
Adapt every observation without relabeling it, reject incomplete arithmetic
alias publication, and separately stage its valid diagnostic exact contests.
The generated include stays temporary; this test never fits or installs policy.
"""

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
    root = Path(__file__).resolve().parents[5]
    kernels = root / "tests/v2/performance/kernels"
    if str(kernels) not in sys.path:
        sys.path.insert(0, str(kernels))
    from native_quantised_dispatch.format_registry import (
        arithmetic_policy_aliases, format_spec,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        output = tmp / "ROCmNativeQuantisedDecodeDispatchGenerated.inc"
        summary = tmp / "summary.txt"
        common = tmp / "adapted.csv"
        base = root / "src/v2/kernels/rocm/gemm/ROCmNativeQuantisedDecodeDispatchGenerated.inc"

        # This is an explicit historical observation reader, not a disguised
        # generation request. In particular physical24 must survive adaptation.
        subprocess.run(
            [sys.executable, str(args.generator), "--input", str(args.input),
             "--adapt-only", "--output", str(common)], check=True,
        )
        with common.open(newline="") as handle:
            observations = list(csv.DictReader(handle))
        source_xs = [row for row in observations if row["source_format"] == "IQ4_XS"]
        if not source_xs or any(row["source_codebook_id"] != "4" or
                                row["runtime_codebook_id"] != "24" for row in source_xs):
            raise SystemExit("adaptation lost IQ4_XS source4/physical24 identity")
        source_nl = [row for row in observations if row["source_format"] == "IQ4_NL"]
        if not source_nl or any(row["source_codebook_id"] != "4" or
                                row["runtime_codebook_id"] != "4" for row in source_nl):
            raise SystemExit("adaptation lost IQ4_NL source4/physical4 identity")

        with args.input.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            original_rows = list(reader)
        if len(observations) != len(original_rows):
            raise SystemExit("historical adaptation dropped or duplicated an observation")
        # This old smoke fixture records NL and XS at different geometries.
        # Their policy4 key is shared, so it cannot certify either missing
        # co-located alias. Do not invent or copy observations to fill that gap.
        incomplete = subprocess.run(
            [sys.executable, str(args.generator), "--input", str(args.input),
             "--exact-only", "--base-include", str(base),
             "--output", str(tmp / "incomplete.inc")],
            text=True, capture_output=True, check=False,
        )
        if incomplete.returncode == 0 or "no eligible candidate covers every required surface" not in incomplete.stderr:
            raise SystemExit("incomplete source-policy alias contest was not rejected precisely")
        if (tmp / "incomplete.inc").exists():
            raise SystemExit("incomplete arithmetic alias contest published an artifact")

        # Independent diagnostic contests whose physical family does not split
        # a source-arithmetic key remain useful generator/codebook witnesses.
        # Their missing broad production aliases/modes are explicitly outside
        # this smoke test's qualification; --profile quick cannot certify them.
        diagnostic_rows = []
        for row in original_rows:
            policy = format_spec(row["source_format"]).arithmetic_policy_codebook("rocm")
            physical = {format_spec(alias).runtime_codebook("rocm")
                        for alias in arithmetic_policy_aliases("rocm", policy)}
            if len(physical) == 1:
                diagnostic_rows.append(row)
        if {row["source_codebook"] for row in diagnostic_rows} != {"5", "7", "19"}:
            raise SystemExit("diagnostic fixture no longer owns its expected codebook contests")
        diagnostic = tmp / "diagnostic.csv"
        with diagnostic.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(diagnostic_rows)

        subprocess.run(
            [
                sys.executable,
                str(args.generator),
                "--input",
                str(diagnostic),
                "--exact-only",
                "--base-include",
                str(base),
                "--profile",
                "quick",
                "--output",
                str(output),
                "--summary",
                str(summary),
            ],
            check=True,
        )

        text = output.read_text()
        summary_text = summary.read_text()
        structural_fragments = [
            "packROCmNativeQuantisedDecodeDispatchKey(int m, int n, int k)",
            "selectROCmNativeQuantisedDecodeGenerated",
            "ROCmNativeQuantisedDecodeDispatchConfig",
        ]
        for fragment in structural_fragments:
            if fragment not in text:
                raise SystemExit(f"missing retained selector contract: {fragment}")
        begin = "    // BEGIN COMMON ROCM NATIVEQUANTISED DECODE EXACT OVERLAY"
        end = "    // END COMMON ROCM NATIVEQUANTISED DECODE EXACT OVERLAY"
        overlay = text.split(begin, 1)[1].split(end, 1)[0]
        required_fragments = [
            "CB=5 M=1",
            "CB=7 M=1",
            "CB=19 M=1",
            "M=1",
            "{16, 4}",
            "rocm.native_quantised.decode.fast.kb16",
            "alias_q4_1",
        ]
        for fragment in required_fragments:
            if fragment not in overlay and fragment not in summary_text:
                raise SystemExit(f"missing expected generated fragment: {fragment}")
        with summary.open(newline="") as handle:
            delta = list(csv.DictReader(handle))
        expected = {(row["shape"], row["n"], row["k"]) for row in diagnostic_rows}
        actual = {(row["shape"], row["n"], row["k"]) for row in delta}
        if actual != expected or len(delta) != len(expected):
            raise SystemExit("diagnostic exact output dropped or duplicated a recorded contest")
        if {row["arithmetic_policy_codebook"] for row in delta} != {"5", "7", "19"}:
            raise SystemExit("diagnostic exact keys do not match their source-arithmetic policies")

        subprocess.run(
            [sys.executable, str(args.dispatch_validator), str(output)],
            check=True,
        )

        # Negative fixture retains source4 while claiming retired execution4.
        # The generator must reject it; old measured corpora are never upgraded.
        stale_rows = [dict(row) for row in original_rows]
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
             "--adapt-only", "--output", str(tmp / "must-not-generate.csv")],
            text=True, capture_output=True, check=False,
        )
        if rejected.returncode == 0 or "execution codebook disagrees with registry" not in rejected.stderr:
            raise SystemExit("retired IQ4_XS execution4 was not rejected precisely")

        # The dispatch validator independently rejects unknown or mislabeled
        # physical identities. Physical24 provenance was proved by the reader;
        # it must not be mistaken for this incomplete fixture's policy4 output.
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

    print("ROCm NativeQuantised decode trainer generator smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
