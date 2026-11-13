#!/usr/bin/env python3
"""Round-trip the existing device-free C++ CSV writer through the canonical Python consumer."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    """Exercise the actual native writer and imported campaign artifact consumer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--writer", type=Path, required=True)
    args = parser.parse_args()
    writer = args.writer.resolve(strict=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts/ci"))
    import run_production_parity_campaigns as campaigns
    with tempfile.TemporaryDirectory(prefix="qfn-query-csv-") as directory:
        for phase, name in (("decode", "ProgramOmissionCSVKeepsNativeFieldsBlank"),
                            ("prefill", "ProgramOmissionPrefillCSVKeepsNativeFieldsBlank")):
            output = Path(directory) / f"{phase}.csv"
            environment = os.environ | {"LLAMINAR_FORCE_CPU_ONLY_STARTUP": "1",
                "LLAMINAR_SKIP_CUDA_STARTUP": "1", "LLAMINAR_SKIP_ROCM_STARTUP": "1",
                "LLAMINAR_QUERY_OMISSION_CSV_EXPORT": str(output)}
            result = subprocess.run([str(writer), f"--gtest_filter=QwenSparseQueryParityEvidence.{name}"],
                env=environment, text=True, capture_output=True, timeout=30)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            with output.open(newline="") as stream:
                rows = list(csv.reader(stream))
            expected_header = next(csv.reader([campaigns.REQUIRED_CSV_HEADERS[f"{phase}_stages.csv"]]))
            if rows[0] != expected_header:
                raise RuntimeError(f"Native {phase} query CSV schema differs from canonical consumer")
            error = campaigns.validate_stage_checkpoint_evidence(rows, decode=phase == "decode")
            if error:
                raise RuntimeError(error)
            print(f"Native C++ {phase} omission CSV -> canonical Python validator PASS")


if __name__ == "__main__":
    main()
