#!/usr/bin/env python3
"""Regression tests for the common-backed CPU verifier policy analyzer."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
ANALYZER = (
    REPO_ROOT
    / "tests"
    / "v2"
    / "performance"
    / "kernels"
    / "cpu"
    / "analyze_cpu_native_vnni_verifier_trainer.py"
)


class CPUNativeVNNIVerifierTrainerTest(unittest.TestCase):
    """Exercise byte eligibility and common exact winner emission."""

    @staticmethod
    def row(
        candidate: str,
        median_us: float,
        *,
        mismatches: int = 0,
        build_isa: str = "AVX512",
        runtime_isa: str = "AVX512",
    ) -> dict[str, object]:
        digest = (
            "fnv1a64:different000000"
            if mismatches else "fnv1a64:equal00000000000"
        )
        return {
            "backend": "cpu",
            "phase": "verifier_rows",
            "source_format": "Q4_0",
            "source_codebook": 0,
            "execution_codebook": 0,
            "shape": "Qwen36_FFN_GateUp",
            "execution_mode": "eager",
            "m": 3,
            "n": 17408,
            "k": 5120,
            "candidate_id": candidate,
            "build_isa": build_isa,
            "runtime_isa_requested": runtime_isa,
            "runtime_isa_effective": runtime_isa,
            "threads": 28,
            "weight_bytes": 1000000,
            "warmup_count": 2,
            "sample_count": 3,
            "min_us": median_us,
            "median_us": median_us,
            "p95_us": median_us,
            "mad_us": 0,
            "cv": 0.01,
            "serial_median_us": 20,
            "speedup": 20 / median_us,
            "bit_mismatches": mismatches,
            "first_bit_mismatch": 7 if mismatches else 0,
            "repeat_byte_mismatches": 0,
            "max_abs": 1.0e-6 if mismatches else 0,
            "relative_l2": 1.0e-7 if mismatches else 0,
            "cosine": 1,
            "symmetric_kld": 0,
            "grouped_output_digest": digest,
            "serial_output_digest": "fnv1a64:equal00000000000",
            "timing_sample_digest": f"fnv1a64:{candidate.lower():0<16}"[:24],
            "route_counter_ok": 1,
            "observed_candidate_id": candidate,
            "k_tiles": 1,
            "n_block_chunks": 2,
            "numerical_correctness": 1,
            "correctness_pass": 0 if mismatches else 1,
            "is_winner": 0,
        }

    def run_analyzer(
        self,
        rows: list[dict[str, object]],
        *extra: str,
    ) -> tuple[subprocess.CompletedProcess[str], str, list[dict[str, str]]]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_csv = root / "trainer.csv"
            output = root / "generated.inc"
            summary = root / "summary.csv"
            common = root / "common.csv"
            with input_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ANALYZER),
                    str(input_csv),
                    "--output",
                    str(output),
                    "--summary",
                    str(summary),
                    "--common-observations",
                    str(common),
                    *extra,
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            generated = output.read_text(encoding="utf-8") if output.exists() else ""
            summary_rows = []
            if summary.exists():
                with summary.open(newline="", encoding="utf-8") as handle:
                    summary_rows = list(csv.DictReader(handle))
            return result, generated, summary_rows

    def test_fastest_byte_exact_policy_is_emitted_through_common_oracle(self) -> None:
        result, generated, summary = self.run_analyzer([
            self.row("Pairwise", 10.0),
            self.row("WideRows", 6.0),
        ])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("common NativeVNNI alias-robust exact oracle", generated)
        self.assertIn("CPUNativeVNNIVerifierRowsPolicy::WideRows", generated)
        self.assertEqual(summary[0]["policy"], "WideRows")

    def test_inexact_faster_candidate_is_filtered_before_selection(self) -> None:
        result, generated, summary = self.run_analyzer([
            self.row("Pairwise", 10.0),
            self.row("WideRows", 5.0, mismatches=1),
        ])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(summary[0]["policy"], "Pairwise")
        self.assertIn("CPUNativeVNNIVerifierRowsPolicy::Pairwise", generated)

    def test_build_and_runtime_isa_regimes_train_independent_rules(self) -> None:
        result, generated, summary = self.run_analyzer([
            self.row("Pairwise", 10.0),
            self.row("WideRows", 6.0),
            self.row(
                "Pairwise",
                8.0,
                build_isa="AVX512",
                runtime_isa="AVX2",
            ),
            self.row(
                "Pairwise",
                9.0,
                build_isa="AVX2",
                runtime_isa="AVX2",
            ),
        ])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CPUNativeVNNIBuildISA::AVX512", generated)
        self.assertIn("CPUNativeVNNIRuntimeISA::AVX2", generated)
        self.assertIn("CPUNativeVNNIBuildISA::AVX2", generated)
        policies = {
            (row["build_isa"], row["runtime_isa"]): row["policy"]
            for row in summary
        }
        self.assertEqual(policies[("AVX512", "AVX512")], "WideRows")
        self.assertEqual(policies[("AVX512", "AVX2")], "Pairwise")
        self.assertEqual(policies[("AVX2", "AVX2")], "Pairwise")

    def test_require_complete_rejects_missing_depths_and_candidates(self) -> None:
        result, _, _ = self.run_analyzer(
            [self.row("Pairwise", 10.0)],
            "--require-complete",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("verifier M matrix incomplete", result.stderr)

    def test_required_isa_matrix_rejects_single_build_corpus(self) -> None:
        result, _, _ = self.run_analyzer(
            [self.row("Pairwise", 10.0)],
            "--require-isa-matrix",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CPU ISA regime matrix incomplete", result.stderr)

    def test_relaxed_legacy_csv_is_rejected(self) -> None:
        row = self.row("Pairwise", 10.0)
        del row["bit_mismatches"]

        result, _, _ = self.run_analyzer([row])

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing strong CPU verifier columns", result.stderr)


if __name__ == "__main__":
    unittest.main()
