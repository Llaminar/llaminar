#!/usr/bin/env python3
"""Regression tests for the common-backed ROCm decode policy analyzer."""

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
    / "rocm"
    / "analyze_rocm_native_vnni_decode_trainer.py"
)


class ROCmNativeVNNIDecodeTrainerTest(unittest.TestCase):
    """Exercise strong adaptation and alias/mode-robust current-ABI emission."""

    @staticmethod
    def row(
        candidate: str,
        execution_mode: str,
        median_us: float,
        *,
        m: int = 1,
    ) -> dict[str, object]:
        """Construct one complete Q8_0 candidate observation."""

        verifier = m > 1
        kb = 32 if verifier else int(candidate.removeprefix("KB"))
        canonical_candidate = "INHERIT_SERIAL_M1" if verifier else candidate
        return {
            "backend": "rocm",
            "phase": "decode",
            "source_format": "Q8_0",
            "source_codebook": 19,
            "execution_codebook": 19,
            "shape": "35BMoE_Expert_GateUp",
            "execution_mode": execution_mode,
            "m": m,
            "n": 512,
            "k": 2048,
            "candidate_id": canonical_candidate,
            "kb": kb,
            "target_waves": 4,
            "weight_bytes": 1114112,
            "warmup_count": 2,
            "sample_count": 3,
            "min_us": median_us,
            "median_us": median_us,
            "p95_us": median_us,
            "mad_us": 0,
            "cv": 0.01,
            "effective_bandwidth_gbs": 30,
            "bit_mismatches": 0,
            "first_bit_mismatch": 0,
            "repeat_byte_mismatches": 0,
            "max_abs": 0,
            "relative_l2": 0,
            "cosine": 1,
            "symmetric_kld": 0,
            "grouped_output_digest": "fnv1a64:equal00000000000",
            "serial_output_digest": "fnv1a64:equal00000000000",
            "timing_sample_digest": f"fnv1a64:{candidate.lower():0<16}"[:24],
            "route_counter_ok": 1,
            "observed_candidate_id": canonical_candidate,
            "observed_path": "split_reduce",
            "serial_m1_kb": 32,
            "serial_m1_target_waves": 4,
            "serial_route_counter_ok": 1,
            "numerical_correctness": 1,
            "correctness_pass": 1,
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
            if result.returncode == 0:
                self.assertTrue(common.exists())
                with common.open(newline="", encoding="utf-8") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), len(rows))
            return result, generated, summary_rows

    def test_opposing_execution_modes_choose_lowest_worst_surface_regret(self) -> None:
        rows = [
            self.row("KB8", "eager", 10.0),
            self.row("KB8", "graph_captured", 100.0),
            self.row("KB32", "eager", 20.0),
            self.row("KB32", "graph_captured", 20.0),
            self.row("INHERIT_SERIAL_M1", "eager", 30.0, m=2),
            self.row("INHERIT_SERIAL_M1", "graph_captured", 31.0, m=2),
        ]

        result, generated, summary = self.run_analyzer(rows)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("common alias/mode-robust exact oracle", generated)
        self.assertIn("{32, 4}", generated)
        self.assertNotIn("M=2 512x2048", generated)
        self.assertEqual(summary[0]["candidate_id"], "rocm.nvnni.decode.fast.kb32")
        self.assertEqual(summary[0]["certified_verifier_key_count"], "1")

    def test_explicit_fast_kb_is_rejected_for_verifier_depth(self) -> None:
        row = self.row("INHERIT_SERIAL_M1", "eager", 30.0, m=2)
        row["candidate_id"] = "KB32"
        row["observed_candidate_id"] = "KB32"

        result, _, _ = self.run_analyzer([row])

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not support Verifier", result.stderr)

    def test_require_complete_rejects_partial_alias_candidate_and_depth_matrix(self) -> None:
        result, _, _ = self.run_analyzer(
            [self.row("KB32", "eager", 20.0)],
            "--require-complete",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("canonical alias/mode coverage is incomplete", result.stderr)

    def test_legacy_weak_csv_is_rejected(self) -> None:
        row = self.row("KB32", "eager", 20.0)
        del row["repeat_byte_mismatches"]

        result, _, _ = self.run_analyzer([row])

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing strong ROCm decode columns", result.stderr)


if __name__ == "__main__":
    unittest.main()
