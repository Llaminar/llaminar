#!/usr/bin/env python3
"""Regression tests for the common-backed ROCm MoE dispatch analyzer."""

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
    / "analyze_rocm_moe_grouped_prefill_trainer.py"
)

FIELDNAMES = [
    "backend", "phase", "source_format", "source_codebook",
    "execution_codebook", "shape", "role", "candidate_id", "m", "n", "k",
    "tile_m", "tile_n", "warmup_count", "sample_count", "timed_replays",
    "min_us", "graph_us", "p95_us", "mad_us", "cv", "pipeline_gops",
    "bit_mismatches", "first_bit_mismatch", "repeat_bit_mismatches",
    "max_abs", "relative_l2", "cosine", "symmetric_kld",
    "grouped_output_digest", "serial_output_digest", "timing_sample_digest",
    "route_counter_ok", "observed_candidate_id", "is_winner",
]

SOURCE_CODEBOOKS = {"Q8_0": 19, "Q8_1": 20, "Q8_K": 21}


class ROCmMoEGroupedPrefillTrainerTest(unittest.TestCase):
    """Exercise strict adaptation, robust alias selection, and C++ emission."""

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
                writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ANALYZER),
                    str(input_csv),
                    "--output",
                    str(output),
                    "--summary-csv",
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
            summary_rows: list[dict[str, str]] = []
            if summary.exists():
                with summary.open(newline="", encoding="utf-8") as handle:
                    summary_rows = list(csv.DictReader(handle))
            if result.returncode == 0:
                self.assertTrue(common.exists())
                with common.open(newline="", encoding="utf-8") as handle:
                    common_rows = list(csv.DictReader(handle))
                self.assertEqual(len(common_rows), len(rows))
                self.assertEqual(
                    common_rows[0]["semantic_contract"],
                    "VerifierSerialM1Bitwise",
                )
            return result, generated, summary_rows

    @staticmethod
    def row(
        source_format: str,
        tile_m: int,
        tile_n: int,
        graph_us: float,
        bit_mismatches: int = 0,
    ) -> dict[str, object]:
        """Build one strong codebook-19 gate/up candidate observation."""

        candidate = f"tm{tile_m}_tn{tile_n}"
        grouped_digest = (
            "fnv1a64:bad0000000000000"
            if bit_mismatches
            else "fnv1a64:equal00000000000"
        )
        return {
            "backend": "rocm",
            "phase": "grouped_prefill",
            "source_format": source_format,
            "source_codebook": SOURCE_CODEBOOKS[source_format],
            "execution_codebook": 19,
            "shape": "gate_ratio_1_4",
            "role": "gateup",
            "candidate_id": candidate,
            "m": 32,
            "n": 512,
            "k": 2048,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "warmup_count": 2,
            "sample_count": 3,
            "timed_replays": 12,
            "min_us": graph_us,
            "graph_us": graph_us,
            "p95_us": graph_us,
            "mad_us": 0.0,
            "cv": 0.0,
            "pipeline_gops": 1.0,
            "bit_mismatches": bit_mismatches,
            "first_bit_mismatch": 7 if bit_mismatches else 0,
            "repeat_bit_mismatches": 0,
            "max_abs": 1.0e-6 if bit_mismatches else 0.0,
            "relative_l2": 1.0e-7 if bit_mismatches else 0.0,
            "cosine": 1.0,
            "symmetric_kld": 0.0,
            "grouped_output_digest": grouped_digest,
            "serial_output_digest": "fnv1a64:equal00000000000",
            "timing_sample_digest": f"fnv1a64:timing{tile_m:02d}{tile_n:03d}",
            "route_counter_ok": 1,
            "observed_candidate_id": candidate,
            "is_winner": 0,
        }

    def test_inexact_alias_rejects_candidate_for_shared_codebook(self) -> None:
        rows = [
            self.row("Q8_0", 4, 64, 10.0),
            self.row("Q8_1", 4, 64, 11.0),
            self.row("Q8_K", 4, 64, 9.0, bit_mismatches=1),
            self.row("Q8_0", 16, 128, 20.0),
            self.row("Q8_1", 16, 128, 21.0),
            self.row("Q8_K", 16, 128, 19.0),
        ]

        result, generated, summary = self.run_analyzer(rows)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["execution_codebook"], "19")
        self.assertEqual(summary[0]["tile_m"], "16")
        self.assertEqual(summary[0]["tile_n"], "128")
        self.assertEqual(summary[0]["source_format_count"], "3")
        self.assertIn("common NativeVNNI alias-robust exact oracle", generated)
        self.assertIn("{19, 0, -2, 32, {16, 128}}", generated)

    def test_opposing_aliases_choose_lowest_worst_surface_regret(self) -> None:
        rows = [
            self.row("Q8_0", 8, 128, 12.0),
            self.row("Q8_1", 8, 128, 100.0),
            self.row("Q8_K", 8, 128, 13.0),
            self.row("Q8_0", 12, 128, 20.0),
            self.row("Q8_1", 12, 128, 20.0),
            self.row("Q8_K", 12, 128, 20.0),
        ]

        result, _, summary = self.run_analyzer(rows)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(summary[0]["tile_m"], "12")
        self.assertEqual(summary[0]["median_graph_us"], "20.000000")
        self.assertLess(float(summary[0]["max_surface_regret"]), 1.0)

    def test_require_complete_rejects_partial_sweep(self) -> None:
        result, _, _ = self.run_analyzer(
            [self.row("Q8_0", 16, 128, 20.0)],
            "--require-complete",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incomplete grouped-prefill sweep", result.stderr)

    def test_legacy_weak_csv_is_rejected_instead_of_promoted(self) -> None:
        weak = self.row("Q8_0", 16, 128, 20.0)
        del weak["repeat_bit_mismatches"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_csv = root / "weak.csv"
            output = root / "generated.inc"
            with input_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(weak))
                writer.writeheader()
                writer.writerow(weak)
            result = subprocess.run(
                [sys.executable, str(ANALYZER), str(input_csv), "--output", str(output)],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing strong ROCm MoE trainer columns", result.stderr)


if __name__ == "__main__":
    unittest.main()
