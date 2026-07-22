#!/usr/bin/env python3
"""Regression tests for resumable CPU NativeVNNI prefill shards."""

from __future__ import annotations

import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch import validate_cpu_prefill_partial as validator  # noqa: E402


class CPUNativeVNNIPrefillPartialTest(unittest.TestCase):
    """Prove interrupted shards are reusable only at complete M boundaries."""

    @staticmethod
    def _write(path: Path, values: tuple[int, ...]) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "m",
                    "candidate_id",
                    "complete_round_probe_duration_us",
                ),
            )
            writer.writeheader()
            for value in values:
                writer.writerow({
                    "m": value,
                    "candidate_id": f"candidate-{value}",
                    "complete_round_probe_duration_us": "1",
                })

    def test_exact_inventory_is_required_for_final_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = Path(directory) / "aggregate.csv"
            timing = Path(directory) / "timing.csv"
            self._write(aggregate, (64, 256))
            self._write(timing, (64, 256))
            with patch.object(validator, "adapt_cpu_prefill_csv"):
                observed = validator.validate_cpu_prefill_partial(
                    aggregate,
                    timing,
                    expected_m_values=(64, 256),
                )
                self.assertEqual(observed, (64, 256))
                with self.assertRaisesRegex(ValueError, "exact planned M inventory"):
                    validator.validate_cpu_prefill_partial(
                        aggregate,
                        timing,
                        expected_m_values=(64, 256, 1024),
                    )

    def test_resume_cli_prints_only_the_missing_ordered_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = Path(directory) / "aggregate.csv"
            timing = Path(directory) / "timing.csv"
            self._write(aggregate, (64, 256))
            self._write(timing, (64, 256))
            stdout = io.StringIO()
            argv = [
                "validate_cpu_prefill_partial",
                "--input",
                str(aggregate),
                "--timing-sidecar",
                str(timing),
                "--planned-m-values",
                "64,256,1024,2048",
                "--print-missing-m-values",
                "--require-append-compatible",
            ]
            with (
                patch.object(validator, "adapt_cpu_prefill_csv"),
                patch.object(sys, "argv", argv),
                redirect_stdout(stdout),
            ):
                self.assertEqual(validator.main(), 0)
            self.assertEqual(stdout.getvalue(), "1024,2048\n")

    def test_resume_rejects_a_nonprefix_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = Path(directory) / "aggregate.csv"
            timing = Path(directory) / "timing.csv"
            self._write(aggregate, (64, 1024))
            self._write(timing, (64, 1024))
            argv = [
                "validate_cpu_prefill_partial",
                "--input",
                str(aggregate),
                "--timing-sidecar",
                str(timing),
                "--planned-m-values",
                "64,256,1024",
            ]
            with (
                patch.object(validator, "adapt_cpu_prefill_csv"),
                patch.object(sys, "argv", argv),
                self.assertRaisesRegex(ValueError, "not an ordered plan prefix"),
            ):
                validator.main()

    def test_repeated_or_mismatched_m_phases_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = Path(directory) / "aggregate.csv"
            timing = Path(directory) / "timing.csv"
            self._write(aggregate, (64, 256, 64))
            self._write(timing, (64, 256, 64))
            with patch.object(validator, "adapt_cpu_prefill_csv"):
                with self.assertRaisesRegex(ValueError, "repeats a completed M phase"):
                    validator.validate_cpu_prefill_partial(aggregate, timing)

            self._write(aggregate, (64, 256))
            self._write(timing, (64,))
            with patch.object(validator, "adapt_cpu_prefill_csv"):
                with self.assertRaisesRegex(ValueError, "inventories differ"):
                    validator.validate_cpu_prefill_partial(aggregate, timing)

    def test_append_rejects_a_legacy_aggregate_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = Path(directory) / "aggregate.csv"
            timing = Path(directory) / "timing.csv"
            aggregate.write_text("m,candidate_id\n64,candidate\n")
            self._write(timing, (64,))
            with patch.object(validator, "adapt_cpu_prefill_csv"):
                with self.assertRaisesRegex(ValueError, "legacy append schema"):
                    validator.validate_cpu_prefill_partial(
                        aggregate,
                        timing,
                        require_append_compatible=True,
                    )


if __name__ == "__main__":
    unittest.main()
