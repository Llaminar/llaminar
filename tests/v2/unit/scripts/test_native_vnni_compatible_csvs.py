#!/usr/bin/env python3
"""Regression tests for cross-generation NativeVNNI CSV publication."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.combine_compatible_csvs import (  # noqa: E402
    combine_compatible_csvs,
)


class NativeVNNICompatibleCSVTest(unittest.TestCase):
    """Prove that additive provenance columns do not invalidate old evidence."""

    def test_column_union_preserves_old_and_new_rows(self) -> None:
        """Blank-fill fields that did not exist in an immutable older schema."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old.csv"
            new = root / "new.csv"
            output = root / "combined.csv"
            old.write_text("candidate_id,sample_count\nold,17\n", encoding="utf-8")
            new.write_text(
                "candidate_id,sample_count,stationary_sample_begin,"
                "stationary_sample_count,stationary_duration_us\n"
                "new,23,8,15,250000\n",
                encoding="utf-8",
            )

            combine_compatible_csvs(output, (old, new))

            with output.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(
                list(rows[0]),
                [
                    "candidate_id",
                    "sample_count",
                    "stationary_sample_begin",
                    "stationary_sample_count",
                    "stationary_duration_us",
                ],
            )
            self.assertEqual(rows[0]["candidate_id"], "old")
            self.assertEqual(rows[0]["stationary_sample_begin"], "")
            self.assertEqual(rows[1]["candidate_id"], "new")
            self.assertEqual(rows[1]["stationary_sample_count"], "15")
            self.assertFalse(output.with_name(f"{output.name}.inprogress").exists())

    def test_reordered_columns_are_merged_by_name(self) -> None:
        """Accept a harness generation that reorders unchanged timing fields."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old.csv"
            new = root / "new.csv"
            output = root / "combined.csv"
            old.write_text(
                "candidate_id,sample_count,stationary_sample_count\n"
                "old,17,12\n",
                encoding="utf-8",
            )
            new.write_text(
                "candidate_id,stationary_sample_count,sample_count\n"
                "new,15,23\n",
                encoding="utf-8",
            )

            combine_compatible_csvs(output, (old, new))

            with output.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[0]["sample_count"], "17")
            self.assertEqual(rows[0]["stationary_sample_count"], "12")
            self.assertEqual(rows[1]["sample_count"], "23")
            self.assertEqual(rows[1]["stationary_sample_count"], "15")


if __name__ == "__main__":
    unittest.main()
