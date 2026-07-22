#!/usr/bin/env python3
"""Regressions for authenticated common-observation schema upgrades."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_ROOT = REPO_ROOT / "tests/v2/performance/kernels"
if str(KERNEL_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_ROOT))

from native_vnni_dispatch.common_observation_migration import (  # noqa: E402
    ALLOWED_ADDED_COLUMNS,
    validate_common_observation_migration,
)


class NativeVNNICommonObservationMigrationTest(unittest.TestCase):
    """Require exact historical evidence across derived schema evolution."""

    def write_csv(
        self,
        path: Path,
        fields: tuple[str, ...],
        rows: list[dict[str, str]],
    ) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_reviewed_schema_upgrade_preserves_every_shared_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            retained = root / "retained.csv"
            replayed = root / "replayed.csv"
            old_fields = ("learner_version", "candidate_id", "median_us")
            new_fields = (*old_fields, *ALLOWED_ADDED_COLUMNS)
            self.write_csv(retained, old_fields, [{
                "learner_version": "v9",
                "candidate_id": "candidate-a",
                "median_us": "1.25",
            }])
            self.write_csv(replayed, new_fields, [{
                "learner_version": "v26",
                "candidate_id": "candidate-a",
                "median_us": "1.25",
                "launch_k_tiles": "4",
                "launch_n_block_chunks": "2",
                "adaptive_timing_evidence": "False",
            }])

            report = validate_common_observation_migration(
                retained,
                replayed,
                require_learner_transition=True,
            )

            self.assertEqual(report.row_count, 1)
            self.assertEqual(report.retained_learner_version, "v9")
            self.assertEqual(report.replayed_learner_version, "v26")

    def test_measured_value_change_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            retained = root / "retained.csv"
            replayed = root / "replayed.csv"
            old_fields = ("learner_version", "candidate_id", "median_us")
            new_fields = (*old_fields, *ALLOWED_ADDED_COLUMNS)
            self.write_csv(retained, old_fields, [{
                "learner_version": "v9",
                "candidate_id": "candidate-a",
                "median_us": "1.25",
            }])
            self.write_csv(replayed, new_fields, [{
                "learner_version": "v26",
                "candidate_id": "candidate-b",
                "median_us": "1.25",
                "launch_k_tiles": "4",
                "launch_n_block_chunks": "2",
                "adaptive_timing_evidence": "False",
            }])

            with self.assertRaisesRegex(ValueError, "changed measured evidence"):
                validate_common_observation_migration(retained, replayed)

    def test_unreviewed_added_column_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            retained = root / "retained.csv"
            replayed = root / "replayed.csv"
            self.write_csv(
                retained,
                ("learner_version",),
                [{"learner_version": "v9"}],
            )
            self.write_csv(
                replayed,
                ("learner_version", "mystery_feature"),
                [{"learner_version": "v26", "mystery_feature": "1"}],
            )

            with self.assertRaisesRegex(ValueError, "unreviewed column set"):
                validate_common_observation_migration(retained, replayed)

    def test_nonidentical_already_current_replay_fails_closed(self) -> None:
        """Only the wrapper's preceding byte-identity check may skip migration."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            retained = root / "retained.csv"
            replayed = root / "replayed.csv"
            fields = (
                "learner_version",
                "candidate_id",
                *ALLOWED_ADDED_COLUMNS,
            )
            retained_row = {
                "learner_version": "v26",
                "candidate_id": "candidate-a",
                "launch_k_tiles": "4",
                "launch_n_block_chunks": "2",
                "adaptive_timing_evidence": "False",
            }
            replayed_row = {**retained_row, "candidate_id": "candidate-b"}
            self.write_csv(retained, fields, [retained_row])
            self.write_csv(replayed, fields, [replayed_row])

            with self.assertRaisesRegex(ValueError, "unreviewed column set"):
                validate_common_observation_migration(
                    retained,
                    replayed,
                    require_learner_transition=True,
                )


if __name__ == "__main__":
    unittest.main()
