#!/usr/bin/env python3
"""Prove explicit prerequisite reuse rebuilds first and rejects stale evidence.

The entrypoint shares the existing complete-inventory validator. These tests
exercise its command boundary without running compilers, CTest or accelerators.
The validator's build/timestamp/inventory negative controls remain in the
canonical production-parity campaign suite.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
import run_production_prerequisites as entrypoint


class PrerequisiteReuseTests(unittest.TestCase):
    """A receipt request must never become an implicit alternate test mode."""

    def test_rebuild_precedes_unchanged_evidence_validation(self):
        """Source edits reach Ninja before the canonical receipt can be admitted."""
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary).resolve()
            report = build / "prior report.json"
            order = []
            def build_current(command):
                order.append("build")
                self.assertEqual(command, ["cmake", "--build", str(build), "--parallel",
                    "--target", "v2_unit_gate", "v2_production_test_preflight_gate"])
                return subprocess.CompletedProcess(command, 0)
            def validate(actual_build, actual_report):
                order.append("validate")
                self.assertEqual((actual_build, actual_report), (build, report))
                return 0, 0.0, ("unit", "preflight")
            with patch.object(entrypoint.subprocess, "run", side_effect=build_current), \
                 patch.object(entrypoint, "validate_source_freshness", side_effect=lambda *_: order.append("source")), \
                 patch.object(entrypoint, "reuse_unchanged_production_test_preflight", side_effect=validate), \
                 patch.object(entrypoint, "run_production_test_preflight") as full:
                self.assertEqual(entrypoint.main(["--build-dir", str(build), "--reuse-report", str(report)]), 0)
                self.assertEqual(order, ["build", "source", "validate"])
                full.assert_not_called()

    def test_failed_build_cannot_reuse_evidence(self):
        """An old report cannot certify sources whose current rebuild failed."""
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(entrypoint.subprocess, "run", return_value=subprocess.CompletedProcess([], 17)), \
             patch.object(entrypoint, "reuse_unchanged_production_test_preflight") as reuse:
            self.assertEqual(entrypoint.main(["--build-dir", temporary, "--reuse-report", "prior.json"]), 17)
            reuse.assert_not_called()

    def test_stale_receipt_fails_without_running_an_alternate_gate(self):
        """Explicit stale evidence is rejected even after a successful rebuild."""
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(entrypoint.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
             patch.object(entrypoint, "validate_source_freshness"), \
             patch.object(entrypoint, "reuse_unchanged_production_test_preflight", side_effect=ValueError("stale build")), \
             patch.object(entrypoint, "run_production_test_preflight") as full:
            with self.assertRaises(SystemExit) as stopped:
                entrypoint.main(["--build-dir", temporary, "--reuse-report", "prior.json"])
            self.assertEqual(stopped.exception.code, 2)
            full.assert_not_called()

    def test_new_evidence_still_runs_the_complete_authority(self):
        """Normal qualification retains the existing complete Unit/preflight transaction."""
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(entrypoint, "run_production_test_preflight", return_value=(0, 1.0, ())) as full, \
             patch.object(entrypoint, "reuse_unchanged_production_test_preflight") as reuse:
            build = Path(temporary).resolve()
            output = build / "new evidence"
            self.assertEqual(entrypoint.main(["--build-dir", str(build), "--output", str(output)]), 0)
            full.assert_called_once_with(build, None, installed_build_receipt=None, artifact_directory=output)
            reuse.assert_not_called()

    def test_interpreted_source_edits_and_deletions_reject_reuse(self):
        """Real metadata detects changed scripts even when Ninja has no work."""
        for mutation in ("unchanged", "edit", "backdated_edit", "delete", "add"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                source = root / "scripts/ci/example.py"
                source.parent.mkdir(parents=True)
                source.write_text("original\n")
                build = root / "build"
                build.mkdir()
                (build / "CMakeCache.txt").write_text(
                    f"CMAKE_HOME_DIRECTORY:INTERNAL={root / 'src/v2'}\n")
                report = build / "prerequisites.json"
                report.touch()
                completed = time.time_ns()
                report.write_text(json.dumps({"preflight_completed_ns": completed}))
                # Keep receipt/build output outside Git's source inventory.
                names = b"scripts/ci/example.py\0"
                if mutation in ("edit", "backdated_edit"):
                    source.write_text("changed\n")
                    if mutation == "backdated_edit":
                        os.utime(source, ns=(1, 1))
                elif mutation == "delete":
                    source.unlink()
                elif mutation == "add":
                    (source.parent / "new.py").write_text("new\n")
                    names += b"scripts/ci/new.py\0"
                with patch.object(entrypoint, "__file__", str(root / "scripts/ci/run_production_prerequisites.py")), \
                     patch.object(entrypoint.subprocess, "check_output", return_value=names):
                    if mutation == "unchanged":
                        entrypoint.validate_source_freshness(build, report)
                    else:
                        with self.assertRaisesRegex(ValueError, "source changed"):
                            entrypoint.validate_source_freshness(build, report)

    def test_receipt_cannot_admit_a_build_from_another_checkout(self):
        """A selectable build must retain the source ownership of this hook."""
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            (build / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=/unrelated/src/v2\n")
            with self.assertRaisesRegex(ValueError, "this checkout"):
                entrypoint.validate_source_freshness(build, build / "missing.json")


if __name__ == "__main__":
    unittest.main()
