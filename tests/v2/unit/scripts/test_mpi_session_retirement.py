## @file test_mpi_session_retirement.py
#  @brief Device-free scheduling-boundary regressions for native MPI file retirement.
"""Python must observe the native owner, never remove files to manufacture a pass."""
from __future__ import annotations

from pathlib import Path
import sys
import json
import subprocess
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts/ci"))
import mpi_session_retirement as retirement
import run_production_parity_campaigns as parity
import prebuilt_test_image


class MPISessionRetirementTests(unittest.TestCase):
    """Exercise dirty predecessor, delayed retirement, failure and exact argv ownership."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="mpi-retirement-unit-")
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)

    def inventory(self) -> dict:
        """Use original CTest command data, including an application path with spaces."""
        return {"tests": [{"name": "owned", "command": ["/build/llaminar_mpi_session",
            "--session-base", str(self.base), "--", "/program with spaces"],
            "properties": [{"name": "LABELS", "value": ["ProcessOwned"]}]}]}

    def test_original_registration_owns_the_base(self) -> None:
        self.assertEqual(retirement.owned_session_bases(self.inventory()), {"owned": self.base})

    def test_missing_owner_is_rejected(self) -> None:
        inventory = self.inventory()
        inventory["tests"][0]["command"] = ["/program"]
        with self.assertRaisesRegex(RuntimeError, "Missing declared native"):
            retirement.owned_session_bases(inventory)

    def test_relative_or_missing_base_is_rejected(self) -> None:
        for path in ("relative", str(self.base / "absent")):
            inventory = self.inventory()
            inventory["tests"][0]["command"][2] = path
            with self.assertRaisesRegex(RuntimeError, "Unavailable declared native"):
                retirement.owned_session_bases(inventory)

    def test_two_jobs_cannot_share_the_same_base(self) -> None:
        inventory = self.inventory()
        second = self.inventory()["tests"][0]
        second["name"] = "another"
        inventory["tests"].append(second)
        with self.assertRaisesRegex(RuntimeError, "share a session base"):
            retirement.owned_session_bases(inventory)

    def test_unmanaged_host_tests_are_not_assigned_an_owner(self) -> None:
        inventory = self.inventory()
        inventory["tests"][0]["properties"] = []
        self.assertEqual(retirement.owned_session_bases(inventory), {})

    def test_dirty_predecessor_blocks_before_admission(self) -> None:
        pending = self.base / "llaminar-mpi-pending"
        pending.mkdir()
        entered = False
        with self.assertRaisesRegex(RuntimeError, "retirement is incomplete"):
            with retirement.retirement_scope([self.base]):
                entered = True
        self.assertFalse(entered)
        self.assertTrue(pending.is_dir())

    def test_observer_waits_for_native_deletion(self) -> None:
        pending = self.base / "llaminar-mpi-pending"
        pending.mkdir()
        with mock.patch.object(retirement.time, "sleep", side_effect=lambda _: pending.rmdir()) as sleep:
            retirement.require_retired([self.base])
        sleep.assert_called_once()
        self.assertFalse(pending.exists())

    def test_failed_body_retains_retirement_obligation(self) -> None:
        pending = self.base / "llaminar-mpi-pending"
        with mock.patch.object(retirement.time, "sleep", side_effect=lambda _: pending.rmdir()):
            with self.assertRaisesRegex(ValueError, "original test failure"):
                with retirement.retirement_scope([self.base]):
                    pending.mkdir()
                    raise ValueError("original test failure")
        self.assertFalse(pending.exists())

    def test_ambiguous_replacement_is_preserved(self) -> None:
        pending = self.base / "llaminar-mpi-replacement"
        pending.symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "backing retained"):
            retirement.require_retired([self.base], timeout_seconds=0)
        self.assertTrue(pending.is_symlink())

    def test_foreign_files_are_never_removed(self) -> None:
        sentinel = self.base / "another-owner"
        sentinel.write_text("keep")
        retirement.require_retired([self.base])
        self.assertEqual(sentinel.read_text(), "keep")

    def test_missing_original_base_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "base disappeared"):
            retirement.require_retired([self.base / "gone"])


class CanonicalPrerequisiteRetirementTests(unittest.TestCase):
    """Execute the actual canonical scheduling boundary without a build or CTest process."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="prerequisite-retirement-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.build = self.root / "build"
        self.build.mkdir()
        self.names = ("V2_Unit_A", "V2_Integration_Host", "V2_Integration_Exclusive")
        self.unselected = "UnselectedDiagnostic"
        self.bases = {}
        for name in (*self.names, self.unselected):
            base = self.root / name
            base.mkdir()
            self.bases[name] = base
        self.registrations = tuple(parity.PreflightRegistration(
            name, frozenset(("Integration", "ProductionTestPreflight", "ProcessOwned")),
            None, lane, ctest_index=index,
        ) for name, lane, index in (
            (self.names[1], parity.PreflightExecutionLane.HOST, 2),
            (self.names[2], parity.PreflightExecutionLane.EXCLUSIVE, 3),
        ))
        self.inventory = {"tests": [{
            "name": name,
            "command": ["/build/llaminar_mpi_session", "--session-base", str(base),
                        "--", "/build/owned-test"],
            "properties": [{"name": "LABELS", "value": ["ProcessOwned"]}],
        } for name, base in self.bases.items()]}

    def dirty(self, name: str) -> Path:
        """Retain an original-owner-shaped predecessor with a content witness."""
        pending = self.bases[name] / "llaminar-mpi-old-boot"
        pending.mkdir()
        (pending / "owner-evidence").write_text("original owner backing must remain")
        return pending

    def witness(self, pending: Path) -> tuple:
        """Compare original inode/content/metadata, excluding access-time observation."""
        return tuple((str(path.relative_to(pending)), path.stat().st_ino,
                      path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns,
                      path.read_bytes() if path.is_file() else None)
                     for path in (pending, *sorted(pending.rglob("*"))))

    def run_boundary(self, *, dirty_name: str | None = None):
        """Mock only installed build validation, canonical discovery and CTest execution."""
        pending = self.dirty(dirty_name) if dirty_name else None
        before = self.witness(pending) if pending else None
        # Installed images already own both binaries. The existing receipt path
        # avoids a build process so ANY process call here would launch tests.
        with mock.patch.object(prebuilt_test_image, "validate") as validate, \
             mock.patch.object(parity, "discover_production_parity_unit_tests", return_value=self.names[:1]), \
             mock.patch.object(parity, "discover_production_test_preflight_registrations", return_value=self.registrations), \
             mock.patch.object(parity.subprocess, "run", return_value=subprocess.CompletedProcess(
                 [], 0, json.dumps(self.inventory), "")) as discover, \
             mock.patch.object(parity, "_validate_preflight_index_selection") as preview, \
             mock.patch.object(parity, "_run_process", return_value=0) as execute:
            if dirty_name in self.names:
                with self.assertRaisesRegex(RuntimeError, "retirement is incomplete"):
                    parity.run_production_test_preflight(self.build, None, self.root / "installed-receipt")
                execute.assert_not_called()
                preview.assert_not_called()
                result = None
            else:
                result = parity.run_production_test_preflight(self.build, None, self.root / "installed-receipt")
                self.assertEqual(result.return_code, 0)
                self.assertEqual(result.tests, self.names)
                self.assertEqual(result.qualification["selected"], list(self.names))
                self.assertEqual(result.qualification["unqualified"], {})
                self.assertEqual(execute.call_count, 3)
                commands = [call.args[0] for call in execute.call_args_list]
                self.assertTrue(all(command[0] == "ctest" and "--no-tests=error" in command
                                    for command in commands))
                self.assertEqual(commands[0][commands[0].index("-R") + 1], "^V2_Unit_")
                self.assertEqual([call.args[2] for call in preview.call_args_list],
                                 [(self.names[1],), (self.names[2],)])
            validate.assert_called_once()
            discover.assert_called_once_with(
                ["ctest", "--test-dir", str(self.build), "--show-only=json-v1"],
                check=True, text=True, capture_output=True)
        if pending:
            self.assertEqual(self.witness(pending), before)
        return result

    def test_dirty_selected_future_lanes_reject_before_any_test_process(self) -> None:
        for name in self.names:
            with self.subTest(name=name):
                try:
                    self.run_boundary(dirty_name=name)
                finally:
                    # Test-only cleanup after the observation, even for the old
                    # failing body, keeps each selected lane independently dirty.
                    # Production deletion stays exclusively the native owner's job.
                    pending = self.bases[name] / "llaminar-mpi-old-boot"
                    (pending / "owner-evidence").unlink()
                    pending.rmdir()

    def test_clean_selected_roots_execute_the_unchanged_complete_inventory(self) -> None:
        self.run_boundary()

    def test_unselected_dirty_root_does_not_block_complete_selected_inventory(self) -> None:
        self.run_boundary(dirty_name=self.unselected)


if __name__ == "__main__":
    unittest.main()
