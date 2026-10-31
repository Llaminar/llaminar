## @file test_mpi_session_retirement.py
#  @brief Device-free scheduling-boundary regressions for native MPI file retirement.
"""Python must observe the native owner, never remove files to manufacture a pass."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts/ci"))
import mpi_session_retirement as retirement


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


if __name__ == "__main__":
    unittest.main()
