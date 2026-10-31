## @file test_mpi_ctest_session_isolation.py
#  @brief Device-free adversarial inventory checks for the production MPI owner.
"""Keep raw launchers and competing file-location authorities out of CTest."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2] / "integration/scripts/test_mpi_ctest_session_isolation.py"
SPEC = importlib.util.spec_from_file_location("mpi_ctest_owner", SOURCE)
assert SPEC is not None and SPEC.loader is not None
OWNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OWNER)


class MPICTestOwnerTests(unittest.TestCase):
    """Validate argv boundaries and real directory identity without starting MPI."""

    def setUp(self) -> None:
        """Create only this case's disposable build and helper fixture."""
        self.temporary = tempfile.TemporaryDirectory(prefix="mpi inventory ")
        self.addCleanup(self.temporary.cleanup)
        self.build = Path(self.temporary.name)
        self.helper = self.build / "llaminar_mpi_session"
        self.helper.touch()

    def registration(self, name: str = "first") -> dict:
        """Return one canonical wrapped CTest command, retaining argv with spaces."""
        base = self.build / "mpi-sessions" / name
        base.mkdir(parents=True, exist_ok=True)
        return {"name": name,
                "command": [str(self.helper), "--session-base", str(base), "--",
                            "/usr/bin/mpirun", "-np", "2", "--", "program with spaces"],
                "properties": [{"name": "LABELS", "value": ["ProductionTestPreflight", "ProcessOwned"]},
                               {"name": "TIMEOUT_SIGNAL_NAME", "value": "SIGTERM"},
                               {"name": "TIMEOUT_SIGNAL_GRACE_PERIOD", "value": 36}]}

    def validate(self, *tests: dict) -> dict:
        """Invoke the same validator that audits the generated CTest inventory."""
        return OWNER._validate_session_ownership({"tests": list(tests)}, self.build)

    def test_independent_owners_and_spaces(self) -> None:
        self.assertEqual(set(self.validate(self.registration(), self.registration("second"))),
                         {"first", "second"})

    def test_raw_original_launcher_is_rejected(self) -> None:
        test = self.registration()
        test["command"] = test["command"][4:]
        with self.assertRaisesRegex(AssertionError, "no original MPI"):
            self.validate(test)

    def test_missing_separator_is_rejected(self) -> None:
        test = self.registration()
        del test["command"][3]
        with self.assertRaisesRegex(AssertionError, "no original MPI"):
            self.validate(test)

    def test_helper_must_be_built(self) -> None:
        test = self.registration()
        self.helper.unlink()
        with self.assertRaisesRegex(AssertionError, "unavailable MPI session helper"):
            self.validate(test)

    def test_root_must_be_build_owned(self) -> None:
        test = self.registration()
        test["command"][2] = "/tmp"
        with self.assertRaisesRegex(AssertionError, "unavailable session root"):
            self.validate(test)

    def test_root_must_exist(self) -> None:
        test = self.registration()
        Path(test["command"][2]).rmdir()
        with self.assertRaisesRegex(AssertionError, "unavailable session root"):
            self.validate(test)

    def test_shared_base_is_rejected(self) -> None:
        first, second = self.registration(), self.registration("second")
        second["command"][2] = first["command"][2]
        with self.assertRaisesRegex(AssertionError, "shares its MPI session root"):
            self.validate(first, second)

    def test_explicit_mpi_location_override_is_rejected(self) -> None:
        for option in ("orte_tmpdir_base", "prte_tmpdir_base", "btl_vader_backing_directory",
                       "btl_sm_backing_directory"):
            with self.subTest(option=option):
                test = self.registration()
                test["command"][5:5] = ["--mca", option, "/tmp"]
                with self.assertRaisesRegex(AssertionError, "overrides its MPI owner's"):
                    self.validate(test)

    def test_host_only_tests_are_not_mpi_owners(self) -> None:
        host = {"name": "host", "command": ["python3", "unit.py"],
                "properties": [{"name": "LABELS", "value": ["Unit"]}]}
        self.assertEqual(set(self.validate(host, self.registration())), {"first"})

    def test_missing_mpi_inventory_fails(self) -> None:
        with self.assertRaisesRegex(AssertionError, "No MPI"):
            self.validate()

    def test_timeout_must_allow_actual_retirement(self) -> None:
        test = self.registration()
        test["properties"].pop()
        with self.assertRaisesRegex(AssertionError, "timeout cleanup"):
            self.validate(test)

    def test_native_application_options_are_not_mpi_options(self) -> None:
        test = self.registration()
        test["command"] += ["--mca", "orte_tmpdir_base", "prompt words"]
        self.assertEqual(set(self.validate(test)), {"first"})

    def test_missing_native_mpi_separator_is_rejected(self) -> None:
        test = self.registration()
        del test["command"][7]
        with self.assertRaisesRegex(AssertionError, "MPI/application boundary"):
            self.validate(test)

    def test_singleton_and_script_jobs_keep_process_ownership(self) -> None:
        test = self.registration()
        test["command"][4:] = ["python3", "script.py"]
        self.assertEqual(set(self.validate(test)), {"first"})
        test["command"] = test["command"][4:]
        with self.assertRaisesRegex(AssertionError, "no original MPI"):
            self.validate(test)

    def test_script_mpirun_argument_is_not_a_launcher(self) -> None:
        test = self.registration()
        test["command"][4:] = ["python3", "script.py", "--mpirun", "/usr/bin/mpirun"]
        self.assertEqual(set(self.validate(test)), {"first"})

    def test_native_aliases_keep_the_same_owner_contract(self) -> None:
        for alias in ("orterun", "prterun", "mpirun.openmpi", "mpiexec.openmpi"):
            with self.subTest(alias=alias):
                test = self.registration()
                test["command"][4] = "/usr/bin/" + alias
                self.assertEqual(set(self.validate(test)), {"first"})
                test["command"] = test["command"][4:]
                with self.assertRaisesRegex(AssertionError, "no original MPI"):
                    self.validate(test)

    def test_opaque_mca_value_does_not_end_native_options(self) -> None:
        test = self.registration()
        test["command"][5:5] = ["--mca", "opaque", "--", "--mca", "orte_tmpdir_base", "/tmp"]
        with self.assertRaisesRegex(AssertionError, "overrides its MPI owner's"):
            self.validate(test)


if __name__ == "__main__":
    unittest.main()
