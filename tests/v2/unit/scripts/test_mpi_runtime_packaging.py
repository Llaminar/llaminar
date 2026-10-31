## @file test_mpi_runtime_packaging.py
#  @brief Keep the native MPI owner module inside installed-image evidence.
"""A test image cannot omit or alter the helper's required native module."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts/ci"))
import prebuilt_test_image as installed


class MPIRuntimePackagingTests(unittest.TestCase):
    """Replay actual image-inventory validation with tiny metadata-only fixtures."""

    def setUp(self) -> None:
        """Create a disposable installed tree and its exact registered module operand."""
        temporary = tempfile.TemporaryDirectory(prefix="mpi-runtime-packaging-")
        self.addCleanup(temporary.cleanup)
        self.build = Path(temporary.name)
        for name in ("CMakeCache.txt", "compile_commands.json", "build.ninja", "CMakeFiles/rules.ninja"):
            path = self.build / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
        self.module = self.build / "llaminar_mpi_components/mca_schizo_llaminar.so"
        self.module.parent.mkdir()
        self.module.write_bytes(b"original fixture module")
        registration = {"tests": [{"name": "V2_Integration_MPISessionNativeDaemon",
            "command": ["python3", "native-replay.py", "--module", str(self.module)],
            "properties": [{"name": "LABELS", "value": ["ProductionTestPreflight"]}]}]}
        query = mock.patch.object(installed.subprocess, "check_output", return_value=json.dumps(registration))
        query.start()
        self.addCleanup(query.stop)

    def test_native_module_is_part_of_the_installed_inventory(self) -> None:
        receipt = installed.inventory(self.build)
        entry = receipt["files"][str(self.module.relative_to(self.build))]
        self.assertEqual(entry["bytes"], self.module.stat().st_size)

    def test_required_missing_module_rejects_the_image(self) -> None:
        self.module.unlink()
        with self.assertRaisesRegex(ValueError, "installed test executable is missing"):
            installed.inventory(self.build)

    def test_changed_native_module_invalidates_the_sealed_receipt(self) -> None:
        receipt = self.build / "installed.json"
        receipt.write_text(json.dumps({"schema": 1, "inventory": installed.inventory(self.build)}))
        self.module.write_bytes(b"different installed native module")
        with self.assertRaisesRegex(ValueError, "receipt does not match"):
            installed.validate(receipt, self.build)


if __name__ == "__main__":
    unittest.main()
