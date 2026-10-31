#!/usr/bin/env python3
"""Exercise the shared module's actual NCCL/RCCL production path-property guard.

LANGUAGES NONE fixtures query actual CMake source properties without finding
an SDK, compiler, library or GPU. Missing/wrong/duplicate bindings reject in
both backends, including a correct definition attached to the wrong directory.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
MODULE = ROOT / "tests/v2/cmake/V2NCCLConfiguredLibraryTests.cmake"


class NCCLConfiguredLibraryCMakeTests(unittest.TestCase):
    """Protect the real source definition beyond the C++ spy's sentinel macro."""

    backend = "NCCL"
    runtime = "CUDA"
    library = "libllaminar_nccl.so.2"

    def binding(self, path=None):
        """Return the exact expected compiler definition for one selected path."""
        selected = path or "/selected private sdk/lib/" + self.library
        return f'LLAMINAR_{self.backend}_LIBRARY_PATH="{selected}"'

    def configure(self, definitions, wrong_scope=False):
        """Evaluate the actual module guard in a compiler-free CMake source fixture."""
        with tempfile.TemporaryDirectory(prefix=self.backend.lower() + "-path-cmake-") as directory:
            source = Path(directory) / "source"
            relative = f"collective/backends/{self.backend}DynamicLoader.cpp"
            loader = source / "src/v2" / relative
            loader.parent.mkdir(parents=True)
            loader.write_text("// Configuration witness only; never compiled.\n")
            property_command = ""
            if definitions:
                values = " ".join('"' + item.replace('"', '\\"') + '"' for item in definitions)
                target = str(loader) if wrong_scope else relative
                property_command = (f'set_property(SOURCE "{target}" APPEND PROPERTY '
                                    f'COMPILE_DEFINITIONS {values})\n')
            (source / "src/v2/CMakeLists.txt").write_text("" if wrong_scope else property_command)
            (source / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.25)\nproject(PathWitness LANGUAGES NONE)\n'
                f'set(REPO_ROOT "{source}")\n'
                f'set(HAVE_{self.runtime} TRUE)\nset(HAVE_{self.backend} TRUE)\n'
                'set(V2_PERF_TESTS_ONLY TRUE)\n'
                f'set({self.backend}_LIBRARY "/selected private sdk/lib/{self.library}")\n'
                + (property_command if wrong_scope else "")
                + f'add_subdirectory(src/v2)\ninclude("{MODULE}")\n')
            ninja = shutil.which("ninja")
            self.assertIsNotNone(ninja)
            result = subprocess.run(["cmake", "-S", str(source), "-B", str(Path(directory) / "build"),
                                     "-G", "Ninja", "-DCMAKE_MAKE_PROGRAM=" + ninja],
                                    capture_output=True, text=True, timeout=10)
            return result.returncode, result.stdout + result.stderr

    def test_exact_selected_path_property_passes(self):
        """The selected path survives whitespace and its real owning scope."""
        status, output = self.configure([self.binding()])
        self.assertEqual(status, 0, output)

    def test_removed_or_different_binding_is_fatal(self):
        """The spy cannot hide missing or stale production source definitions."""
        for definitions in ([], ['OTHER_DEFINITION=1'], [self.binding(self.library)],
                            [self.binding("/wrong sdk/" + self.library)]):
            with self.subTest(definitions=definitions):
                status, output = self.configure(definitions)
                self.assertNotEqual(status, 0)
                self.assertIn(f"must bind exactly the selected {self.backend}_LIBRARY", output)

    def test_duplicate_path_definition_is_fatal(self):
        """A valid definition cannot hide a competing later macro value."""
        status, output = self.configure([self.binding(), self.binding("/wrong sdk/" + self.library)])
        self.assertNotEqual(status, 0)
        self.assertIn("must bind exactly", output)

    def test_definition_in_wrong_directory_scope_is_fatal(self):
        """The property must reach the actual src/v2 loader translation unit."""
        status, output = self.configure([self.binding()], wrong_scope=True)
        self.assertNotEqual(status, 0)
        self.assertIn("must bind exactly", output)


class RCCLConfiguredLibraryCMakeTests(NCCLConfiguredLibraryCMakeTests):
    """Run the same actual-property witness against the selected RCCL source."""

    backend = "RCCL"
    runtime = "ROCM"
    library = "librccl.so.1"


if __name__ == "__main__":
    unittest.main()
