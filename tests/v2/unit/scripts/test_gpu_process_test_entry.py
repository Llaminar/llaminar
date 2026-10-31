#!/usr/bin/env python3
"""Replay the production CMake target policy without a compiler or GPU runtime.

The real link graph determines whether a default GTest entry needs native
retirement. Independent tests must retain their upstream, runtime-free entry.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
POLICY = ROOT / "tests/v2/cmake/V2GPUProcessTestEntryPolicy.cmake"


class GPUProcessTestEntry(unittest.TestCase):
    """Authenticate actual target mutation, including transitive and cyclic links."""

    def configure(self, body):
        """Configure a language-free graph through the same installed helper."""
        with tempfile.TemporaryDirectory(prefix="llaminar-gpu-entry-") as directory:
            root = Path(directory)
            (root / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.25)
project(GPUEntryPolicy NONE)
add_library(llaminar2_core INTERFACE)
add_library(GTest::gtest_main INTERFACE IMPORTED)
add_library(canonical_main INTERFACE)
add_executable(probe IMPORTED GLOBAL)
set_target_properties(probe PROPERTIES IMPORTED_LOCATION "${{CMAKE_COMMAND}}")
include("{POLICY}")
{body}
get_target_property(links probe LINK_LIBRARIES)
get_target_property(owner probe V2_GPU_PROCESS_ENTRY)
file(WRITE "${{CMAKE_BINARY_DIR}}/result.txt" "${{links}}\\n${{owner}}\\n")
''')
            ninja = shutil.which("ninja")
            self.assertIsNotNone(ninja)
            result = subprocess.run(
                ["cmake", "-S", str(root), "-B", str(root / "build"), "-G", "Ninja",
                 "-DCMAKE_MAKE_PROGRAM=" + ninja], capture_output=True, text=True,
                timeout=10)
            text = (root / "build/result.txt").read_text() if result.returncode == 0 else ""
            return result, text

    def test_direct_core_default_entry_is_owned_and_idempotent(self):
        result, text = self.configure('''
set_property(TARGET probe PROPERTY LINK_LIBRARIES "llaminar2_core;GTest::gtest_main")
v2_bind_gpu_process_test_entry(probe canonical_main)
v2_bind_gpu_process_test_entry(probe canonical_main)
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(text, "llaminar2_core;canonical_main\ncanonical_main\n")

    def test_transitive_static_private_cycle_retains_native_owner(self):
        result, text = self.configure('''
add_library(first INTERFACE)
add_library(second INTERFACE)
set_property(TARGET first PROPERTY INTERFACE_LINK_LIBRARIES "second")
set_property(TARGET second PROPERTY INTERFACE_LINK_LIBRARIES "first;$<LINK_ONLY:llaminar2_core>")
set_property(TARGET probe PROPERTY LINK_LIBRARIES "first;GTest::gtest_main")
v2_bind_gpu_process_test_entry(probe canonical_main)
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(text, "first;canonical_main\ncanonical_main\n")

    def test_runtime_independent_default_entry_remains_lean(self):
        result, text = self.configure('''
add_library(header_fixture INTERFACE)
set_property(TARGET probe PROPERTY LINK_LIBRARIES "header_fixture;GTest::gtest_main")
v2_bind_gpu_process_test_entry(probe canonical_main)
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(text, "header_fixture;GTest::gtest_main\nowner-NOTFOUND\n")

    def test_core_and_main_aliases_bind_the_actual_owners(self):
        result, text = self.configure('''
add_library(project_core ALIAS llaminar2_core)
add_library(project_main ALIAS GTest::gtest_main)
set_property(TARGET probe PROPERTY LINK_LIBRARIES "project_core;project_main")
v2_bind_gpu_process_test_entry(probe canonical_main)
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(text, "project_core;canonical_main\ncanonical_main\n")

    def test_transitive_default_main_is_preceded_by_owned_entry(self):
        result, text = self.configure('''
add_library(wrapped_fixture INTERFACE)
set_property(TARGET wrapped_fixture PROPERTY INTERFACE_LINK_LIBRARIES "llaminar2_core;GTest::gtest_main")
set_property(TARGET probe PROPERTY LINK_LIBRARIES "wrapped_fixture")
v2_bind_gpu_process_test_entry(probe canonical_main)
v2_bind_gpu_process_test_entry(probe canonical_main)
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(text, "canonical_main;wrapped_fixture\ncanonical_main\n")

    def test_conflicting_entry_cannot_replace_existing_owner(self):
        result, _ = self.configure('''
set_property(TARGET probe PROPERTY LINK_LIBRARIES "llaminar2_core;GTest::gtest_main")
v2_bind_gpu_process_test_entry(probe canonical_main)
add_library(second_main INTERFACE)
v2_bind_gpu_process_test_entry(probe second_main)
''')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Conflicting GPU process entry", result.stderr)

    def test_custom_entry_keeps_its_explicit_owner(self):
        result, text = self.configure('''
set_property(TARGET probe PROPERTY LINK_LIBRARIES "llaminar2_core")
v2_bind_gpu_process_test_entry(probe canonical_main)
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(text, "llaminar2_core\nowner-NOTFOUND\n")

    def test_missing_required_entry_is_a_configuration_error(self):
        result, _ = self.configure('''
set_property(TARGET probe PROPERTY LINK_LIBRARIES "llaminar2_core;GTest::gtest_main")
v2_bind_gpu_process_test_entry(probe missing_entry)
''')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing canonical core GTest entry", result.stderr)


if __name__ == "__main__":
    unittest.main()
