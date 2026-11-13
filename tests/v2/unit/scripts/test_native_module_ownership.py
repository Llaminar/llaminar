#!/usr/bin/env python3
"""Replay actual native-test link calls without an SDK, compiler, or GPU.

Production policy chooses the unique module owner. Imported graph nodes model
libraries; the ordinary group wrapper only records its arguments. The three
real module calls are replayed before/after the actual canonical-main policy.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
POLICY = ROOT / "tests/v2/cmake/V2NativeModuleOwnership.cmake"
ENTRY = ROOT / "tests/v2/cmake/V2GPUProcessTestEntryPolicy.cmake"
MODULES = {
    "V2CUDAAttentionCentroidFirstCaptureTests.cmake": "v2_integration_cuda_attention_centroid_first_capture",
    "V2CUDAHardwareMetadataTests.cmake": "v2_integration_cuda_hardware_metadata_native",
    "V2HIPProjectionCaptureLifetimeTests.cmake": "v2_integration_hip_projection_capture_lifetime",
}


def commands(source, name):
    """Read complete balanced calls for one narrowly named CMake operation."""
    found = []
    start = 0
    marker = name + "("
    while (position := source.find(marker, start)) >= 0:
        end = position + len(marker)
        depth = 1
        while depth:
            depth += (source[end] == "(") - (source[end] == ")")
            end += 1
        found.append(source[position:end])
        start = end
    return found


class NativeModuleOwnership(unittest.TestCase):
    """Authenticate real target graph ownership independently of archive order."""

    def real_calls(self):
        """Use each focused target's actual production link call, not a copy."""
        calls = []
        for module, target in MODULES.items():
            source = (POLICY.parent / module).read_text()
            matching = [call for call in commands(source, "v2_link_native_test_core")
                        if call.split("(", 1)[1].split()[0] == target]
            self.assertEqual(len(matching), 1, target)
            for operation in ("target_link_libraries", "target_link_libraries_with_groups"):
                direct = [call for call in commands(source, operation)
                          if call.split("(", 1)[1].split()[0] == target]
                self.assertEqual(direct, [], target + " bypasses its unique-owner link API")
            calls += matching
        return "\n".join(calls)

    def configure(self, kind, body=None, bind_main=False):
        """Configure imported actual-type nodes; no compiler detection or linking."""
        with tempfile.TemporaryDirectory(prefix="llaminar-native-owner-") as directory:
            root = Path(directory)
            targets = "\n".join(
                f'add_executable({target} IMPORTED GLOBAL)\n'
                f'set_target_properties({target} PROPERTIES IMPORTED_LOCATION "${{CMAKE_COMMAND}}")'
                for target in MODULES.values())
            writes = "\n".join(
                f'get_target_property(libraries {target} LINK_LIBRARIES)\n'
                f'get_target_property(owner {target} V2_NATIVE_MODULE_OWNER)\n'
                f'file(APPEND "${{CMAKE_BINARY_DIR}}/result.txt" "{target}|${{libraries}}|${{owner}}@@")'
                for target in MODULES.values())
            bindings = "\n".join(f'v2_bind_gpu_process_test_entry({target} canonical_main)'
                                 for target in MODULES.values()) if bind_main else ""
            source = f'''cmake_minimum_required(VERSION 3.25)
project(NativeOwnership NONE)
add_library(llaminar2_core {kind} IMPORTED GLOBAL)
set_target_properties(llaminar2_core PROPERTIES IMPORTED_LOCATION "${{CMAKE_COMMAND}}")
add_library(cuda_backend STATIC IMPORTED GLOBAL)
add_library(rocm_backend STATIC IMPORTED GLOBAL)
add_library(GTest::gtest_main INTERFACE IMPORTED)
add_library(CUDA::cudart INTERFACE IMPORTED)
add_library(hip::host INTERFACE IMPORTED)
add_library(canonical_main INTERFACE)
set_property(TARGET canonical_main PROPERTY INTERFACE_LINK_LIBRARIES llaminar2_core)
add_library(distinct_probe_kernel INTERFACE)
function(target_link_libraries_with_groups target)
    set_property(TARGET "${{target}}" APPEND PROPERTY LINK_LIBRARIES "-Wl,--start-group;${{ARGN}};-Wl,--end-group")
endfunction()
{targets}
include("{POLICY}")
include("{ENTRY}")
{self.real_calls() if body is None else body}
{bindings}
{writes}
'''
            (root / "CMakeLists.txt").write_text(source)
            ninja = shutil.which("ninja")
            self.assertIsNotNone(ninja)
            result = subprocess.run(["cmake", "-S", str(root), "-B", str(root / "build"),
                                     "-G", "Ninja", "-DCMAKE_MAKE_PROGRAM=" + ninja],
                                    capture_output=True, text=True, timeout=10)
            data = {}
            if result.returncode == 0:
                for record in (root / "build/result.txt").read_text().split("@@"):
                    if record:
                        target, libraries, owner = record.split("|")
                        data[target] = (libraries.split(";"), owner)
            return result, data

    def test_shared_core_owns_all_three_native_modules(self):
        for bind_main in (False, True):
            result, data = self.configure("SHARED", bind_main=bind_main)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for target, (libraries, owner) in data.items():
                self.assertEqual(owner, "llaminar2_core", target)
                self.assertNotIn("cuda_backend", libraries, target)
                self.assertNotIn("rocm_backend", libraries, target)
                self.assertNotIn("-Wl,--whole-archive", libraries, target)
                self.assertNotIn("-Wl,--no-whole-archive", libraries, target)
                self.assertIn("canonical_main" if bind_main else "GTest::gtest_main", libraries)

    def test_static_core_retains_exactly_one_requested_backend(self):
        result, data = self.configure("STATIC", bind_main=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for target, (libraries, owner) in data.items():
            expected = "rocm_backend" if "hip_projection" in target else "cuda_backend"
            self.assertEqual(owner, expected)
            self.assertEqual(libraries.count(expected), 1)
            self.assertEqual(libraries.count("-Wl,--whole-archive"), 1)
            self.assertEqual(libraries.count("-Wl,--no-whole-archive"), 1)
            archive = libraries.index(expected)
            self.assertEqual(libraries[archive - 1], "-Wl,--whole-archive")
            self.assertEqual(libraries[archive + 1], "-Wl,--no-whole-archive")
            self.assertLess(libraries.index("llaminar2_core"), archive - 1)
            self.assertGreater(libraries.index("canonical_main"), archive + 1)

    def test_distinct_test_kernel_target_is_preserved(self):
        result, data = self.configure("SHARED", '''
v2_link_native_test_core(v2_integration_cuda_attention_centroid_first_capture cuda_backend distinct_probe_kernel GTest::gtest_main CUDA::cudart)
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        libraries, owner = data["v2_integration_cuda_attention_centroid_first_capture"]
        self.assertIn("distinct_probe_kernel", libraries)
        self.assertEqual(owner, "llaminar2_core")
        self.assertNotIn("cuda_backend", libraries)

    def test_repeated_backend_alias_is_rejected(self):
        result, _ = self.configure("SHARED", '''
add_library(alias_backend ALIAS cuda_backend)
v2_link_native_test_core(v2_integration_cuda_attention_centroid_first_capture cuda_backend alias_backend)
''')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must not repeat a native backend archive", result.stderr)

    def test_original_direct_archive_control_keeps_duplicate_owner(self):
        result, data = self.configure("SHARED", '''
target_link_libraries_with_groups(v2_integration_cuda_attention_centroid_first_capture llaminar2_core cuda_backend GTest::gtest_main CUDA::cudart)
''', bind_main=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        libraries, owner = data["v2_integration_cuda_attention_centroid_first_capture"]
        self.assertIn("cuda_backend", libraries)
        self.assertIn("llaminar2_core", libraries)
        self.assertIn("canonical_main", libraries)
        self.assertEqual(owner, "owner-NOTFOUND")


if __name__ == "__main__":
    unittest.main()
