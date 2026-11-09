#!/usr/bin/env python3
"""Exercise actual CMake/CTest ISA registration and explicit preflight membership.

An imported inert executable stands in for the native binary only for metadata
generation. No test runs here: the real policy GTests remain in Unit/preflight.
Both shipping configure modes must emit exactly their supported runtime ISAs.
"""
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]


class CPUExecutionExactRegistrationTests(unittest.TestCase):
    """Read generated CTest output rather than assert copied source spellings."""

    def test_both_shipping_builds_have_only_registered_preflight_aliases(self):
        for isa, expected in (("AVX2", {"AVX2"}), ("AVX512", {"AVX2", "AVX512"})):
            with self.subTest(isa=isa), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary)
                helper = ROOT / "tests/v2/cmake/V2CPUExecutionExactPolicyTests.cmake"
                (source / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.24)
project(ExecutionExactRegistration NONE)
enable_testing()
add_executable(v2_integration_cpu_native_quantised_fused_completion IMPORTED)
set_property(TARGET v2_integration_cpu_native_quantised_fused_completion PROPERTY IMPORTED_LOCATION /bin/true)
function(add_v2_test name)
  cmake_parse_arguments(TEST "NO_MPI;NO_MODELS" "MPI_PROCS;TIMEOUT" "COMMAND;LABELS;ENVIRONMENT" ${ARGN})
  add_test(NAME ${name} COMMAND ${TEST_COMMAND})
  set_tests_properties(${name} PROPERTIES LABELS "${TEST_LABELS}" ENVIRONMENT "${TEST_ENVIRONMENT}")
endfunction()
include("''' + helper.as_posix() + '''")
foreach(name IN LISTS V2_CPU_EXECUTION_EXACT_POLICY_PREFLIGHT_TESTS)
  if(NOT TEST ${name})
    message(FATAL_ERROR "Explicit preflight entry is not registered: ${name}")
  endif()
endforeach()
file(WRITE "${CMAKE_BINARY_DIR}/members.txt" "${V2_CPU_EXECUTION_EXACT_POLICY_PREFLIGHT_TESTS}")
''')
                build = source / "build"
                subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
                                "-DCMAKE_MAKE_PROGRAM=" + shutil.which("ninja"),
                                "-DLLAMINAR_CPU_ISA_UPPER=" + isa], check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                inventory = json.loads(subprocess.check_output(
                    ["ctest", "--test-dir", str(build), "--show-only=json-v1"], text=True))
                names = {test["name"] for test in inventory["tests"]}
                wanted = {"V2_Integration_CPU_NativeQuantisedExecutionExactPolicy_" + mode for mode in expected}
                self.assertEqual(names, wanted)
                self.assertEqual(set((build / "members.txt").read_text().split(";")), wanted)
                for test in inventory["tests"]:
                    properties = {row["name"]: row["value"] for row in test["properties"]}
                    self.assertTrue({"Unit", "Integration", "ProductionTestPreflight"} <= set(properties["LABELS"]))
                    runtime = test["name"].rsplit("_", 1)[1].lower()
                    self.assertIn("LLAMINAR_ISA_LEVEL=" + runtime, properties["ENVIRONMENT"])


if __name__ == "__main__":
    unittest.main()
