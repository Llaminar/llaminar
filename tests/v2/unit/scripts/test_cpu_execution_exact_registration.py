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
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
from run_production_parity_campaigns import discover_production_parity_unit_tests


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
  set_tests_properties(${name} PROPERTIES LABELS "${TEST_LABELS}" ENVIRONMENT "${TEST_ENVIRONMENT}" TIMEOUT "${TEST_TIMEOUT}")
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
                # Replay the installed-image sealer's actual Unit inventory
                # admission. Mixed Unit/Integration labels previously passed
                # this configure fixture but failed after a full image build.
                unit_names = set(discover_production_parity_unit_tests(build))
                names = {test["name"] for test in inventory["tests"]}
                wanted = {f"V2_{scope}_CPU_NativeQuantisedExecutionExactPolicy_{mode}"
                          for scope in ("Unit", "Integration") for mode in expected}
                self.assertEqual(names, wanted)
                preflight_names = {name for name in wanted if name.startswith("V2_Integration_")}
                self.assertEqual(set((build / "members.txt").read_text().split(";")), preflight_names)
                self.assertEqual(unit_names, wanted - preflight_names)
                for test in inventory["tests"]:
                    properties = {row["name"]: row["value"] for row in test["properties"]}
                    labels = set(properties["LABELS"])
                    if test["name"] in unit_names:
                        self.assertIn("Unit", labels)
                        self.assertFalse({"Integration", "ProductionTestPreflight"} & labels)
                    else:
                        self.assertTrue({"Integration", "ProductionTestPreflight"} <= labels)
                        self.assertNotIn("Unit", labels)
                    runtime = test["name"].rsplit("_", 1)[1].lower()
                    self.assertIn("LLAMINAR_ISA_LEVEL=" + runtime, properties["ENVIRONMENT"])


if __name__ == "__main__":
    unittest.main()
