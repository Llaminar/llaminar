#!/usr/bin/env python3
"""Verify publication admission's model-free, non-MPI registration owner.

Configure the production declaration module and Release selection functions with
inert imported executables. No compiler, model, MPI launcher or device is needed.
This checks selection and process/model flags, not native linking or numerical
execution; the executable's MPI tripwires enforce its actual startup boundary.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
OWNER = "v2_integration_cpu_native_quantised_publication_admission"


class PublicationAdmissionRegistration(unittest.TestCase):
    """Keep every supported ISA on the isolated executable in all build modes."""

    def test_native_isa_and_release_owner_selection(self):
        for isa, runtime, expected in (("AVX2", "avx2", {"AVX2"}),
                                       ("AVX512", "avx512", {"AVX2", "AVX512"}),
                                       ("NEON", "neon", {"NEON"})):
            for mode in ("integration", "release-omitted", "release-selected"):
                with self.subTest(isa=isa, mode=mode), tempfile.TemporaryDirectory() as tmp:
                    source = Path(tmp)
                    source.joinpath("CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.24)
project(PublicationAdmissionRegistration NONE)
enable_testing()
# Load the actual selectors without creating native-owner validation jobs:
# these deliberately imported inert owners prove registration metadata only.
include("${SOURCE_ROOT}/tests/v2/cmake/V2TestTargetSelection.cmake")
set(V2_PERF_TESTS_ONLY ${TEST_PERF_ONLY})
macro(add_executable target)
  v2_test_target_selected("${target}" selected)
  if(selected)
    _add_executable(${target} IMPORTED)
    set_target_properties(${target} PROPERTIES IMPORTED_LOCATION "${TEST_STUB}")
    file(APPEND "${CMAKE_BINARY_DIR}/owners.txt" "${target}\n")
  endif()
endmacro()
function(target_link_libraries)
endfunction()
function(target_include_directories)
endfunction()
function(add_v2_test name)
  v2_test_registration_selected("${name}" selected ${ARGN})
  if(NOT selected)
    return()
  endif()
  cmake_parse_arguments(T "NO_MPI;NO_MODELS" "MPI_PROCS" "COMMAND;LABELS;ENVIRONMENT" ${ARGN})
  if(NOT T_NO_MPI OR NOT T_NO_MODELS OR NOT T_MPI_PROCS EQUAL 1)
    message(FATAL_ERROR "Publication admission must be model-free and non-MPI: ${name}")
  endif()
  add_test(NAME ${name} COMMAND ${T_COMMAND})
  set_tests_properties(${name} PROPERTIES LABELS "${T_LABELS}" ENVIRONMENT "${T_ENVIRONMENT}")
endfunction()
include("${SOURCE_ROOT}/tests/v2/cmake/V2CPUPublicationAdmissionTests.cmake")
file(WRITE "${CMAKE_BINARY_DIR}/preflight.txt" "${V2_PRODUCTION_TEST_PREFLIGHT_TESTS}")
''')
                    build = source / "build"
                    subprocess.run([
                        "cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
                        "-DCMAKE_MAKE_PROGRAM=" + shutil.which("ninja"),
                        "-DSOURCE_ROOT=" + str(ROOT), "-DTEST_STUB=" + shutil.which("true"),
                        "-DPython3_EXECUTABLE=" + sys.executable,
                        "-DLLAMINAR_CPU_ISA_UPPER=" + isa,
                        "-DV2_CPU_BASELINE_RUNTIME_ISA=" + runtime,
                        "-DTEST_PERF_ONLY=" + ("OFF" if mode == "integration" else "ON"),
                        "-DLLAMINAR_RELEASE_NATIVE_TEST_TARGETS=" + (OWNER if mode == "release-selected" else ""),
                    ], check=True, capture_output=True, text=True)
                    tests = json.loads(subprocess.check_output([
                        "ctest", "--test-dir", str(build), "--show-only=json-v1"
                    ], text=True))["tests"]
                    if mode == "release-omitted":
                        self.assertEqual(tests, [])
                        self.assertFalse((build / "owners.txt").exists())
                        continue
                    self.assertEqual((build / "owners.txt").read_text().splitlines(), [OWNER])
                    names = {"V2_Integration_CPU_NativeQuantisedPublicationAdmission_" + cell
                             for cell in expected}
                    names.add("V2_Integration_CPU_NativeQuantisedPublicationStartupGuard")
                    self.assertEqual({test["name"] for test in tests}, names)
                    self.assertEqual(set((build / "preflight.txt").read_text().split(";")), names)
                    for test in tests:
                        props = {p["name"]: p["value"] for p in test["properties"]}
                        self.assertIn("ProductionTestPreflight", props["LABELS"])
                        if test["name"].endswith("PublicationStartupGuard"):
                            self.assertIn("--binary", test["command"])
                            continue
                        self.assertIn("--gtest_filter=CPUNativeQuantisedPublicationAdmission.*", test["command"])
                        cell = test["name"].rsplit("_", 1)[1].lower()
                        self.assertEqual(props["ENVIRONMENT"], ["LLAMINAR_ISA_LEVEL=" + cell])


if __name__ == "__main__":
    unittest.main()
