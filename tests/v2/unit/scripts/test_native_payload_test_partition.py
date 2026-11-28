#!/usr/bin/env python3
"""Exercise native-payload CTest filter ownership through real CMake configuration.

The imported binary is inert: this device-free registration regression never
substitutes for native numerical tests. It checks that the focused registrations
and residual suite cover each declared fixture case exactly once per ISA, while
new cases remain covered by the residual suite. Production GTests still execute
through their original binaries and resource contracts. Admission metadata uses
actual Release target selection for both omitted and explicitly selected owners,
so property setters cannot require tests that the build intentionally excludes.
"""
import fnmatch
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]


def selected(name, expression):
    """Apply GTest's positive/negative wildcard filter semantics to one case."""
    positive, _, negative = expression.partition("-")
    return (any(fnmatch.fnmatchcase(name, p) for p in (positive or "*").split(":"))
            and not any(fnmatch.fnmatchcase(name, p) for p in negative.split(":") if p))


class NativePayloadPartitionTests(unittest.TestCase):
    """Validate generated registrations, including full and residual ownership."""

    def test_admission_owner_and_kpart_worker_reservations(self):
        """Configure actual leaf/GEMV declarations without building numerical code."""
        declarations = (ROOT / "tests/v2/CMakeLists.txt").read_text()
        def section(begin, end):
            first = declarations.index(begin)
            return declarations[first:declarations.index(end, first)]
        leaves = section("if(V2_CPU_TARGET_X86)\n    add_executable(v2_unit_cpu_x86_coded_leaves",
                         "# Q6_K and IQ4_XS coded CPU units")
        gemv = section("set(V2_NATIVE_PAYLOAD_KPART_THREADS 32)",
                       'include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/V2CPUPublicationAdmissionTests.cmake")')
        heuristic = section("# x86 shared-K prefill and verifier-route cases;",
                            "foreach(V2_PREFILL_ISA IN LISTS V2_CPU_NATIVE_ISA_CELLS)")
        cells_and_modes = ((cells, x86, selection)
                           for cells, x86 in (("AVX2", "ON"), ("AVX2;AVX512", "ON"), ("NEON", "OFF"))
                           for selection in ("integration", "release-omitted", "release-selected"))
        for cells, x86, selection in cells_and_modes:
            build_owners = []
            if selection == "release-selected":
                build_owners = ["v2_integration_cpu_native_quantised_gemv"]
                if x86 == "ON":
                    build_owners.append("v2_unit_cpu_x86_coded_leaves")
            with self.subTest(cells=cells, selection=selection), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp)
                for filename in ("unit/kernels/cpu/Test__CPUX86CodedLeaves.cpp",
                                 "integration/kernels/cpu/native_quantised/Test__CPUNativeQuantised_GEMV.cpp"):
                    stub = source / filename
                    stub.parent.mkdir(parents=True, exist_ok=True)
                    stub.write_text("int main() { return 0; }\n")
                source.joinpath("CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.24)
project(NativePayloadAdmission CXX)
enable_testing()
add_library(llaminar2_core INTERFACE)
add_library(GTest::gtest INTERFACE IMPORTED)
add_library(GTest::gtest_main INTERFACE IMPORTED)
include("${REGISTRATION_ROOT}/tests/v2/cmake/V2TestTargetSelection.cmake")
function(v2_cpu_native_isa_runtime isa result)
  string(TOLOWER "${isa}" runtime)
  set(${result} "${runtime}" PARENT_SCOPE)
endfunction()
function(add_v2_test name)
  v2_test_registration_selected("${name}" selected ${ARGN})
  if(NOT selected)
    return()
  endif()
  cmake_parse_arguments(T "NO_MPI;NO_MODELS" "MPI_PROCS;TIMEOUT" "COMMAND;LABELS;ENVIRONMENT" ${ARGN})
  add_test(NAME ${name} COMMAND ${T_COMMAND})
  set_tests_properties(${name} PROPERTIES LABELS "${T_LABELS}" ENVIRONMENT "${T_ENVIRONMENT}" PROCESSORS 4)
endfunction()
''' + leaves + gemv + heuristic + '''
if(TARGET v2_integration_cpu_native_quantised_gemv)
  get_target_property(definitions v2_integration_cpu_native_quantised_gemv COMPILE_DEFINITIONS)
  file(WRITE "${CMAKE_BINARY_DIR}/definitions.txt" "${definitions}")
endif()
file(WRITE "${CMAKE_BINARY_DIR}/preflight.txt" "${V2_PRODUCTION_TEST_PREFLIGHT_TESTS}")
''')
                build = source / "build"
                subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
                                "-DCMAKE_MAKE_PROGRAM=" + shutil.which("ninja"),
                                "-DV2_CPU_NATIVE_ISA_CELLS=" + cells, "-DV2_CPU_TARGET_X86=" + x86,
                                "-DREGISTRATION_ROOT=" + str(ROOT),
                                "-DV2_PERF_TESTS_ONLY=" + ("OFF" if selection == "integration" else "ON"),
                                "-DLLAMINAR_RELEASE_NATIVE_TEST_TARGETS=" + ";".join(build_owners)],
                               check=True, capture_output=True, text=True)
                # Build only inert local stubs so CTest publishes resolved commands.
                build_command = ["cmake", "--build", str(build), "--parallel"]
                if build_owners:
                    build_command.extend(["--target", *build_owners])
                subprocess.run(build_command, check=True, capture_output=True, text=True)
                tests = json.loads(subprocess.check_output(
                    ["ctest", "--test-dir", str(build), "--show-only=json-v1"], text=True))["tests"]
                by_name = {t["name"]: t for t in tests}
                if selection == "release-omitted":
                    self.assertEqual(by_name, {})
                    self.assertFalse((build / "definitions.txt").exists())
                    continue
                definitions = (build / "definitions.txt").read_text().split(";")
                width = int(next(d.split("=", 1)[1] for d in definitions
                                 if d.startswith("LLAMINAR_TEST_NATIVE_PAYLOAD_KPART_THREADS=")))
                self.assertGreater(width, 24)
                if x86 == "ON":
                    for name in ("V2_Integration_CPUNativeQuantised_GEMV",
                                 "V2_Integration_CPU_NativeQuantised_PrefillHeuristic"):
                        props = {p["name"]: p["value"] for p in by_name[name]["properties"]}
                        self.assertEqual(props["PROCESSORS"], width)
                    for isa in cells.split(";"):
                        name = "V2_Integration_CPUX86CodedTileAdmission_" + isa
                        self.assertIn(name, (build / "preflight.txt").read_text().split(";"))
                        owners = []
                        for test in tests:
                            if not test["name"].startswith("V2_Integration_CPUX86") or not test["name"].endswith("_" + isa):
                                continue
                            expr = next(arg.split("=", 1)[1] for arg in test["command"]
                                        if arg.startswith("--gtest_filter="))
                            if selected("CPUX86CodedLeaves.MalformedOrUnpairedEntriesRejectBeforeAnyWrite", expr):
                                owners.append(test["name"])
                        self.assertEqual(owners, [name])
                        q6 = "CPUX86CodedLeaves.Q6KOrderedNaNsKeepEarlierBlockAndSerialBytes"
                        q6_owners = []
                        for test in tests:
                            if not test["name"].startswith("V2_Integration_CPUX86") or not test["name"].endswith("_" + isa):
                                continue
                            expr = next(arg.split("=", 1)[1] for arg in test["command"]
                                        if arg.startswith("--gtest_filter="))
                            if selected(q6, expr):
                                q6_owners.append(test["name"])
                        self.assertEqual(q6_owners, ["V2_Integration_CPUX86FusedOrderedNaN_" + isa])
                else:
                    self.assertFalse(any("CPUX86CodedTileAdmission" in n for n in by_name))

    def test_fused_contracts_have_one_owner_per_runtime_isa(self):
        """Compile both x86 envelopes and NEON registration without native SDKs."""
        folder = ROOT / "tests/v2/integration/kernels/cpu/native_quantised"
        sources = [folder / "Test__CPUNativeQuantisedFusedCompletion.cpp"]
        seen = set()
        names = set()
        while sources:
            path = sources.pop()
            if path in seen:
                continue
            seen.add(path)
            text = path.read_text()
            names.update(f"{suite}.{case}" for suite, case in re.findall(
                r"\bTEST\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)", text))
            sources.extend(folder / inc for inc in re.findall(r'#include "([^"/]+\.inc)"', text))
        self.assertGreater(len(names), 10)
        names.add("CPUNativeQuantisedFusedCompletion.FutureRegression")
        module = ROOT / "tests/v2/cmake/V2NativePayloadFusedTests.cmake"
        for cells, x86 in (("AVX2", "ON"), ("AVX2;AVX512", "ON"), ("NEON", "OFF")):
            with self.subTest(cells=cells), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp)
                source.joinpath("CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.24)
project(NativePayloadPartition NONE)
enable_testing()
add_executable(v2_integration_cpu_native_quantised_fused_completion IMPORTED)
set_property(TARGET v2_integration_cpu_native_quantised_fused_completion PROPERTY IMPORTED_LOCATION "''' + shutil.which("true") + '''")
function(v2_cpu_native_isa_runtime isa result)
  string(TOLOWER "${isa}" runtime)
  set(${result} "${runtime}" PARENT_SCOPE)
endfunction()
function(add_v2_test name)
  cmake_parse_arguments(T "NO_MPI;NO_MODELS" "MPI_PROCS;TIMEOUT" "COMMAND;LABELS;ENVIRONMENT" ${ARGN})
  add_test(NAME ${name} COMMAND ${T_COMMAND})
  set_tests_properties(${name} PROPERTIES LABELS "${T_LABELS}" ENVIRONMENT "${T_ENVIRONMENT}")
endfunction()
include("''' + module.as_posix() + '''")
''')
                build = source / "build"
                subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
                                "-DCMAKE_MAKE_PROGRAM=" + shutil.which("ninja"),
                                "-DV2_CPU_NATIVE_ISA_CELLS=" + cells, "-DV2_CPU_TARGET_X86=" + x86],
                               check=True, capture_output=True, text=True)
                tests = json.loads(subprocess.check_output(
                    ["ctest", "--test-dir", str(build), "--show-only=json-v1"], text=True))["tests"]
                for isa in cells.split(";"):
                    registrations = [t for t in tests if t["name"].endswith("_" + isa)]
                    self.assertGreater(len(registrations), 1)
                    for name in names:
                        owners = []
                        for test in registrations:
                            filters = [arg.split("=", 1)[1] for arg in test["command"]
                                       if arg.startswith("--gtest_filter=")]
                            self.assertEqual(len(filters), 1)
                            if selected(name, filters[0]):
                                owners.append(test["name"])
                        self.assertEqual(len(owners), 1, (isa, name, owners))


if __name__ == "__main__":
    unittest.main()
