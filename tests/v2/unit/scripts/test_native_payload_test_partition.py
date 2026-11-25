#!/usr/bin/env python3
"""Exercise native-payload CTest filter ownership through real CMake configuration.

The imported binary is inert: this device-free registration regression never
substitutes for native numerical tests. It checks that the focused registrations
and residual suite cover each declared fixture case exactly once per ISA, while
new cases remain covered by the residual suite. Production GTests still execute
through their original binaries and resource contracts.
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
