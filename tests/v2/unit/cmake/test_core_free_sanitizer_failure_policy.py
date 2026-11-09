#!/usr/bin/env python3
"""Prove real core-free CTest registrations make UBSan failure reporting fatal.

Only CMake/CTest metadata is configured: no runtime compiler, model or accelerator
is needed. A minimal directory registrar exercises the wrapper interface, not a
replacement native dependency or a claim that the complete engine gate ran.
Named CPU-team resources must follow the same registration/omission boundary.
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
MODULE = ROOT / "tests/v2/cmake/V2CoreFreeTestRegistration.cmake"
CMAKE, CTEST, NINJA = (shutil.which(name) for name in ("cmake", "ctest", "ninja"))


class CoreFreeSanitizerFailurePolicy(unittest.TestCase):
    """Inspect actual generated CTest properties in both supported directory contexts."""

    def registration(self, wrapped: bool, perf_only: bool = False) -> dict:
        """Configure a real registration or its directory-owned perf-only omission."""
        cmake, ctest, ninja = CMAKE, CTEST, NINJA
        self.assertTrue(cmake and ctest and ninja, "genuine CMake/CTest/Ninja are required, never skipped")
        registrar = """
function(add_v2_test name)
  if(V2_PERF_TESTS_ONLY AND NOT "${name}" MATCHES "^V2_Perf_")
    return()
  endif()
  cmake_parse_arguments(PARSE_ARGV 1 OWN "NO_MPI;NO_MODELS" "MPI_PROCS;TIMEOUT" "COMMAND;LABELS;ENVIRONMENT;RESOURCE_LOCK")
  if(NOT OWN_NO_MPI OR NOT OWN_NO_MODELS OR NOT OWN_MPI_PROCS EQUAL 1)
    message(FATAL_ERROR "core-free test escaped original device-free ownership")
  endif()
  add_test(NAME ${name} COMMAND ${OWN_COMMAND})
  set_tests_properties(${name} PROPERTIES LABELS "${OWN_LABELS}" TIMEOUT "${OWN_TIMEOUT}" ENVIRONMENT "${OWN_ENVIRONMENT}" RESOURCE_LOCK "${OWN_RESOURCE_LOCK}")
endfunction()
""" if wrapped else ""
        with tempfile.TemporaryDirectory(prefix="llaminar-core-free-sanitizer-policy-") as directory:
            source = Path(directory) / "source"
            build = Path(directory) / "build"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 4.0)\n"
                "project(CoreFreeSanitizerPolicy LANGUAGES NONE)\n"
                "enable_testing()\n" + ("set(V2_PERF_TESTS_ONLY TRUE)\n" if perf_only else "") + registrar +
                f'include("{MODULE.as_posix()}")\n'
                'v2_add_core_free_test(PolicyProbe COMMAND "${CMAKE_COMMAND}" -E true LABELS V2 Unit DeviceFree RESOURCE_LOCK OriginalCPUTeam)\n'
            )
            configured = subprocess.run(
                [cmake, "-S", str(source), "-B", str(build), "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja}"],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)
            shown = subprocess.run(
                [ctest, "--test-dir", str(build), "--show-only=json-v1"],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(shown.returncode, 0, shown.stdout + shown.stderr)
            tests = json.loads(shown.stdout)["tests"]
            if perf_only:
                self.assertTrue(wrapped, "only the full directory registrar owns perf-only omission")
                self.assertEqual(tests, [])
                return {}
            self.assertEqual(len(tests), 1)
            test = tests[0]
            self.assertEqual(test["name"], "PolicyProbe")
            self.assertEqual(test["command"][1:], ["-E", "true"])
            self.assertTrue(Path(test["command"][0]).is_file())
            return {prop["name"]: prop["value"] for prop in test["properties"]}

    def check(self, wrapped: bool) -> None:
        """Require fatal diagnostics without modifying arithmetic or source-owned timeout."""
        properties = self.registration(wrapped)
        self.assertEqual(properties["TIMEOUT"], 30.0)
        self.assertIn("DeviceFree", properties["LABELS"])
        self.assertIn("UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1", properties["ENVIRONMENT"])
        self.assertEqual(sum(value.startswith("UBSAN_OPTIONS=") for value in properties["ENVIRONMENT"]), 1)
        self.assertEqual(properties["RESOURCE_LOCK"], ["OriginalCPUTeam"])

    def test_standalone_registration_is_fatal(self) -> None:
        """The standalone portability registration must not recover a sanitizer error."""
        self.check(False)

    def test_wrapped_registration_is_fatal(self) -> None:
        """The complete-build directory interface receives the same common failure policy."""
        self.check(True)

    def test_wrapped_perf_only_omits_without_decorating_absent_test(self) -> None:
        """A legal directory-owned omission must not become a configure-time failure."""
        self.assertEqual(self.registration(True, True), {})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cmake", default=CMAKE)
    parser.add_argument("--ctest", default=CTEST)
    parser.add_argument("--ninja", default=NINJA)
    tools, remaining = parser.parse_known_args()
    CMAKE, CTEST, NINJA = tools.cmake, tools.ctest, tools.ninja
    unittest.main(argv=[sys.argv[0], *remaining])
