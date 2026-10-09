#!/usr/bin/env python3
"""Exercise the real CMake test-target selector without a model or GPU.

Release excludes Unit executables, including standalone host/device arithmetic
contracts that declare their own language level. Their property declarations
must be excluded with them, including startup device ownership, while admitted targets retain
every requirement and CMake must still reject invalid features. Native HTTP
affinity registrations follow the same selection and preserve their exact
OpenMP startup policy. These probes configure tiny CPU-only projects; they
never build or run an inference or test executable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[2] / "cmake/V2TestTargetSelection.cmake"
HTTP_REGISTRATION = MODULE.with_name("V2HttpServiceThreadAffinityTests.cmake")


class TestTargetSelection(unittest.TestCase):
    """Verify selection and requirement propagation using CMake itself."""

    cmake = "cmake"
    ninja = "ninja"

    def configure(self, performance_only: bool, feature: str = "cxx_std_20", *,
                  scope: str = "FullInventory", extra: str = ""):
        """Configure an isolated fixture with the selected production module."""
        with tempfile.TemporaryDirectory(prefix="llaminar-test-targets-") as tmp:
            root = Path(tmp)
            (root / "main.cpp").write_text("int main() { return 0; }\n")
            # Reuse the module, not copied wrappers or source-string assertions.
            # Include twice to exercise its guard against nested macro wrapping.
            (root / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.20)
project(TestTargetSelection LANGUAGES CXX)
set(V2_PERF_TESTS_ONLY {"ON" if performance_only else "OFF"})
include("{MODULE.as_posix()}")
include("{MODULE.as_posix()}")
add_library(requirements INTERFACE)
foreach(target v2_test_contract v2_integration_contract v2_perf_contract)
    add_executable(${{target}} main.cpp)
    target_link_libraries(${{target}} PRIVATE requirements)
    target_include_directories(${{target}} PRIVATE "${{CMAKE_CURRENT_SOURCE_DIR}}")
    target_compile_definitions(${{target}} PRIVATE CONTRACT_CHECK=1)
    target_compile_options(${{target}} PRIVATE -Wall)
    target_compile_features(${{target}} PRIVATE {feature})
    v2_test_device_scope(${{target}} {scope})
    if(V2_PERF_TESTS_ONLY AND NOT target MATCHES "^v2_perf_")
        if(TARGET ${{target}})
            message(FATAL_ERROR "Excluded target was materialized: ${{target}}")
        endif()
    else()
        if(NOT TARGET ${{target}})
            message(FATAL_ERROR "Admitted target is missing: ${{target}}")
        endif()
        foreach(property LINK_LIBRARIES INCLUDE_DIRECTORIES COMPILE_DEFINITIONS
                         COMPILE_OPTIONS COMPILE_FEATURES V2_TEST_DEVICE_SCOPE)
            get_target_property(value ${{target}} ${{property}})
            if(NOT value)
                message(FATAL_ERROR "Lost ${{property}} for ${{target}}")
            endif()
        endforeach()
        get_target_property(features ${{target}} COMPILE_FEATURES)
        get_target_property(device_scope ${{target}} V2_TEST_DEVICE_SCOPE)
        if(NOT device_scope STREQUAL "{scope}")
            message(FATAL_ERROR "Lost startup device ownership for ${{target}}")
        endif()
        if(NOT "{feature}" IN_LIST features)
            message(FATAL_ERROR "Lost language contract for ${{target}}")
        endif()
    endif()
endforeach()
{extra}
''', encoding="utf-8")
            return subprocess.run(
                [self.cmake, "-S", str(root), "-B", str(root / "build"),
                 "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={self.ninja}"],
                capture_output=True, text=True, timeout=15, check=False,
            )

    def test_creation_and_all_requirements_follow_selection(self):
        """Both inventories configure and retain every admitted requirement."""
        for performance_only in (False, True):
            with self.subTest(performance_only=performance_only):
                result = self.configure(performance_only)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_admitted_targets_still_reject_unknown_features(self):
        """The Release wrapper must not suppress validation on a live target."""
        result = self.configure(True, "cxx_not_a_real_compile_feature")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cxx_not_a_real_compile_feature", result.stderr)
        self.assertIn("v2_perf_contract", result.stderr)

    def test_unknown_scope_is_not_hidden_by_release_exclusion(self):
        """Invalid startup ownership fails even on a skipped declaration."""
        result = self.configure(True, scope="InventedScope")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown V2 test device scope", result.stderr)

    def test_missing_live_target_is_not_silently_ignored(self):
        """Only intentional Release exclusions may have no target owner."""
        for performance_only, target in ((False, "v2_integration_missing"),
                                         (True, "v2_perf_missing")):
            with self.subTest(performance_only=performance_only):
                result = self.configure(performance_only,
                    extra=f"v2_test_device_scope({target} FullInventory)")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(target, result.stderr)

    def configure_http_registration(self, performance_only: bool, *, declare_target: bool = True):
        """Generate actual production registrations without a compiler or device.

        The imported executable names CMake itself only to make CTest's
        inventory resolver independent of an unbuilt binary. It is never run.
        Production target selection still owns whether that target exists.
        """
        with tempfile.TemporaryDirectory(prefix="llaminar-http-registration-") as tmp:
            root = Path(tmp)
            target = """
add_executable(v2_test_server_mode IMPORTED)
if(TARGET v2_test_server_mode)
    set_target_properties(v2_test_server_mode PROPERTIES IMPORTED_LOCATION "${CMAKE_COMMAND}")
endif()
""" if declare_target else ""
            (root / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.20)
project(HttpServiceRegistration LANGUAGES NONE)
enable_testing()
set(V2_PERF_TESTS_ONLY {"ON" if performance_only else "OFF"})
include("{MODULE.as_posix()}")
{target}
include("{HTTP_REGISTRATION.as_posix()}")
include("{HTTP_REGISTRATION.as_posix()}")
""", encoding="utf-8")
            configured = subprocess.run(
                [self.cmake, "-S", str(root), "-B", str(root / "build"),
                 "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={self.ninja}"],
                capture_output=True, text=True, timeout=15, check=False)
            if configured.returncode:
                return configured, None
            inventory = subprocess.run(
                [str(Path(self.cmake).with_name("ctest")), "--test-dir", str(root / "build"),
                 "--show-only=json-v1"],
                capture_output=True, text=True, timeout=15, check=True)
            return configured, json.loads(inventory.stdout)["tests"]

    def test_http_service_registration_follows_target_selection(self):
        """Both native policies survive Integration and leave no Release target reference."""
        for performance_only in (False, True):
            with self.subTest(performance_only=performance_only):
                configured, tests = self.configure_http_registration(performance_only)
                self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)
                expected = {} if performance_only else {
                    "V2_Integration_HTTPServiceThreadAffinity":
                        ["OMP_PROC_BIND=close", "OMP_PLACES=cores", "OMP_NUM_THREADS=2"],
                    "V2_Integration_HTTPServiceThreadAffinityUnbound":
                        ["OMP_PROC_BIND=false", "OMP_NUM_THREADS=2"],
                }
                self.assertEqual({test["name"] for test in tests}, set(expected))
                self.assertEqual(len(tests), len(expected))
                for test in tests:
                    properties = {value["name"]: value["value"] for value in test["properties"]}
                    self.assertEqual(set(properties["LABELS"]), {
                        "V2", "Integration", "ProductionTestPreflight", "HTTP",
                        "Observability", "Threading", "Regression", "DeviceFree"})
                    self.assertEqual(properties["TIMEOUT"], 30)
                    command = test["command"]
                    self.assertEqual(Path(command[0]).name, "cmake")
                    self.assertEqual(command[1:-2], ["-E", "env", *expected[test["name"]]])
                    self.assertEqual(command[-2], command[0])
                    self.assertEqual(command[-1], "--gtest_filter=HttpServiceThreadAffinity.*")

    def test_http_service_registration_rejects_missing_admitted_target(self):
        """An absent Integration executable remains a fatal configure error."""
        configured, tests = self.configure_http_registration(False, declare_target=False)
        self.assertNotEqual(configured.returncode, 0)
        self.assertIsNone(tests)
        self.assertIn('No target "v2_test_server_mode"', configured.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--ninja", default="ninja")
    args, remaining = parser.parse_known_args()
    TestTargetSelection.cmake = args.cmake
    TestTargetSelection.ninja = args.ninja
    unittest.main(argv=[__file__, *remaining])
