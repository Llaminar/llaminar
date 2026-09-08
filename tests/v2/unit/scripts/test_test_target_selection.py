#!/usr/bin/env python3
"""Exercise the real CMake test-target selector without a model or GPU.

Release excludes Unit executables, including standalone host/device arithmetic
contracts that declare their own language level. Their property declarations
must be excluded with them, while admitted performance targets must retain
every requirement and CMake must still reject invalid features. These probes
configure tiny CPU-only projects; they never build or run an executable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[2] / "cmake/V2TestTargetSelection.cmake"


class TestTargetSelection(unittest.TestCase):
    """Verify selection and requirement propagation using CMake itself."""

    cmake = "cmake"
    ninja = "ninja"

    def configure(self, performance_only: bool, feature: str = "cxx_std_20"):
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
    if(V2_PERF_TESTS_ONLY AND NOT target MATCHES "^v2_perf_")
        if(TARGET ${{target}})
            message(FATAL_ERROR "Excluded target was materialized: ${{target}}")
        endif()
    else()
        if(NOT TARGET ${{target}})
            message(FATAL_ERROR "Admitted target is missing: ${{target}}")
        endif()
        foreach(property LINK_LIBRARIES INCLUDE_DIRECTORIES COMPILE_DEFINITIONS
                         COMPILE_OPTIONS COMPILE_FEATURES)
            get_target_property(value ${{target}} ${{property}})
            if(NOT value)
                message(FATAL_ERROR "Lost ${{property}} for ${{target}}")
            endif()
        endforeach()
        get_target_property(features ${{target}} COMPILE_FEATURES)
        if(NOT "{feature}" IN_LIST features)
            message(FATAL_ERROR "Lost language contract for ${{target}}")
        endif()
    endif()
endforeach()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--ninja", default="ninja")
    args, remaining = parser.parse_known_args()
    TestTargetSelection.cmake = args.cmake
    TestTargetSelection.ninja = args.ninja
    unittest.main(argv=[__file__, *remaining])
