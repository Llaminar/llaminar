#!/usr/bin/env python3
"""Configure the actual codegen target-manifest module with two enabled languages.

CMake evaluates file(GENERATE) once per enabled language. A CUDA build with an
option guarded by $<COMPILE_LANGUAGE:CUDA> therefore produced different content
for one manifest path and failed to configure on GB10. CUDA is not needed to
reproduce this: C and C++ exercise the same per-language evaluation. The test
configures the real module, requires the C++ view only, and keeps an
original-code control that must still fail.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[2] / "cmake" / "V2CPUCodegenTargetManifest.cmake"

PROJECT = """cmake_minimum_required(VERSION 3.20)
project(manifest_probe C CXX)
add_executable(probe probe.cpp)
target_compile_options(probe PRIVATE $<$<COMPILE_LANGUAGE:C>:-DONLY_C_VIEW> -DCXX_VIEW)
target_compile_definitions(probe PRIVATE $<$<COMPILE_LANGUAGE:C>:C_DEFINITION> CXX_DEFINITION)
{body}
"""

FIXED = 'include("{module}")\nv2_write_cpu_codegen_target_manifest(probe "${{CMAKE_BINARY_DIR}}/target.txt" "-O2")'
ORIGINAL = ('file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/target.txt" CONTENT '
            '"options=$<TARGET_GENEX_EVAL:probe,$<TARGET_PROPERTY:probe,COMPILE_OPTIONS>>\\n")')


@unittest.skipUnless(shutil.which("cmake"), "cmake is required")
class CPUCodegenTargetManifestTest(unittest.TestCase):
    """The manifest must configure with several languages and record only C++."""

    def configure(self, body: str) -> tuple[subprocess.CompletedProcess[str], Path]:
        """Configure one isolated project and return the result and build directory."""
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (root / "probe.cpp").write_text("int main() { return 0; }\n")
        (root / "CMakeLists.txt").write_text(PROJECT.format(body=body))
        build = root / "build"
        result = subprocess.run(["cmake", "-S", str(root), "-B", str(build)],
                                capture_output=True, text=True, check=False)
        return result, build

    def test_multi_language_configure_records_only_the_cxx_view(self) -> None:
        """Two language views of one target generate exactly one C++ manifest."""
        result, build = self.configure(FIXED.format(module=MODULE.as_posix()))
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = (build / "target.txt").read_text()
        self.assertIn("options=-DCXX_VIEW\n", manifest)
        self.assertIn("definitions=CXX_DEFINITION\n", manifest)
        self.assertIn("flags=-O2\n", manifest)
        self.assertNotIn("ONLY_C_VIEW", manifest)
        self.assertNotIn("C_DEFINITION;", manifest)

    def test_original_unconditional_generation_still_fails(self) -> None:
        """Control: the former unconditional generation is a configure error."""
        result, _ = self.configure(ORIGINAL)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("written multiple times with different content", " ".join(result.stderr.split()))


if __name__ == "__main__":
    unittest.main()
