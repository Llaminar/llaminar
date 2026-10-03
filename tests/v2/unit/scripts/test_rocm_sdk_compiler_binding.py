#!/usr/bin/env python3
"""Device-free negative controls for the canonical HIP SDK compile boundary.

A compiler found under a staged SDK may still inherit /opt/rocm headers. Require
explicit matching header and bitcode roots on all HIP commands, including .cpp
translation units. This suite tests the validator, while production preflight
applies it to the certified build's complete compile database.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest

from test_accelerator_build_type_flags import verify_hip_sdk_command_binding


class ROCmSDKCompilerBindingTests(unittest.TestCase):
    """Prove stale/implicit compiler roots cannot certify a different SDK."""

    SDK = "/isolated/core-sdk"

    def entry(self, options: str, source: str = "kernel.hip") -> dict[str, str]:
        """Make a small real driver-command representation, not a source scan."""
        return {"file": source, "command": f"clang++ {options} -x hip -c {source}"}

    def test_matching_explicit_roots_pass(self):
        """The exact configured SDK owns both header and bitcode selection."""
        verify_hip_sdk_command_binding(
            [self.entry(f"--hip-path={self.SDK} --rocm-path={self.SDK}")], self.SDK)

    def test_implicit_default_roots_are_rejected(self):
        """Compiler location alone cannot prove which HIP headers it uses."""
        with self.assertRaisesRegex(AssertionError, "--hip-path"):
            verify_hip_sdk_command_binding([self.entry("")], self.SDK)

    def test_wrong_header_root_is_rejected(self):
        """New device bitcode cannot make old host/runtime declarations valid."""
        with self.assertRaisesRegex(AssertionError, "--hip-path"):
            verify_hip_sdk_command_binding(
                [self.entry(f"--hip-path=/opt/rocm --rocm-path={self.SDK}")], self.SDK)

    def test_wrong_bitcode_root_is_rejected(self):
        """Matching headers alone do not bind math/device libraries."""
        with self.assertRaisesRegex(AssertionError, "--rocm-path"):
            verify_hip_sdk_command_binding(
                [self.entry(f"--hip-path={self.SDK} --rocm-path=/opt/rocm")], self.SDK)

    def test_conflicting_duplicate_options_are_rejected(self):
        """A later matching flag must not hide a second competing SDK choice."""
        with self.assertRaisesRegex(AssertionError, "--hip-path"):
            verify_hip_sdk_command_binding(
                [self.entry(f"--hip-path=/old --hip-path={self.SDK} --rocm-path={self.SDK}")],
                self.SDK)

    def test_hip_compiled_cpp_files_are_checked(self):
        """Backend bridges are HIP translation units despite their suffix."""
        with self.assertRaisesRegex(AssertionError, "bridge.cpp"):
            verify_hip_sdk_command_binding([self.entry("", "bridge.cpp")], self.SDK)

    def test_non_hip_commands_do_not_supply_evidence(self):
        """CUDA/CPU commands cannot accidentally stand in for HIP coverage."""
        with self.assertRaisesRegex(AssertionError, "no HIP"):
            verify_hip_sdk_command_binding(
                [{"file": "host.cpp", "command": "g++ -c host.cpp"}], self.SDK)


class ROCmSDKHeaderDiscoveryTests(unittest.TestCase):
    """Execute CMake's real optional-component resolver without a compiler/GPU."""

    def test_absent_component_never_uses_another_sdk(self):
        """A stale cache and search prefix cannot donate another SDK's headers."""
        self._exercise(selected_header=False)

    def test_selected_component_overrides_stale_cache(self):
        """The current SDK wins even when an old include directory still exists."""
        self._exercise(selected_header=True)

    def _exercise(self, *, selected_header: bool):
        """Configure the production function with two conflicting fake SDKs."""
        module = Path(__file__).resolve().parents[4] / "cmake/ROCmSDKHeaders.cmake"
        with tempfile.TemporaryDirectory(prefix="llaminar-sdk-headers-") as temporary:
            root = Path(temporary)
            selected, stale = root / "selected", root / "stale"
            (stale / "include/ck").mkdir(parents=True)
            (stale / "include/ck/ck.hpp").write_text("// incompatible SDK\n")
            (selected / "include").mkdir(parents=True)
            if selected_header:
                (selected / "include/ck").mkdir()
                (selected / "include/ck/ck.hpp").write_text("// selected SDK\n")
            source = root / "source"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.24)\nproject(SDKHeaders NONE)\n"
                f'include("{module.as_posix()}")\n'
                "llaminar_find_rocm_sdk_header(CK_INCLUDE_DIR ck/ck.hpp)\n"
                'file(WRITE "${CMAKE_BINARY_DIR}/resolved.txt" "${CK_INCLUDE_DIR}")\n')
            build = root / "build"
            command = ["cmake", "-S", str(source), "-B", str(build),
                       f"-DROCM_PATH={selected}",
                       f"-DCK_INCLUDE_DIR:PATH={stale / 'include'}",
                       f"-DCMAKE_PREFIX_PATH={stale}"]
            result = subprocess.run(command, text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            resolved = (build / "resolved.txt").read_text()
            self.assertEqual(resolved, str(selected / "include") if selected_header
                             else "CK_INCLUDE_DIR-NOTFOUND")
            # Reuse the same cache with a different selected SDK. This catches
            # the original warmed-build failure, not just clean configuration.
            command[command.index(f"-DROCM_PATH={selected}")] = f"-DROCM_PATH={stale}"
            result = subprocess.run(command, text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((build / "resolved.txt").read_text(), str(stale / "include"))


if __name__ == "__main__":
    unittest.main()
