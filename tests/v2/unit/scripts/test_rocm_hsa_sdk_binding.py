#!/usr/bin/env python3
"""Execute actual SDK HSA resolution without a compiler, SDK, DSO or accelerator.

Language-free CMake fixtures keep competing system/old-SDK paths present, reuse
one build tree, and verify the configured physical ABI1 path or precise failure.
"""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]


class ROCmHSASDKBinding(unittest.TestCase):
    """The selected SDK alone owns all HSA discovery, including warmed caches."""

    def test_selected_warmed_and_missing_sdk_never_search_old_library(self):
        """Real-path output moves with ROCM_PATH and rejects a missing new DSO."""
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.27)\nproject(HSA NONE)\n'
                f'include("{ROOT}/cmake/ROCmHSARuntime.cmake")\n'
                'llaminar_find_rocm_hsa_runtime(ROCM_HSA_RUNTIME_LIBRARY)\n'
                'file(WRITE "${CMAKE_BINARY_DIR}/selected.txt" "${ROCM_HSA_RUNTIME_LIBRARY}")\n')
            libraries = []
            for name in ("old SDK", "new SDK"):
                sdk = base / name
                (sdk / "lib").mkdir(parents=True)
                library = sdk / "lib/libhsa-runtime64.so.1.0"
                library.write_text("inert selected-library fixture")
                (sdk / "lib/libhsa-runtime64.so.1").symlink_to(library.name)
                libraries.append(library)
            build = base / "build"
            def configure(sdk):
                return subprocess.run(['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja',
                    '-DROCM_PATH=' + str(sdk), '-DCMAKE_INCLUDE_PATH=' + str(libraries[0].parent),
                    '-DROCM_HSA_RUNTIME_LIBRARY=' + str(libraries[0])],
                    capture_output=True, text=True, timeout=20)
            for library in libraries:
                result = configure(library.parent.parent)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual((build / 'selected.txt').read_text(), str(library.resolve()))
            libraries[1].unlink()
            result = configure(libraries[1].parent.parent)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('libhsa-runtime64.so.1', result.stdout + result.stderr)
            (libraries[1].parent / 'libhsa-runtime64.so.1').unlink()
            (libraries[1].parent / 'libhsa-runtime64.so.1').symlink_to(libraries[0])
            result = configure(libraries[1].parent.parent)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('escapes ROCM_PATH', result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
