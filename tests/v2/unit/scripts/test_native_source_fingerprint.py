#!/usr/bin/env python3
"""Replay native source dependency aliases through real CMake and Ninja.

The fixture has no compiler or GPU language. It exercises generated build-file
ownership, source hash freshness and the original duplicate-output failure.
"""
from pathlib import Path
import hashlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
MODULE = ROOT / "cmake/NativeSourceFingerprint.cmake"


class NativeSourceFingerprint(unittest.TestCase):
    """Different caller directories must watch one normalized source identity."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native source fingerprint ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.build = self.root / "build"
        self.source.mkdir()
        (self.source / "child").mkdir()
        self.watched = self.source / "input.cpp"
        self.watched.write_text("// original native creator\n")
        self.cmake = shutil.which("cmake")
        self.ninja = shutil.which("ninja")
        self.assertIsNotNone(self.cmake, "Canonical test image must provide CMake")
        self.assertIsNotNone(self.ninja, "Canonical test image must provide Ninja")

    def prepare(self, original=False):
        """Load production helper, or the recorded original registration control."""
        module = MODULE
        if original:
            module = self.root / "original.cmake"
            module.write_text('''function(llaminar_native_source_fingerprint path output)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${path}")
    file(SHA256 "${path}" digest)
    set(${output} "${digest}" PARENT_SCOPE)
endfunction()
''')
        (self.source / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.24)
project(NativeSourceDependencyReplay NONE)
include([==[%s]==])
llaminar_native_source_fingerprint("${CMAKE_CURRENT_SOURCE_DIR}/input.cpp" digest)
file(WRITE "${CMAKE_BINARY_DIR}/core.sha" "${digest}")
add_subdirectory(child)
''' % module.as_posix())
        (self.source / "child/CMakeLists.txt").write_text('''llaminar_native_source_fingerprint("${CMAKE_CURRENT_SOURCE_DIR}/../input.cpp" digest)
file(WRITE "${CMAKE_BINARY_DIR}/child.sha" "${digest}")
''')

    def run_command(self, command):
        """Run build-graph operations only; no compiler or inference is invoked."""
        return subprocess.run(command, capture_output=True, text=True, timeout=10)

    def configure(self):
        return self.run_command([self.cmake, "-S", str(self.source), "-B", str(self.build),
                                 "-G", "Ninja", "-DCMAKE_MAKE_PROGRAM=" + self.ninja])

    def build_graph(self):
        return self.run_command([self.cmake, "--build", str(self.build), "--parallel",
                                 "--target", "help"])

    def require_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_actual_ninja_accepts_aliases_and_reconfigures_after_source_change(self):
        self.prepare()
        self.require_success(self.configure())
        self.require_success(self.build_graph())
        before = (self.build / "core.sha").read_text()
        self.assertEqual(before, hashlib.sha256(self.watched.read_bytes()).hexdigest())
        self.assertEqual(before, (self.build / "child.sha").read_text())
        self.watched.write_text("// changed native creator; keep automatic dependency\n")
        self.require_success(self.build_graph())
        after = hashlib.sha256(self.watched.read_bytes()).hexdigest()
        self.assertNotEqual(before, after)
        self.assertEqual((self.build / "core.sha").read_text(), after)
        self.assertEqual((self.build / "child.sha").read_text(), after)

    def test_missing_creator_source_rejects_instead_of_fingerprinting_another_file(self):
        self.prepare()
        self.watched.unlink()
        result = self.configure()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("input.cpp", result.stdout + result.stderr)

    def test_original_registration_reproduces_duplicate_ninja_output(self):
        self.prepare(original=True)
        self.require_success(self.configure())
        result = self.build_graph()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("defined as an output multiple times", result.stdout + result.stderr)

    def prepare_event_registration(self, original=False):
        """Replay the actual event module's dependency declarations in both scopes.

        Only the prefix before extraction/target creation is relevant: it is
        evaluated verbatim by CMake, with real files and the production source
        fingerprint helper. No compiler, fake executable or GPU is involved.
        """
        module = ROOT / "tests/v2/cmake/V2NativeEventDestructionTests.cmake"
        prefix, marker, _ = module.read_text().partition("execute_process(")
        self.assertTrue(marker, "Actual event extraction boundary is required")
        if original:
            fixed = 'get_filename_component(_event_replay_source "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)'
            self.assertEqual(prefix.count(fixed), 1)
            prefix = prefix.replace(fixed, 'set(_event_replay_source "${CMAKE_CURRENT_SOURCE_DIR}/../..")')
        event_dir = self.source / "tests/v2"
        (event_dir / "unit/scripts").mkdir(parents=True)
        (event_dir / "unit/scripts/generate_native_event_destruction_replay.py").write_text("# Dependency only; never executed.\n")
        for name in ("cuda/NvidiaDeviceContext.cu", "cuda/CUDABackend.cu",
                     "rocm/AMDDeviceContext.cpp", "rocm/ROCmBackend.cpp"):
            path = self.source / "src/v2/backends" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("// Native creator dependency replay\n")
        self.watched = self.source / "src/v2/backends/cuda/CUDABackend.cu"
        (self.source / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.24)
project(ActualEventDependencyReplay NONE)
include([==[%s]==])
llaminar_native_source_fingerprint("${CMAKE_CURRENT_SOURCE_DIR}/src/v2/backends/cuda/CUDABackend.cu" digest)
file(WRITE "${CMAKE_BINARY_DIR}/core.sha" "${digest}")
add_subdirectory(tests/v2)
''' % MODULE.as_posix())
        (event_dir / "CMakeLists.txt").write_text(prefix + "\nendif()\n")

    def test_actual_event_registration_has_one_dependency_identity_and_stays_fresh(self):
        """The original Integration registration coexists with the core fingerprint."""
        self.prepare_event_registration()
        self.require_success(self.configure())
        self.require_success(self.build_graph())
        self.watched.write_text("// Changed actual event cleanup dependency\n")
        self.require_success(self.build_graph())
        self.assertEqual((self.build / "core.sha").read_text(),
                         hashlib.sha256(self.watched.read_bytes()).hexdigest())

    def test_original_event_registration_reproduces_duplicate_ninja_output(self):
        """The pre-fix module body fails the real Ninja parser, without GPU setup."""
        self.prepare_event_registration(original=True)
        self.require_success(self.configure())
        result = self.build_graph()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("defined as an output multiple times", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
