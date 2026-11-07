#!/usr/bin/env python3
"""Exercise the production oneDNN phase owner without a compiler, SDK or model.

A LANGUAGES NONE/custom-command fixture produces one named artifact and an
explicitly labelled lifecycle-only install file. It implements no oneDNN API and
is never used by an inference build. Real CMake/Ninja operations prove failure
stream propagation, retained configured work, install publication and identity
invalidation; no production implementation is copied into the test.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
CMAKE = "cmake"
NINJA = "ninja"
GIT = "git"
MODULE = ROOT / "cmake/OneDNNBuild.cmake"


class OneDNNBuildLifecycleTest(unittest.TestCase):
    """Adversarial configured/installed transitions through the actual phase API."""

    def setUp(self) -> None:
        """Create an isolated custom-command producer, never a fake toolchain."""
        self.temp = tempfile.TemporaryDirectory(prefix="llaminar-owned-dependency-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "producer"
        self.source.mkdir()
        self.build = self.root / "owned-build"
        (self.source / "lifecycle-only.txt").write_text(
            "Lifecycle fixture only. No oneDNN API, SDK or executable implementation.\n")
        (self.source / "gate.cmake").write_text('''
if(NOT EXISTS "${ALLOW_FILE}")
    message(FATAL_ERROR "FIXTURE_BUILD_STDERR")
endif()
''')
        (self.source / "CMakeLists.txt").write_text('''
cmake_minimum_required(VERSION 3.18)
project(OwnedPhaseFixture LANGUAGES NONE)
if(FIXTURE_CONFIGURE_FAIL)
    message(STATUS "FIXTURE_CONFIG_STDOUT")
    message(FATAL_ERROR "FIXTURE_CONFIG_STDERR")
endif()
add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/generated.txt"
    COMMAND "${CMAKE_COMMAND}" -E touch "${CMAKE_BINARY_DIR}/generated.txt")
add_custom_target(producer ALL DEPENDS "${CMAKE_BINARY_DIR}/generated.txt")
add_custom_target(gate ALL
    COMMAND "${CMAKE_COMMAND}" -E echo FIXTURE_BUILD_STDOUT
    COMMAND "${CMAKE_COMMAND}" "-DALLOW_FILE=${CMAKE_BINARY_DIR}/allow"
        -P "${CMAKE_SOURCE_DIR}/gate.cmake")
add_dependencies(gate producer)
install(FILES lifecycle-only.txt DESTINATION include/oneapi/dnnl RENAME dnnl.hpp)
install(FILES lifecycle-only.txt DESTINATION include/oneapi/dnnl RENAME dnnl_types.h)
install(FILES lifecycle-only.txt DESTINATION include RENAME dnnl.hpp)
# These files are explicitly labelled lifecycle data, never linked/loaded or
# presented as a working SDK. Their protocol names exercise manifest closure.
install(FILES lifecycle-only.txt DESTINATION lib RENAME libdnnl.so)
''')
        self.runner = self.root / "run.cmake"
        self.runner.write_text('''
cmake_minimum_required(VERSION 3.18)
set(CMAKE_GENERATOR Ninja)
include("${PHASE_MODULE}")
_llaminar_install_onednn("${PRODUCER_SOURCE}" "${OWNED_BUILD}" "${IDENTITY}" consumer_library
    "-DFIXTURE_CONFIGURE_FAIL=${CONFIGURE_FAIL}")
file(WRITE "${OWNED_BUILD}/consumer-library.txt" "${consumer_library}")
''')

    def invoke(self, identity: str = "lifecycle-fixture-v1", configure_fail: bool = False) -> subprocess.CompletedProcess[str]:
        """Run the real private phase owner with opaque fixture identity/arguments."""
        return subprocess.run([
            CMAKE, f"-DCMAKE_MAKE_PROGRAM={NINJA}", f"-DPHASE_MODULE={MODULE}",
            f"-DPRODUCER_SOURCE={self.source}", f"-DOWNED_BUILD={self.build}",
            f"-DIDENTITY={identity}", f"-DCONFIGURE_FAIL={'ON' if configure_fail else 'OFF'}",
            "-P", str(self.runner)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def complete_install(self) -> tuple[str, int]:
        """Observe a failed producer, then resume it without destroying completed work."""
        failed = self.invoke()
        self.assertNotEqual(failed.returncode, 0, failed.stdout)
        self.assertIn("FIXTURE_BUILD_STDOUT", failed.stdout)
        self.assertIn("FIXTURE_BUILD_STDERR", failed.stdout)
        configured = self.build / ".llaminar-onednn-configured"
        installed = self.build / ".llaminar-onednn-commit"
        self.assertTrue(configured.is_file())
        self.assertFalse(installed.exists(), "failed producer published an installation")
        artifact = self.build / "generated.txt"
        before = artifact.stat().st_mtime_ns
        identity = configured.read_text()
        (self.build / "allow").touch()
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertEqual(artifact.stat().st_mtime_ns, before, "configured work was deleted/rebuilt")
        self.assertEqual(installed.read_text(), identity)
        return identity, before

    def probe_origin(self, expected: str) -> subprocess.CompletedProcess[str]:
        """Run real Git authentication through the production read-only boundary."""
        probe = self.root / "origin.cmake"
        probe.write_text('''
cmake_minimum_required(VERSION 3.18)
include("${PHASE_MODULE}")
llaminar_require_onednn_origin("${GIT_PROGRAM}" "${PROBE_SOURCE}" "${EXPECTED_ORIGIN}")
''')
        return subprocess.run([
            CMAKE, f"-DPHASE_MODULE={MODULE}", f"-DGIT_PROGRAM={GIT}",
            f"-DPROBE_SOURCE={self.source}", f"-DEXPECTED_ORIGIN={expected}",
            "-P", str(probe)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def initialize_origin(self) -> str:
        """Create a genuine local Git fixture without network or global configuration."""
        origin = "https://example.invalid/lifecycle-fixture.git"
        subprocess.run([GIT, "init", "--quiet", str(self.source)], check=True)
        subprocess.run([GIT, "-C", str(self.source), "remote", "add", "origin", origin], check=True)
        return origin

    def test_origin_authentication_accepts_the_real_local_git_observation(self) -> None:
        """A matching existing origin is accepted without changing its config."""
        origin = self.initialize_origin()
        config = self.source / ".git/config"
        before = config.read_bytes()
        result = self.probe_origin(origin)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(config.read_bytes(), before)

    def test_origin_mismatch_preserves_existing_source_and_configuration(self) -> None:
        """A wrong origin rejects rather than recursively deleting user source/builds."""
        self.initialize_origin()
        config = self.source / ".git/config"
        before = config.read_bytes()
        result = self.probe_origin("https://example.invalid/a-different-source.git")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("oneDNN origin mismatch", result.stdout)
        self.assertEqual(config.read_bytes(), before)
        self.assertTrue((self.source / "CMakeLists.txt").exists())

    def test_failed_git_probe_is_not_a_destructive_origin_mismatch(self) -> None:
        """A real failing Git command exposes its diagnostic and retains source."""
        # Deliberately no Git repository. This is an ordinary input error,
        # not an executable pretending to be Git or a compiler SDK.
        result = self.probe_origin("https://example.invalid/lifecycle-fixture.git")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("oneDNN origin authentication failed", result.stdout)
        self.assertIn("stderr:", result.stdout)
        self.assertIn("not a git repository", result.stdout)
        self.assertTrue((self.source / "CMakeLists.txt").exists())

    def test_failed_build_resumes_and_complete_install_reuses(self) -> None:
        """A failed build is resumable, while a complete receipt performs no new build."""
        _, before = self.complete_install()
        (self.build / "allow").unlink()
        reused = self.invoke()
        self.assertEqual(reused.returncode, 0, reused.stdout)
        self.assertIn("Reusing authenticated", reused.stdout)
        self.assertEqual((self.build / "generated.txt").stat().st_mtime_ns, before)

    def test_identity_change_invalidates_install_and_configured_artifacts(self) -> None:
        """A different source/compiler/flag identity cannot reuse an old completed tree."""
        old_identity, _ = self.complete_install()
        changed = self.invoke(identity="lifecycle-fixture-v2")
        self.assertNotEqual(changed.returncode, 0, changed.stdout)
        self.assertFalse((self.build / "allow").exists())
        self.assertFalse((self.build / ".llaminar-onednn-commit").exists())
        self.assertNotEqual((self.build / ".llaminar-onednn-configured").read_text(), old_identity)
        self.assertIn("Invalidating unproven/different", changed.stdout)

    def test_failed_configure_exposes_both_streams_without_publication(self) -> None:
        """Failed configuration names both output streams and publishes neither state."""
        failed = self.invoke(configure_fail=True)
        self.assertNotEqual(failed.returncode, 0, failed.stdout)
        self.assertIn("FIXTURE_CONFIG_STDOUT", failed.stdout)
        self.assertIn("FIXTURE_CONFIG_STDERR", failed.stdout)
        self.assertFalse((self.build / ".llaminar-onednn-configured").exists())
        self.assertFalse((self.build / ".llaminar-onednn-commit").exists())

    def test_missing_artifact_retires_success_before_failed_producer(self) -> None:
        """Missing required output cannot retain an install receipt through a failed repair."""
        self.complete_install()
        (self.build / "include/oneapi/dnnl/dnnl.hpp").unlink()
        (self.build / "allow").unlink()
        failed = self.invoke()
        self.assertNotEqual(failed.returncode, 0, failed.stdout)
        self.assertFalse((self.build / ".llaminar-onednn-commit").exists())
        self.assertTrue((self.build / ".llaminar-onednn-configured").exists())

    def test_missing_library_retires_receipt_and_reinstalls_the_owned_closure(self) -> None:
        """A surviving public header cannot certify a deleted installed library."""
        self.complete_install()
        library = self.build / "lib/libdnnl.so"
        library.unlink()
        self.assertTrue((self.build / "include/oneapi/dnnl/dnnl.hpp").exists())
        (self.build / "allow").unlink()
        failed = self.invoke()
        self.assertNotEqual(failed.returncode, 0, failed.stdout)
        self.assertFalse((self.build / ".llaminar-onednn-commit").exists())
        self.assertFalse((self.build / ".llaminar-onednn-installed-artifacts").exists())
        # The invalid install surfaces retire together, not just their receipts;
        # configured compilation artifacts remain available for the next install.
        self.assertFalse((self.build / "include/oneapi/dnnl/dnnl.hpp").exists())
        self.assertTrue((self.build / "generated.txt").exists())
        (self.build / "allow").touch()
        repaired = self.invoke()
        self.assertEqual(repaired.returncode, 0, repaired.stdout)
        self.assertTrue(library.is_file())

    def test_missing_other_required_header_cannot_reuse_installation(self) -> None:
        """Every installed header belongs to closure, not only the name checked by find_library."""
        self.complete_install()
        (self.build / "include/oneapi/dnnl/dnnl_types.h").unlink()
        (self.build / "allow").unlink()
        failed = self.invoke()
        self.assertNotEqual(failed.returncode, 0, failed.stdout)
        self.assertFalse((self.build / ".llaminar-onednn-commit").exists())
        self.assertIn("missing installed artifact", failed.stdout)

    def test_modified_artifact_is_not_an_authenticated_reuse(self) -> None:
        """Content changes are detected even while the manifest and all names survive."""
        self.complete_install()
        header = self.build / "include/oneapi/dnnl/dnnl_types.h"
        header.write_text("modified lifecycle artifact\n")
        (self.build / "allow").unlink()
        failed = self.invoke()
        self.assertNotEqual(failed.returncode, 0, failed.stdout)
        self.assertFalse((self.build / ".llaminar-onednn-commit").exists())

    def test_successful_repair_restores_same_size_same_mtime_modified_header(self) -> None:
        """A timestamp-based up-to-date install must not re-certify modified bytes."""
        self.complete_install()
        header = self.build / "include/oneapi/dnnl/dnnl_types.h"
        original = header.read_bytes()
        metadata = header.stat()
        compiled = (self.build / "generated.txt").stat().st_mtime_ns
        changed = bytearray(original)
        changed[0] ^= 1
        header.write_bytes(changed)
        os.utime(header, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        repaired = self.invoke()
        self.assertEqual(repaired.returncode, 0, repaired.stdout)
        self.assertEqual(header.read_bytes(), original)
        self.assertEqual((self.build / "generated.txt").stat().st_mtime_ns, compiled)

    def test_consumer_library_is_manifested_not_an_unlisted_priority_shadow(self) -> None:
        """An unlisted lib/ file cannot replace the authenticated lib64/ consumer."""
        source = self.source / "CMakeLists.txt"
        source.write_text(source.read_text().replace(
            "DESTINATION lib RENAME libdnnl.so", "DESTINATION lib64 RENAME libdnnl.so"))
        self.complete_install()
        shadow = self.build / "lib/libdnnl.so"
        shadow.parent.mkdir()
        shadow.write_text("unlisted lifecycle-only shadow, never linked or loaded")
        reused = self.invoke()
        self.assertEqual(reused.returncode, 0, reused.stdout)
        selected = Path((self.build / "consumer-library.txt").read_text())
        self.assertEqual(selected, self.build / "lib64/libdnnl.so")
        self.assertNotEqual(selected, shadow)

    def test_rejected_surface_stays_rejected_across_two_invocations(self) -> None:
        """A removed receipt must not allow the next attempt to install outside its prefix."""
        self.complete_install()
        outside = self.root / "outside-owned-prefix"
        outside.mkdir()
        include = self.build / "include"
        shutil.rmtree(include)
        include.symlink_to(outside, target_is_directory=True)
        diagnostics = []
        for _ in range(2):
            failed = self.invoke()
            self.assertNotEqual(failed.returncode, 0, failed.stdout)
            # Observe the outside tree before checking new marker/diagnostic
            # structure: the old implementation must reach its unsafe retry.
            self.assertEqual(list(outside.rglob("*")), [], "retry wrote through the rejected symlink")
            diagnostics.append(failed.stdout)
        for diagnostic in diagnostics:
            self.assertIn("not an owned directory", diagnostic)
        self.assertTrue((self.build / ".llaminar-onednn-retirement-needed").exists())
        self.assertFalse((self.build / ".llaminar-onednn-commit").exists())
        include.unlink()
        repaired = self.invoke()
        self.assertEqual(repaired.returncode, 0, repaired.stdout)
        self.assertEqual(list(outside.rglob("*")), [])
        self.assertFalse((self.build / ".llaminar-onednn-retirement-needed").exists())

    def test_interrupted_partial_retirement_resumes_before_recertifying(self) -> None:
        """Durable intent survives a crash after receipts and only some surfaces retire."""
        identity, compiled = self.complete_install()
        marker = self.build / ".llaminar-onednn-retirement-needed"
        marker.write_text(identity)
        (self.build / ".llaminar-onednn-commit").unlink()
        (self.build / ".llaminar-onednn-installed-artifacts").unlink()
        shutil.rmtree(self.build / "lib")
        header = self.build / "include/oneapi/dnnl/dnnl_types.h"
        original, metadata = header.read_bytes(), header.stat()
        changed = bytearray(original)
        changed[0] ^= 1
        header.write_bytes(changed)
        os.utime(header, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        resumed = self.invoke()
        self.assertEqual(resumed.returncode, 0, resumed.stdout)
        self.assertEqual(header.read_bytes(), original)
        self.assertFalse(marker.exists())
        self.assertEqual((self.build / "generated.txt").stat().st_mtime_ns, compiled)

    def test_nested_directory_link_rejects_before_unpublished_retry_writes(self) -> None:
        """A benign compile retry cannot install through a nested directory alias."""
        first = self.invoke()
        self.assertNotEqual(first.returncode, 0, first.stdout)
        self.assertFalse((self.build / ".llaminar-onednn-commit").exists())
        self.assertFalse((self.build / ".llaminar-onednn-retirement-needed").exists())
        outside = self.root / "outside-nested-prefix"
        outside.mkdir()
        include = self.build / "include"
        include.mkdir()
        (include / "oneapi").symlink_to(outside, target_is_directory=True)
        (self.build / "allow").touch()
        for _ in range(2):
            retry = self.invoke()
            self.assertNotEqual(retry.returncode, 0, retry.stdout)
            self.assertEqual(list(outside.rglob("*")), [], "producer followed nested directory link")
        self.assertTrue((self.build / ".llaminar-onednn-retirement-needed").exists())

    def test_owned_versioned_library_file_links_remain_legal(self) -> None:
        """The ancestry guard permits ordinary existing in-prefix library-file symlinks."""
        library = self.build / "lib"
        library.mkdir(parents=True)
        real = library / "libdnnl.so.3"
        real.write_text("lifecycle-only data; not a shared library implementation")
        (library / "libdnnl.so").symlink_to(real.name)
        script = self.root / "safe-links.cmake"
        script.write_text('''
cmake_minimum_required(VERSION 3.18)
include("${PHASE_MODULE}")
_llaminar_check_onednn_surfaces("${OWNED_BUILD}" "explicit-fixture-identity")
''')
        result = subprocess.run([CMAKE, f"-DPHASE_MODULE={MODULE}",
                                 f"-DOWNED_BUILD={self.build}", "-P", str(script)],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertTrue((library / "libdnnl.so").is_symlink())
        self.assertFalse((self.build / ".llaminar-onednn-retirement-needed").exists())

    def test_compiler_arg1_is_forwarded_and_changes_identity(self) -> None:
        """Serialize synthetic command metadata through the actual production builder; no SDK invoked."""
        script = self.root / "arguments.cmake"
        script.write_text('''
cmake_minimum_required(VERSION 3.18)
include("${PHASE_MODULE}")
set(CMAKE_C_COMPILER_ARG1 "-DREQUIRED_C=1")
set(CMAKE_CXX_COMPILER_ARG1 "--gcc-toolchain=/fixture/sdk-a")
_llaminar_onednn_toolchain_arguments(arguments identity_a)
if(NOT "-DCMAKE_C_COMPILER_ARG1=-DREQUIRED_C=1" IN_LIST arguments OR
   NOT "-DCMAKE_CXX_COMPILER_ARG1=--gcc-toolchain=/fixture/sdk-a" IN_LIST arguments)
    message(FATAL_ERROR "mandatory compiler arguments were not forwarded: ${arguments}")
endif()
set(CMAKE_CXX_COMPILER_ARG1 "--gcc-toolchain=/fixture/sdk-b")
_llaminar_onednn_toolchain_arguments(changed identity_b)
if(identity_a STREQUAL identity_b OR NOT identity_b MATCHES "sdk-b")
    message(FATAL_ERROR "changed mandatory compiler argument did not change identity")
endif()
''')
        result = subprocess.run([CMAKE, f"-DPHASE_MODULE={MODULE}", "-P", str(script)],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", default=CMAKE)
    parser.add_argument("--ninja", default=NINJA)
    parser.add_argument("--git", default=GIT)
    options, remaining = parser.parse_known_args()
    CMAKE, NINJA, GIT = options.cmake, options.ninja, options.git
    unittest.main(argv=[__file__, *remaining])
