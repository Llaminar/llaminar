# @file test_arm_witness_toolchain_manifest.py
# @brief Device-free regressions for the ARM64 witness toolchain closure manifest.
#
# The manifest must reproduce the builder's exact closure bytes and symlink
# shape, and any drift (content, retarget, missing entry) must fail closed with
# a precise diagnostic. These tests build a small synthetic closure with the
# same symlink/alternatives shape the proof authenticates.

import json
import os
from pathlib import Path
import tempfile
import unittest
import arm_witness_toolchain_manifest as manifest


class WitnessToolchainManifestTests(unittest.TestCase):
    def _closure(self, root: Path) -> None:
        (root / "usr" / "bin").mkdir(parents=True)
        (root / "etc" / "alternatives").mkdir(parents=True)
        (root / "usr" / "libexec" / "gcc").mkdir(parents=True)
        (root / "usr" / "include" / "c++" / "14").mkdir(parents=True)
        # Real compiler binary with a stable content marker.
        (root / "usr" / "bin" / "aarch64-linux-gnu-g++-14").write_bytes(b"compiler-bytes")
        # The exact alternatives chain the proof records:
        #   c++ -> /etc/alternatives/c++ -> g++ -> /etc/alternatives/g++ -> g++-14 -> real
        os.symlink("/etc/alternatives/c++", root / "usr" / "bin" / "c++")
        os.symlink("/usr/bin/g++", root / "etc" / "alternatives" / "c++")
        os.symlink("/etc/alternatives/g++", root / "usr" / "bin" / "g++")
        os.symlink("/usr/bin/g++-14", root / "etc" / "alternatives" / "g++")
        os.symlink("aarch64-linux-gnu-g++-14", root / "usr" / "bin" / "g++-14")
        os.symlink("aarch64-linux-gnu-as", root / "usr" / "bin" / "as")
        (root / "usr" / "bin" / "aarch64-linux-gnu-as").write_bytes(b"assembler-bytes")
        # Absolute program path like the proof's cc1plus job.
        (root / "usr" / "libexec" / "gcc" / "14").mkdir(parents=True)
        (root / "usr" / "libexec" / "gcc" / "14" / "cc1plus").write_bytes(b"cc1plus-bytes")
        (root / "usr" / "include" / "c++" / "14" / "version").write_bytes(b"#define V 1\n")

    def test_capture_verify_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "closure"
            self._closure(root)
            captured = manifest.capture(root)
            self.assertEqual(captured["schema"], manifest.SCHEMA)
            self.assertEqual(captured["tool"], manifest.TOOL)
            # Symlinks are recorded by link text, never followed.
            self.assertEqual(captured["entries"]["usr/bin/c++"]["target"], "/etc/alternatives/c++")
            self.assertEqual(captured["entries"]["usr/bin/g++-14"]["target"], "aarch64-linux-gnu-g++-14")
            manifest.verify(root, captured)  # pass: unchanged closure

    def test_content_drift_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "closure"
            self._closure(root)
            captured = manifest.capture(root)
            (root / "usr" / "bin" / "aarch64-linux-gnu-g++-14").write_bytes(b"tampered-compiler")
            with self.assertRaises(ValueError) as error:
                manifest.verify(root, captured)
            self.assertIn("usr/bin/aarch64-linux-gnu-g++-14", str(error.exception))
            self.assertIn("content changed", str(error.exception))

    def test_symlink_retarget_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "closure"
            self._closure(root)
            captured = manifest.capture(root)
            (root / "etc" / "alternatives" / "g++").unlink()
            os.symlink("/usr/bin/g++-13", root / "etc" / "alternatives" / "g++")
            with self.assertRaises(ValueError) as error:
                manifest.verify(root, captured)
            self.assertIn("retargeted", str(error.exception))
            self.assertIn("etc/alternatives/g++", str(error.exception))

    def test_missing_entry_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "closure"
            self._closure(root)
            captured = manifest.capture(root)
            (root / "usr" / "libexec" / "gcc" / "14" / "cc1plus").unlink()
            with self.assertRaises(ValueError) as error:
                manifest.verify(root, captured)
            self.assertIn("missing", str(error.exception))

    def test_foreign_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "closure"
            self._closure(root)
            foreign = {"schema": 999, "tool": manifest.TOOL, "entries": {}}
            with self.assertRaises(ValueError):
                manifest.verify(root, foreign)

    def test_capture_requires_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                manifest.capture(Path(directory))


if __name__ == "__main__":
    unittest.main()
