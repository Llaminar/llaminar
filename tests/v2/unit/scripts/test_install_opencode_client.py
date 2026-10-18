#!/usr/bin/env python3
"""Verify client pinning and atomic executable publication without network access.

Tiny tar fixtures exercise the actual installer and executable-version check.
The test pin is explicitly substituted with each fixture's code-artifact SRI;
the production CLI exposes no version, URL or integrity override.
"""
import base64
import hashlib
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts/ci'))
import install_opencode_client as client


class OpenCodeInstallationTests(unittest.TestCase):
    """Never replace a working client until artifact and executable both agree."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive, self.target = self.root / 'client.tgz', self.root / 'opencode'
        self.target.write_bytes(b'previous client')

    def fixture(self, *, version=client.VERSION, duplicate=False, symlink=False, missing=False):
        """Create a benign tiny executable whose stdout supplies a fixed version."""
        with tarfile.open(self.archive, 'w:gz') as archive:
            if not missing:
                for _ in range(2 if duplicate else 1):
                    data = f'#!/bin/sh\nprintf "%s\\n" "{version}"\n'.encode()
                    entry = tarfile.TarInfo(client.EXECUTABLE_MEMBER)
                    entry.size = len(data)
                    if symlink:
                        entry.type, entry.linkname, entry.size = tarfile.SYMTYPE, '/outside', 0
                    archive.addfile(entry, io.BytesIO(data))
            unrelated = tarfile.TarInfo('../../must-not-be-extracted')
            archive.addfile(unrelated, io.BytesIO())
        return 'sha512-' + base64.b64encode(hashlib.sha512(self.archive.read_bytes()).digest()).decode()

    def test_exact_archive_and_native_version_publish_atomically(self):
        integrity = self.fixture()
        with patch.object(client, 'PACKAGE_INTEGRITY', integrity):
            receipt = client.install_client(self.archive, self.target)
        self.assertTrue(receipt['version_verified'])
        self.assertEqual(receipt['version'], client.VERSION)
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o755)
        self.assertFalse(list(self.root.glob('.opencode-*')))

    def test_changed_artifact_never_executes_or_overwrites_existing_client(self):
        self.fixture()
        with patch.object(client.subprocess, 'check_output', side_effect=AssertionError('Unverified execution')):
            with self.assertRaisesRegex(ValueError, 'integrity'):
                client.install_client(self.archive, self.target)
        self.assertEqual(self.target.read_bytes(), b'previous client')

    def test_missing_duplicate_and_symlink_executables_fail_closed(self):
        for arguments in ({'missing': True}, {'duplicate': True}, {'symlink': True}):
            with self.subTest(arguments=arguments):
                integrity = self.fixture(**arguments)
                with patch.object(client, 'PACKAGE_INTEGRITY', integrity), self.assertRaisesRegex(ValueError, 'regular native'):
                    client.install_client(self.archive, self.target)
                self.assertEqual(self.target.read_bytes(), b'previous client')

    def test_wrong_native_version_removes_temporary_binary_and_preserves_prior_client(self):
        integrity = self.fixture(version='0.0.0')
        with patch.object(client, 'PACKAGE_INTEGRITY', integrity), self.assertRaisesRegex(ValueError, 'differs from tested version'):
            client.install_client(self.archive, self.target)
        self.assertEqual(self.target.read_bytes(), b'previous client')
        self.assertFalse(list(self.root.glob('.opencode-*')))

    def test_version_check_failure_also_retires_only_the_owned_temporary(self):
        integrity = self.fixture()
        with patch.object(client, 'PACKAGE_INTEGRITY', integrity), \
             patch.object(client.subprocess, 'check_output', side_effect=OSError('cannot execute')), \
             self.assertRaisesRegex(OSError, 'cannot execute'):
            client.install_client(self.archive, self.target)
        self.assertEqual(self.target.read_bytes(), b'previous client')
        self.assertFalse(list(self.root.glob('.opencode-*')))


if __name__ == '__main__':
    unittest.main()
