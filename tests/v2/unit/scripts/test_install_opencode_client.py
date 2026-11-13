#!/usr/bin/env python3
"""Check authenticated source builds and atomic qualified-client publication.

Tiny archives and mocked build subprocesses keep Unit device-free and fast.
The real four-control native lifecycle proof runs in the production installer.
"""
import base64
import hashlib
import io
import json
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
    """No changed source or failed lifecycle may replace the installed client."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary, self.target = self.root / 'candidate', self.root / 'opencode'
        self.binary.write_text(f'#!/bin/sh\nprintf "%s\\n" "{client.VERSION}"\n')
        self.target.write_bytes(b'previous client')

    def test_native_version_and_lifecycle_pass_before_atomic_publication(self):
        with patch.object(client, 'qualify_client', return_value={'passed': True}) as qualify:
            result = client.publish_client(self.binary, self.target, {'upstream_commit': client.UPSTREAM_COMMIT})
        self.assertTrue(result['version_verified'])
        self.assertEqual(result['version'], '1.18.34-llaminar.1')
        self.assertEqual(result['binary_sha256'], hashlib.sha256(self.target.read_bytes()).hexdigest())
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o755)
        qualify.assert_called_once()
        self.assertFalse(list(self.root.glob('.opencode-*')))

    def test_failed_lifecycle_keeps_previous_installation(self):
        with patch.object(client, 'qualify_client', side_effect=ValueError('lifecycle failed')):
            with self.assertRaisesRegex(ValueError, 'lifecycle failed'):
                client.publish_client(self.binary, self.target, {})
        self.assertEqual(self.target.read_bytes(), b'previous client')
        self.assertFalse(list(self.root.glob('.opencode-*')))

    def test_wrong_version_never_runs_lifecycle_or_replaces_previous_client(self):
        self.binary.write_text('#!/bin/sh\nprintf "1.18.34\\n"\n')
        with patch.object(client, 'qualify_client') as qualify:
            with self.assertRaisesRegex(ValueError, 'differs from tested version'):
                client.publish_client(self.binary, self.target, {})
        qualify.assert_not_called()
        self.assertEqual(self.target.read_bytes(), b'previous client')
        self.assertFalse(list(self.root.glob('.opencode-*')))

    def test_changed_source_is_rejected_before_extraction(self):
        archive = self.root / 'changed.tgz'
        archive.write_bytes(b'untrusted source')
        with patch.object(client.tarfile, 'open', side_effect=AssertionError('unverified extraction')):
            with self.assertRaisesRegex(ValueError, 'integrity'):
                client.extract_source(archive, self.root / 'source')

    def test_source_fixture_is_authenticated_independently(self):
        archive = self.root / 'source.tgz'
        data = b'changed models metadata'
        with tarfile.open(archive, 'w:gz') as output:
            entry = tarfile.TarInfo(f'opencode-{client.UPSTREAM_COMMIT}/{client.MODELS_FIXTURE}')
            entry.size = len(data)
            output.addfile(entry, io.BytesIO(data))
        with patch.object(client, 'SOURCE_SHA256', hashlib.sha256(archive.read_bytes()).hexdigest()):
            with self.assertRaisesRegex(ValueError, 'integrity'):
                client.extract_source(archive, self.root / 'extracted')

    def test_native_qualification_requires_every_control_exactly_once(self):
        controls = [{'mode': mode, 'passed': True} for mode in
                    ('stop', 'tool_calls', 'tool_stop', 'provider_overflow')]
        for changed in (controls[:-1], controls + controls[:1], [dict(row, passed=False) for row in controls]):
            with self.subTest(changed=changed), patch.object(client.subprocess, 'check_output', return_value=json.dumps(changed)):
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    client.qualify_client(self.binary)
        with patch.object(client.subprocess, 'check_output', return_value=json.dumps(controls)):
            self.assertTrue(client.qualify_client(self.binary)['passed'])

    def test_source_build_uses_frozen_lock_and_authenticated_fixture(self):
        source = self.root / 'source'
        source.mkdir()
        (source / 'bun.lock').write_bytes(b'pinned lock')
        (source / client.MODELS_FIXTURE).parent.mkdir(parents=True)
        (source / client.MODELS_FIXTURE).write_bytes(b'pinned metadata')
        bun = self.root / 'bun.tgz'
        with tarfile.open(bun, 'w:gz') as archive:
            data = b'#!/bin/sh\nexit 0\n'
            entry = tarfile.TarInfo('package/bin/bun')
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
        integrity = 'sha512-' + base64.b64encode(hashlib.sha512(bun.read_bytes()).digest()).decode()
        with patch.object(client, 'extract_source', return_value=source), \
             patch.object(client, 'BUN_INTEGRITY', integrity), \
             patch.object(client.subprocess, 'run') as run, \
             patch.object(client, 'publish_client', return_value={'published': True}) as publish:
            result = client.install_client(self.root / 'source.tgz', self.target, bun_archive=bun)
        self.assertTrue(result['published'])
        self.assertEqual(run.call_count, 3)
        self.assertIn('--fuzz=0', run.call_args_list[0].args[0])
        self.assertEqual(run.call_args_list[1].args[0][1:], ['install', '--frozen-lockfile'])
        build = run.call_args_list[2]
        self.assertEqual(build.args[0][1:], ['run', 'script/build.ts', '--single', '--skip-install', '--skip-embed-web-ui'])
        self.assertEqual(build.kwargs['env']['MODELS_DEV_API_JSON'], str(source / client.MODELS_FIXTURE))
        self.assertEqual(build.kwargs['env']['OPENCODE_VERSION'], client.VERSION)
        self.assertFalse(publish.call_args.args[2]['models_catalog_qualified'])

    def test_bun_sri_rejects_changed_bytes(self):
        expected = 'sha512-' + base64.b64encode(hashlib.sha512(self.binary.read_bytes()).digest()).decode()
        client.authenticate(self.binary, expected, 'sha512')
        self.binary.write_bytes(b'changed Bun')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            client.authenticate(self.binary, expected, 'sha512')


if __name__ == '__main__':
    unittest.main()
