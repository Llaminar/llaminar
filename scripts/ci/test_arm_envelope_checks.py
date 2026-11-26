# @file test_arm_envelope_checks.py
# @brief Exercise failure aggregation and immutable build evidence without ARM execution.
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from arm_envelope_checks import run


class EnvelopeChecks(unittest.TestCase):
    def test_stage_controls(self):
        for failure in (None, 0, 1, 2, 'missing'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                calls = []
                def execute(command, **kwargs):
                    index = len(calls)
                    calls.append(command)
                    if command[0] == 'cmake' and '--build' in command:
                        proof = Path(command[2]) / 'cpu_native_tile_registers'
                        proof.mkdir(parents=True)
                        (proof / 'driver.json').write_text('build')
                    if command[0] == 'ctest':
                        (Path(command[2]) / 'cpu_native_tile_registers/driver.json').write_text('verify')
                    if failure == 'missing':
                        raise FileNotFoundError('missing tool')
                    return SimpleNamespace(returncode=int(index % 3 == failure))
                self.assertEqual(run('.', directory, execute, test_identity=True), int(failure is not None))
                attempt, = Path(directory).iterdir()
                records = json.loads((attempt / 'stages.json').read_text())
                self.assertEqual(len(records), 12)
                self.assertEqual(len({(r['isa'], r['configuration']) for r in records}), 4)
                self.assertEqual(len(list(attempt.glob('*.log'))), 12)
                self.assertTrue((attempt / 'identity.json').exists())
                for snapshot in attempt.glob('*-build-evidence'):
                    witness = snapshot / 'cpu_native_tile_registers/driver.json'
                    if witness.exists():
                        self.assertEqual(witness.read_text(), 'build')
                self.assertIn('-DCMAKE_MAKE_PROGRAM:FILEPATH=', ' '.join(calls[0]))

    def test_source_and_manifest_tampering(self):
        import hashlib
        import os
        from unittest.mock import patch
        from arm_envelope_checks import source_identity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = root / 'header.h'
            header.write_text('original')
            manifest = root / 'arm-envelope-manifest.json'
            manifest.write_text(json.dumps(dict(commit='commit', tree='tree', files={'header.h': hashlib.sha256(header.read_bytes()).hexdigest()})))
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            with patch.dict(os.environ, ARM_ENVELOPE_MANIFEST_SHA256=digest):
                self.assertEqual(source_identity(root)['tree'], 'tree')
                header.write_text('tampered')
                with self.assertRaisesRegex(ValueError, 'source mismatch'):
                    source_identity(root)
                manifest.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'manifest digest'):
                    source_identity(root)

    def test_identity_failure_records_every_cell_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            def execute(*args, **kwargs):
                self.fail('invalid identity must not execute stages')
            self.assertEqual(run('.', directory, execute, test_identity=False), 1)
            attempt, = Path(directory).iterdir()
            records = json.loads((attempt / 'stages.json').read_text())
            self.assertEqual(len(records), 12)
            self.assertTrue(all(r['returncode'] and r['status'] == 'skipped-after-failure' for r in records))
