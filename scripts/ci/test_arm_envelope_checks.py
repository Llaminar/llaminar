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
            manifest.write_text(json.dumps(dict(commit='a'*40, tree='b'*40, files={'header.h': {'sha256': hashlib.sha256(header.read_bytes()).hexdigest()}})))
            (root / 'arm-envelope.Dockerfile').write_text('FROM toolchain')
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            with patch.dict(os.environ, ARM_ENVELOPE_MANIFEST_SHA256=digest):
                self.assertEqual(source_identity(root)['tree'], 'b'*40)
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

    def test_identity_admission_truth_table(self):
        from arm_envelope_checks import identity_bound
        for compiler, source, expected in ((True, True, True), (True, False, False),
                                           (False, True, False), (False, False, False)):
            with self.subTest(compiler=compiler, source=source):
                self.assertIs(identity_bound(compiler, source), expected)

    def test_noncanonical_compiler_with_authenticated_source(self):
        import hashlib
        import os
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'source'
            root.mkdir()
            (root / 'arm-envelope.Dockerfile').write_text('FROM toolchain')
            manifest = root / 'arm-envelope-manifest.json'
            manifest.write_text(json.dumps(dict(commit='a'*40, tree='b'*40, files={})))
            compiler = Path(directory) / 'noncanonical-compiler'
            compiler.write_text('#!/bin/sh\necho noncanonical compiler\n')
            compiler.chmod(0o755)
            calls = []
            def execute(command, **kwargs):
                calls.append(command)
                raise FileNotFoundError('diagnostic missing compiler')
            with patch.dict(os.environ, ARM_ENVELOPE_MANIFEST_SHA256=hashlib.sha256(manifest.read_bytes()).hexdigest(), CC=str(compiler), CXX=str(compiler)):
                self.assertEqual(run(root, Path(directory) / 'out', execute), 1)
            attempt, = (Path(directory) / 'out').iterdir()
            identity = json.loads((attempt / 'identity.json').read_text())
            self.assertTrue(identity['source_bound'])
            self.assertFalse(identity['compiler_bound'])
            self.assertNotIn('source_error', identity)
            self.assertEqual(len(calls), 1)
            records = json.loads((attempt / 'stages.json').read_text())
            self.assertEqual(len(records), 12)
            self.assertTrue(all(r['status'] == 'skipped-after-failure' and r['returncode'] for r in records))

    def test_exact_inventory_and_symlink_targets(self):
        import hashlib
        import os
        from unittest.mock import patch
        from arm_envelope_checks import source_identity
        for change in ('extra', 'commit', 'symlink'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'header').write_text('data')
                (root / 'link').symlink_to('header')
                (root / 'arm-envelope.Dockerfile').write_text('FROM toolchain')
                manifest = dict(commit='a'*40, tree='b'*40, files={'header': {'sha256': hashlib.sha256(b'data').hexdigest()}, 'link': {'symlink': 'header'}})
                if change == 'commit':
                    manifest['commit'] = 'invalid'
                path = root / 'arm-envelope-manifest.json'
                path.write_text(json.dumps(manifest))
                if change == 'extra':
                    (root / 'extra').write_text('extra')
                if change == 'symlink':
                    (root / 'link').unlink()
                    (root / 'link').symlink_to('other')
                with patch.dict(os.environ, ARM_ENVELOPE_MANIFEST_SHA256=hashlib.sha256(path.read_bytes()).hexdigest()):
                    with self.assertRaises(ValueError):
                        source_identity(root)

    def test_failed_attempt_is_host_readable(self):
        import os
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            def execute(*args, **kwargs):
                return SimpleNamespace(returncode=7)
            self.assertEqual(run('.', directory, execute, test_identity=False), 1)
            attempt, = Path(directory).iterdir()
            self.assertEqual(attempt.stat().st_mode & 0o777, 0o755)
            self.assertTrue(all(path.stat().st_mode & 0o004 for path in attempt.rglob('*') if path.is_file()))
            if os.name == 'posix' and os.geteuid() == 0:
                # Exercise real Linux DAC with a root-created failed attempt,
                # not root's ability to read its own private output.
                Path(directory).chmod(0o755)
                child = os.fork()
                if child == 0:
                    try:
                        os.setgroups([])
                        os.setgid(65534)
                        os.setuid(65534)
                        for path in Path(directory).rglob('*'):
                            if path.is_file():
                                path.read_bytes()
                        os._exit(0)
                    except BaseException:
                        os._exit(1)
                _, status = os.waitpid(child, 0)
                self.assertEqual(os.waitstatus_to_exitcode(status), 0)
