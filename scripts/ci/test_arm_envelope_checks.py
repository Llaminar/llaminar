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
                self.assertEqual(run('.', directory, execute), int(failure is not None))
                attempt, = Path(directory).iterdir()
                records = json.loads((attempt / 'stages.json').read_text())
                self.assertEqual(len(records), 12)
                self.assertEqual(len({(r['isa'], r['configuration']) for r in records}), 4)
                self.assertEqual(len(list(attempt.glob('*.log'))), 12)
                self.assertTrue((attempt / 'identity.json').exists())
                for snapshot in attempt.glob('*-build-evidence'):
                    self.assertEqual((snapshot / 'cpu_native_tile_registers/driver.json').read_text(), 'build')
                self.assertIn('-DCMAKE_MAKE_PROGRAM:FILEPATH=', ' '.join(calls[0]))
