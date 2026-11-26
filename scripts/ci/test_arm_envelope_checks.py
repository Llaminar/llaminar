# @file test_arm_envelope_checks.py
# @brief Fail-closed orchestration regression without executing ARM instructions.
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from arm_envelope_checks import run


class EnvelopeChecks(unittest.TestCase):
    def test_failure_retains_all_stages_and_envelopes(self):
        calls = []
        def execute(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=1 if len(calls) == 1 else 0)
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(run('.', directory, execute), 1)
            self.assertEqual(len(calls), 12)
            self.assertEqual(len(list(Path(directory).glob('*.log'))), 12)
            self.assertTrue((Path(directory) / 'stages.json').exists())
