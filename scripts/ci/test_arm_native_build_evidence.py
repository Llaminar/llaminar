# @file test_arm_native_build_evidence.py
# @brief Failed native operations retain diagnostics but cannot admit a builder.
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import arm_native_build_evidence as evidence


class EvidenceTests(unittest.TestCase):
    def test_original_exit_and_tamper(self):
        for code in (0, 7):
            with tempfile.TemporaryDirectory() as directory, patch.object(evidence, 'ROOT', Path(directory) / 'evidence'):
                evidence.capture(['sh', '-c', f'echo original; exit {code}'])
                self.assertTrue((evidence.ROOT / 'operation.log').exists())
                if code:
                    with self.assertRaises(SystemExit) as error:
                        evidence.enforce()
                    self.assertEqual(error.exception.code, code)
                else:
                    evidence.enforce()
                (evidence.ROOT / 'operation.log').write_text('tampered')
                with self.assertRaises(AssertionError):
                    evidence.enforce()

    def test_missing_and_stale(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(evidence, 'ROOT', Path(directory) / 'evidence'):
            with self.assertRaises(FileNotFoundError):
                evidence.enforce()
            evidence.capture(['true'])
            with patch.dict(os.environ, LLAMINAR_SOURCE_TREE='different'):
                with self.assertRaises(AssertionError):
                    evidence.enforce()
