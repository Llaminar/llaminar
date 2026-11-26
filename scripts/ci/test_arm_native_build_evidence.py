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

    def test_nonce_and_operation_chokepoint(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(evidence, 'ROOT', Path(directory) / 'evidence'):
            evidence.capture(['true'])
            with patch.dict(os.environ, LLAMINAR_EXPECTED_BUILD_NONCE='wrong'):
                with self.assertRaisesRegex(AssertionError, 'nonce'):
                    evidence.enforce()
        dockerfile = (Path(__file__).resolve().parents[2] / 'Dockerfile').read_text()
        references = [line for line in dockerfile.splitlines() if 'builder-integration-operation' in line]
        self.assertEqual(references, ['FROM toolchain AS builder-integration-operation',
                                    'COPY --from=builder-integration-operation /src/native-build-evidence/ /',
                                    'FROM builder-integration-operation AS builder'])
