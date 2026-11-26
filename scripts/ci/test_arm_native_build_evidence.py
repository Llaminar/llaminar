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
            with tempfile.TemporaryDirectory() as directory, patch.object(evidence, 'ROOT', Path(directory) / 'evidence'), patch.dict(os.environ, LLAMINAR_TEST_RUNNER_INVENTORY='arm64-pr' if code else 'unit-only'):
                with patch.dict(os.environ, LLAMINAR_TEST_RUNNER_INVENTORY='arm64-pr' if code else 'unit-only'):
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
                with self.assertRaisesRegex(AssertionError, 'different attempts'):
                    evidence.enforce()
        dockerfile = (Path(__file__).resolve().parents[2] / 'Dockerfile').read_text()
        references = [line for line in dockerfile.splitlines() if 'builder-integration-operation' in line]
        self.assertEqual(references, ['FROM toolchain AS builder-integration-operation',
                                    'COPY --from=builder-integration-operation /src/native-build-evidence/ /',
                                    'FROM builder-integration-operation AS builder'])

    def test_fulltree_receipts_and_bounded_log(self):
        old = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                proof = Path('build_v2_integration/tests/v2/cpu_native_tile_registers')
                proof.mkdir(parents=True)
                for name in ('driver.json', 'stack-gate.json', 'fused-stage.json'):
                    (proof / name).write_text('{}')
                evidence.capture(['python3', '-c', 'print("x" * 3000000); print("TAIL-MARKER")'])
                import json
                status = json.loads((evidence.ROOT / 'status.json').read_text())
                self.assertFalse(status['missing_stage_evidence'])
                self.assertTrue(status['operation_log']['truncated'])
                self.assertIn('TAIL-MARKER', (evidence.ROOT / 'operation.log').read_text())
                self.assertEqual(status['binding']['operation_build_type'], 'Integration')
                self.assertLess((evidence.ROOT / 'operation.log').stat().st_size, 3000000)
            finally:
                os.chdir(old)

    def test_non_arm_failure_is_not_cacheable(self):
        for inventory in ('full-matrix', 'model-free', 'unit-only'):
            with self.subTest(inventory=inventory), tempfile.TemporaryDirectory() as directory, patch.object(evidence, 'ROOT', Path(directory) / 'evidence'), patch.dict(os.environ, LLAMINAR_TEST_RUNNER_INVENTORY=inventory):
                with self.assertRaises(SystemExit) as error:
                    evidence.capture(['sh', '-c', 'exit 7'])
                self.assertEqual(error.exception.code, 7)
                self.assertTrue((evidence.ROOT / 'status.json').exists())

    def test_collector_failure_preserves_status(self):
        import json
        with tempfile.TemporaryDirectory() as directory, patch.object(evidence, 'ROOT', Path(directory) / 'evidence'), patch('arm_native_build_evidence.subprocess.check_output', side_effect=OSError('broken version')):
            evidence.capture(['true'])
            status = json.loads((evidence.ROOT / 'status.json').read_text())
            self.assertTrue(status['evidence_errors'])
            with self.assertRaisesRegex(AssertionError, 'collected evidence'):
                evidence.enforce()
