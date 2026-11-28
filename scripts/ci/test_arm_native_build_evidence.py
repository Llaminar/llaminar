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
        for inventory in ('arm64-pr', 'unit-only', 'full-matrix', 'model-free'):
            with self.subTest(inventory=inventory), tempfile.TemporaryDirectory() as directory, patch.object(evidence, 'ROOT', Path(directory) / 'evidence'), patch.dict(os.environ, LLAMINAR_TEST_RUNNER_INVENTORY=inventory), patch('arm_native_build_evidence.subprocess.check_output', side_effect=OSError('broken version')):
                evidence.capture(['true'])
                status = json.loads((evidence.ROOT / 'status.json').read_text())
                self.assertTrue(status['evidence_errors'])
                if inventory == 'arm64-pr':
                    with self.assertRaisesRegex(AssertionError, 'collected evidence'):
                        evidence.enforce()
                else:
                    evidence.enforce()

    def test_workflow_waits_for_complete_progress_and_preserves_exit(self):
        """Execute both actual workflow pipelines with a large-output build stub."""
        import re
        import shlex
        import subprocess
        import sys
        workflow = (Path(__file__).resolve().parents[2] / '.github/workflows/develop-pr-arm64.yml').read_text()
        pipelines = re.findall(r'\{ docker buildx build .*?3>&1[^\n]*', workflow, re.S)
        self.assertEqual(len(pipelines), 2)
        marker = 'FINAL-PROGRESS-MARKER'
        for index, pipeline in enumerate(pipelines):
            pipeline = re.sub(r'\$\{\{.*?\}\}', 'fixture-cache', pipeline)
            for code in (0, 7):
                with self.subTest(pipeline=index, code=code), tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / 'llaminar-arm64-evidence'
                    output.mkdir()
                    producer = f'import sys; sys.stderr.write("x"*3000000+"{marker}")'
                    script = ('set -e -o pipefail\nargs=()\n'
                              f'docker() {{ {shlex.quote(sys.executable)} -c {shlex.quote(producer)}; return {code}; }}\n'
                              + pipeline + '\nexit "${build_status:-0}"\n')
                    result = subprocess.run(['bash', '-c', script], env=dict(os.environ, RUNNER_TEMP=directory, TOOLCHAIN_CONTEXT='fixture', EXPECTED_NONCE='fixture'), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.assertEqual(result.returncode, code)
                    name = 'canonical-export.rawjson' if index == 0 else 'canonical-main.rawjson'
                    self.assertEqual((output / name).read_text(), 'x' * 3000000 + marker)
