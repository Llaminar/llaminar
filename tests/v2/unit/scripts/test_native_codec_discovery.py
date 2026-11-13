#!/usr/bin/env python3
## @file test_native_codec_discovery.py
# @brief Device-free compiled native discovery and original Release owner contracts.
"""Generate actual CTest metadata from inert compiled-inventory stand-ins."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[4]


class NativeCodecDiscovery(unittest.TestCase):
    """Exercise registration without running MPI, devices, or native fixtures."""

    cmake = 'cmake'
    ninja = 'ninja'

    def discover(self, listing, backends='CUDA;ROCm', timeout='120'):
        """Run the actual shared discovery script and inspect CTest's metadata."""
        with tempfile.TemporaryDirectory(prefix='native codec discovery ') as directory:
            root = Path(directory)
            helper = Path(os.environ['LLAMINAR_TEST_MPI_SESSION_HELPER'])
            self.assertTrue(helper.is_file())
            executable = root / 'inventory'
            executable.write_text(f'#!{sys.executable}\nprint({listing!r})\n')
            executable.chmod(0o755)
            # Exact literal metadata must survive every emitted codec iteration.
            working_directory = root / 'literal "quote" and \\backslash'
            working_directory.mkdir()
            generated = root / 'cases.cmake'
            result = subprocess.run([self.cmake, '-DMODEL_FREE_NATIVE=TRUE',
                f'-DNATIVE_BACKENDS={backends}', '-DMPI_PROCS=1', f'-DTIMEOUT={timeout}',
                '-DCORES_PER_SOCKET=4', f'-DTEST_EXECUTABLE={executable}',
                f'-DMPI_SESSION_HELPER={helper}', f'-DMPI_SESSION_ROOT={root / "sessions"}',
                f'-DCTEST_FILE={generated}', '-DTEST_PREFIX=V2_Integration_NativeFixedHeadPipeline',
                '-DLABELS=V2;Integration;ProductionTestPreflight;HardwareOptional;Devices2;FullDeviceInventory',
                f'-DWORKING_DIR={working_directory}', '-P', str(SOURCE / 'tests/v2/cmake/V2ParityTestDiscovery.cmake')],
                capture_output=True, text=True, timeout=10)
            if result.returncode:
                return result, []
            (root / 'CTestTestfile.cmake').write_text(f'include("{generated}")\n')
            shown = subprocess.run([str(Path(shutil.which(self.cmake)).with_name('ctest')),
                '--test-dir', str(root), '--show-only=json-v1'], capture_output=True,
                text=True, check=True, timeout=10)
            # Consume the generated inventory through the actual gate, before
            # this disposable CTest directory retires; never infer MPI width.
            campaign_path = SOURCE / 'scripts/ci/run_production_parity_campaigns.py'
            spec = importlib.util.spec_from_file_location('native_codec_preflight_consumer', campaign_path)
            self.assertIsNotNone(spec)
            consumer = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = consumer
            sys.path.insert(0, str(campaign_path.parent))
            try:
                spec.loader.exec_module(consumer)
            finally:
                sys.path.pop(0)
            registrations = consumer.discover_production_test_preflight_registrations(root)
            requirements = consumer.preflight_optional_hardware_requirements(registrations)
            tests = json.loads(shown.stdout)['tests']
            self.assertEqual({row.name for row in registrations}, {test['name'] for test in tests})
            for row in registrations:
                self.assertIsNone(row.mpi_ranks)
                self.assertIs(row.lane, consumer.PreflightExecutionLane.EXCLUSIVE)
                self.assertNotIn('StageSizes1x1', row.labels)
                backend = 'cuda' if '_CUDA2Formats_' in row.name else 'rocm'
                self.assertEqual(requirements[row.name], {
                    'backend': backend, 'degree': 2,
                    'locality': 'original-numa-local-rank'})
            return result, tests

    @staticmethod
    def listing():
        """Representative names test mechanics; production owns its canonical roster."""
        return ('Unrelated.\n  Ignore\n'
                'AllNativeCodecs/NativeFixedHeadPipelineFormats.\n'
                '  CUDA2/FP16  # opaque parameter\n  CUDA2/Q4_K\n'
                '  ROCm2/FP16\n  ROCm2/Q4_K\n')

    def test_each_compiled_codec_has_original_singleton_owner_and_strict_metadata(self):
        result, tests = self.discover(self.listing())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({test['name'] for test in tests}, {
            f'V2_Integration_NativeFixedHeadPipeline_{backend}2Formats_{codec}'
            for backend in ('CUDA', 'ROCm') for codec in ('FP16', 'Q4_K')})
        for test in tests:
            props = {p['name']: p['value'] for p in test['properties']}
            command = test['command']
            self.assertEqual(Path(command[0]), Path(os.environ['LLAMINAR_TEST_MPI_SESSION_HELPER']))
            self.assertEqual(command[1], '--session-base')
            self.assertEqual(command[3], '--')
            self.assertEqual(len([x for x in command if x.startswith('--gtest_filter=')]), 1)
            backend = 'cuda' if '_CUDA2Formats_' in test['name'] else 'rocm'
            self.assertIn(f'--pipeline-native-family={backend}', command)
            self.assertIn('--pipeline-native-degree=2', command)
            self.assertNotIn('mpirun', command)
            self.assertEqual(props['WORKING_DIRECTORY'], str(
                Path(command[4]).parent / 'literal "quote" and \\backslash'))
            self.assertEqual(props['TIMEOUT'], 120)
            self.assertEqual(props['TIMEOUT_SIGNAL_NAME'], 'SIGTERM')
            self.assertIn('ProductionTestPreflight', props['LABELS'])
            self.assertIn('ProcessOwned', props['LABELS'])
            self.assertIn('FullDeviceInventory', props['LABELS'])
            self.assertIn('Integration_Serial', props['RESOURCE_LOCK'])
            self.assertTrue(any('SKIPPED' in p for p in props['FAIL_REGULAR_EXPRESSION']))
            self.assertTrue(any('0 tests?' in p for p in props['FAIL_REGULAR_EXPRESSION']))
            self.assertNotIn('FIXTURES_REQUIRED', props)
            self.assertNotIn('SKIP_REGULAR_EXPRESSION', props)
            self.assertFalse(any('PRODUCTION_PARITY' in e for e in props['ENVIRONMENT']))
            for entry in (
                'LLAMINAR_LOG_LEVEL=DEBUG', 'LLAMINAR_TP_COLLECT_TIMEOUT_MS=30000',
                'OMP_NUM_THREADS=4', 'OMP_PLACES=sockets', 'OMP_PROC_BIND=close',
                'KMP_AFFINITY=granularity=fine,compact,1,0',
                'OMPI_MCA_mpi_leave_pinned=1', 'OMPI_MCA_btl_openib_allow_ib=1'):
                self.assertIn(entry, props['ENVIRONMENT'])

    def test_canonical_preflight_consumer_admits_each_original_singleton_backend(self):
        """Each configured family needs two same-owner GPUs, with no external MPI claim."""
        for backend in ('CUDA', 'ROCm'):
            with self.subTest(backend=backend):
                result, tests = self.discover(self.listing(), backends=backend)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual({test['name'] for test in tests}, {
                    f'V2_Integration_NativeFixedHeadPipeline_{backend}2Formats_{codec}'
                    for codec in ('FP16', 'Q4_K')})

    def test_missing_duplicate_malformed_and_asymmetric_populations_fail_closed(self):
        original = self.listing()
        for listing, backends in [
            (original.replace('  ROCm2/FP16\n  ROCm2/Q4_K\n', ''), 'CUDA;ROCm'),
            (original + '  CUDA2/FP16\n', 'CUDA;ROCm'),
            (original.replace('CUDA2/FP16', 'CUDA2/bad-label'), 'CUDA;ROCm'),
            (original.replace('  ROCm2/Q4_K\n', ''), 'CUDA;ROCm'),
            (original, 'CUDA;CUDA'), (original, 'CPU')]:
            with self.subTest(listing=listing, backends=backends):
                result, _ = self.discover(listing, backends)
                self.assertNotEqual(result.returncode, 0)
        result, _ = self.discover(original, timeout='121')
        self.assertNotEqual(result.returncode, 0)

    def test_explicit_release_owner_controls_native_discovery(self):
        """Configure the real registration function without compiling any source."""
        source = (SOURCE / 'tests/v2/CMakeLists.txt').read_text()
        start = source.index('function(discover_v2_parity_tests TARGET)')
        function = source[start:source.index('\nendfunction()', start) + len('\nendfunction()')]
        for selected in (False, True):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                shutil.copytree(SOURCE / 'tests/v2/cmake', root / 'cmake')
                (root / 'dummy.cpp').write_text('int main() {return 0;}\n')
                (root / 'CMakeLists.txt').write_text(
                    'cmake_minimum_required(VERSION 3.27)\nproject(NativeDiscovery CXX)\nenable_testing()\n'
                    'include(cmake/V2TestTargetSelection.cmake)\n'
                    'set(V2_PERF_TESTS_ONLY TRUE)\n'
                    + ('set(LLAMINAR_RELEASE_NATIVE_TEST_TARGETS native)\n' if selected else '')
                    + 'add_executable(native dummy.cpp)\nadd_executable(llaminar_mpi_session IMPORTED)\n'
                    + f'set_target_properties(llaminar_mpi_session PROPERTIES IMPORTED_LOCATION "{os.environ["LLAMINAR_TEST_MPI_SESSION_HELPER"]}")\n'
                    + 'add_test(NAME UnitProbe COMMAND "${CMAKE_COMMAND}" -E true)\n'
                    'set_tests_properties(UnitProbe PROPERTIES LABELS Unit)\n'
                    + function + '\ndiscover_v2_parity_tests(native MODEL_FREE_NATIVE NATIVE_BACKENDS CUDA '
                    'TEST_PREFIX Native LABELS Integration MPI_PROCS 1 TIMEOUT 120)\n'
                    'get_property(owners GLOBAL PROPERTY V2_DISCOVERED_PREFLIGHT_GATE_TARGETS)\n'
                    'file(WRITE "${CMAKE_BINARY_DIR}/owners" "${owners}")\n')
                result = subprocess.run([self.cmake, '-S', str(root), '-B', str(root / 'build'),
                    '-G', 'Ninja', f'-DCMAKE_MAKE_PROGRAM={self.ninja}'], capture_output=True,
                    text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                owners = (root / 'build/owners').read_text()
                self.assertEqual(bool(owners), selected)
                if selected:
                    self.assertEqual(owners, 'native;llaminar_mpi_session')
                    identity = next((root / 'build').glob('*discovery_identity*')).read_text()
                    self.assertIn('model_free_native=1', identity)
                    self.assertIn('native_backends=CUDA\n', identity)
                    self.assertIn('test_prefix=Native\n', identity)
                    self.assertNotIn('LLAMINAR_PARITY_SOURCE_DIR',
                                     (root / 'build/build.ninja').read_text())
                    self.assertIn('_undiscovered_', (root / 'build/native_discovered_tests.cmake').read_text())
                    ctest = str(Path(shutil.which(self.cmake)).with_name('ctest'))
                    unit = subprocess.run([ctest, '--test-dir', str(root / 'build'),
                        '-L', '^Unit$', '--output-on-failure'], capture_output=True,
                        text=True, timeout=10)
                    self.assertEqual(unit.returncode, 0, unit.stdout + unit.stderr)
                    inventory = subprocess.run([ctest, '--test-dir', str(root / 'build'),
                        '--show-only=json-v1'], capture_output=True, text=True,
                        check=True, timeout=10)
                    listed = json.loads(inventory.stdout)['tests']
                    self.assertEqual({test['name'] for test in listed},
                                     {'UnitProbe', 'Native_DiscoveryRequired'})
                    sentinel = next(test for test in listed if test['name'].endswith('DiscoveryRequired'))
                    self.assertEqual(sentinel['command'][0], os.environ['LLAMINAR_TEST_MPI_SESSION_HELPER'])
                    selected_preflight = subprocess.run([ctest, '--test-dir', str(root / 'build'),
                        '-L', '^ProductionTestPreflight$', '--output-on-failure'],
                        capture_output=True, text=True, timeout=10)
                    self.assertNotEqual(selected_preflight.returncode, 0)
                    self.assertIn('Build native before required native codec discovery',
                                  selected_preflight.stdout + selected_preflight.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cmake', default='cmake')
    parser.add_argument('--ninja', default='ninja')
    args, remaining = parser.parse_known_args()
    NativeCodecDiscovery.cmake = args.cmake
    NativeCodecDiscovery.ninja = args.ninja
    unittest.main(argv=[sys.argv[0], *remaining])
