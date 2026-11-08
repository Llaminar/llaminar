#!/usr/bin/env python3
## @file test_optional_gpu_preflight_registration.py
#  @brief Prove native test admission and build obligations with the real CMake authorities.
"""Execute optional-GPU CMake registration in Integration and Release modes.

The real helper and native topology loops run in a language-free CMake project
with imported executable placeholders. CTest's generated metadata proves both
wide-hardware skip ownership and safe omission in Release. The actual ROCm
owner-partition cache/owning-TU blocks additionally prove Automatic/explicit mode
configuration and conditional native-test registration. CUDA GDN admission
executes the real registration and gate finalizer for every configured backend
pair, including a literal original-code control. No compiler, model, GPU
discovery, native test launch or replacement registration helper is involved.
"""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[4]


def registration_source() -> str:
    """Load executable CMake policy from its canonical source-owned boundaries."""
    source = (SOURCE / 'tests/v2/CMakeLists.txt').read_text()
    start = source.index('function(add_v2_test TEST_NAME)')
    end = source.index('endfunction()', start) + len('endfunction()')
    helper = source[start:end]
    start = source.index('    foreach(_devices 2 4 8)')
    end = source.index('    message(STATUS "V2 Tests: Added RCCLCoordinator', start)
    rccl = source[start:end]
    start = source.index('    foreach(_order CUDA_ROCm ROCm_CUDA)')
    end = source.index('\nendif()', start)
    native = source[start:end]
    start = source.index('foreach(device_count IN ITEMS 4 8)',
                         source.index('# Independent PP stages preserve'))
    end = source.index('# Test: device-owned ExpertOverlay epoch publication', start)
    owner = f'include("{SOURCE}/tests/v2/fixtures/cmake/NativeMPISessionOwner.cmake")'
    return '\n'.join((owner, helper, rccl, native, source[start:end]))


class OptionalGPUPreflightRegistration(unittest.TestCase):
    """Exercise the same ownership rules in both production build modes."""

    def configure(self, build_type: str) -> list[dict]:
        """Generate actual CTest metadata without native tools or runtime work."""
        with tempfile.TemporaryDirectory(prefix='llaminar-optional-preflight-') as directory:
            root = Path(directory)
            prefix = f'''cmake_minimum_required(VERSION 3.25)
project(OptionalGPUPreflight NONE)
enable_testing()
set(CMAKE_BUILD_TYPE {build_type})
set(V2_PERF_TESTS_ONLY {'ON' if build_type == 'Release' else 'OFF'})
set(V2_TEST_SOCKETS 2)
set(V2_TEST_CORES_PER_SOCKET 2)
set(DEFAULT_LOG_LEVEL ERROR)
set(Python3_EXECUTABLE "{sys.executable}")
'''
            for target in ('v2_integration_rccl_unused_communicator_retirement',
                           'v2_integration_pipeline_domain_generation'):
                prefix += f'add_executable({target} IMPORTED GLOBAL)\n'
                prefix += f'set_target_properties({target} PROPERTIES IMPORTED_LOCATION "${{CMAKE_COMMAND}}")\n'
            # Release deliberately has no unit fixture targets. A direct
            # add_test() bypassing the real helper must therefore fail CMake's
            # generator-expression validation, just as it would in production.
            if build_type != 'Release':
                for target in ('v2_test_moe_optimization_pipeline', 'v2_test_prefix_cache_prefill_flow',
                               'v2_test_chat_completion_handler'):
                    prefix += f'add_executable({target} IMPORTED GLOBAL)\n'
                    prefix += f'set_target_properties({target} PROPERTIES IMPORTED_LOCATION "${{CMAKE_COMMAND}}")\n'
            (root / 'CMakeLists.txt').write_text(prefix + registration_source())
            ninja = shutil.which('ninja')
            self.assertIsNotNone(ninja)
            configured = subprocess.run(['cmake', '-S', str(root), '-B', str(root / 'build'),
                                         '-G', 'Ninja', '-DCMAKE_MAKE_PROGRAM=' + ninja],
                                        text=True, capture_output=True, timeout=10)
            self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)
            result = subprocess.run(['ctest', '--test-dir', str(root / 'build'), '--show-only=json-v1'],
                                    text=True, capture_output=True, check=True, timeout=10)
            return json.loads(result.stdout)['tests']

    def test_integration_preserves_each_native_cell_and_exact_hardware_skip(self):
        """Only eight-device layouts carry optional-hardware skip expressions."""
        tests = self.configure('Integration')
        self.assertEqual(len(tests), 49)
        optional = 0
        for test in tests:
            properties = {item['name']: item['value'] for item in test['properties']}
            self.assertIn('ProductionTestPreflight', properties['LABELS'])
            self.assertNotIn('DISABLED', properties)
            labels = set(properties['LABELS'])
            if 'Devices8' in labels:
                optional += 1
                pattern = ('Requires 8 ROCm devices' if 'RCCLUnusedRetirement_' in test['name']
                           else 'Requires 4 CUDA and ROCm devices')
                self.assertEqual(properties['SKIP_REGULAR_EXPRESSION'], [pattern])
            else:
                self.assertNotIn('SKIP_REGULAR_EXPRESSION', properties)
        self.assertEqual(optional, 14)

    def test_release_omits_native_tests_and_their_properties_together(self):
        """A performance-only build cannot attach metadata to nonexistent tests."""
        self.assertEqual(self.configure('Release'), [])



class CUDAGDNPreflightRegistration(unittest.TestCase):
    """Prove configured CUDA owns both native test and prerequisite build admission."""

    TEST_NAME = 'V2_Integration_CUDA_GDNTensorCaptureOwnership'
    TARGET = 'v2_integration_cuda_gemm_parity'

    def configure(self, *, cuda, rocm, original=False):
        """Execute real registration/finalization and return inventory plus built markers."""
        source = (SOURCE / 'tests/v2/CMakeLists.txt').read_text()
        start = source.index('function(add_v2_test TEST_NAME)')
        end = source.index('endfunction()', start) + len('endfunction()')
        helper = source[start:end]
        start = source.index('# Exact stage-level regression for external event prejoins')
        end = source.index('if(TEST V2_Integration_ROCmNativeVNNITrainerOpenMPRuntimeBinding)', start)
        registration = source[start:end]
        if original:
            registration = (SOURCE / 'tests/v2/fixtures/cmake/OriginalCUDAGDNPreflightRegistration.cmake').read_text()
        with tempfile.TemporaryDirectory(prefix='llaminar-cuda-gdn-admission-') as directory:
            root = Path(directory)
            prefix = f'''cmake_minimum_required(VERSION 3.25)
project(CUDAGDNPreflightAdmission NONE)
enable_testing()
set(CMAKE_BUILD_TYPE Integration)
set(V2_PERF_TESTS_ONLY OFF)
set(HAVE_CUDA {'ON' if cuda else 'OFF'})
set(HAVE_ROCM {'ON' if rocm else 'OFF'})
set(V2_TEST_SOCKETS 2)
set(V2_TEST_CORES_PER_SOCKET 2)
set(V2_TEST_THREADS_PER_CORE 1)
set(DEFAULT_LOG_LEVEL ERROR)
include("{SOURCE}/tests/v2/fixtures/cmake/NativeMPISessionOwner.cmake")
add_custom_target(baseline_marker COMMAND ${{CMAKE_COMMAND}} -E touch "${{CMAKE_BINARY_DIR}}/baseline.built")
set_property(GLOBAL PROPERTY V2_UNIT_GATE_TARGETS baseline_marker)
add_test(NAME Baseline COMMAND ${{CMAKE_COMMAND}} -E true)
set_tests_properties(Baseline PROPERTIES LABELS "Integration;ProductionTestPreflight;DeviceFree")
set_property(GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_Baseline baseline_marker)
add_custom_target(cuda_marker COMMAND ${{CMAKE_COMMAND}} -E touch "${{CMAKE_BINARY_DIR}}/cuda.built")
add_executable({self.TARGET} IMPORTED GLOBAL)
set_target_properties({self.TARGET} PROPERTIES IMPORTED_LOCATION "${{CMAKE_COMMAND}}" V2_TEST_DEVICE_SCOPE FullInventory)
add_dependencies({self.TARGET} cuda_marker)
'''
            suffix = f'''
file(WRITE "${{CMAKE_BINARY_DIR}}/named-inventory.txt" "${{V2_PRODUCTION_TEST_PREFLIGHT_TESTS}}")
include("{SOURCE}/tests/v2/cmake/V2TestGateFinalization.cmake")
'''
            (root / 'CMakeLists.txt').write_text(prefix + helper + '\n' + registration + suffix)
            ninja = shutil.which('ninja')
            self.assertIsNotNone(ninja)
            configured = subprocess.run(['cmake', '-S', str(root), '-B', str(root / 'build'),
                                         '-G', 'Ninja', '-DCMAKE_MAKE_PROGRAM=' + ninja],
                                        text=True, capture_output=True, timeout=10)
            self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)
            built = subprocess.run(['cmake', '--build', str(root / 'build'), '--parallel',
                                    '--target', 'v2_production_test_preflight_gate'],
                                   text=True, capture_output=True, timeout=10)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            listed = subprocess.run(['ctest', '--test-dir', str(root / 'build'), '--show-only=json-v1'],
                                    text=True, capture_output=True, check=True, timeout=10)
            tests = [test for test in json.loads(listed.stdout)['tests'] if test['name'] != 'Baseline']
            named = (root / 'build/named-inventory.txt').read_text().split(';')
            markers = {path.stem for path in (root / 'build').glob('*.built')}
            return tests, [name for name in named if name], markers

    def test_disabled_cuda_omits_native_test_inventory_and_build(self):
        """CPU-only and ROCm-only gates cannot acquire a CUDA compilation obligation."""
        for rocm in (False, True):
            with self.subTest(rocm=rocm):
                tests, named, markers = self.configure(cuda=False, rocm=rocm)
                self.assertEqual(tests, [])
                self.assertEqual(named, [])
                self.assertEqual(markers, {'baseline'})

    def test_enabled_cuda_preserves_native_test_inventory_and_build(self):
        """CUDA-only and combined builds retain the exact filter, labels and native target."""
        for rocm in (False, True):
            with self.subTest(rocm=rocm):
                tests, named, markers = self.configure(cuda=True, rocm=rocm)
                self.assertEqual([test['name'] for test in tests], [self.TEST_NAME])
                self.assertEqual(named, [self.TEST_NAME])
                self.assertEqual(markers, {'baseline', 'cuda'})
                properties = {item['name']: item['value'] for item in tests[0]['properties']}
                self.assertTrue({'CUDA', 'ProductionTestPreflight', 'FullDeviceInventory'} <= set(properties['LABELS']))
                self.assertNotIn('DISABLED', properties)
                self.assertNotIn('SKIP_REGULAR_EXPRESSION', properties)
                self.assertIn('--gtest_filter=Test__CUDAGemmParity.GDNProjectionStageFusesCUDAQuantizedQKVAndZSmallM:'
                              'Test__CUDAGemmParity.GDNProjectionStageCapturedMixedBF16AndFP32UsesPhysicalFormatSubgroups',
                              tests[0]['command'])

    def test_original_registration_exposes_disabled_cuda_build_obligation(self):
        """The literal former body wrongly registers and builds CUDA in both disabled configurations."""
        for rocm in (False, True):
            with self.subTest(rocm=rocm):
                tests, named, markers = self.configure(cuda=False, rocm=rocm, original=True)
                self.assertEqual([test['name'] for test in tests], [self.TEST_NAME])
                self.assertEqual(named, [self.TEST_NAME])
                self.assertEqual(markers, {'baseline', 'cuda'})


class ROCmOwnerPartitionConfiguration(unittest.TestCase):
    """Execute the actual source-owned cache and definition blocks without a compiler."""

    def configure_policy(self, policy=None, rocm=True):
        """Evaluate real CMake declarations in a tiny language-free Ninja project."""
        source = (SOURCE / 'src/v2/CMakeLists.txt').read_text()
        start = source.index('set(LLAMINAR_ROCM_MOE_M1_OWNER_PARTITIONS ')
        end = source.index('set(LLAMINAR_SHARED_CORE_DEFAULT ', start)
        declaration = source[start:end]
        start = source.index('    if(LLAMINAR_ROCM_MOE_M1_OWNER_PARTITIONS STREQUAL ')
        end = source.index('\n    message(STATUS "V2: Compiled ROCm M1 owner partition', start)
        definitions = source[start:end]
        with tempfile.TemporaryDirectory(prefix='llaminar-owner-policy-') as directory:
            root = Path(directory)
            prefix = 'cmake_minimum_required(VERSION 3.25)\nproject(OwnerPolicy NONE)\n'
            prefix += 'set(HAVE_ROCM ' + ('ON' if rocm else 'OFF') + ')\n'
            if policy is not None:
                prefix += 'set(LLAMINAR_ROCM_MOE_M1_OWNER_PARTITIONS "' + policy + '" CACHE STRING "")\n'
            suffix = '\nif(HAVE_ROCM)\n' + definitions + '\nendif()\n'
            suffix += 'get_source_file_property(owner_definition kernels/rocm/moe/ROCmMoEKernel.cpp COMPILE_DEFINITIONS)\n'
            suffix += 'file(WRITE "${CMAKE_BINARY_DIR}/policy.txt" "${LLAMINAR_ROCM_MOE_M1_OWNER_PARTITIONS}\\n${owner_definition}\\n")\n'
            (root / 'CMakeLists.txt').write_text(prefix + declaration + suffix)
            ninja = shutil.which('ninja')
            self.assertIsNotNone(ninja)
            result = subprocess.run(['cmake', '-S', str(root), '-B', str(root / 'build'),
                                     '-G', 'Ninja', '-DCMAKE_MAKE_PROGRAM=' + ninja],
                                    text=True, capture_output=True, timeout=10)
            metadata = (root / 'build/policy.txt').read_text().splitlines() if result.returncode == 0 else []
            return result, metadata

    def test_default_and_explicit_choices_bind_only_the_actual_owner_source(self):
        """Fresh default and authored cache choices retain exact0/1/2 definitions."""
        for requested, expected, value in ((None, 'automatic', 2), ('automatic', 'automatic', 2),
                                            ('serial', 'serial', 0), ('four_worker_planes', 'four_worker_planes', 1)):
            with self.subTest(policy=requested):
                result, metadata = self.configure_policy(requested)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(metadata, [expected, 'LLAMINAR_ROCM_MOE_M1_OWNER_PARTITION_PLANES=' + str(value)])

    def test_invalid_or_explicit_four_without_rocm_rejects(self):
        """Invalid intent and unsupported explicit backend demand cannot configure."""
        for requested, rocm in (('bad', True), ('Four', True), ('four_worker_planes', False)):
            with self.subTest(policy=requested, rocm=rocm):
                result, _ = self.configure_policy(requested, rocm)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('require the ROCm backend' if not rocm else
                              'Invalid LLAMINAR_ROCM_MOE_M1_OWNER_PARTITIONS', result.stdout + result.stderr)
        for requested, expected in ((None, 'automatic'), ('serial', 'serial')):
            result, metadata = self.configure_policy(requested, False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(metadata, [expected, 'NOTFOUND'])

    def test_automatic_native_registration_cannot_run_as_explicit_mode(self):
        """Real conditional registration emits its exact native filter only for Automatic."""
        source = (SOURCE / 'tests/v2/CMakeLists.txt').read_text()
        entry = source.index('    add_v2_test(V2_Integration_ROCmMoEAutomaticOwnerDefault')
        start = source.rfind('\nif(', 0, entry) + 1
        end = source.index('\nendif()', entry) + len('\nendif()')
        registration = source[start:end]
        helper_start = source.index('function(add_v2_test TEST_NAME)')
        helper_end = source.index('endfunction()', helper_start) + len('endfunction()')
        for policy in ('automatic', 'serial', 'four_worker_planes'):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory(prefix='llaminar-auto-native-registration-') as directory:
                root = Path(directory)
                prefix = '''cmake_minimum_required(VERSION 3.25)
project(AutomaticNativeRegistration NONE)
enable_testing()
set(CMAKE_BUILD_TYPE Integration)
set(HAVE_ROCM ON)
set(V2_PERF_TESTS_ONLY OFF)
set(V2_TEST_SOCKETS 2)
set(V2_TEST_CORES_PER_SOCKET 2)
set(DEFAULT_LOG_LEVEL ERROR)
add_executable(v2_integration_moe_projection_pipeline IMPORTED GLOBAL)
set_target_properties(v2_integration_moe_projection_pipeline PROPERTIES IMPORTED_LOCATION "${CMAKE_COMMAND}")
'''
                prefix += 'set(LLAMINAR_ROCM_MOE_M1_OWNER_PARTITIONS ' + policy + ')\n'
                prefix += f'include("{SOURCE}/tests/v2/fixtures/cmake/NativeMPISessionOwner.cmake")\n'
                (root / 'CMakeLists.txt').write_text(prefix + source[helper_start:helper_end] + '\n' + registration)
                ninja = shutil.which('ninja')
                self.assertIsNotNone(ninja)
                result = subprocess.run(['cmake', '-S', str(root), '-B', str(root / 'build'), '-G', 'Ninja',
                                         '-DCMAKE_MAKE_PROGRAM=' + ninja], text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                tests = json.loads(subprocess.run(['ctest', '--test-dir', str(root / 'build'), '--show-only=json-v1'],
                    text=True, capture_output=True, check=True, timeout=10).stdout)['tests']
                self.assertEqual([test['name'] for test in tests],
                    ['V2_Integration_ROCmMoEAutomaticOwnerDefault'] if policy == 'automatic' else [])
                if tests:
                    self.assertIn('--gtest_filter=MoEModelPartitionBinding.AutomaticMeasured*ROCm4', tests[0]['command'])


if __name__ == '__main__':
    unittest.main()
