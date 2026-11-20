#!/usr/bin/env python3
## @file test_optional_gpu_preflight_registration.py
#  @brief Execute device-free native topology registration with the real process-owner target.
"""Execute optional-GPU CMake registration in Integration and Release modes.

The real helper and native topology loops run in a language-free CMake project
with imported executable placeholders. CTest's generated metadata proves both
physical-hardware qualification and safe omission in Release. The actual ROCm
owner-partition cache/owning-TU blocks additionally prove Automatic/explicit mode
configuration and conditional native-test registration. No compiler, model,
GPU discovery, executable launch or replacement registration helper is involved.
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
    selection = f'include("{SOURCE}/tests/v2/cmake/V2TestTargetSelection.cmake")'
    return '\n'.join((owner, selection, helper, rccl, native, source[start:end]))


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

    def test_physical_cardinality_interface_rejects_ambiguous_declarations(self):
        """Execute the real CMake builder with local, mixed and malformed contracts."""
        source = (SOURCE / 'tests/v2/CMakeLists.txt').read_text()
        start = source.index('function(add_v2_test TEST_NAME)')
        end = source.index('endfunction()', start) + len('endfunction()')
        cases = (
            ('LABELS CUDA GPU_DEVICES 2', {'CUDA', 'Devices2'}),
            ('LABELS CUDA ROCm CUDA_DEVICES 1 ROCM_DEVICES 1', {'CUDADevices1', 'ROCmDevices1', 'Devices2'}),
            ('LABELS CUDA GPU_DEVICES 0', None),
            ('LABELS CUDA GPU_DEVICES', None),
            ('LABELS CUDA ROCm GPU_DEVICES 2', None),
            ('LABELS CUDA ROCm CUDA_DEVICES 1', None),
            ('LABELS ROCm CUDA_DEVICES 2', None),
            ('LABELS CUDA HardwareOptional GPU_DEVICES 2', None),
            ('LABELS CUDA GPU_DEVICES 2 CUDA_DEVICES 2', None),
        )
        for declaration, expected in cases:
            with self.subTest(declaration=declaration), tempfile.TemporaryDirectory(prefix='llaminar-count-contract-') as directory:
                root = Path(directory)
                root.joinpath('CMakeLists.txt').write_text(f"""cmake_minimum_required(VERSION 3.25)
project(PhysicalCounts NONE)
enable_testing()
set(CMAKE_BUILD_TYPE Integration)
set(V2_PERF_TESTS_ONLY OFF)
set(V2_TEST_SOCKETS 1)
set(V2_TEST_CORES_PER_SOCKET 2)
set(DEFAULT_LOG_LEVEL ERROR)
set(Python3_EXECUTABLE "{sys.executable}")
include("{SOURCE}/tests/v2/fixtures/cmake/NativeMPISessionOwner.cmake")
include("{SOURCE}/tests/v2/cmake/V2TestTargetSelection.cmake")
""" + source[start:end] + f'\nadd_v2_test(V2_Integration_Count COMMAND "${{CMAKE_COMMAND}}" {declaration} MPI_PROCS 1 NO_MPI NO_MODELS)\n')
                configured = subprocess.run(['cmake', '-S', str(root), '-B', str(root / 'build'),
                    '-G', 'Ninja', '-DCMAKE_MAKE_PROGRAM=' + shutil.which('ninja')],
                    capture_output=True, text=True, timeout=10)
                if expected is None:
                    self.assertNotEqual(configured.returncode, 0)
                    self.assertIn('CMake Error', configured.stderr)
                    continue
                self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)
                tests = json.loads(subprocess.check_output(['ctest', '--test-dir', str(root / 'build'),
                    '--show-only=json-v1'], text=True, timeout=10))['tests']
                labels = next(prop['value'] for prop in tests[0]['properties'] if prop['name'] == 'LABELS')
                self.assertTrue(expected | {'HardwareOptional', 'LocalHostDevices'} <= set(labels))

    def test_representative_identity_matrices_preserve_pairwise_and_boundary_coverage(self):
        """Execute native registration and prove all interaction pairs and exact thresholds."""
        from itertools import combinations, product
        source = (SOURCE / 'tests/v2/CMakeLists.txt').read_text()
        start = source.index('function(add_v2_test TEST_NAME)')
        end = source.index('endfunction()', start) + len('endfunction()')
        with tempfile.TemporaryDirectory(prefix='llaminar-identity-matrix-') as directory:
            root = Path(directory)
            root.joinpath('CMakeLists.txt').write_text(f"""cmake_minimum_required(VERSION 3.25)
project(IdentityMatrix NONE)
enable_testing()
set(CMAKE_BUILD_TYPE Integration)
set(V2_PERF_TESTS_ONLY OFF)
set(V2_TEST_SOCKETS 1)
set(V2_TEST_CORES_PER_SOCKET 2)
set(DEFAULT_LOG_LEVEL ERROR)
set(HAVE_CUDA ON)
set(HAVE_ROCM ON)
set(Python3_EXECUTABLE "{sys.executable}")
add_executable(v2_test_qwen4exp_device_primitives IMPORTED GLOBAL)
set_target_properties(v2_test_qwen4exp_device_primitives PROPERTIES IMPORTED_LOCATION "${{CMAKE_COMMAND}}")
include("{SOURCE}/tests/v2/fixtures/cmake/NativeMPISessionOwner.cmake")
include("{SOURCE}/tests/v2/cmake/V2TestTargetSelection.cmake")
""" + source[start:end] + f'\ninclude("{SOURCE}/tests/v2/cmake/V2PooledExactIdentityNativeTests.cmake")\n')
            configured = subprocess.run(['cmake', '-S', str(root), '-B', str(root / 'build'),
                '-G', 'Ninja', '-DCMAKE_MAKE_PROGRAM=' + shutil.which('ninja')],
                text=True, capture_output=True, timeout=10)
            self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)
            tests = json.loads(subprocess.check_output(['ctest', '--test-dir', str(root / 'build'),
                '--show-only=json-v1'], text=True, timeout=10))['tests']
        self.assertEqual(len(tests), 64)  # formerly 88; no formats or dispatch thresholds removed
        for backend in ('CUDA', 'ROCm'):
            names = [test['name'].split(f'{backend}_PooledImmutable_')[1]
                     for test in tests if f'{backend}_PooledImmutable_' in test['name']]
            for family, axes in (
                    ('CooperativeRejection', (('Pin', 'Free', 'Victim'), ('65', '257'), ('Row32', 'Row64'))),
                    ('Record', (('PinSlotBounds', 'PinOrdinal', 'PinCommitFlags'), ('Early', 'Late'), ('Row32', 'Row64')))):
                rows = [name.split('_')[1:] for name in names if name.startswith(family + '_')]
                self.assertEqual(len(rows), 6)
                for first, second in combinations(range(3), 2):
                    self.assertEqual({(row[first], row[second]) for row in rows},
                                     set(product(axes[first], axes[second])))
            boundaries = {str(n) for n in (63, 64, 65, 127, 128, 129, 255, 256, 257, 513)}
            for trace in ('Pin', 'RetainedPinsAndAllocation'):
                self.assertEqual({name.split('_')[2] for name in names
                                  if name.startswith('ExactIdentity_' + trace + '_')}, boundaries)
        for test in tests:
            labels = next(prop['value'] for prop in test['properties'] if prop['name'] == 'LABELS')
            self.assertIn('ProductionTestPreflight', labels)
            self.assertEqual(sum(arg.startswith('--gtest_filter=') for arg in test['command']), 1)

    def test_integration_preserves_each_native_cell_and_exact_hardware_qualification(self):
        """Every missing physical geometry is qualified before launch; native bodies remain strict."""
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
                if 'RCCLUnusedRetirement_' in test['name']:
                    self.assertTrue({'HardwareOptional', 'LocalHostDevices'} <= labels)
                    self.assertNotIn('SKIP_REGULAR_EXPRESSION', properties)
                else:
                    self.assertTrue({'HardwareOptional', 'DistinctNUMARanks', 'CUDADevices4', 'ROCmDevices4'} <= labels)
                    self.assertNotIn('SKIP_REGULAR_EXPRESSION', properties)
            else:
                self.assertNotIn('SKIP_REGULAR_EXPRESSION', properties)
        self.assertEqual(optional, 14)

    def test_addressed_optional_geometry_keeps_all_original_strict_cases(self):
        """Execute the actual addressed registration block; optionality never disables or skips its body."""
        source = (SOURCE / 'tests/v2/CMakeLists.txt').read_text()
        start = source.index('function(add_v2_test TEST_NAME)')
        end = source.index('endfunction()', start) + len('endfunction()')
        helper = source[start:end]
        module = (SOURCE / 'tests/v2/cmake/V2NativeCountedBroadcastTests.cmake').read_text()
        start = module.index('# Addressed original-communicator cases')
        end = module.index('# Original connector observations', start)
        with tempfile.TemporaryDirectory(prefix='llaminar-addressed-eligibility-') as directory:
            root = Path(directory)
            owner = SOURCE / 'tests/v2/fixtures/cmake/NativeMPISessionOwner.cmake'
            selection = SOURCE / 'tests/v2/cmake/V2TestTargetSelection.cmake'
            root.joinpath('CMakeLists.txt').write_text(f"""cmake_minimum_required(VERSION 3.25)
project(AddressedEligibility NONE)
enable_testing()
set(CMAKE_BUILD_TYPE Integration)
set(V2_PERF_TESTS_ONLY OFF)
set(V2_TEST_SOCKETS 2)
set(V2_TEST_CORES_PER_SOCKET 2)
set(DEFAULT_LOG_LEVEL ERROR)
set(Python3_EXECUTABLE "{sys.executable}")
set(HAVE_CUDA ON)
set(HAVE_ROCM ON)
set(HAVE_NCCL ON)
set(HAVE_RCCL ON)
add_executable(v2_integration_captured_transfer_channel IMPORTED GLOBAL)
set_target_properties(v2_integration_captured_transfer_channel PROPERTIES IMPORTED_LOCATION "${{CMAKE_COMMAND}}")
include("{owner}")
include("{selection}")
""" + helper + module[start:end])
            ninja = shutil.which('ninja')
            self.assertIsNotNone(ninja)
            subprocess.run(['cmake', '-S', str(root), '-B', str(root / 'build'), '-G', 'Ninja',
                '-DCMAKE_MAKE_PROGRAM=' + ninja], check=True, capture_output=True, text=True, timeout=10)
            listing = subprocess.check_output(['ctest', '--test-dir', str(root / 'build'),
                '--show-only=json-v1'], text=True, timeout=10)
            tests = json.loads(listing)['tests']
        self.assertEqual(len(tests), 12)
        optional = set()
        for test in tests:
            props = {item['name']: item['value'] for item in test['properties']}
            labels = set(props['LABELS'])
            self.assertIn('ProductionTestPreflight', labels)
            self.assertNotIn('DISABLED', props)
            self.assertNotIn('SKIP_REGULAR_EXPRESSION', props)
            self.assertNotIn('SKIP_RETURN_CODE', props)
            self.assertTrue({'HardwareOptional', 'LocalHostDevices'} <= labels)
            self.assertTrue({'Devices2', 'Devices4', 'Devices8'} & labels)
            optional.add(test['name'])
        self.assertEqual(len(optional), 12)

    def test_original_mapped_components_partition_exact_native_cohorts(self):
        """Execute the real per-body registration, retaining honest local mixed eligibility."""
        source = (SOURCE / 'tests/v2/CMakeLists.txt').read_text()
        helper_start = source.index('function(add_v2_test TEST_NAME)')
        helper_end = source.index('endfunction()', helper_start) + len('endfunction()')
        marker = source.index('# BEGIN original mapped component cohort registration')
        begin = source.index('    set(_original_component_cases', marker)
        end = source.index('# END original mapped component cohort registration', begin)
        # Source ownership is tested by the enclosing real target_sources block;
        # imported placeholders execute only the executable registration itself.
        registration = source[begin:end].rsplit('endif()', 1)[0]
        self.assertNotIn('v2_integration_mapped_activation_packet_cuda_rocm', source)
        with tempfile.TemporaryDirectory(prefix='llaminar-original-components-') as directory:
            root = Path(directory)
            prefix = f"""cmake_minimum_required(VERSION 3.25)
project(OriginalComponents NONE)
enable_testing()
set(CMAKE_BUILD_TYPE Integration)
set(V2_PERF_TESTS_ONLY OFF)
set(V2_TEST_SOCKETS 2)
set(V2_TEST_CORES_PER_SOCKET 2)
set(DEFAULT_LOG_LEVEL ERROR)
set(Python3_EXECUTABLE "{sys.executable}")
add_executable(v2_integration_captured_transfer_channel IMPORTED GLOBAL)
set_target_properties(v2_integration_captured_transfer_channel PROPERTIES IMPORTED_LOCATION "${{CMAKE_COMMAND}}")
include("{SOURCE}/tests/v2/fixtures/cmake/NativeMPISessionOwner.cmake")
include("{SOURCE}/tests/v2/cmake/V2TestTargetSelection.cmake")
"""
            (root / 'CMakeLists.txt').write_text(prefix + source[helper_start:helper_end] + '\n' + registration)
            ninja = shutil.which('ninja')
            self.assertIsNotNone(ninja)
            result = subprocess.run(['cmake', '-S', str(root), '-B', str(root / 'build'), '-G', 'Ninja',
                '-DCMAKE_MAKE_PROGRAM=' + ninja], text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            tests = json.loads(subprocess.check_output(['ctest', '--test-dir', str(root / 'build'),
                '--show-only=json-v1'], text=True, timeout=10))['tests']
        self.assertEqual(len(tests), 25)
        filters = set()
        local_mixed = []
        return_cases = []
        prefill_cases = {
            'V2_Integration_CanonicalRoutes_CUDA2_Prefill': 'CUDA',
            'V2_Integration_CanonicalRoutes_ROCm2_Prefill': 'ROCm',
        }
        for test in tests:
            props = {item['name']: item['value'] for item in test['properties']}
            labels = set(props['LABELS'])
            self.assertTrue({'ProductionTestPreflight', 'HardwareOptional', 'ProcessOwned'} <= labels)
            self.assertNotIn('DISABLED', props)
            self.assertNotIn('SKIP_REGULAR_EXPRESSION', props)
            self.assertNotIn('SKIP_RETURN_CODE', props)
            command = test['command']
            self.assertIn('-np', command)
            self.assertEqual(command[command.index('-np') + 1], '2')
            exact = [arg for arg in command if arg.startswith('--gtest_filter=')]
            self.assertEqual(len(exact), 1)
            self.assertNotIn(exact[0], filters)
            filters.add(exact[0])
            if {'CUDA', 'ROCm'} <= labels:
                local_mixed.append(test)
                self.assertTrue({'Devices2', 'CUDADevices1', 'ROCmDevices1'} <= labels)
                self.assertNotIn('DistinctNUMARanks', labels)
            if 'ReturnValidation_' in test['name']:
                return_cases.append(test)
                self.assertIn('Devices1', labels)
                self.assertIn('/MoEOverlayReturnValidation.', exact[0])
            elif 'TicketAckRace_' in test['name']:
                self.assertIn('Devices1', labels)
            else:
                self.assertIn('Devices2', labels)
            if test['name'] in prefill_cases:
                backend = prefill_cases.pop(test['name'])
                self.assertIn(backend, labels)
                self.assertEqual(props['TIMEOUT'], 120)
                self.assertEqual(exact[0], '--gtest_filter=Test__MappedActivationPacketCUDAAndROCm.'
                                 f'{backend}SparseCanonicalRoutesRetainPrefillBytesAcrossTwentyEpochs')
        self.assertFalse(prefill_cases, prefill_cases)
        self.assertEqual(len(local_mixed), 2)
        self.assertEqual(len(return_cases), 8)
        self.assertNotIn('V2_Integration_MappedActivationPacket_CUDA_ROCm', {test['name'] for test in tests})
        self.assertNotIn('V2_Integration_CanonicalRoutePrefillReplay', {test['name'] for test in tests})
        # Both original metadata bodies remain covered by the real device-free alias.
        source_map = source[source.index('add_v2_test(V2_Integration_CPUJointRecordedSourceMapContract'):]
        source_map = source_map[:source_map.index('list(APPEND')]
        self.assertIn('JointRecordedSourceMapEpochAndZeroHoleContract', source_map)
        self.assertIn('JointSourceFixtureHasCompleteOriginalColdPlacement', source_map)
        self.assertIn('DeviceFree', source_map)
        self.assertIn('NO_MPI', source_map)

    def test_retained_mi50_registration_matches_native_preflight_scheduler(self):
        """Run the production scheduler over native mandatory and optional CTest rows."""
        source = (SOURCE / 'tests/v2/CMakeLists.txt').read_text()
        begin = source.index('function(add_v2_test TEST_NAME)')
        end = source.index('endfunction()', begin) + len('endfunction()')
        helper = source[begin:end]
        begin = source.index('        add_v2_test(V2_Integration_MoEGroupedFixedFP16RouterReplay_ROCm')
        end = source.index('        add_v2_test(', begin + len('        add_v2_test('))
        fixed_router = source[begin:end]
        begin = source.index('        foreach(case IN ITEMS Default Off Portable)')
        end = source.index('        endforeach()', begin) + len('        endforeach()')
        hc_down = source[begin:end]
        begin = source.index('        foreach(peer_push_case IN ITEMS Replay')
        end = source.index('        # Use the original rank-owned native fixture', begin)
        peer_push = source[begin:end]
        begin = source.index('foreach(planning_sample_backend IN ITEMS CUDA ROCm)')
        end = source.index('\nendforeach()', begin) + len('\nendforeach()')
        planning = source[begin:end]
        modules = SOURCE / 'tests/v2/cmake'
        retained = (modules / 'V2RetainedM1HCFFNTests.cmake').read_text()
        begin = retained.index('    add_v2_test(V2_Integration_ROCmGroupedFFNFormatProgramDemand')
        end = retained.index('\nendif()\n\nif(NOT LLAMINAR_QFN_RETAINED_TP_GGUF', begin)
        numerics = (modules / 'V2ROCmRetainedM1NumericsTests.cmake').read_text()
        with tempfile.TemporaryDirectory(prefix='llaminar-retained-native-registration-') as work:
            root = Path(work)
            prefix = f'''cmake_minimum_required(VERSION 3.25)
project(RetainedNativePreflight NONE)
enable_testing()
set(CMAKE_BUILD_TYPE Integration)
set(V2_PERF_TESTS_ONLY OFF)
set(V2_TEST_SOCKETS 2)
set(V2_TEST_CORES_PER_SOCKET 2)
set(HAVE_CUDA ON)
set(HAVE_ROCM ON)
set(DEFAULT_LOG_LEVEL ERROR)
set(Python3_EXECUTABLE "{sys.executable}")
include("{SOURCE}/tests/v2/fixtures/cmake/NativeMPISessionOwner.cmake")
include("{modules}/V2TestTargetSelection.cmake")
'''
            for target in ('v2_integration_moe_intermediate_ffn_finalization',
                           'v2_integration_rocm_retained_m1_numerics', 'v2_integration_moe_live_rows',
                           'v2_test_qwen4exp_device_primitives', 'v2_integration_rocm_peer_push',
                           'v2_integration_planning_mpi_transfers'):
                prefix += f'add_executable({target} IMPORTED GLOBAL)\n'
                prefix += f'set_target_properties({target} PROPERTIES IMPORTED_LOCATION "${{CMAKE_COMMAND}}")\n'
            # Only executable placeholders; each add_v2_test call and its MPI
            # wrapping run exactly as in the actual configured Integration tree.
            (root / 'CMakeLists.txt').write_text('\n'.join(
                (prefix, helper, fixed_router, hc_down, peer_push, planning, retained[begin:end],
                 numerics[numerics.index('set(qfn_wave_address_test '):])))
            configured = subprocess.run(['cmake', '-S', str(root), '-B', str(root / 'build'), '-G', 'Ninja'],
                                        capture_output=True, text=True, timeout=10)
            self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)
            sys.path.insert(0, str(SOURCE / 'scripts/ci'))
            try:
                import run_production_parity_campaigns as campaigns
                rows = campaigns.discover_production_test_preflight_registrations(root / 'build')
                requirements = campaigns.preflight_optional_hardware_requirements(rows)
            finally:
                sys.path.pop(0)
            document = subprocess.check_output(['ctest', '--test-dir', str(root / 'build'),
                '--show-only=json-v1'], text=True, timeout=10)

        singleton = {'backend': 'rocm', 'degree': 1, 'locality': 'original-numa-local-rank',
                     'native_isa': 'gfx906'}
        retained_replays = {name for name, req in requirements.items() if req == singleton}
        self.assertTrue({
            'V2_Integration_ROCmGroupedFFNFormatProgramDemand',
            'V2_Integration_FullNativeHCFFN_EqualF160GroupedFrontierDefaultReplay_ROCm',
            'V2_Integration_QFNFixedMTPConditionRowReplay',
        } <= retained_replays)
        names = {row.name for row in rows}
        self.assertTrue({
            'V2_Integration_MoEGroupedFixedFP16RouterReplay_ROCm',
            'V2_Integration_ROCmRetainedM1Numerics',
            'V2_Integration_ROCmGroupedVerifierWaveAddressReplay',
            'V2_Integration_ROCmGroupedHCDownReducePublishDefault',
            'V2_Integration_ROCmGroupedHCDownReducePublishOff',
        } <= names - requirements.keys())

        def hardware(rocm_nodes, rocm_isas, peer_access):
            return {
                'qfn_source_hardware_schema': 2, 'scope': 'local-host-all-ranks',
                'devices': [
                    {'backend': 'cpu', 'ordinal': 0, 'numa_node': node} for node in (0, 1)
                ] + [
                    {'backend': 'rocm', 'ordinal': ordinal, 'numa_node': numa,
                     'native_isa': isa, 'uuid': f'rocm-{ordinal}'}
                    for ordinal, (numa, isa) in enumerate(zip(rocm_nodes, rocm_isas, strict=True))
                ] + [{'backend': 'cuda', 'ordinal': 0, 'numa_node': -1, 'uuid': 'cuda-0'}],
                'directed_peer_access': [
                    {'backend': 'rocm', 'accessor': a, 'peer': b, 'available': (a, b) in peer_access}
                    for a in range(len(rocm_nodes)) for b in range(len(rocm_nodes)) if a != b
                ],
            }

        def qualified(observation):
            return {name for name, requirement in requirements.items()
                    if campaigns.optional_native_cohort_absence(requirement, observation) is None}

        peer = 'V2_Integration_ROCmPeerPushCapturedReplay'
        portable = 'V2_Integration_ROCmGroupedHCDownReducePublishPortable'
        planning_rocm = 'V2_Integration_AutomaticPlanningNativeContextHandoff_ROCm'
        planning_cuda = 'V2_Integration_AutomaticPlanningNativeContextHandoff_CUDA'
        self.assertFalse(qualified(hardware([-1], ['gfx906:sramecc+:xnack-'], set())))
        self.assertEqual(qualified(hardware([0], ['gfx906'], set())), retained_replays)
        self.assertEqual(qualified(hardware([-1], ['gfx1100'], set())), {portable})
        one_way = hardware([0, 0], ['gfx906'] * 2, {(0, 1)})
        self.assertEqual(qualified(one_way), retained_replays | {planning_rocm})
        two_way = hardware([0, 0], ['gfx906'] * 2, {(0, 1), (1, 0)})
        self.assertIn(peer, qualified(two_way))
        self.assertNotIn(planning_cuda, qualified(two_way))
        self.assertNotIn(peer, qualified(hardware([0, 1], ['gfx906'] * 2, {(0, 1), (1, 0)})))
        replacement = 'V2_Integration_ROCmPeerPushAutomaticOwnerReplacement'
        links = {(a, b) for a in range(1, 5) for b in range(1, 5) if a != b}
        self.assertIn(replacement, qualified(hardware([0] * 5, ['gfx906'] * 5, links)))
        self.assertNotIn(replacement, qualified(hardware([0] * 5, ['gfx906'] * 5, links - {(1, 4)})))
        filters = [arg for row in json.loads(document)['tests']
                   if row['name'].startswith('V2_Integration_GroupedM6IndexedGateReuse')
                   for arg in row['command'] if arg.startswith('--gtest_filter=')]
        self.assertEqual(len(filters), 2)
        self.assertEqual({case for filt in filters for case in filt.removeprefix('--gtest_filter=').split(':')}, {
            'OriginalSources/GroupedM6Family.Deferred/G09_*', 'OriginalSources/GroupedM6Family.Static/G10_*',
            'OriginalSources/GroupedM6Family.Static/G11_*', 'OriginalSources/GroupedM6Family.Deferred/G12_*',
        })

    def test_release_omits_native_tests_and_their_properties_together(self):
        """A performance-only build cannot attach metadata to nonexistent tests."""
        self.assertEqual(self.configure('Release'), [])



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
                prefix += f'include("{SOURCE}/tests/v2/cmake/V2TestTargetSelection.cmake")\n'
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
