#!/usr/bin/env python3
"""Execute optional-GPU CMake registration in Integration and Release modes.

The real helper and native topology loops run in a language-free CMake project
with imported executable placeholders. CTest's generated metadata proves both
wide-hardware skip ownership and safe omission in Release. No compiler, model,
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
    return '\n'.join((helper, rccl, native, source[start:end]))


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


if __name__ == '__main__':
    unittest.main()
