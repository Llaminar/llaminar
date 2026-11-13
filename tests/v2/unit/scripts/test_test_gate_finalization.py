#!/usr/bin/env python3
"""Prove actual CMake gate finalization includes declarations after its include.

LANGUAGES NONE fixtures build marker targets with the real production module.
They need no compiler, model, SDK or accelerator. The preserved former eager
body exposes both omitted build dependencies and unnormalized late CPU budgets.
"""
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
MODULE = ROOT / 'tests/v2/cmake/V2TestGateFinalization.cmake'
ORIGINAL = ROOT / 'tests/v2/fixtures/cmake/V2EagerTestGateFinalization.cmake'


def run(arguments):
    """Run a bounded real CMake/CTest operation, preserving a failure's diagnostic."""
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise AssertionError(' '.join(map(str, arguments)) + '\n' + result.stdout + result.stderr)
    return result.stdout


class TestGateFinalization(unittest.TestCase):
    """Exercise early/late/duplicate includes and the literal former failure."""

    def configure_and_build(self, module, *, duplicate=False, performance=False):
        """Return actual marker inventory and CTest properties after complete configure."""
        with tempfile.TemporaryDirectory(prefix='llaminar-test-gate-') as directory:
            source = Path(directory) / 'source'
            build = Path(directory) / 'build'
            source.mkdir()
            ninja = shutil.which('ninja')
            self.assertIsNotNone(ninja, 'The canonical Ninja executable must be available')
            include = 'include([=[' + str(module) + ']=])\n'
            source.joinpath('CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.24)
project(TestGateFinalization NONE)
enable_testing()
set(V2_TEST_THREADS_PER_CORE 2)
set(V2_PERF_TESTS_ONLY ''' + ('ON' if performance else 'OFF') + ''')
function(register_fixture name labels budget)
    add_custom_target(${name}_target
        COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_BINARY_DIR}/${name}.built")
    add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -E true)
    set_tests_properties(${name} PROPERTIES LABELS "${labels}")
    if(NOT "${budget}" STREQUAL "")
        set_tests_properties(${name} PROPERTIES PROCESSORS ${budget})
    endif()
    if("Unit" IN_LIST labels)
        set_property(GLOBAL APPEND PROPERTY V2_UNIT_GATE_TARGETS ${name}_target)
    endif()
    set_property(GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_${name} ${name}_target)
endfunction()
register_fixture(EarlyUnit "V2;Unit" "")
register_fixture(EarlyPreflight "V2;Integration;ProductionTestPreflight" 5)
''' + include + (include if duplicate else '') + '''
register_fixture(LateUnit "V2;Unit" 3)
register_fixture(LatePreflight "V2;Integration;ProductionTestPreflight" 7)
# Script-only entries require scheduling but correctly contribute no build target.
add_test(NAME LateScript COMMAND ${CMAKE_COMMAND} -E true)
set_tests_properties(LateScript PROPERTIES LABELS "V2;Unit;ProductionTestPreflight")
''')
            run(['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja',
                 '-DCMAKE_MAKE_PROGRAM:FILEPATH=' + ninja])
            if not performance:
                run(['cmake', '--build', str(build), '--parallel', '--target',
                     'v2_unit_gate', 'v2_production_test_preflight_gate'])
            else:
                targets = run(['cmake', '--build', str(build), '--target', 'help'])
                self.assertNotIn('v2_unit_gate', targets)
                self.assertNotIn('v2_production_test_preflight_gate', targets)
            listing = json.loads(run(['ctest', '--test-dir', str(build), '--show-only=json-v1']))
            properties = {test['name']: {p['name']: p['value'] for p in test['properties']}
                          for test in listing['tests']}
            return {path.stem for path in build.glob('*.built')}, properties

    def test_late_targets_and_cpu_budgets_are_included_exactly_once(self):
        """Actual production finalization sees the full directory after repeated includes."""
        built, properties = self.configure_and_build(MODULE, duplicate=True)
        self.assertEqual(built, {'EarlyUnit', 'LateUnit', 'EarlyPreflight', 'LatePreflight'})
        self.assertEqual(properties['EarlyUnit']['PROCESSORS'], 2)
        self.assertEqual(properties['LateUnit']['PROCESSORS'], 6)
        self.assertEqual(properties['LateScript']['PROCESSORS'], 2)
        self.assertEqual(properties['EarlyPreflight']['PROCESSORS'], 5)
        self.assertEqual(properties['LatePreflight']['PROCESSORS'], 7)

    def test_original_body_exposes_both_late_registration_defects(self):
        """The literal old implementation runs, but omits both late build targets."""
        built, properties = self.configure_and_build(ORIGINAL)
        self.assertEqual(built, {'EarlyUnit', 'EarlyPreflight'})
        self.assertEqual(set(properties), {'EarlyUnit', 'LateUnit', 'EarlyPreflight', 'LatePreflight', 'LateScript'})
        self.assertEqual(properties['EarlyUnit']['PROCESSORS'], 2)
        self.assertEqual(properties['LateUnit']['PROCESSORS'], 3)
        self.assertNotIn('PROCESSORS', properties['LateScript'])

    def test_performance_only_keeps_gate_targets_absent(self):
        """The existing performance-only exclusion survives deferred execution."""
        built, properties = self.configure_and_build(MODULE, performance=True)
        self.assertEqual(built, set())
        self.assertEqual(properties['LateUnit']['PROCESSORS'], 6)


if __name__ == '__main__':
    unittest.main()
