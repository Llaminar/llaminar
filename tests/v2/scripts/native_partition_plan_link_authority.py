#!/usr/bin/env python3
"""Link authority and registration fixtures for the ordered-partition plan reporters.

CPUNativeOrderedPartitionPlan.h declares its fatal reporters out of line; the
OBJECT library llaminar2_native_partition_plan (cmake/NativePartitionPlan.cmake)
is their sole provider. Builder mode, run as build work of the Unit and
preflight gates, observes two fixture projects with the configured tools:

- link: the real authority module and the configured compiler build one header
  consumer that links the authority and the same consumer without it. The
  linked consumer must build; the unlinked one must fail on both reporters.
- registration: the real add_v2_test helper and the real
  NativeSuperblockWalkContract registrations are configured in a language-free
  project. They must register both tests as single, MPI-free processes and add
  the Integration test to the preflight inventory; the same registrations with
  the Integration test's MPI_PROCS removed must fail configuration.

The receipt binds every input's SHA256, the extracted CMake blocks and the
configure-recorded compiler identity. Verify mode, which installed runners
execute without a compiler, CMake or Ninja, re-applies every assertion to the
retained observations and executes no tool.
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tests/v2/scripts'))
from check_cpu_contraction_witness import compiler_identity  # noqa: E402

SCHEMA = 1
INPUTS = ('cmake/NativePartitionPlan.cmake',
          'src/v2/kernels/cpu/gemm/CPUNativeOrderedPartitionPlan.h',
          'src/v2/kernels/cpu/gemm/CPUNativeOrderedPartitionPlan.cpp',
          'tests/v2/scripts/native_partition_plan_link_authority.py',
          'tests/v2/fixtures/cmake/NativeMPISessionOwner.cmake',
          'tests/v2/cmake/V2TestTargetSelection.cmake')
REPORTERS = ('failSuperblockWalk', 'failOrderedPartitionPlan')
UNIT_TEST = 'V2_Unit_NativeSuperblockWalkContract'
PREFLIGHT_TEST = 'V2_Integration_NativeSuperblockWalkContract'
MULTIPLE_RANKS = f'NO_MPI test declares multiple ranks: {PREFLIGHT_TEST}'
CASES = ('link', 'registration', 'registration-without-mpi-procs')

LINK_PROJECT = r'''
cmake_minimum_required(VERSION 3.20)
project(NativePartitionPlanLinkFixture LANGUAGES CXX)
include("${AUTHORITY}")
foreach(consumer linked_consumer unlinked_consumer)
    add_executable(${consumer} consumer.cpp)
    target_compile_features(${consumer} PRIVATE cxx_std_20)
    target_include_directories(${consumer} PRIVATE "${SOURCE_ROOT}/src/v2")
endforeach()
target_link_libraries(linked_consumer PRIVATE llaminar2_native_partition_plan)
'''
# Volatile operands keep both fatal admissions, and therefore both reporters,
# referenced: the optimizer cannot prove either admission always holds.
CONSUMER = r'''
#include "kernels/cpu/gemm/CPUNativeOrderedPartitionPlan.h"
using namespace llaminar2::cpu::native_quantised;
volatile int operand = 1;
int main() {
    const int value = operand;
    requireSuperblockWalk(0, value, 8);
    return requireOrderedPartitionWidth(0, value, 8, value) - 1;
}
'''
REGISTRATION_PREFIX = '''cmake_minimum_required(VERSION 3.27)
project(NativePartitionPlanRegistrationFixture NONE)
enable_testing()
set(CMAKE_BUILD_TYPE Integration)
set(V2_PERF_TESTS_ONLY OFF)
set(V2_TEST_SOCKETS 2)
set(V2_TEST_CORES_PER_SOCKET 2)
set(DEFAULT_LOG_LEVEL ERROR)
set(Python3_EXECUTABLE "{python}")
include("{root}/tests/v2/fixtures/cmake/NativeMPISessionOwner.cmake")
include("{root}/tests/v2/cmake/V2TestTargetSelection.cmake")
add_executable(v2_unit_native_superblock_walk_contract IMPORTED GLOBAL)
set_target_properties(v2_unit_native_superblock_walk_contract PROPERTIES IMPORTED_LOCATION "${{CMAKE_COMMAND}}")
'''
REGISTRATION_SUFFIX = 'file(WRITE "${CMAKE_BINARY_DIR}/preflight.txt" "${V2_PRODUCTION_TEST_PREFLIGHT_TESTS}")\n'


def registration_blocks():
    """Extract the canonical add_v2_test helper and the walk-contract registrations."""
    source = (ROOT / 'tests/v2/CMakeLists.txt').read_text()
    start = source.index('function(add_v2_test TEST_NAME)')
    helper = source[start:source.index('endfunction()', start) + len('endfunction()')]
    start = source.index(f'add_v2_test({UNIT_TEST}')
    tail = f'list(APPEND V2_PRODUCTION_TEST_PREFLIGHT_TESTS {PREFLIGHT_TEST})'
    registration = source[start:source.index(tail, start) + len(tail)]
    return helper, registration


def without_preflight_mpi_procs(registration):
    """Remove exactly the Integration registration's MPI_PROCS declaration."""
    start = registration.index(f'add_v2_test({PREFLIGHT_TEST}')
    body = registration[start:]
    if body.count(' MPI_PROCS 1') != 1:
        raise RuntimeError(f'{PREFLIGHT_TEST} must declare MPI_PROCS 1 exactly once')
    return registration[:start] + body.replace(' MPI_PROCS 1', '', 1)


def sources():
    """Bind every input file and the two extracted CMake blocks."""
    digests = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in INPUTS}
    for name, text in zip(('add_v2_test', 'walk_contract_registration'), registration_blocks()):
        digests[name] = hashlib.sha256(text.encode()).hexdigest()
    return digests


def clean_environment():
    """Ambient compiler-flag variables would leak into the fixtures' commands."""
    return {key: value for key, value in os.environ.items()
            if key not in ('CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS')}


def run(command, environment):
    """Run one tool and keep its complete observable outcome."""
    result = subprocess.run(command, capture_output=True, text=True, env=environment)
    return dict(returncode=result.returncode, output=result.stdout + result.stderr)


def observe_link(root, tools, compiler):
    """Configure the authority fixture, then build its linked and unlinked consumers."""
    project = root / 'link'
    project.mkdir()
    (project / 'CMakeLists.txt').write_text(LINK_PROJECT)
    (project / 'consumer.cpp').write_text(CONSUMER)
    build = project / 'build'
    environment = clean_environment()
    observation = dict(configure=run([tools['cmake'], '-S', str(project), '-B', str(build), '-G', 'Ninja',
                                      '-DCMAKE_MAKE_PROGRAM=' + tools['ninja'],
                                      '-DCMAKE_CXX_COMPILER=' + compiler, '-DCMAKE_BUILD_TYPE=Release',
                                      '-DAUTHORITY=' + str(ROOT / 'cmake/NativePartitionPlan.cmake'),
                                      '-DSOURCE_ROOT=' + str(ROOT)], environment))
    if observation['configure']['returncode'] == 0:
        for consumer in ('linked_consumer', 'unlinked_consumer'):
            observation[consumer] = run([tools['ninja'], '-C', str(build), consumer], environment)
    return observation


def observe_registration(root, name, tools, helper, registration, session_helper):
    """Configure the real registrations and record CTest's generated metadata."""
    project = root / name
    project.mkdir()
    (project / 'CMakeLists.txt').write_text(
        REGISTRATION_PREFIX.format(python=sys.executable, root=ROOT) + helper + '\n' + registration + '\n'
        + REGISTRATION_SUFFIX)
    build = project / 'build'
    environment = clean_environment()
    environment['LLAMINAR_TEST_MPI_SESSION_HELPER'] = session_helper
    observation = dict(configure=run([tools['cmake'], '-S', str(project), '-B', str(build), '-G', 'Ninja',
                                      '-DCMAKE_MAKE_PROGRAM=' + tools['ninja']], environment),
                       tests=None, preflight=None)
    if observation['configure']['returncode'] == 0:
        listing = run([tools['ctest'], '--test-dir', str(build), '--show-only=json-v1'], environment)
        observation['tests'] = [dict(name=test['name'], command=test.get('command', []),
                                     labels=next((p['value'] for p in test.get('properties', [])
                                                  if p['name'] == 'LABELS'), []))
                                for test in json.loads(listing['output'])['tests']]
        observation['preflight'] = (build / 'preflight.txt').read_text().split(';')
    return observation


def require(condition, case, message):
    """Fail one case with its name."""
    if not condition:
        raise RuntimeError(f'{case}: {message}')


def verify_link(observation):
    """The authority supplies both reporters; a consumer without it cannot link."""
    case = 'link'
    require(observation['configure']['returncode'] == 0, case,
            'authority fixture configure failed:\n' + observation['configure']['output'])
    linked, unlinked = observation.get('linked_consumer'), observation.get('unlinked_consumer')
    require(linked is not None and unlinked is not None, case, 'consumer builds were not observed')
    require(linked['returncode'] == 0, case, 'consumer linking the authority failed:\n' + linked['output'])
    require(unlinked['returncode'] != 0, case, 'consumer without the authority linked its reporters')
    for reporter in REPORTERS:
        require(reporter in unlinked['output'], case, f'unlinked consumer did not fail on {reporter}')
    require('undefined' in unlinked['output'].lower(), case, 'unlinked consumer failed for another reason')


def verify_registration(observation):
    """Both registrations are single MPI-free processes; the Integration test is preflight."""
    case = 'registration'
    require(observation['configure']['returncode'] == 0, case,
            'canonical registration failed:\n' + observation['configure']['output'])
    tests = {test['name']: test for test in observation['tests'] or []}
    for name in (UNIT_TEST, PREFLIGHT_TEST):
        require(name in tests, case, name + ' is not registered')
        require('mpirun' not in tests[name]['command'], case, name + ' launches MPI')
        require('-np' not in tests[name]['command'], case, name + ' declares MPI ranks')
    require('ProductionTestPreflight' in tests[PREFLIGHT_TEST]['labels'], case,
            PREFLIGHT_TEST + ' lacks the preflight label')
    require(PREFLIGHT_TEST in (observation['preflight'] or []), case,
            PREFLIGHT_TEST + ' is absent from the preflight inventory')


def verify_missing_mpi_procs(observation):
    """Without MPI_PROCS the canonical helper rejects the MPI-free Integration registration."""
    case = 'registration-without-mpi-procs'
    require(observation['configure']['returncode'] != 0, case, 'registration without MPI_PROCS configured')
    # CMake wraps message text; compare with whitespace runs collapsed.
    require(MULTIPLE_RANKS in ' '.join(observation['configure']['output'].split()), case,
            'configure failed without the multiple-ranks diagnostic')
    require(observation['tests'] is None, case, 'rejected configuration produced tests')


def verify_report(report, identity):
    """Reject a stale, foreign or incomplete receipt; otherwise re-check every observation."""
    if report.get('schema') != SCHEMA or report.get('sources') != sources():
        raise RuntimeError('Missing, incompatible or stale native partition plan link receipt')
    if report.get('compiler') != identity:
        raise RuntimeError('Native partition plan link receipt compiler differs from the configured compiler')
    observations = report.get('observations', {})
    if list(observations) != list(CASES):
        raise RuntimeError('Native partition plan link receipt has incomplete or reordered cases')
    verify_link(observations['link'])
    verify_registration(observations['registration'])
    verify_missing_mpi_procs(observations['registration-without-mpi-procs'])


def build(args):
    """Refuse a changed compiler, observe every fixture and publish a verified receipt."""
    args.report.unlink(missing_ok=True)
    identity = json.loads(args.identity.read_text())
    if compiler_identity(identity['path']) != identity:
        raise RuntimeError('C++ compiler changed since configuration (path, binary SHA256 or --version); '
                           'reconfigure before building the native partition plan link receipt')
    tools = dict(cmake=args.cmake, ctest=args.ctest, ninja=args.ninja)
    helper, registration = registration_blocks()
    with tempfile.TemporaryDirectory(prefix='native-partition-plan-link-') as directory:
        root = pathlib.Path(directory)
        observations = dict(
            link=observe_link(root, tools, identity['path']),
            registration=observe_registration(root, 'registration', tools, helper, registration,
                                              args.session_helper))
        observations['registration-without-mpi-procs'] = observe_registration(
            root, 'registration-without-mpi-procs', tools, helper,
            without_preflight_mpi_procs(registration), args.session_helper)
    report = dict(schema=SCHEMA, sources=sources(), compiler=identity, tools=tools, observations=observations)
    verify_report(report, identity)
    temporary = args.report.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=1) + '\n')
    temporary.replace(args.report)
    print(f'Native partition plan link receipt: {len(CASES)} fixture cases verified')


def verify(args):
    """Authenticate the retained receipt against the configure-recorded identity; executes no tool."""
    if not args.report.exists():
        raise RuntimeError('Missing native partition plan link receipt: ' + str(args.report))
    verify_report(json.loads(args.report.read_text()), json.loads(args.identity.read_text()))
    print('Native partition plan link receipt authenticated')


def main():
    """Dispatch the builder or the tool-free verifier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('build', 'verify'))
    parser.add_argument('--identity', required=True, type=pathlib.Path)
    parser.add_argument('--report', required=True, type=pathlib.Path)
    parser.add_argument('--cmake')
    parser.add_argument('--ctest')
    parser.add_argument('--ninja')
    parser.add_argument('--session-helper')
    args = parser.parse_args()
    if args.mode == 'build':
        if not all((args.cmake, args.ctest, args.ninja, args.session_helper)):
            parser.error('build requires the configured cmake, ctest, ninja and MPI session helper')
        build(args)
    else:
        verify(args)


if __name__ == '__main__':
    main()
