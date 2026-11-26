#!/usr/bin/env python3
"""Generated-command fixtures for the x86 contraction proof: build-time observation, tool-free verification.

Builder mode configures (never builds) small fixture projects with the real
V2CPUContractionProof.cmake module, the canonical CPUInstructionSet.cmake
authority and the configured compiler. It records each configure's raw
observations in a source- and compiler-bound receipt. Verify mode, which Unit
runs on installed runners without a compiler SDK, re-applies every assertion
to that receipt with the current derivation code and executes no tool.
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile

from check_cpu_contraction_witness import (ENVELOPES, SETTINGS, compiler_identity, derive_command,
                                           expected_counts)

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tests/v2/cmake'))
from CPUCodegenEnvelope import target_command  # noqa: E402

SCHEMA = 1
INPUTS = ('tests/v2/cmake/V2CPUContractionProof.cmake', 'cmake/CPUInstructionSet.cmake',
          'tests/v2/scripts/cpu_contraction_generated_commands.py',
          'tests/v2/scripts/check_cpu_contraction_witness.py', 'tests/v2/cmake/CPUCodegenEnvelope.py')
PROOF_TEST = 'V2_Integration_CPUContributionContraction'
UNIT_TESTS = ('V2_Unit_CPUContractionProofAdmission', 'V2_Unit_CPUContractionProofGeneratedCommands')
QUALIFIED = tuple((isa, build_type) for isa in ('AVX2', 'AVX512') for build_type in ('Release', 'Integration'))
# name -> (generator, build type, exact status text); none registers the proof.
UNQUALIFIED = {
    'Debug': ('Ninja', 'Debug', "UNQUALIFIED for build type 'Debug'"),
    'none': ('Ninja', None, "UNQUALIFIED for build type '<none>'"),
    'multi-config': ('Ninja Multi-Config', None,
                     "UNQUALIFIED for multi-config generator 'Ninja Multi-Config'"),
}
MANUAL = 'manual-isa'
CASES = tuple(f'{isa}-{build_type}' for isa, build_type in QUALIFIED) + tuple(UNQUALIFIED) + (MANUAL,)
# Generation only: the derived commands are recorded, never compiled here.
PROJECT = r'''
cmake_minimum_required(VERSION 3.20)
project(ContractionFixture LANGUAGES CXX)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
# A user-supplied manual flag, as an ordinary CMAKE_CXX_FLAGS entry would
# appear in the production command.
set(CMAKE_CXX_FLAGS "${MANUAL_FLAGS} ${CMAKE_CXX_FLAGS}")
include("${AUTHORITY}")
llaminar_configure_cpu_instruction_set()
add_library(core SHARED core.cpp)
target_compile_features(core PUBLIC cxx_std_23)
# Stand-in for OpenMP::OpenMP_CXX: a target compile option the witness must inherit.
target_compile_options(core PRIVATE -fopenmp)
target_compile_definitions(core PUBLIC CORE_PUBLIC_DEFINITION)
set(V2_CPU_TARGET_X86 ON)
set(V2_CPU_CONTRACTION_CODEGEN_TARGET core)
set(V2_CPU_CONTRACTION_CODEGEN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/core.cpp")
enable_testing()
include("${MODULE}")
'''


def sources():
    """Bind the module, the ISA authority and every derivation/verification input."""
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in INPUTS}


def observe(root, name, tools, compiler, generator, build_type, isa, extra=()):
    """Configure one fixture tree and record its raw, tool-independent observations."""
    build = root / name
    command = [tools['cmake'], '-S', str(root), '-B', str(build), '-G', generator,
               '-DCMAKE_MAKE_PROGRAM=' + tools['ninja'], '-DCMAKE_CXX_COMPILER=' + compiler,
               '-DLLAMINAR_CPU_ISA=' + isa, '-DPython3_EXECUTABLE=' + sys.executable,
               '-DAUTHORITY=' + str(ROOT / 'cmake/CPUInstructionSet.cmake'),
               '-DMODULE=' + str(ROOT / 'tests/v2/cmake/V2CPUContractionProof.cmake')]
    if build_type is not None:
        command.append('-DCMAKE_BUILD_TYPE=' + build_type)
    # Ambient compiler-flag variables would leak into the fixture's commands.
    environment = {key: value for key, value in os.environ.items()
                   if key not in ('CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS')}
    result = subprocess.run(command + list(extra), capture_output=True, text=True, env=environment)
    observation = dict(returncode=result.returncode, output=result.stdout + result.stderr,
                       tests=[], plan=None, core_command=None)
    if result.returncode != 0:
        return observation
    listing = subprocess.run([tools['ctest'], '--test-dir', str(build), '-N'], capture_output=True,
                             text=True, check=True).stdout
    observation['tests'] = sorted(line.split(': ', 1)[1].strip() for line in listing.splitlines()
                                  if 'Test #' in line)
    plan_path = build / 'cpu_contraction/plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        observation['plan'] = plan
        observation['core_command'] = target_command(argparse.Namespace(
            compile_database=plan['compile_database'], target_source=plan['codegen_source'],
            compiler=plan['compiler']['path']))[0]
    return observation


def require(condition, case, message):
    """Fail one case with its name."""
    if not condition:
        raise RuntimeError(f'{case}: {message}')


def derived(observation):
    """Re-derive every witness command from a recorded plan and core command."""
    plan = observation['plan']
    return {f'{envelope}/{setting}': derive_command(observation['core_command'], plan['configured'],
                                                   plan['envelopes'][envelope], setting,
                                                   plan['stricter_setting'])
            for envelope in ENVELOPES for setting in SETTINGS}


def verify_qualified(case, isa, build_type, observation, identity):
    """A qualified tree records its plan and registers the proof and both Unit tests."""
    require(observation['returncode'] == 0, case, 'configure failed:\n' + observation['output'])
    plan = observation['plan']
    require(plan is not None, case, 'qualified configuration wrote no plan')
    require(plan['configured']['isa'] == isa and plan['build_type'] == build_type, case,
            'plan names another configuration')
    require(plan['compiler'] == identity, case, 'plan compiler identity differs from the receipt')
    require(sorted(plan['envelopes']) == sorted(ENVELOPES), case, 'plan does not name every envelope')
    commands = derived(observation)
    require(sorted(commands) == sorted(expected_counts()), case, 'incomplete derived commands')
    for key, command in commands.items():
        envelope, setting = key.split('/')
        flags = plan['envelopes'][envelope]['flags'].split()
        require(bool(flags) and flags[0] in command, case, key + ' lacks its envelope flags')
        start = command.index(flags[0])
        require(command[start:start + len(flags)] == flags, case, key + ' splits its envelope flags')
        for token in ('-std=gnu++23', '-fPIC', '-fopenmp', '-O3', '-DCORE_PUBLIC_DEFINITION'):
            require(token in command, case, f'{key} lacks production token {token}')
        require(('-g' in command) == (build_type == 'Integration'), case, key + ' changed debug info')
        if envelope == 'AVX2':
            require(not [t for t in command if t.startswith('-mavx512')], case, key + ' leaks AVX-512')
        require(command.count('-ffp-contract=fast') == int(setting == 'stricter'), case,
                key + ' has the wrong contraction setting')
    for test in (PROOF_TEST,) + UNIT_TESTS:
        require(test in observation['tests'], case, test + ' is not registered')


def verify_unqualified(case, status, observation):
    """An unqualified tree configures, states UNQUALIFIED and registers only the tool-free Unit tests."""
    require(observation['returncode'] == 0, case, 'configure failed:\n' + observation['output'])
    require(status in observation['output'], case, 'missing explicit UNQUALIFIED status')
    require(observation['plan'] is None, case, 'unqualified configuration wrote a plan')
    require(PROOF_TEST not in observation['tests'], case, 'unqualified configuration registered the proof')
    for test in UNIT_TESTS:
        require(test in observation['tests'], case, test + ' is not registered')


def verify_manual(observation):
    """A manual ISA flag reaches the plan's core command and the derivation rejects it."""
    require(observation['returncode'] == 0, MANUAL, 'configure failed:\n' + observation['output'])
    require(observation['plan'] is not None, MANUAL, 'qualified configuration wrote no plan')
    try:
        derived(observation)
    except ValueError as error:
        require('ambiguous manual ISA option' in str(error), MANUAL, 'wrong rejection: ' + str(error))
        return
    raise RuntimeError(MANUAL + ': manual ISA flag was accepted')


def verify_report(report, identity):
    """Reject a stale, foreign or incomplete receipt; otherwise re-check every observation."""
    if report.get('schema') != SCHEMA or report.get('sources') != sources():
        raise RuntimeError('Missing, incompatible or stale generated-command receipt')
    if report.get('compiler') != identity:
        raise RuntimeError('Generated-command receipt compiler identity differs from the configured compiler')
    observations = report.get('observations', {})
    if list(observations) != list(CASES):
        raise RuntimeError('Generated-command receipt has incomplete or reordered cases')
    for isa, build_type in QUALIFIED:
        case = f'{isa}-{build_type}'
        verify_qualified(case, isa, build_type, observations[case], identity)
    for case, (_, _, status) in UNQUALIFIED.items():
        verify_unqualified(case, status, observations[case])
    verify_manual(observations[MANUAL])


def build(args):
    """Refuse a changed compiler, configure every fixture and publish a verified receipt."""
    args.report.unlink(missing_ok=True)
    identity = json.loads(args.identity.read_text())
    if compiler_identity(identity['path']) != identity:
        raise RuntimeError('C++ compiler changed since configuration (path, binary SHA256 or '
                           '--version); reconfigure before building the generated-command receipt')
    tools = dict(cmake=args.cmake, ctest=args.ctest, ninja=args.ninja)
    observations = {}
    with tempfile.TemporaryDirectory(prefix='cpu-contraction-generated-') as directory:
        root = pathlib.Path(directory)
        (root / 'CMakeLists.txt').write_text(PROJECT)
        (root / 'core.cpp').write_text('int core;\n')
        compiler = identity['path']
        for isa, build_type in QUALIFIED:
            observations[f'{isa}-{build_type}'] = observe(
                root, f'{isa}-{build_type}', tools, compiler, 'Ninja', build_type, isa,
                ['-DCMAKE_CXX_FLAGS_INTEGRATION=-O3 -g -DNDEBUG -DLLAMINAR_ENABLE_ASSERTIONS'])
        for case, (generator, build_type, _) in UNQUALIFIED.items():
            observations[case] = observe(root, case, tools, compiler, generator, build_type, 'AVX512')
        observations[MANUAL] = observe(root, MANUAL, tools, compiler, 'Ninja', 'Release', 'AVX2',
                                       ['-DMANUAL_FLAGS=-mavx512f'])
    report = dict(schema=SCHEMA, sources=sources(), compiler=identity, tools=tools,
                  observations=observations)
    verify_report(report, identity)
    temporary = args.report.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=1) + '\n')
    temporary.replace(args.report)
    print(f'Generated-command receipt: {len(CASES)} fixture configurations verified')


def verify(args):
    """Authenticate the retained receipt against the configure-recorded identity; executes no tool."""
    if not args.report.exists():
        raise RuntimeError('Missing generated-command receipt: ' + str(args.report))
    verify_report(json.loads(args.report.read_text()), json.loads(args.identity.read_text()))
    print('Generated-command receipt authenticated')


def main():
    """Dispatch the builder or the tool-free verifier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('build', 'verify'))
    parser.add_argument('--identity', required=True, type=pathlib.Path)
    parser.add_argument('--report', required=True, type=pathlib.Path)
    parser.add_argument('--cmake')
    parser.add_argument('--ctest')
    parser.add_argument('--ninja')
    args = parser.parse_args()
    if args.mode == 'build':
        if not all((args.cmake, args.ctest, args.ninja)):
            parser.error('build requires the configured cmake, ctest and ninja')
        build(args)
    else:
        verify(args)


if __name__ == '__main__':
    main()
