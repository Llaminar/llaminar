#!/usr/bin/env python3
"""Device-free controls for the x86 contraction witness: receipt, identity, commands, matcher.

ContractionProofAdmission replays synthetic plans, compilation databases and
receipts through the actual builder/verifier helpers. ContractionProofGeneratedCommands
configures (never builds) small projects with the real CMake module and the
canonical ISA authority, then audits the commands the proof derives from them.
"""
import copy
import hashlib
import json
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile
import unittest

import cpu_contraction_proof as proof
from check_cpu_contraction_witness import (ENVELOPES, SETTINGS, count_fused, derive_command,
                                           expected_counts, planned_commands, symbol_bodies)

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = pathlib.Path(__file__).with_name('cpu_contraction_proof.py')
AVX2_FLAGS = '-msse4.1 -mavx -mavx2 -mfma -mf16c -mbmi -mbmi2 -mpopcnt'
AVX512_FLAGS = AVX2_FLAGS + ' -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni'


def definitions(avx512):
    """Return the resolver's x86 definitions for one envelope."""
    return ['LLAMINAR_COMPILED_WITH_AVX2=1', f'LLAMINAR_COMPILED_WITH_AVX512={avx512}',
            'LLAMINAR_COMPILED_WITH_NEON=0', 'LLAMINAR_COMPILED_WITH_NEON_DOTPROD=0',
            'LLAMINAR_COMPILED_WITH_NEON_I8MM=0', 'LLAMINAR_COMPILED_WITH_FP16FML=0']


ENVELOPE_PLAN = {'AVX2': {'flags': AVX2_FLAGS, 'definitions': definitions(0)},
                 'AVX512': {'flags': AVX512_FLAGS, 'definitions': definitions(1)}}
CONFIGURED = {'isa': 'AVX512', **ENVELOPE_PLAN['AVX512']}


def core_command(compiler, source, extra=()):
    """Return a representative CMake core compile command under the AVX512 envelope."""
    return ([compiler] + ['-D' + item for item in definitions(1)] + ['-DCORE_PUBLIC', '-I/include',
            '-O3', '-g', '-DNDEBUG'] + AVX512_FLAGS.split() + ['-std=gnu++23', '-fPIC', '-fopenmp']
            + list(extra) + ['-MD', '-MT', 'core.o', '-MF', 'core.o.d', '-o', 'core.o', '-c', source])


class Fixture:
    """A synthetic configured tree: stub compiler, database, plan and valid receipt."""

    def __init__(self, directory, extra=()):
        self.root = pathlib.Path(directory)
        self.compiler = self.root / 'g++'
        self.compiler.write_text('#!/bin/sh\necho "fixture g++ 14.2.0"\n')
        self.compiler.chmod(0o755)
        self.source = self.root / 'core.cpp'
        self.source.write_text('int core;\n')
        self.database = self.root / 'compile_commands.json'
        self.write_database(core_command(str(self.compiler), str(self.source), extra))
        self.plan_path = self.root / 'plan.json'
        self.report_path = self.root / 'proof.json'
        self.plan = dict(schema=1, build_type='Integration', compiler=proof.compiler_identity(
            str(self.compiler)), configured=CONFIGURED, envelopes=ENVELOPE_PLAN,
            codegen_source=str(self.source), compile_database=str(self.database),
            objdump=str(self.compiler), stricter_setting='-ffp-contract=fast')
        self.plan_path.write_text(json.dumps(self.plan))

    def write_database(self, command):
        """Publish one core row exactly as CMake's command form."""
        self.database.write_text(json.dumps([dict(directory=str(self.root), file=str(self.source),
                                                  command=shlex.join(command))]))

    def valid_report(self):
        """Return a receipt the verifier must accept."""
        _, digest, commands = planned_commands(self.plan)
        return proof.make_report(ROOT, self.plan_path, self.plan['compiler'], digest, commands,
                                 expected_counts(), dict(path='objdump'), ['evidence'])

    def verify(self, report):
        """Run the installed verifier on one receipt; return (accepted, stderr)."""
        if report is None:
            self.report_path.unlink(missing_ok=True)
        else:
            self.report_path.write_text(json.dumps(report))
        result = subprocess.run([sys.executable, str(SCRIPT), 'verify', '--plan', str(self.plan_path),
                                 '--report', str(self.report_path)], capture_output=True, text=True)
        return result.returncode == 0, result.stderr


class ContractionProofAdmission(unittest.TestCase):
    """Incomplete, stale or foreign evidence must never admit a builder proof."""

    def test_receipt_negative_controls(self):
        """Replay one valid receipt, then mutate each authenticated field."""
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            valid = fixture.valid_report()
            self.assertEqual(fixture.verify(valid), (True, ''))
            self.assertFalse(fixture.verify(None)[0])
            mutated_counts = copy.deepcopy(valid)
            mutated_counts['counts']['AVX2/canonical']['expertAVX2'] = 1
            missing_key = copy.deepcopy(valid)
            del missing_key['counts']['AVX512/stricter']
            missing_symbol = copy.deepcopy(valid)
            del missing_symbol['counts']['AVX512/canonical']['negativeAVX512']
            foreign_command = copy.deepcopy(valid)
            foreign_command['commands']['AVX2/stricter'].remove('-ffp-contract=fast')
            for name, report in [('stale sources', {**valid, 'sources': {}}),
                                 ('old schema', {**valid, 'version': 1}),
                                 ('mutated count', mutated_counts), ('missing setting', missing_key),
                                 ('missing symbol', missing_symbol), ('foreign command', foreign_command),
                                 ('foreign plan', {**valid, 'plan_sha256': '0' * 64})]:
                with self.subTest(name):
                    self.assertFalse(fixture.verify(report)[0])

    def test_same_path_compiler_upgrade_is_stale(self):
        """A same-path compiler change invalidates the receipt and refuses a rebuild."""
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            valid = fixture.valid_report()
            for field, value in (('sha256', 'f' * 64), ('version', 'fixture g++ 14.3.0\n')):
                with self.subTest(field):
                    report = copy.deepcopy(valid)
                    report['compiler'][field] = value
                    accepted, error = fixture.verify(report)
                    self.assertFalse(accepted)
                    self.assertIn('compiler identity', error)
            # Reconfigured plan after the upgrade: the retained receipt is stale.
            plan = copy.deepcopy(fixture.plan)
            plan['compiler']['sha256'] = 'e' * 64
            fixture.plan_path.write_text(json.dumps(plan))
            self.assertFalse(fixture.verify(valid)[0])
            # Upgrade in place without reconfiguring: the builder refuses before compiling.
            fixture.plan_path.write_text(json.dumps(fixture.plan))
            fixture.compiler.write_text('#!/bin/sh\necho "fixture g++ 14.2.0"\n# rebuilt\n')
            result = subprocess.run([sys.executable, str(SCRIPT), 'compile', '--plan',
                                     str(fixture.plan_path), '--report', str(fixture.report_path)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('compiler changed since configuration', result.stderr)
            self.assertFalse(fixture.report_path.exists())

    def test_core_command_change_is_stale(self):
        """Changing any retained production flag invalidates the receipt."""
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            valid = fixture.valid_report()
            command = core_command(str(fixture.compiler), str(fixture.source))
            command[command.index('-O3')] = '-O2'
            fixture.write_database(command)
            accepted, error = fixture.verify(valid)
            self.assertFalse(accepted)
            self.assertIn('core compile command changed', error)

    def test_derivation_parity_and_rejections(self):
        """Only the named ISA block, its definitions and the stricter setting may differ."""
        core = core_command('g++', 'core.cpp')
        for envelope in ENVELOPES:
            for setting in SETTINGS:
                with self.subTest(envelope=envelope, setting=setting):
                    command = derive_command(core, CONFIGURED, ENVELOPE_PLAN[envelope], setting,
                                             '-ffp-contract=fast')
                    flags = ENVELOPE_PLAN[envelope]['flags'].split()
                    start = command.index(flags[0])
                    self.assertEqual(command[start:start + len(flags)], flags)
                    if envelope == 'AVX2':
                        self.assertFalse([t for t in command if t.startswith('-mavx512')])
                    self.assertEqual([t[2:] for t in command if t.startswith('-DLLAMINAR_COMPILED_WITH_')],
                                     ENVELOPE_PLAN[envelope]['definitions'])
                    self.assertEqual(command[-1] == '-ffp-contract=fast', setting == 'stricter')
                    self.assertEqual(command.count('-ffp-contract=fast'), int(setting == 'stricter'))

                    def normal(tokens):
                        return [t for t in tokens if not t.startswith(('-m', '-DLLAMINAR_COMPILED_WITH_'))
                                and t != '-ffp-contract=fast']
                    self.assertEqual(normal(command), normal(core))
        rejections = {
            'manual ISA flag': (core_command('g++', 'core.cpp', ['-mavx512f']), 'ambiguous manual ISA'),
            'manual -march': (core_command('g++', 'core.cpp', ['-march=native']), 'ambiguous manual ISA'),
            'missing contribution': ([t for t in core if t != '-mbmi2'], 'exactly once'),
            'duplicated contribution': (core[:-2] + AVX512_FLAGS.split() + core[-2:], 'exactly once'),
            'split definition': (core[:1] + ['-D', 'LLAMINAR_COMPILED_WITH_AVX2=1'] + core[2:], 'split'),
            'foreign definition': ([t.replace('AVX512=1', 'AVX512=0') for t in core], 'differ'),
        }
        for name, (command, message) in rejections.items():
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, message):
                    derive_command(command, CONFIGURED, ENVELOPE_PLAN['AVX2'], 'canonical',
                                   '-ffp-contract=fast')

    def test_matcher_self_test(self):
        """Every FMA family form is counted per symbol; other vector opcodes are not."""
        disassembly = (
            '0000000000000000 <expertAVX2>:\n'
            '   0:\tc5 fc 58 c1          \tvaddps %ymm1,%ymm0,%ymm0\n'
            '   4:\tc5 fc 59 c1          \tvmulps %ymm1,%ymm0,%ymm0\n'
            '   8:\t62 f3 7d 28 54 c1 00 \tvfixupimmps $0x0,%ymm1,%ymm0,%ymm0\n'
            '  10:\t62 f3 7d 28 66 c1 00 \tvfpclassps $0x0,%ymm1,%k0\n'
            '0000000000000020 <nativeAVX2>:\n'
            '  20:\tc4 e2 75 b6 c2       \tvfmaddsub231ps %ymm2,%ymm1,%ymm0\n'
            '  25:\tc4 e2 75 b7 c2       \tvfmsubadd231pd %ymm2,%ymm1,%ymm0\n'
            '  2a:\tc4 e2 71 b9 c2       \tvfmadd231ss %xmm2,%xmm1,%xmm0\n'
            '  2f:\tc4 e2 f1 bf c2       \tvfnmsub231sd %xmm2,%xmm1,%xmm0\n'
            '  34:\t62 f2 75 48 bc c2    \tvfnmadd231ps %zmm2,%zmm1,%zmm0\n'
            '  3a:\tc4 e2 75 a8 c2       \tvfmadd213ps %ymm2,%ymm1,%ymm0\n')
        bodies = symbol_bodies(disassembly)
        self.assertEqual(sorted(bodies), ['expertAVX2', 'nativeAVX2'])
        self.assertEqual(count_fused(bodies['expertAVX2']), 0)
        self.assertEqual(count_fused(bodies['nativeAVX2']), 6)


class ContractionProofGeneratedCommands(unittest.TestCase):
    """Configure the real module and authority; audit the commands it derives."""

    PROJECT = r'''
cmake_minimum_required(VERSION 3.20)
project(ContractionFixture LANGUAGES CXX)
# Generation only: the derived commands are audited, never compiled here.
set(CMAKE_SYSTEM_PROCESSOR x86_64)
# A user-supplied manual flag after compiler detection, as an ordinary
# CMAKE_CXX_FLAGS entry would appear in the production command.
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

    def configure(self, root, name, build_type, isa='AVX512', extra=(), generator='Ninja'):
        """Configure one fixture tree; return (result, build directory)."""
        tools = {key: os.environ.get('LLAMINAR_TEST_' + key) for key in ('CXX', 'CMAKE', 'CTEST', 'NINJA')}
        missing = [key for key, value in tools.items() if not value]
        self.assertFalse(missing, 'configured tool paths are required: ' + ' '.join(missing))
        build = root / name
        command = [tools['CMAKE'], '-S', str(root), '-B', str(build), '-G', generator,
                   '-DCMAKE_MAKE_PROGRAM=' + tools['NINJA'], '-DCMAKE_CXX_COMPILER=' + tools['CXX'],
                   '-DLLAMINAR_CPU_ISA=' + isa, '-DPython3_EXECUTABLE=' + sys.executable,
                   '-DAUTHORITY=' + str(ROOT / 'cmake/CPUInstructionSet.cmake'),
                   '-DMODULE=' + str(ROOT / 'tests/v2/cmake/V2CPUContractionProof.cmake')]
        if build_type is not None:
            command.append('-DCMAKE_BUILD_TYPE=' + build_type)
        return subprocess.run(command + list(extra), capture_output=True, text=True), build

    def registered(self, build):
        """Return the registered CTest names of one configured tree."""
        listing = subprocess.run([os.environ['LLAMINAR_TEST_CTEST'], '--test-dir', str(build), '-N'], capture_output=True,
                                 text=True, check=True).stdout
        return {line.split(': ', 1)[1].strip() for line in listing.splitlines() if 'Test #' in line}

    def test_generated_commands(self):
        """Qualified configurations derive production commands; others are explicit."""
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / 'CMakeLists.txt').write_text(self.PROJECT)
            (root / 'core.cpp').write_text('int core;\n')
            for isa in ('AVX2', 'AVX512'):
                for build_type in ('Release', 'Integration'):
                    with self.subTest(isa=isa, build_type=build_type):
                        extra = ['-DCMAKE_CXX_FLAGS_INTEGRATION=-O3 -g -DNDEBUG -DLLAMINAR_ENABLE_ASSERTIONS']
                        result, build = self.configure(root, f'{isa}-{build_type}', build_type, isa, extra)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        plan = json.loads((build / 'cpu_contraction/plan.json').read_text())
                        self.assertEqual(plan['configured']['isa'], isa)
                        self.assertEqual(plan['compiler']['sha256'], hashlib.sha256(
                            pathlib.Path(plan['compiler']['realpath']).read_bytes()).hexdigest())
                        _, _, commands = planned_commands(plan)
                        self.assertEqual(sorted(commands), sorted(expected_counts()))
                        for key, command in commands.items():
                            envelope, setting = key.split('/')
                            flags = plan['envelopes'][envelope]['flags'].split()
                            start = command.index(flags[0])
                            self.assertEqual(command[start:start + len(flags)], flags, key)
                            for token in ('-std=gnu++23', '-fPIC', '-fopenmp', '-O3',
                                          '-DCORE_PUBLIC_DEFINITION'):
                                self.assertIn(token, command, key)
                            self.assertEqual('-g' in command, build_type == 'Integration', key)
                            if envelope == 'AVX2':
                                self.assertFalse([t for t in command if t.startswith('-mavx512')], key)
                            self.assertEqual(command.count('-ffp-contract=fast'),
                                             int(setting == 'stricter'), key)
                        tests = self.registered(build)
                        self.assertIn('V2_Integration_CPUContributionContraction', tests)
                        self.assertIn('V2_Unit_CPUContractionProofAdmission', tests)
            for build_type, label in (('Debug', 'Debug'), (None, '<none>')):
                with self.subTest(build_type=label):
                    result, build = self.configure(root, f'unqualified-{label.strip("<>")}', build_type)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(f"UNQUALIFIED for build type '{label}'", result.stdout)
                    self.assertFalse((build / 'cpu_contraction/plan.json').exists())
                    tests = self.registered(build)
                    self.assertNotIn('V2_Integration_CPUContributionContraction', tests)
                    self.assertIn('V2_Unit_CPUContractionProofAdmission', tests)
            with self.subTest('manual ISA flag'):
                result, build = self.configure(root, 'manual', 'Release', 'AVX2',
                                               ['-DMANUAL_FLAGS=-mavx512f'])
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                plan = json.loads((build / 'cpu_contraction/plan.json').read_text())
                with self.assertRaisesRegex(ValueError, 'ambiguous manual ISA option'):
                    planned_commands(plan)
            with self.subTest('multi-config'):
                result, _ = self.configure(root, 'multi', None, generator='Ninja Multi-Config')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('multi-config generator', result.stderr)


if __name__ == '__main__':
    unittest.main()
