#!/usr/bin/env python3
"""Tool-free controls for the x86 contraction witness: receipts, identity, commands, matcher.

ContractionProofAdmission replays synthetic plans, compilation databases and
receipts through the actual builder/verifier helpers. GeneratedCommandsReceipt
replays synthetic generated-command receipts through the actual installed
verifier. No compiler, CMake or disassembler is executed; the fixture compiler
is a shell stub that only prints a banner.
"""
import copy
import json
import pathlib
import shlex
import subprocess
import sys
import tempfile
import unittest

import cpu_contraction_generated_commands as generated
import cpu_contraction_proof as proof
from check_cpu_contraction_witness import (ENVELOPES, SETTINGS, count_fused, derive_command,
                                           expected_counts, planned_commands, symbol_bodies)

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = pathlib.Path(__file__).with_name('cpu_contraction_proof.py')
GENERATED = pathlib.Path(__file__).with_name('cpu_contraction_generated_commands.py')
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


def core_command(compiler, source, extra=(), isa='AVX512', debug_info=True):
    """Return a representative CMake core compile command under one configured envelope."""
    return ([compiler] + ['-D' + item for item in ENVELOPE_PLAN[isa]['definitions']]
            + ['-DCORE_PUBLIC_DEFINITION', '-I/include', '-O3'] + (['-g'] if debug_info else [])
            + ['-DNDEBUG'] + ENVELOPE_PLAN[isa]['flags'].split() + ['-std=gnu++23', '-fPIC', '-fopenmp']
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


def synthetic_observations(identity):
    """Observations a correct fixture configure of every generated-command case records."""
    observations = {}
    for isa, build_type in generated.QUALIFIED:
        plan = dict(schema=1, build_type=build_type, compiler=identity,
                    configured={'isa': isa, **ENVELOPE_PLAN[isa]}, envelopes=ENVELOPE_PLAN,
                    codegen_source='core.cpp', compile_database='compile_commands.json',
                    objdump='objdump', stricter_setting='-ffp-contract=fast')
        core = core_command('g++', 'core.cpp', isa=isa, debug_info=build_type == 'Integration')
        observations[f'{isa}-{build_type}'] = dict(
            returncode=0, output='', plan=plan, core_command=core[:core.index('-MD')],
            tests=sorted((generated.PROOF_TEST,) + generated.UNIT_TESTS))
    for case, (_, _, status) in generated.UNQUALIFIED.items():
        observations[case] = dict(returncode=0, output='-- V2: x86 ... ' + status, plan=None,
                                  core_command=None, tests=sorted(generated.UNIT_TESTS))
    manual = copy.deepcopy(observations['AVX2-Release'])
    manual['core_command'].insert(1, '-mavx512f')
    observations[generated.MANUAL] = manual
    return observations


class GeneratedCommandsReceipt(unittest.TestCase):
    """The tool-free verifier admits only a complete, current receipt for the configured compiler."""

    def run_verify(self, root, identity, report):
        """Run the installed verifier; return (accepted, stderr)."""
        identity_path, report_path = root / 'compiler.json', root / 'generated_commands.json'
        identity_path.write_text(json.dumps(identity))
        if report is None:
            report_path.unlink(missing_ok=True)
        else:
            report_path.write_text(json.dumps(report))
        result = subprocess.run([sys.executable, str(GENERATED), 'verify', '--identity', str(identity_path),
                                 '--report', str(report_path)], capture_output=True, text=True)
        return result.returncode == 0, result.stderr

    def test_receipt_negative_controls(self):
        """Replay a valid receipt, then remove, stale or mutate each authenticated part."""
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            identity = dict(path='/usr/bin/g++', realpath='/usr/bin/g++-14', sha256='a' * 64,
                            version='g++ 14.2.0\n')
            valid = dict(schema=generated.SCHEMA, sources=generated.sources(), compiler=identity,
                         tools={}, observations=synthetic_observations(identity))
            self.assertEqual(self.run_verify(root, identity, valid), (True, ''))

            def mutated(case, field, value):
                report = copy.deepcopy(valid)
                report['observations'][case][field] = value
                return report
            multi_registered = mutated('multi-config', 'tests', sorted(
                (generated.PROOF_TEST,) + generated.UNIT_TESTS))
            multi_fatal = mutated('multi-config', 'returncode', 1)
            missing_status = mutated('Debug', 'output', '')
            leaked = copy.deepcopy(valid)
            leaked['observations']['AVX2-Release']['core_command'].append('-march=native')
            accepted_manual = mutated(generated.MANUAL, 'core_command',
                                      valid['observations']['AVX2-Release']['core_command'])
            no_openmp = copy.deepcopy(valid)
            no_openmp['observations']['AVX512-Integration']['core_command'].remove('-fopenmp')
            dropped = copy.deepcopy(valid)
            del dropped['observations']['none']
            upgraded = {**identity, 'sha256': 'b' * 64}
            for name, report, configured, message in [
                    ('missing receipt', None, identity, 'Missing generated-command receipt'),
                    ('stale sources', {**valid, 'sources': {}}, identity, 'stale'),
                    ('old schema', {**valid, 'schema': 0}, identity, 'stale'),
                    ('same-path compiler upgrade', valid, upgraded, 'compiler identity'),
                    ('foreign receipt compiler', {**valid, 'compiler': upgraded}, identity, 'compiler identity'),
                    ('dropped case', dropped, identity, 'incomplete'),
                    ('multi-config registered proof', multi_registered, identity, 'registered the proof'),
                    ('multi-config fatal', multi_fatal, identity, 'configure failed'),
                    ('missing UNQUALIFIED status', missing_status, identity, 'UNQUALIFIED'),
                    ('manual flag in qualified tree', leaked, identity, 'ambiguous manual ISA'),
                    ('manual flag accepted', accepted_manual, identity, 'was accepted'),
                    ('lost OpenMP', no_openmp, identity, '-fopenmp')]:
                with self.subTest(name):
                    accepted, error = self.run_verify(root, configured, report)
                    self.assertFalse(accepted)
                    self.assertIn(message, error)


if __name__ == '__main__':
    unittest.main()
