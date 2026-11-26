#!/usr/bin/env python3
"""Derive production witness commands, compile real x86 helpers and attribute every fused instruction.

Each witness command is the production core translation unit's actual ordered
command from CMake's compilation database. Only the resolver's ISA flags and
LLAMINAR_COMPILED_WITH_* definitions are substituted per envelope; the
separately recorded stricter setting additionally appends -ffp-contract=fast.
The unguarded negative control must fuse under the same compiler invocation;
otherwise absence of fusion in the production expert is not evidence.
"""
import argparse
import hashlib
import pathlib
import re
import shlex
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURE = ROOT / 'tests/v2/unit/kernels/cpu/CPUContractionWitness.cpp'
sys.path.insert(0, str(ROOT / 'tests/v2/cmake'))
from CPUCodegenEnvelope import command_digest, target_command  # noqa: E402

# Every packed/scalar FMA family member, including fmaddsub/fmsubadd and the
# negated forms; ordinary adds, multiplies and other vf* opcodes never match.
FUSED = re.compile(r'\bv(?:f|fn)m(?:add|sub)\w*\b')
ENVELOPES = ('AVX2', 'AVX512')
SETTINGS = ('canonical', 'stricter')
POLICIES = (('expert', 0), ('native', 1), ('negative', 1))
DEFINITION = re.compile(r'-D(LLAMINAR_COMPILED_WITH_\w+)=(.*)')


def count_fused(body):
    """Count fused multiply-add/subtract instructions in one disassembled body."""
    return len(FUSED.findall(body))


def symbol_bodies(disassembly):
    """Split objdump -d output into per-symbol instruction bodies."""
    return dict(re.findall(r'^[0-9a-f]+ <([^>]+)>:\n(.*?)(?=^[0-9a-f]+ <|\Z)',
                           disassembly, re.M | re.S))


def expected_counts():
    """Return the exact per-envelope, per-setting fused-instruction contract."""
    counts = {}
    for envelope in ENVELOPES:
        suffixes = ('AVX2',) if envelope == 'AVX2' else ('AVX2', 'AVX512')
        for setting in SETTINGS:
            counts[f'{envelope}/{setting}'] = {
                policy + suffix: count for suffix in suffixes for policy, count in POLICIES}
    return counts


def isa_option(token):
    """Classify ISA-selecting options exactly as cpu_host_compile_contract.py does."""
    return token.startswith('-m') and token != '-m64' and not token.startswith('-mcmodel=')


def derive_command(core, configured, envelope, setting, stricter):
    """Substitute one envelope into the core command; reject any ambiguous ISA input.

    The resolver's configured flags must appear exactly once as one contiguous
    block, and no other ISA option may exist: user-supplied manual ISA flags
    are never silently stripped. All other tokens keep their order.
    """
    flags = shlex.split(configured['flags'])
    if not flags:
        raise ValueError('empty configured resolver ISA contribution')
    starts = [index for index in range(len(core) - len(flags) + 1)
              if core[index:index + len(flags)] == flags]
    if len(starts) != 1:
        raise ValueError('core command does not carry the resolver ISA contribution exactly once')
    start = starts[0]
    manual = [token for index, token in enumerate(core)
              if not start <= index < start + len(flags) and isa_option(token)]
    if manual:
        raise ValueError('ambiguous manual ISA option outside the resolver contribution: '
                         + ' '.join(manual))
    declared, positions = {}, []
    for index, token in enumerate(core):
        if token == '-D' and index + 1 < len(core) and core[index + 1].startswith('LLAMINAR_COMPILED_WITH_'):
            raise ValueError('split CPU envelope definition')
        match = DEFINITION.fullmatch(token)
        if match:
            if match.group(1) in declared:
                raise ValueError('duplicated CPU envelope definition: ' + match.group(1))
            declared[match.group(1)] = match.group(2)
            positions.append(index)
    if declared != dict(item.split('=', 1) for item in configured['definitions']):
        raise ValueError('core command CPU envelope definitions differ from the resolver')
    command = []
    for index, token in enumerate(core):
        if index == start:
            command.extend(shlex.split(envelope['flags']))
        if start <= index < start + len(flags):
            continue
        if index in positions:
            if index == positions[0]:
                command.extend('-D' + item for item in envelope['definitions'])
            continue
        command.append(token)
    if setting == 'stricter':
        command.append(stricter)
    return command


def planned_commands(plan):
    """Return the database directory, core command digest and every derived witness command."""
    namespace = argparse.Namespace(compile_database=plan['compile_database'],
                                   target_source=plan['codegen_source'],
                                   compiler=plan['compiler']['path'])
    core, directory = target_command(namespace)
    if sorted(plan['envelopes']) != sorted(ENVELOPES):
        raise ValueError('contraction plan does not name every x86 envelope')
    commands = {f'{envelope}/{setting}': derive_command(core, plan['configured'],
                                                       plan['envelopes'][envelope], setting,
                                                       plan['stricter_setting'])
                for envelope in ENVELOPES for setting in SETTINGS}
    return directory, command_digest(core, directory), commands


def file_identity(path):
    """Return the resolved path and SHA256 of an executable."""
    real = pathlib.Path(path).resolve()
    return str(real), hashlib.sha256(real.read_bytes()).hexdigest()


def compiler_identity(path):
    """Measure the compiler binary and banner exactly as configure recorded them."""
    real, digest = file_identity(path)
    version = subprocess.check_output([path, '--version'], text=True)
    return dict(path=path, realpath=real, sha256=digest, version=version)


def compile_witness(directory, commands, objdump, workdir):
    """Compile every derived command and count fused instructions per witness symbol."""
    counts, evidence = {}, []
    expected = expected_counts()
    for key, command in commands.items():
        obj = pathlib.Path(workdir) / (key.replace('/', '-') + '.o')
        full = command + ['-c', str(FIXTURE), '-o', str(obj)]
        evidence.append(shlex.join(full))
        subprocess.run(full, cwd=directory, check=True)
        bodies = symbol_bodies(subprocess.check_output([objdump, '-d', str(obj)], text=True))
        counts[key] = {}
        for symbol, count in expected[key].items():
            if symbol not in bodies:
                raise RuntimeError(f'{key}: witness symbol {symbol} is missing')
            counts[key][symbol] = count_fused(bodies[symbol])
            evidence.append(f'{key} {symbol}: fused={counts[key][symbol]}, expected={count}')
    return counts, evidence
