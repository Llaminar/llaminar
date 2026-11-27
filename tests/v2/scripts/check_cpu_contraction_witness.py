#!/usr/bin/env python3
"""Derive production witness commands, compile real x86 helpers and attribute every fused instruction.

Each witness command is the production core translation unit's actual ordered
command from CMake's compilation database. Only the resolver's ISA flags and
LLAMINAR_COMPILED_WITH_* definitions are substituted per envelope; the
separately recorded stricter setting additionally appends -ffp-contract=fast.
The unguarded negative control must fuse under the same compiler invocation;
otherwise absence of fusion in the production expert is not evidence.

The packed-prefill attribution witness compiles the whole real
CPUNativeX86PackedPrefill.cpp translation unit with its own production command
per envelope, canonically and with -ffp-contract=off. Under -ffp-contract=off
GCC forms no FMA from a separate multiply and add, so every fused instruction it
emits comes from an explicit source FMA (_mm256_fmadd_ps etc.). A canonical
fused instruction is therefore attributed to source exactly when its function
body is instruction-identical to the -ffp-contract=off body; any other body is a
compiler-introduced contraction and fails the proof with its symbol. A control
fixture compiled with the same commands proves the comparison detects a
contracted _mm256_mul_ps/_mm256_add_ps pair.
"""
import argparse
import hashlib
import json
import pathlib
import re
import shlex
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURE = ROOT / 'tests/v2/unit/kernels/cpu/CPUContractionWitness.cpp'
ATTRIBUTION_CONTROL = ROOT / 'tests/v2/unit/kernels/cpu/CPUContractionAttributionControl.cpp'
sys.path.insert(0, str(ROOT / 'tests/v2/cmake'))
from CPUCodegenEnvelope import command_digest, target_command  # noqa: E402

# Every packed/scalar FMA family member, including fmaddsub/fmsubadd and the
# negated forms; ordinary adds, multiplies and other vf* opcodes never match.
FUSED = re.compile(r'\bv(?:f|fn)m(?:add|sub)\w*\b')
ENVELOPES = ('AVX2', 'AVX512')
SETTINGS = ('canonical', 'stricter')
POLICIES = (('expert', 0), ('native', 1), ('negative', 1))
DEFINITION = re.compile(r'-D(LLAMINAR_COMPILED_WITH_\w+)=(.*)')
ATTRIBUTION_SETTINGS = ('canonical', 'contract-off')
CONTRACT_OFF = '-ffp-contract=off'
# The ordered-panel family's noinline entry points; each must appear as at least
# one instantiated symbol (mangled within x86_panel's anonymous namespace).
PREFILL_FUNCTIONS = ('orderedPanel', 'stripeOrdered', 'run', 'pooledChunk')
PREFILL_SCOPE = '9x86_panel12_GLOBAL__N_1'
INSTRUCTION_ADDRESS = re.compile(r'^\s*[0-9a-f]+:\s*')
BRANCH_TARGET = re.compile(r'\b[0-9a-f]+ <')


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


def attribution_command(core, configured, envelope, setting):
    """Derive one packed-prefill attribution command: canonical, or with -ffp-contract=off appended."""
    if setting not in ATTRIBUTION_SETTINGS:
        raise ValueError('unknown attribution setting: ' + setting)
    command = derive_command(core, configured, envelope, 'canonical', None)
    return command + ([CONTRACT_OFF] if setting == 'contract-off' else [])


def planned_attribution_commands(plan):
    """Return the database directory, packed-prefill command digest and every attribution command."""
    namespace = argparse.Namespace(compile_database=plan['compile_database'],
                                   target_source=plan['prefill_source'],
                                   compiler=plan['compiler']['path'])
    core, directory = target_command(namespace)
    commands = {f'{envelope}/{setting}': attribution_command(core, plan['configured'],
                                                             plan['envelopes'][envelope], setting)
                for envelope in ENVELOPES for setting in ATTRIBUTION_SETTINGS}
    return directory, command_digest(core, directory), commands


def normalized_body(body):
    """Instruction lines of one symbol without addresses, so identical code compares equal."""
    return [BRANCH_TARGET.sub('<', INSTRUCTION_ADDRESS.sub('', line)).strip()
            for line in body.splitlines() if INSTRUCTION_ADDRESS.match(line)]


def unattributed_symbols(canonical, contract_off):
    """Symbols whose canonical fused instructions are not all explicit source FMAs.

    A canonical body without fused instructions contains no contraction. A body
    with fused instructions is attributed to source only when -ffp-contract=off
    produces the identical instruction sequence for the same symbol.
    """
    return sorted(symbol for symbol, body in canonical.items()
                  if count_fused(body) and (symbol not in contract_off
                                            or normalized_body(body) != normalized_body(contract_off[symbol])))


def expected_attribution_control():
    """Per envelope, the control symbols the attribution must flag and the source FMAs it must keep."""
    control = {}
    for envelope in ENVELOPES:
        suffixes = ('AVX2',) if envelope == 'AVX2' else ('AVX2', 'AVX512')
        control[envelope] = dict(unattributed=['attributionContracted' + suffix for suffix in suffixes],
                                 source_fused={'attributionSource' + suffix: 1 for suffix in suffixes})
    return control


def require_attribution(attribution):
    """Reject incomplete evidence, undetected controls, uncovered functions or compiler-introduced FMAs."""
    if sorted(attribution) != sorted(ENVELOPES):
        raise RuntimeError('packed-prefill attribution does not cover every x86 envelope')
    control = expected_attribution_control()
    for envelope in ENVELOPES:
        result = attribution[envelope]
        if result.get('control') != control[envelope]:
            raise RuntimeError(f'{envelope}: attribution control was not detected exactly: '
                               f'{result.get("control")}')
        prefill = result.get('prefill', {})
        missing = [name for name in PREFILL_FUNCTIONS if not prefill.get('functions', {}).get(name)]
        if sorted(prefill.get('functions', {})) != sorted(PREFILL_FUNCTIONS) or missing:
            raise RuntimeError(f'{envelope}: packed-prefill functions without disassembled symbols: {missing}')
        if not prefill.get('symbols'):
            raise RuntimeError(f'{envelope}: no disassembled CPUNativeX86PackedPrefill.cpp symbols')
        if prefill.get('unattributed') != []:
            raise RuntimeError(f'{envelope}: compiler-introduced contraction in CPUNativeX86PackedPrefill.cpp: '
                               + ', '.join(prefill.get('unattributed', ['<missing evidence>'])))


def disassembled_bodies(command, source, objdump, directory, obj):
    """Compile one source with one command and return its per-symbol bodies and the full command."""
    full = command + ['-c', str(source), '-o', str(obj)]
    subprocess.run(full, cwd=directory, check=True)
    disassembly = subprocess.check_output([objdump, '-d', '--no-show-raw-insn', str(obj)], text=True)
    return symbol_bodies(disassembly), shlex.join(full)


def compile_attribution(directory, commands, prefill_source, objdump, workdir):
    """Compile the real packed-prefill TU and the control under both settings; attribute every FMA."""
    attribution, evidence = {}, []
    for envelope in ENVELOPES:
        bodies = {}
        for name, source in (('prefill', prefill_source), ('control', ATTRIBUTION_CONTROL)):
            for setting in ATTRIBUTION_SETTINGS:
                obj = pathlib.Path(workdir) / f'attribution-{envelope}-{name}-{setting}.o'
                bodies[name, setting], full = disassembled_bodies(commands[f'{envelope}/{setting}'], source,
                                                                  objdump, directory, obj)
                evidence.append(full)
        prefill = bodies['prefill', 'canonical']
        functions = {name: sum(f'{PREFILL_SCOPE}{len(name)}{name}I' in symbol for symbol in prefill)
                     for name in PREFILL_FUNCTIONS}
        control = bodies['control', 'canonical']
        attribution[envelope] = dict(
            prefill=dict(symbols=len(prefill), fused=sum(count_fused(body) for body in prefill.values()),
                         functions=functions,
                         unattributed=unattributed_symbols(prefill, bodies['prefill', 'contract-off'])),
            control=dict(unattributed=unattributed_symbols(control, bodies['control', 'contract-off']),
                         source_fused={symbol: count_fused(body) for symbol, body in control.items()
                                       if symbol.startswith('attributionSource')}))
        evidence.append(f'{envelope} packed-prefill attribution: {json.dumps(attribution[envelope], sort_keys=True)}')
    return attribution, evidence


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
