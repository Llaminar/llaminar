#!/usr/bin/env python3
# @file cpu_host_compile_contract.py
# @brief Audit actual host compiler forwarding against the CMake-resolved CPU envelope.
#
# Every compile-database row is bound to exactly one declared envelope. The
# canonical envelope owns every row except the exact object rows of declared
# secondary-envelope targets (tests/v2/cmake/V2CPUSecondaryEnvelope.cmake), which
# are verified against their own flags and complete definitions. A row with a
# reduced envelope that is not bound to a declared secondary object still fails
# against the canonical envelope, and every declared object must be compiled
# exactly once.
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import re


@dataclass(frozen=True)
class SecondaryEnvelope:
    """One test-only CPU envelope bound to the exact objects of one target."""

    flags: str
    definitions: tuple
    objects: tuple


def _envelope(flags, definitions):
    """Return validated expected flags and definition values of one envelope."""
    expected = shlex.split(flags)
    if not expected:
        raise ValueError('empty expected CPU flags')
    if not definitions:
        raise ValueError('empty expected CPU definitions')
    return expected, dict(token.split('=', 1) for token in definitions)


def _object_path(row):
    """Return the normalized absolute object path of one row, or None if unrecorded."""
    output = row.get('output')
    if not output:
        return None
    return os.path.normpath(os.path.join(row.get('directory', ''), output))


def _secondary_bindings(secondary, canonical_definitions):
    """Map each declared secondary object path to its target and envelope."""
    bindings = {}
    for target, envelope in (secondary or {}).items():
        expected, definitions = _envelope(envelope.flags, list(envelope.definitions))
        if set(definitions) != set(canonical_definitions):
            raise ValueError(f'{target}: secondary envelope must declare every CPU definition')
        if not envelope.objects:
            raise ValueError(f'{target}: secondary envelope has no objects')
        for path in envelope.objects:
            normalized = os.path.normpath(path)
            if not os.path.isabs(normalized):
                raise ValueError(f'{target}: secondary object path is not absolute: {path}')
            if normalized in bindings:
                raise ValueError(f'{target}: secondary object bound twice: {path}')
            bindings[normalized] = (target, expected, definitions)
    return bindings


def verify(database, flags, definitions=None, compilers=None, secondary=None):
    """Reject each declared-envelope command missing an exact host option.

    @param secondary Optional mapping of target name to SecondaryEnvelope.
    """
    canonical_expected, canonical_definitions = _envelope(flags, definitions)
    bindings = _secondary_bindings(secondary, canonical_definitions)
    compiled = dict.fromkeys(bindings, 0)
    checked = 0
    for row in json.loads(Path(database).read_text()):
        command = row.get('arguments') or shlex.split(row['command'])
        expected, expected_definitions = canonical_expected, canonical_definitions
        bound = bindings.get(_object_path(row))
        if bound is not None:
            compiled[_object_path(row)] += 1
            _, expected, expected_definitions = bound
        declared = {}
        for token in command:
            match = re.fullmatch(r'-D(LLAMINAR_COMPILED_WITH_\w+)=(.*)', token)
            if match:
                name, value = match.groups()
                if value not in ('0', '1') or name in declared and declared[name] != value:
                    raise ValueError('contradictory CPU definition: ' + name)
                declared[name] = value
        if not declared:
            raise ValueError('missing CPU envelope declarations')
        checked += 1
        if declared != expected_definitions:
            raise ValueError('mismatched CPU envelope declarations')
        language = row.get('host_language') if compilers is None else None
        if compilers is not None:
            matches = {name for name, compiler in compilers.items() if compiler and any(Path(token).resolve() == Path(compiler).resolve() for token in command if not token.startswith('-'))}
            if len(matches) > 1:
                explicit = None
                if '--offload-arch' in ' '.join(command) or any(command[i:i+2] == ['-x', 'hip'] for i in range(len(command)-1)):
                    explicit = 'HIP'
                elif any(command[i:i+2] == ['-x', 'c'] for i in range(len(command)-1)) or '-xc' in command:
                    explicit = 'C'
                elif any(command[i:i+2] == ['-x', 'c++'] for i in range(len(command)-1)) or '-xc++' in command:
                    explicit = 'CXX'
                elif row['file'].endswith('.c'):
                    explicit = 'C'
                elif row['file'].endswith(('.cpp', '.cc', '.cxx')):
                    explicit = 'CXX'
                if explicit in matches:
                    matches = {explicit}
            if len(matches) != 1:
                raise ValueError('unidentified host compiler executable')
            language = matches.pop()
        if language is None:
            raise ValueError('explicit compiler identity required')
        options = []
        i = 0
        while i < len(command):
            token = command[i]
            if token.startswith(('-Xcompiler=', '--compiler-options=')):
                options.extend((i, 'compiler', option) for option in token.split('=', 1)[1].split(','))
            elif token in ('-Xcompiler', '--compiler-options', '-Xarch_host', '-Xarch_device'):
                if i + 1 == len(command):
                    raise ValueError('missing forwarded compiler option')
                position = {'-Xarch_host': 'host', '-Xarch_device': 'device'}.get(token, 'compiler')
                values = command[i+1].split(',') if position == 'compiler' else [command[i+1]]
                options.extend((i, position, option) for option in values)
                i += 1
            elif token.startswith('-m'):
                options.append((i, 'raw', token))
            i += 1
        def isa_option(option):
            # Address width and code model do not select optional CPU ISA features.
            return option.startswith('-m') and option != '-m64' and not option.startswith('-mcmodel=')
        hip_spill_option = '-mllvm=-amdgpu-spill-cfi-saved-regs=false'
        # GpuSpillGuard.cmake owns this exact device-only allocation diagnostic;
        # ROCmArchitectures.cmake owns wave64. Neither is a host ISA option.
        if language == 'HIP':
            options = [(index, position, option) for index, position, option in options
                       if not (position == 'raw' and option == '-mwavefrontsize64')
                       and not (position == 'device' and option == hip_spill_option)]
        if any(position == 'device' and isa_option(option) for _, position, option in options):
            raise ValueError('CPU option leaks to device')
        if any(option not in expected for _, _, option in options if isa_option(option)):
            raise ValueError('contrary CPU compiler option')
        for flag in expected:
            if language == 'CUDA':
                present = '-Xcompiler=' + flag in command
            elif language == 'HIP':
                present = any(command[i:i+2] == ['-Xarch_host', flag] for i in range(len(command)-1))
                if flag in command and any(token == flag and (i == 0 or command[i-1] != '-Xarch_host') for i, token in enumerate(command)):
                    raise ValueError('CPU option leaks to HIP device')
            elif language in ('C', 'CXX'):
                present = flag in command
            else:
                raise ValueError('unknown host compiler language')
            if not present:
                raise ValueError(f"{row['file']}: missing host option {flag}")
    if not checked:
        raise ValueError('no declared CPU envelopes in compile database')
    for path, count in compiled.items():
        if count != 1:
            raise ValueError(f'{bindings[path][0]}: secondary object compiled {count} times: {path}')


def _bound_value(token, option, targets):
    """Split one TARGET=VALUE argument and require a declared target."""
    target, separator, value = token.partition('=')
    if not separator or target not in targets:
        raise ValueError(f'{option} names an undeclared secondary envelope: {token}')
    return target, value


def secondary_envelopes(targets, flags, definitions, objects):
    """Build typed secondary envelopes from TARGET-bound command-line arguments.

    Every declared target needs exactly one flags and objects argument and at
    least one definition; no argument may name an undeclared target.
    """
    if len(set(targets)) != len(targets):
        raise ValueError('secondary envelope declared twice')
    declared = {target: dict(flags=None, definitions=[], objects=None) for target in targets}
    for option, tokens, field in (('--secondary-flags', flags, 'flags'),
                                  ('--secondary-objects', objects, 'objects')):
        for token in tokens:
            target, value = _bound_value(token, option, declared)
            if declared[target][field] is not None:
                raise ValueError(f'{option} given twice for {target}')
            declared[target][field] = value
    for token in definitions:
        target, value = _bound_value(token, '--secondary-definition', declared)
        declared[target]['definitions'].append(value)
    result = {}
    for target, fields in declared.items():
        if fields['flags'] is None or fields['objects'] is None or not fields['definitions']:
            raise ValueError(f'incomplete secondary envelope: {target}')
        # CMake joins $<TARGET_OBJECTS> with '|' to keep one argument per target.
        result[target] = SecondaryEnvelope(fields['flags'], tuple(fields['definitions']),
                                           tuple(path for path in fields['objects'].split('|') if path))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', required=True)
    parser.add_argument('--flags', required=True)
    parser.add_argument('--definition', action='append', required=True)
    for language in ('C', 'CXX', 'CUDA', 'HIP'):
        parser.add_argument('--compiler-' + language, default='')
    parser.add_argument('--secondary-envelope', action='append', default=[])
    parser.add_argument('--secondary-flags', action='append', default=[])
    parser.add_argument('--secondary-definition', action='append', default=[])
    parser.add_argument('--secondary-objects', action='append', default=[])
    args = parser.parse_args()
    verify(args.database, args.flags, args.definition, {language: getattr(args, "compiler_" + language) for language in ("C", "CXX", "CUDA", "HIP")},
           secondary_envelopes(args.secondary_envelope, args.secondary_flags,
                               args.secondary_definition, args.secondary_objects))
