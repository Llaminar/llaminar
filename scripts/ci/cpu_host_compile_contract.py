#!/usr/bin/env python3
# @file cpu_host_compile_contract.py
# @brief Audit actual host compiler forwarding against the CMake-resolved CPU envelope.
import argparse
import json
from pathlib import Path
import shlex
import re


def verify(database, flags, definitions=None, compilers=None):
    """Reject each declared-envelope command missing an exact host option."""
    expected = shlex.split(flags)
    if not expected:
        raise ValueError('empty expected CPU flags')
    if not definitions:
        raise ValueError('empty expected CPU definitions')
    expected_definitions = dict(token.split('=', 1) for token in definitions or [])
    checked = 0
    for row in json.loads(Path(database).read_text()):
        command = row.get('arguments') or shlex.split(row['command'])
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', required=True)
    parser.add_argument('--flags', required=True)
    parser.add_argument('--definition', action='append', required=True)
    for language in ('C', 'CXX', 'CUDA', 'HIP'):
        parser.add_argument('--compiler-' + language, default='')
    args = parser.parse_args()
    verify(args.database, args.flags, args.definition, {language: getattr(args, "compiler_" + language) for language in ("C", "CXX", "CUDA", "HIP")})
