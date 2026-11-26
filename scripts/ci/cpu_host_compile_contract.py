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
        if expected_definitions and declared != expected_definitions:
            raise ValueError('mismatched CPU envelope declarations')
        language = row.get('host_language') if compilers is None else None
        if compilers is not None:
            matches = {name for name, compiler in compilers.items() if compiler and any(Path(token).resolve() == Path(compiler).resolve() for token in command if not token.startswith('-'))}
            if matches == {'C', 'CXX'}:
                raise ValueError('ambiguous compiler identity')
            if len(matches) != 1:
                raise ValueError('unidentified host compiler executable')
            language = matches.pop()
        if language is None:
            raise ValueError('explicit compiler identity required')
        options = []
        for i, token in enumerate(command):
            if token.startswith('-Xcompiler='):
                options.extend(token.split('=', 1)[1].split(','))
            elif token == '-Xarch_host' and i+1 < len(command):
                options.append(command[i+1])
            elif token.startswith(('-march=', '-mcpu=', '-mno-', '-mavx', '-msse', '-mfma', '-mf16c', '-mbmi', '-mpopcnt')):
                options.append(token)
        if any(option not in expected for option in options if option != '-fPIC'):
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
    parser.add_argument('--definition', action='append')
    for language in ('C', 'CXX', 'CUDA', 'HIP'):
        parser.add_argument('--compiler-' + language, default='')
    args = parser.parse_args()
    verify(args.database, args.flags, args.definition, {language: getattr(args, "compiler_" + language) for language in ("C", "CXX", "CUDA", "HIP")})
