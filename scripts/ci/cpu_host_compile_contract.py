#!/usr/bin/env python3
# @file cpu_host_compile_contract.py
# @brief Audit actual host compiler forwarding against the CMake-resolved CPU envelope.
import argparse
import json
from pathlib import Path
import shlex
import re


def verify(database, flags, definitions=None):
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
            continue
        checked += 1
        if expected_definitions and declared != expected_definitions:
            raise ValueError('mismatched CPU envelope declarations')
        language = row.get('host_language')
        if language is None:
            if any(Path(token).name == 'nvcc' for token in command):
                language = 'CUDA'
            elif '-x' in command and command[command.index('-x') + 1] == 'hip':
                language = 'HIP'
            elif any(token.startswith('--offload-arch=') for token in command):
                language = 'HIP'
            elif row['file'].endswith('.c'):
                language = 'C'
            elif row['file'].endswith(('.cpp', '.cc', '.cxx')):
                language = 'CXX'
            else:
                raise ValueError('unidentified host compiler language')
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
    args = parser.parse_args()
    verify(args.database, args.flags, args.definition)
