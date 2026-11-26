#!/usr/bin/env python3
# @file cpu_host_compile_contract.py
# @brief Audit actual host compiler forwarding against the CMake-resolved CPU envelope.
import argparse
import json
from pathlib import Path
import shlex


def verify(database, flags):
    """Reject each declared-envelope command missing an exact host option."""
    expected = shlex.split(flags)
    checked = 0
    for row in json.loads(Path(database).read_text()):
        command = row.get('arguments') or shlex.split(row['command'])
        if not any('LLAMINAR_COMPILED_WITH_' in token for token in command):
            continue
        checked += 1
        compiler = next((Path(token).name for token in command if Path(token).name in ('nvcc', 'hipcc')), '')
        for flag in expected:
            if compiler == 'nvcc':
                present = '-Xcompiler=' + flag in command
            elif row['file'].endswith(('.hip', '.hip.cpp')) or compiler == 'hipcc':
                present = any(command[i:i+2] == ['-Xarch_host', flag] for i in range(len(command)-1))
            else:
                present = flag in command
            if not present:
                raise ValueError(f"{row['file']}: missing host option {flag}")
    if not checked:
        raise ValueError('no declared CPU envelopes in compile database')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', required=True)
    parser.add_argument('--flags', required=True)
    args = parser.parse_args()
    verify(args.database, args.flags)
