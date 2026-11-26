#!/usr/bin/env python3
"""Compile real x86 contribution helpers and attribute every fused instruction.

The unguarded negative control must fuse under the same compiler invocation;
otherwise absence of fusion in the production expert is not evidence.
"""
import argparse
import pathlib
import re
import subprocess
import tempfile


def main():
    """Fail closed on missing symbols, unavailable tools, or unexpected fusion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', required=True)
    parser.add_argument('--objdump', default='objdump')
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[3]
    fixture = root / 'tests/v2/unit/kernels/cpu/CPUContractionWitness.cpp'
    print(subprocess.check_output([args.compiler, '--version'], text=True))
    with tempfile.TemporaryDirectory(prefix='cpu-contraction-') as directory:
        for isa in ('avx2', 'avx512f'):
            obj = pathlib.Path(directory) / (isa + '.o')
            command = [args.compiler, '-std=c++20', '-O3', '-ffp-contract=fast',
                       '-mfma', '-m' + isa, '-I' + str(root / 'src/v2'),
                       '-c', str(fixture), '-o', str(obj)]
            print(' '.join(command), flush=True)
            subprocess.run(command, check=True)
            text = subprocess.check_output([args.objdump, '-d', str(obj)], text=True)
            symbols = dict(re.findall(r'^[0-9a-f]+ <([^>]+)>:\n(.*?)(?=^[0-9a-f]+ <|\Z)',
                                      text, re.M | re.S))
            for suffix in (('AVX2',) if isa == 'avx2' else ('AVX2', 'AVX512')):
                for policy, expected in (('expert', 0), ('native', 1), ('negative', 1)):
                    name = policy + suffix
                    body = symbols[name]
                    count = len(re.findall(r'\bv(?:f|fn)m(?:add|sub)\w*\b', body))
                    print(f'{isa} {name}: fused={count}, expected={expected}')
                    if count != expected:
                        raise RuntimeError(name + '\n' + body)


if __name__ == '__main__':
    main()
