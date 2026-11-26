#!/usr/bin/env python3
"""Build and authenticate the bounded x86 contribution-helper witness receipt."""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys


def source_manifest(root):
    """Bind the probe and complete quoted-header closure, not vendor headers."""
    paths = [root / 'tests/v2/unit/kernels/cpu/CPUContractionWitness.cpp',
             pathlib.Path(__file__).resolve(),
             root / 'tests/v2/scripts/check_cpu_contraction_witness.py']
    seen = {}
    while paths:
        path = paths.pop().resolve()
        key = str(path.relative_to(root))
        if key in seen:
            continue
        data = path.read_bytes()
        seen[key] = hashlib.sha256(data).hexdigest()
        for include in re.findall(r'^#include "([^"]+)"', data.decode(), re.M):
            local = path.parent / include
            paths.append(local if local.exists() else root / 'src/v2' / include)
    return seen


def main():
    """Compile only in builder mode; installed verification never invokes tools."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('compile', 'verify'))
    parser.add_argument('--report', required=True, type=pathlib.Path)
    parser.add_argument('--compiler', required=True)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[3]
    manifest = source_manifest(root)
    if args.mode == 'compile':
        args.report.unlink(missing_ok=True)
        result = subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name(
            'check_cpu_contraction_witness.py')), '--compiler', args.compiler],
            check=True, text=True, stdout=subprocess.PIPE)
        print(result.stdout)
        args.report.write_text(json.dumps(dict(version=1, sources=manifest,
            compiler=args.compiler, evidence=result.stdout), sort_keys=True))
    else:
        report = json.loads(args.report.read_text())
        if (report.get('version') != 1 or report.get('sources') != manifest
                or report.get('compiler') != args.compiler):
            raise RuntimeError('Missing, incompatible or stale contraction proof')
        for isa, suffix in (('avx2', 'AVX2'), ('avx512f', 'AVX2'), ('avx512f', 'AVX512')):
            for policy, count in (('expert', 0), ('native', 1), ('negative', 1)):
                expected = f'{isa} {policy}{suffix}: fused={count}, expected={count}'
                if expected not in report['evidence'].splitlines():
                    raise RuntimeError('Incomplete contraction evidence: ' + expected)
        print('Contribution-helper contraction proof authenticated')


if __name__ == '__main__':
    main()
