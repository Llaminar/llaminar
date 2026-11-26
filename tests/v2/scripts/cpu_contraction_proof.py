#!/usr/bin/env python3
"""Build and authenticate the bounded x86 contribution-helper witness receipt.

Builder mode refuses a compiler whose binary or banner changed since the
configure-recorded plan, compiles the witness with the production-derived
commands and publishes the receipt. Installed verification executes no tool:
it compares the receipt with the plan's recorded identity, the source closure
and the commands re-derived from the compilation database.
"""
import argparse
import hashlib
import json
import pathlib
import re
import tempfile

from check_cpu_contraction_witness import (compile_witness, compiler_identity, expected_counts,
                                           file_identity, planned_commands)

VERSION = 2


def source_manifest(root):
    """Bind the probe, the proof tooling and the complete quoted-header closure, not vendor headers."""
    paths = [root / 'tests/v2/unit/kernels/cpu/CPUContractionWitness.cpp',
             pathlib.Path(__file__).resolve(),
             root / 'tests/v2/scripts/check_cpu_contraction_witness.py',
             root / 'tests/v2/cmake/CPUCodegenEnvelope.py']
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


def make_report(root, plan_path, compiler, core_digest, commands, counts, objdump, evidence):
    """Assemble the receipt binding every authenticated input."""
    return dict(version=VERSION, sources=source_manifest(root),
                plan_sha256=hashlib.sha256(pathlib.Path(plan_path).read_bytes()).hexdigest(),
                compiler=compiler, core_command_sha256=core_digest, commands=commands,
                counts=counts, objdump=objdump, evidence='\n'.join(evidence))


def build(root, plan_path, report_path):
    """Compile only in builder mode, after proving the configured compiler is unchanged."""
    report_path.unlink(missing_ok=True)
    plan = json.loads(plan_path.read_text())
    compiler = compiler_identity(plan['compiler']['path'])
    if compiler != plan['compiler']:
        raise RuntimeError('C++ compiler changed since configuration (path, binary SHA256 or '
                           '--version); reconfigure before building the contraction proof')
    directory, core_digest, commands = planned_commands(plan)
    with tempfile.TemporaryDirectory(prefix='cpu-contraction-') as workdir:
        counts, evidence = compile_witness(directory, commands, plan['objdump'], workdir)
    print('\n'.join(evidence))
    if counts != expected_counts():
        raise RuntimeError('Unexpected contraction evidence:\n' + '\n'.join(evidence))
    real, digest = file_identity(plan['objdump'])
    report = make_report(root, plan_path, compiler, core_digest, commands, counts,
                         dict(path=plan['objdump'], realpath=real, sha256=digest), evidence)
    temporary = report_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, sort_keys=True, indent=1) + '\n')
    temporary.replace(report_path)


def verify(root, plan_path, report_path):
    """Authenticate a retained receipt without invoking any compiler or disassembler."""
    report = json.loads(report_path.read_text())
    plan = json.loads(plan_path.read_text())
    if report.get('version') != VERSION or report.get('sources') != source_manifest(root):
        raise RuntimeError('Missing, incompatible or stale contraction proof')
    if report.get('plan_sha256') != hashlib.sha256(plan_path.read_bytes()).hexdigest():
        raise RuntimeError('Contraction proof was built for another configuration plan')
    if report.get('compiler') != plan['compiler']:
        raise RuntimeError('Contraction proof compiler identity differs from the configured compiler')
    _, core_digest, commands = planned_commands(plan)
    if report.get('core_command_sha256') != core_digest or report.get('commands') != commands:
        raise RuntimeError('Production core compile command changed after the contraction proof')
    if report.get('counts') != expected_counts():
        raise RuntimeError('Incomplete or unexpected contraction evidence')
    print('Contribution-helper contraction proof authenticated')


def main():
    """Dispatch the builder or the installed verifier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('compile', 'verify'))
    parser.add_argument('--plan', required=True, type=pathlib.Path)
    parser.add_argument('--report', required=True, type=pathlib.Path)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[3]
    if args.mode == 'compile':
        build(root, args.plan, args.report)
    else:
        verify(root, args.plan, args.report)


if __name__ == '__main__':
    main()
