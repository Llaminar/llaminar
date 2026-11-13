#!/usr/bin/env python3
"""Build or authenticate the native preferred-queue actual-body contract.

The dependency builder extracts complete original/candidate selector bodies and
original QueueInfo/QueueCompare definitions. Installed runners only authenticate
the resulting source/contract/DSO-bound receipt; verification never compiles,
loads a native runtime or enters an accelerator. This is immutable build evidence,
not another runtime queue ownership or readiness authority.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

SCHEMA = 1
CHECKS = 132
ORIGINAL_FAILURES = 54


def digest(path: Path) -> str:
    """Hash one bounded proof/source or explicit dependency artifact."""
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def contract_identity() -> str:
    """Return the ordered producer/fixture identity also declared by the installer."""
    root = Path(__file__).resolve().parent
    return ':'.join(digest(root / name) for name in ('proof.py', 'selector-contract.cpp.in'))


def definition(text: str, signature: str) -> str:
    """Extract the complete native definition without replacing its implementation."""
    start = text.index(signature)
    opening = text.index('{', start)
    depth = 0
    for end in range(opening, len(text)):
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
            if depth == 0:
                return text[start:end + 1]
    raise ValueError('Incomplete native definition: ' + signature)


def verify(document: dict, source_identity: str, library_digest: str,
           expected_contract: str | None = None) -> None:
    """Reject incomplete, foreign or weakened builder evidence without native work.

    Args name the existing canonical source/ordered-patch authority, the actual
    resolved DSO bytes and the current code-owned producer/fixture contract.
    Raises ValueError on any missing or inconsistent proof. Returns None only
    after both complete actual-body outcomes and all finite controls agree.
    """
    if document.get('schema') != SCHEMA or type(document.get('schema')) is not int:
        raise ValueError('HIP preferred-queue proof has an unknown schema')
    if document.get('source_identity') != source_identity:
        raise ValueError('HIP preferred-queue proof has a foreign source identity')
    if document.get('library_sha256') != library_digest:
        raise ValueError('HIP preferred-queue proof has a foreign DSO')
    if document.get('contract_identity') != (expected_contract or contract_identity()):
        raise ValueError('HIP preferred-queue proof has a stale contract')
    import re
    for field in ('original_source_sha256', 'candidate_source_sha256', 'header_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', str(document.get(field, ''))):
            raise ValueError('HIP preferred-queue proof omitted original native input: ' + field)
    if document['original_source_sha256'] == document['candidate_source_sha256']:
        raise ValueError('HIP preferred-queue proof did not retain the original control')
    for variant, exit_code, failures in (('original', 1, ORIGINAL_FAILURES), ('candidate', 0, 0)):
        result = document.get(variant)
        if not isinstance(result, dict):
            raise ValueError('HIP preferred-queue proof omitted ' + variant)
        if (type(result.get('compile_exit')) is not int or result['compile_exit'] != 0 or
                type(result.get('run_exit')) is not int or result['run_exit'] != exit_code or
                type(result.get('failures')) is not int or result['failures'] != failures):
            raise ValueError('HIP preferred-queue proof has an invalid ' + variant + ' outcome')
        if not re.fullmatch('[0-9a-f]{64}', str(result.get('body_sha256', ''))):
            raise ValueError('HIP preferred-queue proof omitted its actual native body')
        cases = result.get('cases')
        if not isinstance(cases, list) or len(cases) != CHECKS:
            raise ValueError('HIP preferred-queue proof has incomplete ' + variant + ' controls')
        for ordinal, case in enumerate(cases, 1):
            if (not isinstance(case, dict) or type(case.get('ordinal')) is not int or
                    case['ordinal'] != ordinal or type(case.get('passed')) is not bool or
                    not isinstance(case.get('name'), str) or not case['name']):
                raise ValueError('HIP preferred-queue proof has malformed controls')
        if sum(not case['passed'] for case in cases) != failures:
            raise ValueError('HIP preferred-queue proof has inconsistent failure counts')
    if document['original']['body_sha256'] == document['candidate']['body_sha256']:
        raise ValueError('HIP preferred-queue proof reused the same native body')
    if ([case['name'] for case in document['original']['cases']] !=
            [case['name'] for case in document['candidate']['cases']]):
        raise ValueError('HIP preferred-queue proof changed the control matrix')
    # The primary old defect must actually be among the failed native decisions.
    if not any(not case['passed'] and case['name'] ==
               'shared preferred must not bypass eligible unshared queue'
               for case in document['original']['cases']):
        raise ValueError('HIP preferred-queue proof did not reproduce the original defect')


def load_verified(path: Path, source_identity: str, library_digest: str,
                  expected_contract: str | None = None) -> dict:
    """Read one proportional immutable receipt and apply the shared strict verifier."""
    if path.stat().st_size > 262144:
        raise ValueError('HIP preferred-queue proof exceeds its bounded contract')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('HIP preferred-queue proof is not an object')
    verify(value, source_identity, library_digest, expected_contract)
    return value


def produce(source_root: Path, patch: Path, compiler: str, library: Path,
            source_identity: str, output: Path) -> None:
    """Replay actual original/candidate native bodies with one unchanged fixture.

    source_root is the exact patched CLR builder tree; patch is only this repair.
    The original is recovered by reverse-applying that patch to a temporary copy,
    never by editing the native tree or substituting a handwritten selector.
    compiler is the admitted SDK host compiler. The output binds final DSO bytes.
    Raises on compile, negative-control, candidate or receipt validation failure.
    """
    relative = Path('rocclr/device/rocm/rocdevice.cpp')
    native = source_root / relative
    header = source_root / relative.with_suffix('.hpp')
    template = Path(__file__).with_name('selector-contract.cpp.in').read_text()
    header_text = header.read_text()
    template = template.replace('@@QUEUE_INFO@@', definition(header_text, 'struct QueueInfo') + ';')
    template = template.replace('@@QUEUE_COMPARE@@', definition(header_text, 'struct QueueCompare') + ';')
    document = {'schema': SCHEMA, 'source_identity': source_identity,
                'library_sha256': digest(library), 'contract_identity': contract_identity(),
                'candidate_source_sha256': digest(native), 'header_sha256': digest(header)}
    with tempfile.TemporaryDirectory(prefix='llaminar-hip-queue-proof-') as name:
        root = Path(name)
        original = root / relative
        original.parent.mkdir(parents=True)
        shutil.copyfile(native, original)
        subprocess.run(['git', 'apply', '--reverse', '--check', str(patch.resolve())],
                       cwd=root, check=True, capture_output=True, text=True)
        subprocess.run(['git', 'apply', '--reverse', str(patch.resolve())],
                       cwd=root, check=True, capture_output=True, text=True)
        document['original_source_sha256'] = digest(original)
        for variant, path in (('original', original), ('candidate', native)):
            body = definition(path.read_text(), 'hsa_queue_t* Device::getQueueFromPool(')
            source = root / (variant + '.cpp')
            binary = root / variant
            source.write_text(template.replace('@@SELECTOR@@', body))
            command = [compiler, '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror',
                       '-Wno-unused-parameter', str(source), '-o', str(binary)]
            compiled = subprocess.run(command, capture_output=True, text=True)
            if compiled.returncode:
                raise ValueError(variant + ' native body compile failed: ' + compiled.stderr)
            tested = subprocess.run([str(binary)], capture_output=True, text=True)
            cases = []
            for row in csv.reader(tested.stdout.splitlines()):
                if row and row[0] == 'CHECK':
                    if len(row) != 4 or row[2] not in ('0', '1'):
                        raise ValueError('Malformed actual native body observation')
                    cases.append({'ordinal': int(row[1]), 'passed': row[2] == '1', 'name': row[3]})
            document[variant] = {'compile_exit': compiled.returncode, 'run_exit': tested.returncode,
                                 'failures': sum(not case['passed'] for case in cases), 'cases': cases,
                                 'body_sha256': hashlib.sha256(body.encode()).hexdigest(), 'compile_argv': command}
    verify(document, source_identity, digest(library))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + '\n')


def main() -> int:
    """Run builder-only actual-body proof or compiler-free installed admission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--patch', type=Path)
    parser.add_argument('--compiler')
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--source-identity', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        load_verified(args.verify, args.source_identity, digest(args.library))
        print('PASS HIP preferred-queue source/contract/DSO proof')
    else:
        if not all((args.source_root, args.patch, args.compiler, args.output)):
            parser.error('builder proof requires source root, patch, compiler and output')
        produce(args.source_root, args.patch, args.compiler, args.library, args.source_identity, args.output)
        print('PASS HIP preferred-queue actual-body original RED / candidate GREEN')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
