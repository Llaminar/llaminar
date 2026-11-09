#!/usr/bin/env python3
"""Source-slice ROCr helper-mask replay and immutable dependency-proof admission.

Builders compile the actual original/candidate helper parser and constructor
with authored OS inputs. Installed checks only verify that bounded evidence and
the configured DSO; they do not load ROCr, compile, or query hardware.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

NAMES = ('one_node_subset', 'multiple_nodes', 'exact_env_zero', 'unrestricted',
         'unknown_topology_all_cpu_semantics', 'affinity_error_all_cpu_semantics',
         'noncontiguous_node_ids')
NATIVE = Path('projects/rocr-runtime/runtime/hsa-runtime/core/util/lnx/os_linux.cpp')


def digest(path: Path) -> str:
    """Authenticate one explicit dependency artifact or bounded proof input."""
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def contract_identity() -> str:
    """Bind the producer and authored host fixture used by the builder."""
    root = Path(__file__).resolve().parent
    return ':'.join(digest(root / name) for name in ('proof.py', 'contract.cpp.in'))


def definition(text: str, signature: str) -> str:
    """Extract the complete native definition; reject a missing or partial body."""
    start = text.index(signature)
    opening = text.index('{', start)
    depth = 0
    for end in range(opening, len(text)):
        depth += (text[end] == '{') - (text[end] == '}')
        if depth == 0:
            return text[start:end + 1]
    raise ValueError('Incomplete ROCr definition: ' + signature)


def verify(document: dict, identity: str, library_digest: str) -> None:
    """Reject stale, incomplete, foreign or weakened native source/DSO evidence."""
    if (document.get('schema') != 'rocr_helper_numa_v1' or
            document.get('source_identity') != identity or
            document.get('contract_identity') != contract_identity() or
            document.get('library_sha256') != library_digest):
        raise ValueError('ROCr helper proof has foreign dependency/contract identity')
    for variant in ('original', 'candidate'):
        result = document.get(variant, {})
        if not re.fullmatch('[0-9a-f]{64}', str(result.get('source_sha256', ''))):
            raise ValueError('ROCr proof omitted exact native input')
        branches = result.get('branches', [])
        if len(branches) != 2:
            raise ValueError('ROCr proof omitted an affinity implementation')
        for attr, branch in enumerate(branches):
            if (branch.get('attr_affinity') != bool(attr) or branch.get('compile_exit') != 0 or
                    branch.get('run_exit') != (1 if variant == 'original' else 0)):
                raise ValueError('ROCr proof has invalid native replay outcome')
            cases = branch.get('cases', [])
            if [row.get('name') for row in cases] != list(NAMES):
                raise ValueError('ROCr proof has incomplete authored topology controls')
            for index, row in enumerate(cases):
                if type(row.get('passed')) is not bool or row.get('creator_unchanged') is not True:
                    raise ValueError('ROCr proof lost creator mask or malformed its outcome')
                masks = (3, 243, 2, 255, 255, 255, 240) if variant == 'candidate' else (255, 255, 2, 255, 255, 255, 255)
                if row.get('helper_mask') != masks[index] or row.get('creator_mask') != (2, 66, 2, 255, 2, 2, 64)[index]:
                    raise ValueError('ROCr proof has inconsistent observed masks')
                expected = variant == 'candidate' or row['name'] not in (
                    'one_node_subset', 'multiple_nodes', 'noncontiguous_node_ids')
                if row['passed'] != expected:
                    raise ValueError('ROCr proof changed original/candidate mask semantics')
    if document['original']['source_sha256'] == document['candidate']['source_sha256']:
        raise ValueError('ROCr proof did not retain original source control')


def load_verified(path: Path, identity: str, library: Path) -> dict:
    """Read proportional immutable evidence without executing a native runtime."""
    if path.stat().st_size > 65536:
        raise ValueError('ROCr helper proof exceeds bounded contract')
    document = json.loads(path.read_text())
    verify(document, identity, digest(library))
    return document


def produce(source_root: Path, patch: Path, compiler: str, library: Path,
            identity: str, output: Path) -> None:
    """Replay complete actual bodies with both pthread-affinity implementations."""
    native = source_root / NATIVE
    template = Path(__file__).with_name('contract.cpp.in').read_text()
    document = dict(schema='rocr_helper_numa_v1', source_identity=identity,
                    contract_identity=contract_identity(), library_sha256=digest(library))
    with tempfile.TemporaryDirectory(prefix='llaminar-rocr-helper-proof-') as temporary:
        root = Path(temporary)
        original = root / NATIVE
        original.parent.mkdir(parents=True)
        shutil.copy2(native, original)
        subprocess.run(['git', 'apply', '--reverse', '--check', str(patch.resolve())],
                       cwd=root, check=True, capture_output=True, text=True)
        subprocess.run(['git', 'apply', '--reverse', str(patch.resolve())],
                       cwd=root, check=True, capture_output=True, text=True)
        for variant, path in (('original', original), ('candidate', native)):
            text = path.read_text()
            helpers = '' if variant == 'original' else '\n'.join(
                definition(text, signature) for signature in
                ('static bool ReadCpuList(', 'static bool FillCpuSetFromCallerNumaNodes('))
            constructor = definition(text, '  explicit os_thread(')
            source = root / (variant + '.cpp')
            source.write_text(template.replace('@HELPERS@', helpers).replace('@CONSTRUCTOR@', constructor))
            result = dict(source_sha256=digest(path), branches=[])
            for attr in (False, True):
                binary = root / (variant + str(int(attr)))
                command = [compiler, '-std=c++17', '-O2', '-pthread', str(source), '-o', str(binary)]
                if attr:
                    command.append('-DHAVE_PTHREAD_ATTR_SETAFFINITY_NP=1')
                compiled = subprocess.run(command, capture_output=True, text=True)
                if compiled.returncode:
                    raise ValueError('ROCr native body compile failed: ' + compiled.stderr)
                tested = subprocess.run([str(binary)], capture_output=True, text=True)
                cases = []
                callers = (2, 66, 2, 255, 2, 2, 64)
                for index, row in enumerate(csv.reader(tested.stdout.splitlines())):
                    if len(row) != 4 or index >= len(callers) or row[1] not in ('0', '1'):
                        raise ValueError('Malformed ROCr native body observation')
                    cases.append(dict(name=row[0], passed=row[1] == '1', helper_mask=int(row[2]),
                                      creator_mask=int(row[3]), creator_unchanged=int(row[3]) == callers[index]))
                result['branches'].append(dict(attr_affinity=attr, compile_exit=compiled.returncode,
                                               run_exit=tested.returncode, cases=cases))
            document[variant] = result
    verify(document, identity, digest(library))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + '\n')


def main() -> int:
    """Produce builder-only replay or authenticate installed dependency evidence."""
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
        load_verified(args.verify, args.source_identity, args.library)
    else:
        if not all((args.source_root, args.patch, args.compiler, args.output)):
            parser.error('builder replay requires source, patch, compiler and output')
        produce(args.source_root, args.patch, args.compiler, args.library, args.source_identity, args.output)
    print('PASS ROCr helper-NUMA original/candidate source/DSO proof')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
