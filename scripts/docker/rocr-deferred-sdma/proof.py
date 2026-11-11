#!/usr/bin/env python3
"""Replay the complete pinned SDMA deferred body and authenticate installed proof.

Builders retain original-code failure plus actual candidate packet decisions.
Installed admission only authenticates bounded source/contract/DSO evidence; it
never invokes ROCr, a compiler, accelerator, or dependency callback.
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

NAMES = ('deferred_on_off', 'deferred_off_on', 'deferred_on_on', 'deferred_off_off',
         'immediate_on_off', 'immediate_off_on', 'reservation_rejection')
PREFIX = Path('projects/rocr-runtime/runtime/hsa-runtime/core')
CPP = PREFIX / 'runtime/amd_blit_sdma.cpp'
HEADER = PREFIX / 'inc/amd_blit_sdma.h'


def digest(path: Path) -> str:
    """Bind one exact dependency/proof artifact, never a model payload."""
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def contract_identity() -> str:
    """Authenticate both producer and the authored external-input/packet sinks."""
    root = Path(__file__).resolve().parent
    return ':'.join(digest(root / name) for name in ('proof.py', 'contract.cpp.in'))


def definition(text: str, signature: str, start: int = 0) -> tuple[str, int]:
    """Retain one complete actual definition, including all native branches."""
    begin = text.index(signature, start)
    opening = text.index('{', begin)
    depth = 0
    for end in range(opening, len(text)):
        depth += (text[end] == '{') - (text[end] == '}')
        if depth == 0:
            return text[begin:end + 1], end + 1
    raise ValueError('Incomplete native SDMA definition: ' + signature)


def render(native: Path) -> str:
    """Use actual callback owner/handler, full submission bodies and declarations."""
    text = (native / CPP).read_text()
    header = (native / HEADER).read_text()
    base, _ = definition(header, 'class BlitSdmaBase :')
    declarations = lambda value: '\n'.join(re.findall(
        r'(?:virtual )?hsa_status_t SubmitCommand\(.*?;', value, re.S))
    base_declarations = declarations(base)
    class_begin = header.index('class BlitSdma :')
    subclass, _ = definition(header, 'class BlitSdma :', class_begin)
    submission_declarations = declarations(subclass)
    bodies = []
    cursor = 0
    signature = 'hsa_status_t BlitSdma<useGCR, scopeFields>::SubmitCommand('
    while text.find(signature, cursor) != -1:
        body, cursor = definition(text, signature, cursor)
        bodies.append('template <bool useGCR, bool scopeFields>\n' + body)
    if len(bodies) not in (1, 2):
        raise ValueError('Unexpected native SDMA submission family')
    callback, _ = definition(text, 'class CommandCallBackData')
    handler, _ = definition(text, 'static bool DepSignalCompleteHandler(')
    template = Path(__file__).with_name('contract.cpp.in').read_text()
    for name, value in (('BASE_DECLARATIONS', base_declarations),
                        ('SUBMISSION_DECLARATIONS', submission_declarations),
                        ('CALLBACK_DATA', callback), ('CALLBACK_HANDLER', handler),
                        ('SUBMISSION_BODIES', '\n'.join(bodies))):
        template = template.replace('@' + name + '@', value)
    if re.search(r'@[A-Z_]+@', template):
        raise ValueError('Unresolved native replay input')
    return template


def verify(document: dict, identity: str, library_digest: str) -> None:
    """Reject stale evidence, policy rereads, relabelled RED, and lifetime gaps."""
    if (document.get('schema') != 'rocr_deferred_sdma_profiling_v1' or
            document.get('source_identity') != identity or
            document.get('contract_identity') != contract_identity() or
            document.get('library_sha256') != library_digest):
        raise ValueError('ROCr deferred SDMA proof has foreign dependency/contract identity')
    for variant in ('original', 'candidate'):
        result = document.get(variant, {})
        if (result.get('compile_exit') != 0 or result.get('run_exit') != (variant == 'original') or
                not all(re.fullmatch('[0-9a-f]{64}', str(result.get(key, '')))
                        for key in ('source_sha256', 'header_sha256'))):
            raise ValueError('ROCr deferred SDMA native replay outcome is incomplete')
        cases = result.get('cases', [])
        if [case.get('name') for case in cases] != list(NAMES):
            raise ValueError('ROCr deferred SDMA proof omits a retained policy transition')
        expected_times = (40, 0, 40, 0, 40, 0, 0) if variant == 'candidate' else (0, 40, 40, 0, 40, 0, 0)
        for index, case in enumerate(cases):
            if (type(case.get('passed')) is not bool or
                    case['passed'] != (variant == 'candidate' or index >= 2) or
                    case.get('timestamp_packets') != expected_times[index] or
                    case.get('callbacks') != (20, 20, 40, 40, 0, 0, 0)[index] or
                    case.get('live_bytes') != (160 if index < 6 else 0) or case.get('packet_unchanged') is not True or
                    case.get('global_unchanged') is not True or case.get('live_callback_allocations') != 0):
                raise ValueError('ROCr deferred SDMA native policy/lifetime observation disagrees')
    if document['original']['source_sha256'] == document['candidate']['source_sha256']:
        raise ValueError('ROCr deferred SDMA proof lost its original-code negative control')


def load_verified(path: Path, identity: str, library: Path) -> dict:
    """Read bounded immutable proof without entering a runtime or compiling."""
    if path.stat().st_size > 32768:
        raise ValueError('ROCr deferred SDMA proof exceeds bounded evidence')
    document = json.loads(path.read_text())
    verify(document, identity, digest(library))
    return document


def produce(source_root: Path, patch: Path, compiler: str, library: Path,
            identity: str, output: Path) -> None:
    """Compile full actual original/candidate bodies against authored HSA inputs."""
    document = dict(schema='rocr_deferred_sdma_profiling_v1', source_identity=identity,
                    contract_identity=contract_identity(), library_sha256=digest(library))
    with tempfile.TemporaryDirectory(prefix='llaminar-rocr-deferred-proof-') as temporary:
        root = Path(temporary)
        original = root / 'original'
        for relative in (CPP, HEADER):
            destination = original / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root / relative, destination)
        subprocess.run(['git', 'apply', '--reverse', '--check', str(patch.resolve())],
                       cwd=original, check=True, capture_output=True, text=True)
        subprocess.run(['git', 'apply', '--reverse', str(patch.resolve())],
                       cwd=original, check=True, capture_output=True, text=True)
        for variant, native in (('original', original), ('candidate', source_root)):
            source = root / (variant + '.cpp')
            source.write_text(render(native))
            binary = root / (variant + ".bin")
            compiled = subprocess.run([compiler, '-std=c++17', '-O2', '-pthread', str(source), '-o', str(binary)],
                                      capture_output=True, text=True)
            if compiled.returncode:
                raise ValueError('Native deferred SDMA body compile failed: ' + compiled.stderr)
            tested = subprocess.run([str(binary)], capture_output=True, text=True)
            cases = []
            for row in csv.reader(tested.stdout.splitlines()):
                if len(row) != 8 or row[1] not in ('0', '1'):
                    raise ValueError('Malformed native deferred SDMA observation')
                cases.append(dict(name=row[0], passed=row[1] == '1', timestamp_packets=int(row[2]),
                                  callbacks=int(row[3]), live_bytes=int(row[4]), packet_unchanged=row[5] == '1',
                                  global_unchanged=row[6] == '1', live_callback_allocations=int(row[7])))
            document[variant] = dict(source_sha256=digest(native / CPP), header_sha256=digest(native / HEADER),
                                     compile_exit=compiled.returncode, run_exit=tested.returncode, cases=cases)
    verify(document, identity, digest(library))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + '\n')


def main() -> int:
    """Produce builder-only source replay or authenticate installed evidence."""
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
    print('PASS ROCr deferred SDMA original/candidate source/DSO proof')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
