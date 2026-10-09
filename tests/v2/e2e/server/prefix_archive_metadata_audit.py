"""Authenticate retired format-3 cache storage using metadata only.

The journal describes exact immutable file generations. This independent audit
replays only journal metadata and stats payload files; it never opens, hashes,
checksums or copies a cache payload. It runs after native retirement so pending
persistence and selected-reader leases have ended before physical reclamation
is assessed. The production cache remains the sole capacity authority.
"""
import argparse
import json
import os
from pathlib import Path
import stat
import struct
import time


def inventory(directory: Path, budget: int) -> dict:
    """Match every committed live generation to exactly one bounded payload file."""
    journals = list(directory.glob('*.kvcache'))
    if len(journals) != 1:
        raise ValueError('Expected one model journal in ' + str(directory))
    journal = journals[0]
    if journal.is_symlink() or not journal.is_file():
        raise ValueError('Journal is not a regular owned file')
    active = {}
    records = metadata_bytes = 0
    with journal.open('rb', buffering=0) as stream:
        header = stream.read(80)
        metadata_bytes += len(header)
        if (len(header) != 80 or header[:8] != b'LLKVC001'
                or struct.unpack('<II', header[8:16]) != (3, 80)
                or header[16:].decode('ascii') + '.kvcache' != journal.name):
            raise ValueError('Unqualified journal format or model identity')
        while preamble := stream.read(56):
            metadata_bytes += len(preamble)
            if len(preamble) != 56 or preamble[:8] != b'LLKVR001':
                raise ValueError('Incomplete journal record')
            version, kind, extent, length, payload, sequence, checksum = struct.unpack('<IIQQQQQ', preamble[8:])
            if version != 3 or kind not in (1, 2, 3) or payload != 0 or not 0 < length <= 1048576 or extent != length + 72:
                raise ValueError('Journal contains payload or invalid metadata extent')
            encoded = stream.read(length)
            footer = stream.read(16)
            metadata_bytes += len(encoded) + len(footer)
            if len(encoded) != length or len(footer) != 16 or footer[:8] != b'LLKVDONE':
                raise ValueError('Uncommitted journal record')
            value = json.loads(encoded)
            key = json.dumps(value['key'], sort_keys=True, separators=(',', ':'))
            if kind == 1:
                identity = value['storage_identity']
                if len(identity) != 2 or any(type(v) is not int or not 0 <= v < 2**64 for v in identity) or identity == [0, 0]:
                    raise ValueError('Invalid immutable storage generation')
                size = value['total_bytes']
                sections = value['sections']
                if (type(size) is not int or size <= 0 or len(sections) != 6
                        or any(type(row['bytes']) is not int or row['bytes'] < 0 for row in sections)
                        or sum(row['bytes'] for row in sections) != size):
                    raise ValueError('Invalid payload metadata extent')
                active[key] = {'name': f'{identity[0]:016x}{identity[1]:016x}.kvblock', 'bytes': size}
            elif kind == 2:
                active.pop(key, None)
            elif key not in active:
                raise ValueError('Touch references absent generation')
            records += 1
    expected = {row['name']: row['bytes'] for row in active.values()}
    if len(expected) != len(active) or sum(expected.values()) > budget:
        raise ValueError('Duplicate storage ownership or live budget exceeded')
    payload_directory = journal.with_name(journal.name + '.blocks')
    if payload_directory.is_symlink() or not payload_directory.is_dir():
        raise ValueError('Missing owned payload directory')
    actual = {}
    allocated = 0
    for path in payload_directory.iterdir():
        state = path.lstat()
        if not stat.S_ISREG(state.st_mode):
            raise ValueError('Non-regular payload entry')
        actual[path.name] = state.st_size
        allocated += state.st_blocks * 512
    if actual != expected:
        raise ValueError('Physical payload files differ from committed live generations')
    journal_state = journal.stat()
    if journal_state.st_size != metadata_bytes:
        raise ValueError('Journal changed after native retirement')
    return {'passed': True, 'journal': journal.name, 'format': 3, 'records': records,
            'entries': len(active), 'active_payload_bytes': sum(expected.values()),
            'payload_files': len(actual), 'payload_allocated_bytes': allocated,
            'journal_bytes': metadata_bytes, 'journal_allocated_bytes': journal_state.st_blocks * 512,
            'metadata_read_bytes': metadata_bytes, 'payload_read_bytes': 0,
            'orphan_payload_files': 0, 'budget_bytes': budget}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--budget-bytes', required=True, type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    results = [inventory(args.directory, args.budget_bytes)]
    if (args.directory / 'ordinary-control').is_dir():
        results.append(inventory(args.directory / 'ordinary-control', args.budget_bytes))
    report = {'passed': True, 'observed': time.time(), 'archives': results}
    if args.output is not None:
        with args.output.open('x') as output:
            json.dump(report, output, indent=2)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
    print(json.dumps(report), flush=True)
