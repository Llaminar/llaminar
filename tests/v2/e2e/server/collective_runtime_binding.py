"""Authenticate the image's collective ELF mappings at serving readiness.

The existing host PID observer provides process start identities. Linux maps
identifies the opened file even when a library name resolves through symlinks.
At retirement the native rank inventory identifies which backends each process
actually owned.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys

IMAGE_LIBRARIES = {
    'rocm': '/usr/local/lib/librccl.so.1',
    'cuda': '/usr/local/lib/libllaminar_nccl.so.2',
}
NATIVE_CALL_DOMAINS = frozenset({
    'tp_allreduce_runtime', 'tp_raw_allgather_runtime', 'tp_rooted_collective_runtime',
})
# The vendor can load separate libnccl-net, tuner and profiler plugins.
# Only the primary collective implementation supplies its image SONAME.
COLLECTIVE_NAMES = re.compile(r'lib(?:llaminar_)?(rccl|nccl)\.so(?:\.| |$)')


def mapping_backend(path: str) -> str | None:
    """Classify a native collective DSO by its saved Linux mapping name."""
    match = COLLECTIVE_NAMES.match(Path(path).name)
    return ('rocm' if match[1] == 'rccl' else 'cuda') if match else None


def validate_mappings(mappings: dict, expected: dict) -> None:
    """Authenticate mapped segments against the same shipped inode and executable rules."""
    if type(mappings) is not dict or set(mappings) - IMAGE_LIBRARIES.keys():
        raise ValueError('Invalid collective mapping backend')
    for backend, rows in mappings.items():
        wanted = expected.get(backend)
        if type(rows) is not list or not rows or wanted is None:
            raise ValueError(f'Unexpected {backend} collective mapping roster')
        for row in rows:
            path = row['path']
            device = row['device']
            inode = row['inode']
            if (not path.startswith('/') or path.endswith(' (deleted)')
                    or mapping_backend(path) != backend
                    or not re.fullmatch(r'[0-9a-f]+:[0-9a-f]+', device)
                    or not re.fullmatch(r'[0-9a-f]+-[0-9a-f]+', row['address'])
                    or not re.fullmatch(r'[r-][w-][x-][ps]', row['permissions'])
                    or not re.fullmatch(r'[0-9a-f]+', row['offset'])
                    or type(inode) is not int or inode <= 0):
                raise ValueError(f'Unexpected {backend} collective mapping: {row}')
            major, minor = (int(part, 16) for part in device.split(':'))
            if (major, minor, inode) != tuple(wanted['identity']):
                raise ValueError(f'Unexpected {backend} collective mapping: {row}')
        if not any('x' in row['permissions'] for row in rows):
            raise ValueError(f'No executable code mapping for {backend} collective DSO')


def bind_mappings(maps: str, expected: dict) -> dict:
    """Extract collective segments from Linux maps, then check their recorded identities."""
    found = {}
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        address, permissions, offset, device, inode, path = fields
        backend = mapping_backend(path)
        if backend is not None:
            found.setdefault(backend, []).append(dict(address=address, permissions=permissions,
                offset=offset, device=device, inode=int(inode), path=path))
    validate_mappings(found, expected)
    return found


def file_fingerprint(path: Path, cached: dict) -> dict:
    """Fingerprint each shipped inode once inside the observed mount namespace."""
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        identity = (os.major(before.st_dev), os.minor(before.st_dev), before.st_ino)
        revision = (identity, before.st_size, before.st_mtime_ns)
        if revision not in cached:
            cached[revision] = dict(identity=identity, bytes=before.st_size,
                sha256=hashlib.file_digest(stream, 'sha256').hexdigest())
        after = os.fstat(stream.fileno())
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError('Collective DSO changed during its cold fingerprint')
    return cached[revision]


def start_ticks(process: Path) -> int:
    """Return Linux process birth ticks independently of comm field contents."""
    return int((process / 'stat').read_text().rsplit(')', 1)[1].split()[19])


def observe(processes: list[dict], *, proc: Path = Path('/proc')) -> dict:
    """Collect passive executable mapping evidence for the sealed server PIDs."""
    cached, ranks = {}, []
    for pin in processes:
        process = proc / str(pin['pid'])
        if start_ticks(process) != pin['start_ticks'] or (process / 'comm').read_text().strip() != 'llaminar2':
            raise ValueError('Serving PID changed before collective observation')
        files = {}
        for backend, library in IMAGE_LIBRARIES.items():
            expected = process / 'root' / library.lstrip('/')
            try:
                target = expected.readlink()
            except OSError:
                if not expected.exists():
                    continue
                target = None
            if target is not None and (target.is_absolute() or len(target.parts) != 1):
                raise ValueError(f'Shipped collective alias points outside its image library directory: {library}')
            files[backend] = dict(path=library, **file_fingerprint(expected, cached))
        mappings = bind_mappings((process / 'maps').read_text(), files)
        if start_ticks(process) != pin['start_ticks']:
            raise ValueError('Serving PID changed during collective observation')
        ranks.append(dict(pid=pin['pid'], start_ticks=pin['start_ticks'], mappings=mappings, shipped=files))
    if not ranks:
        raise ValueError('No serving process to check for collective mappings')
    receipt = dict(schema='llaminar.collective_runtime_binding.v1', processes=ranks)
    validate_process_mappings(receipt)
    return receipt


def validate_process_mappings(receipt: dict) -> None:
    """Replay a complete live observation with the original shipped-file witnesses."""
    if receipt.get('schema') != 'llaminar.collective_runtime_binding.v1' or not receipt.get('processes'):
        raise ValueError('Missing collective runtime binding receipt')
    shipped = None
    for row in receipt['processes']:
        files = row['shipped']
        if type(files) is not dict or set(files) - IMAGE_LIBRARIES.keys():
            raise ValueError('Unexpected shipped collective file roster')
        for backend, file in files.items():
            identity = file['identity']
            if (file['path'] != IMAGE_LIBRARIES[backend] or type(identity) not in (list, tuple)
                    or len(identity) != 3 or any(type(part) is not int or part < 0 for part in identity)
                    or not identity[2] or type(file['bytes']) is not int or file['bytes'] <= 0
                    or not re.fullmatch(r'[0-9a-f]{64}', file['sha256'])):
                raise ValueError(f'Invalid shipped collective file evidence for {backend}')
        if shipped is None:
            shipped = files
        elif files != shipped:
            raise ValueError('Shipped collective identity differs across serving ranks')
        validate_mappings(row['mappings'], files)


def require_native_owners(receipt: dict, ranks, runtime_records: list[dict], *, discovery=None) -> list[dict]:
    """Require mappings on ranks with successful native multi-device submissions."""
    records = {(row['pid'], row['start_ticks']): row['mappings'] for row in receipt['processes']}
    owners = discovery.processes if discovery else ranks
    if set(records) != {(rank.pid, rank.start_ticks) for rank in owners} or len(records) != len(receipt['processes']):
        raise ValueError('Collective evidence does not cover exact native rank lifetimes')
    by_rank = {rank.rank: rank for rank in ranks}
    used = {}
    for row in runtime_records:
        if row.get('domain') not in NATIVE_CALL_DOMAINS or row.get('name') != 'calls':
            continue
        tags = row['tags']
        backend = {'rccl': 'rocm', 'nccl': 'cuda'}.get(tags['backend'])
        if backend is None or int(tags['degree']) <= 1 or row['value'] <= 0:
            continue
        rank = by_rank[row['rank']]
        device = row['device']
        if not device.lower().startswith(backend + ':') or device not in {item.device for item in rank.devices}:
            raise ValueError('Native collective submission names a device outside the owning rank')
        if backend not in records[(rank.pid, rank.start_ticks)]:
            raise ValueError(f'Native rank {rank.rank} has no shipped collective mapping for {backend}')
        used.setdefault((rank.rank, backend), set()).add(row['domain'])
    return [dict(rank=rank, backend=backend, records=sorted(domains))
            for (rank, backend), domains in sorted(used.items())]


def replay_collective_binding(receipt: dict, image: str, ranks, runtime_records: list[dict],
                              *, discovery=None) -> list[dict]:
    """Validate the archived image, rank bindings and observed collective calls."""
    if receipt.get('passed') is not True or receipt.get('image') != image:
        raise ValueError('Collective runtime binding differs from immutable serving image')
    validate_process_mappings(receipt)
    used = require_native_owners(receipt, ranks, runtime_records, discovery=discovery)
    if used != receipt.get('native_communicators'):
        raise ValueError('Collective runtime receipt contradicts ranked native submissions')
    return used


if __name__ == '__main__':
    print(json.dumps(observe(json.loads(sys.argv[1]))))
