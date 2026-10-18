#!/usr/bin/env python3
"""Observe serving resources through OS metadata, without entering a GPU context.

Run in the host PID/cgroup namespace. Cgroup counters own process RAM totals;
RSS rows are diagnostics and are never summed into a second shared-memory total.
Filesystem observations use inode metadata only, including allocated blocks and
deleted-but-open files. This module never reads model or prefix-cache payloads.
PhysicalMemoryAuthority remains the engine's sole admission/allocation authority.
Controller identities bound each process probe. Unrelated cgroups are inspected
only for membership, before opening any task identity, memory or descriptor data.
DRM fdinfo may scale each memory field independently; explicit binary units are
normalized to bytes before resource assertions, without summing aliased clients.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import stat
import subprocess
import time

from resource_process_identity import read_process_status


def drm_memory_bytes(value: str) -> int:
    """Normalize one nonnegative fdinfo memory field using its declared unit.

    Large GPU mappings may be reported in MiB instead of KiB. Bare integer
    values retain the existing byte contract. Unknown units and malformed
    counts fail rather than silently understating a mapping's resource use.
    """
    parts = value.split()
    if (len(parts) not in (1, 2) or not parts[0].isascii()
            or not parts[0].isdecimal()):
        raise ValueError('Malformed DRM memory count: ' + value)
    units = {'B': 1, 'KiB': 1024, 'MiB': 1024**2, 'GiB': 1024**3, 'TiB': 1024**4}
    unit = parts[1] if len(parts) == 2 else 'B'
    if unit not in units:
        raise ValueError('Unknown DRM memory unit: ' + value)
    return int(parts[0]) * units[unit]


def numeric_fields(path: Path, *, kilobytes: bool = False) -> dict[str, int]:
    """Read named integer kernel fields, retaining units explicitly as bytes."""
    result = {}
    for line in path.read_text().splitlines():
        parts = line.replace(':', '').split()
        if len(parts) >= 2 and parts[1].isdigit():
            result[parts[0]] = int(parts[1]) * (1024 if kilobytes and parts[-1] == 'kB' else 1)
    return result


def drm_clients(directory: Path) -> list[dict]:
    """Observe per-client mappings, retaining aliases rather than summing GTT.

    The same host allocation may be mapped into every GPU client. These rows
    describe mappings, not unique physical host bytes; cgroup and host telemetry
    remain separate. Duplicated descriptors for a client are observed only once.
    """
    clients = {}
    for descriptor in directory.iterdir():
        try:
            values = dict(line.split(':', 1) for line in descriptor.read_text().splitlines()
                          if line.startswith('drm-'))
        except FileNotFoundError:
            continue
        if not values:
            continue
        row = {key: value.strip() for key, value in values.items()}
        key = (row['drm-client-id'], row['drm-pdev'])
        if key in clients:
            continue
        for field, value in list(row.items()):
            if field.startswith(('drm-total-', 'drm-shared-', 'drm-resident-', 'drm-memory-')):
                row[field + '_bytes'] = drm_memory_bytes(value)
                del row[field]
        clients[key] = row
    return list(clients.values())


def latest_stats(directory: Path) -> dict | None:
    """Read the final complete observation from the current harness journal.

    This is a bounded metadata read. A writer's incomplete final line is not a
    completed observation; a malformed completed line is a failure. The actual
    server timestamp is retained for independent freshness validation.
    """
    candidates = [p for name in ('stats-tools-observations.jsonl', 'stats-webapp-observations.jsonl')
                  if (p := directory / name).exists() and p.stat().st_size]
    if not candidates:
        return None
    path = max(candidates, key=lambda p: p.stat().st_mtime_ns)
    with path.open('rb') as stream:
        size = os.fstat(stream.fileno()).st_size
        stream.seek(max(0, size - 262144))
        data = stream.read(262144)
    lines = data.split(b'\n')
    if len(lines) < 2:
        raise ValueError('No complete bounded statistics observation')
    return json.loads(lines[-2])


def inode_usage(directory: Path, processes: list[int], *, proc: Path = Path('/proc'),
                namespace_directory: str | None = None) -> dict:
    """Count distinct reachable and open retired inodes using stat, never open.

    Open descriptors retain unlinked payloads, so pathname-only usage would hide
    a leak. Deduplicate hard links and shared reader descriptors by (device,inode).
    Live mutations can remove an entry between enumeration and stat; count these
    races explicitly. Permission and unexpected object errors are failures.
    """
    seen = set()
    groups: dict[str, dict] = {}
    namespaces: dict[str, dict] = {}
    races = 0

    def record(value: os.stat_result, kind: str, namespace: str) -> None:
        nonlocal seen
        key = (value.st_dev, value.st_ino)
        if key in seen:
            return
        seen.add(key)
        for target in (groups, namespaces.setdefault(namespace, {})):
            group = target.setdefault(kind, {'inodes': 0, 'logical_bytes': 0, 'allocated_bytes': 0})
            group['inodes'] += 1
            group['logical_bytes'] += value.st_size
            group['allocated_bytes'] += value.st_blocks * 512

    pending = [directory]
    while pending:
        parent = pending.pop()
        for item in parent.iterdir():
            try:
                value = item.lstat()
            except FileNotFoundError:
                races += 1
                continue
            if stat.S_ISDIR(value.st_mode):
                pending.append(item)
            elif stat.S_ISREG(value.st_mode):
                payload = item.parent.name.endswith('.blocks')
                namespace = str((item.parent.parent if payload else item.parent).relative_to(directory))
                record(value, 'payload' if payload else 'metadata', namespace)
            else:
                raise ValueError(f'Unexpected archive object: {item}')
    for pid in processes:
        for fd in (proc / str(pid) / 'fd').iterdir():
            try:
                target = os.readlink(fd)
                prefix = namespace_directory or str(directory)
                if target.startswith(prefix + '/') and target.endswith(' (deleted)'):
                    relative = Path(target[len(prefix) + 1:].removesuffix(' (deleted)'))
                    parent = relative.parent.parent if relative.parent.name.endswith('.blocks') else relative.parent
                    record(fd.stat(), 'retired_open', str(parent))
            except FileNotFoundError:
                races += 1
    return {'groups': groups, 'namespaces': namespaces, 'raced_entries': races,
            **{name: sum(row[name] for row in groups.values())
               for name in ('inodes', 'logical_bytes', 'allocated_bytes')}}


def process_inventory(container_ids: frozenset[str], *, proc: Path = Path('/proc')) -> dict[str, dict]:
    """Read deeper process metadata only inside exact owned Docker cgroups.

    Full immutable IDs prevent substring matches with another container or a
    neighboring scope. Both Docker's cgroupfs and systemd layouts are admitted.
    No selected container means no process enumeration, including while an
    inference lifecycle is still waiting for admission.
    """
    if not isinstance(container_ids, frozenset) or any(
            type(identity) is not str or len(identity) != 64
            or any(char not in '0123456789abcdef' for char in identity)
            for identity in container_ids):
        raise ValueError('Process observation requires complete immutable Docker identities')
    result = {}
    if not container_ids:
        return result
    for directory in proc.iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            memberships = (directory / 'cgroup').read_text().splitlines()
            unified = next(line[3:] for line in memberships if line.startswith('0::'))
            components = set(unified.split('/'))
            owners = [identity for identity in container_ids if identity in components
                      or 'docker-' + identity + '.scope' in components]
            if not owners:
                continue
            if len(owners) != 1:
                raise ValueError('Process belongs to multiple selected container scopes')
            comm = (directory / 'comm').read_text().strip()
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            row = {'pid': int(directory.name), 'comm': comm, 'start_ticks': int(fields[19])}
            if comm == 'llaminar2':
                row.update(read_process_status(directory / 'status', row['pid']))
                row['drm_clients'] = drm_clients(directory / 'fdinfo')
                kfd = Path('/sys/class/kfd/kfd/proc') / directory.name
                if kfd.exists():
                    row['kfd'] = {f.name: int(f.read_text()) for f in kfd.iterdir()
                                  if f.name.startswith(('vram_', 'sdma_'))}
            result.setdefault(unified, {'container_id': owners[0], 'processes': []})['processes'].append(row)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


def cuda_usage() -> dict:
    """Query NVML through nvidia-smi in the host PID namespace; no CUDA context."""
    result = {}
    for key, option, columns in (
        ('devices', '--query-gpu=', ['index', 'uuid', 'pci.bus_id', 'memory.used', 'memory.total']),
        ('processes', '--query-compute-apps=', ['pid', 'gpu_uuid', 'used_gpu_memory'])):
        command = ['nvidia-smi', option + ','.join(columns), '--format=csv,noheader,nounits']
        output = subprocess.check_output(command, text=True, timeout=10)
        rows = []
        for line in csv.reader(output.splitlines(), skipinitialspace=True):
            if len(line) != len(columns):
                raise ValueError('Incomplete NVML observation: ' + repr(line))
            row = dict(zip(columns, (value.strip() for value in line)))
            for field in columns:
                if field in ('memory.used', 'memory.total', 'used_gpu_memory'):
                    row[field + '_bytes'] = int(row.pop(field)) * 1024**2
                elif field in ('pid', 'index'):
                    row[field] = int(row[field])
            rows.append(row)
        result[key] = rows
    result['resolution_bytes'] = 1024**2
    return result


def collect(manifest: dict) -> dict:
    """Join controller identity, real processes, RAM, device and filesystem metadata."""
    backends = manifest.get('backends')
    if (not isinstance(backends, list) or not backends
            or any(not isinstance(value, str) for value in backends)
            or len(set(backends)) != len(backends)
            or not set(backends) <= {'cpu', 'cuda', 'rocm'}):
        raise ValueError('Resource observation requires an explicit unique backend inventory')
    started = time.monotonic()
    result = {'observed': time.time(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              'host_ram': numeric_fields(Path('/proc/meminfo'), kilobytes=True),
              'cells': {}, 'rocm': {}, 'errors': [], 'schema_version': 2}
    # Read each atomic controller publication once. Its identity selects both
    # the process probe and the resulting row; a later publication must not
    # silently relabel an inventory gathered for a different native process.
    states = {}
    for name, configuration in manifest['cells'].items():
        row = result['cells'][name] = {'errors': []}
        try:
            controller = Path(configuration['controller'])
            if not controller.exists():
                row['phase'] = 'not_started'
                continue
            state = json.loads(controller.read_text())
            row.update({key: state.get(key) for key in ('phase', 'complete', 'container_id', 'model', 'devices')})
            row['stats_observation'] = latest_stats(controller.parent)
            states[name] = state
        except Exception as error:
            row['errors'].append(repr(error))
    inventory = process_inventory(frozenset(state['container_id'] for state in states.values()
        if state.get('container_id') is not None))
    # Telemetry follows selected backend ownership. CPU and ROCm hosts need
    # no NVIDIA runtime, while a selected CUDA probe failure remains fatal.
    if 'cuda' in backends:
        try:
            result['cuda'] = cuda_usage()
        except Exception as error:
            result['errors'].append('NVML: ' + repr(error))
    for device in (Path('/sys/class/drm').glob('card*/device') if 'rocm' in backends else ()):
        if (device / 'mem_info_vram_used').exists():
            result['rocm'][device.resolve().name] = {
                name: int((device / name).read_text()) for name in
                ('mem_info_vram_used', 'mem_info_vram_total', 'mem_info_gtt_used', 'mem_info_gtt_total')}
    for name, state in states.items():
        row = result['cells'][name]
        try:
            identity = state.get('container_id')
            matches = [(path, value) for path, value in inventory.items()
                       if value['container_id'] == identity]
            if len(matches) > 1:
                raise ValueError('Container appears in multiple cgroups')
            if matches:
                path, processes = matches[0]
                row.update(processes)
                cgroup = Path('/sys/fs/cgroup') / path.lstrip('/')
                row['ram'] = {'current_bytes': int((cgroup / 'memory.current').read_text()),
                              'stat': numeric_fields(cgroup / 'memory.stat'),
                              'events': numeric_fields(cgroup / 'memory.events'),
                              'swap_bytes': int((cgroup / 'memory.swap.current').read_text())}
            else:
                row['processes'] = []
            archive = Path('/models') / Path(state.get('archive_host_directory', '/absent')).name
            if archive.is_dir():
                row['archive'] = inode_usage(archive, [p['pid'] for p in row['processes']],
                                             namespace_directory='/prefix-cache')
            row['inference_pids'] = [p['pid'] for p in row['processes'] if p['comm'] == 'llaminar2']
        except Exception as error:
            row['errors'].append(repr(error))
    for label, directory in manifest['filesystems'].items():
        info = os.statvfs(directory)
        result.setdefault('filesystems', {})[label] = {
            'capacity_bytes': info.f_blocks * info.f_frsize,
            'available_bytes': info.f_bavail * info.f_frsize, 'available_inodes': info.f_favail}
    result['collection_seconds'] = time.monotonic() - started
    return result


def main() -> None:
    """Journal periodic snapshots; terminal evidence requires all actual owners gone."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=20)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(exist_ok=True)
    with (args.output / 'observations.jsonl').open('x') as history:
        while True:
            try:
                value = collect(manifest)
            except Exception as error:
                value = {'observed': time.time(), 'errors': [repr(error)], 'cells': {}}
            history.write(json.dumps(value) + '\n')
            history.flush()
            pending = args.output / 'latest.tmp'
            pending.write_text(json.dumps(value, indent=2) + '\n')
            pending.replace(args.output / 'latest.json')
            if (len(value['cells']) == len(manifest['cells']) and
                    all(row.get('complete') and not row.get('inference_pids')
                        for row in value['cells'].values())):
                return
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
