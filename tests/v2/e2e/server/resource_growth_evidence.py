#!/usr/bin/env python3
"""Join OS observations to initialized, rank-authenticated admission evidence.

The server's final PerfStats export contains its immutable ready-for-inference
PhysicalMemoryAuthority attestation. A lifecycle owner seals a pre-workload
observation before starting the client. Native server startup evidence joins
execution ranks to host PID namespaces and physical devices after retirement.
This module replays the observation journal against that sealed baseline after
shutdown. It neither allocates resources nor changes a running request.

Anonymous process memory, swap and each GPU are checked independently. File RSS
and cgroup page cache remain observations, not unexplained anonymous growth.
Pinned host mappings are reported independently because several DRM clients can
map the same physical RAM. Disk capacity is checked through each cache owner;
physical disk peaks remain separate from the required post-retirement archive
ownership audit, which accounts for outstanding writers and reader inodes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Iterable

from ranked_perf_artifacts import collect_ranked_perf_stats, validate_memory_authority
from resource_growth_assertions import GrowthLimit, ResourceGrowthAssertions, natural, remaining_materialization


def timestamp(value, name: str) -> float:
    """Accept finite wall timestamps and durations, never boolean JSON values."""
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a finite nonnegative number')
    return value


def initialized_allowances(perf: dict) -> dict[tuple[int, str], int]:
    """Consume the canonical complete rank proof before deriving unused capacity.

    The one initialized snapshot is the only allowed source. A later snapshot
    cannot ratchet growth limits; a filtered document with missing owners cannot
    grant itself spare capacity. Allocator ordinals are always rank-qualified.
    """
    validate_memory_authority(perf)
    rows = []
    keys = {}
    for row in perf['records']:
        if row.get('domain') != 'physical_memory' or row.get('name') not in (
                'resource_admission', 'owner_attestation'):
            continue
        if (row.get('phase') != 'initialized' or row.get('kind') != 'counter'
                or type(row.get('value')) not in (int, float) or row['value'] != 1):
            raise ValueError('Growth limits require the initialized authority snapshot')
        if row['name'] != 'owner_attestation':
            continue
        tags = row['tags']
        key = (row['rank'], row['device'])
        allocator = f'{key[0]}/{key[1]}'
        keys[allocator] = key
        rows.append({'allocator': allocator, 'owner': tags['owner'], **{
            field: int(tags[field]) for field in (
                'planned_new_bytes', 'planned_resident_bytes', 'materialized_new_bytes',
                'committed_new_bytes', 'adopted_resident_bytes')}})
    return {keys[key]: value for key, value in remaining_materialization(rows).items()}


@dataclass(frozen=True)
class DeviceBinding:
    """Native inventory identity; device ordinals must never select an OS GPU."""

    device: str
    physical_id: str

    def __post_init__(self) -> None:
        """Reject ambiguous NVML ordinals, unknown backends and incomplete BDFs."""
        if re.fullmatch(r'CUDA:[0-9]+', self.device):
            if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', self.physical_id):
                raise ValueError('CUDA resource binding requires the native GPU UUID')
        elif re.fullmatch(r'ROCm:[0-9]+', self.device):
            if not re.fullmatch(r'[0-9a-f]{4,8}:[0-9a-f]{2}:[01][0-9a-f]\.[0-7]', self.physical_id):
                raise ValueError('ROCm resource binding requires the native PCI address')
        else:
            raise ValueError('Unknown GPU allocator in resource binding')


@dataclass(frozen=True)
class RankBinding:
    """A host-PID/start-time identity joined by the lifecycle owner to one rank."""

    rank: int
    pid: int
    start_ticks: int
    devices: tuple[DeviceBinding, ...]

    def __post_init__(self) -> None:
        """Forbid rank/PID aliases or repeated physical devices within one rank."""
        natural(self.rank, 'rank')
        if not natural(self.pid, 'pid') or not natural(self.start_ticks, 'process start'):
            raise ValueError('Missing process lifetime identity')
        if (len({row.device for row in self.devices}) != len(self.devices)
                or len({row.physical_id for row in self.devices}) != len(self.devices)):
            raise ValueError('Duplicate GPU resource binding')


def bind_native_ranks(perf: dict, observation: dict, cell: str) -> tuple[RankBinding, ...]:
    """Join server-authored rank/PID/PCI identity to one sealed host observation.

    Linux NSpid lists the host PID first and the innermost namespace PID last.
    The server emits getpid() in that namespace, while the observer supplies the
    start ticks. This also works for a singleton launched without mpirun. Neither
    environment rank variables nor PID/GPU enumeration order select an owner.
    """
    size = natural(perf['world_size'], 'world size')
    if not size:
        raise ValueError('Missing execution communicator')
    processes = [p for p in observation['cells'][cell]['processes'] if p['comm'] == 'llaminar2']
    namespace = {}
    for process in processes:
        pids = process['namespace_pids']
        if (not isinstance(pids, list) or not pids or
                any(not natural(pid, 'namespace PID') for pid in pids) or
                pids[0] != process['pid'] or pids[-1] in namespace):
            raise ValueError('Missing or ambiguous host PID namespace identity')
        namespace[pids[-1]] = process
    identities, participants = {}, {}
    for row in perf['records']:
        if row.get('domain') != 'server' or row.get('name') not in ('process_identity', 'execution_participant'):
            continue
        rank = natural(row['rank'], 'server evidence rank')
        if (rank >= size or row.get('phase') != 'startup' or row.get('kind') != 'counter' or
                type(row.get('value')) not in (int, float) or row['value'] != 1):
            raise ValueError('Resource binding requires exact native startup evidence')
        tags = row['tags']
        if tags.get('schema') != '1':
            raise ValueError('Unknown native identity schema')
        if row['name'] == 'process_identity':
            if (rank in identities or tags.get('identity_source') != 'native_pid_namespace' or
                    not re.fullmatch(r'[1-9][0-9]*', tags.get('pid', ''))):
                raise ValueError('Missing or duplicate native rank process identity')
            identities[rank] = int(tags['pid'])
            continue
        device = row['device']
        if (tags.get('identity_source') != 'communicator_cluster_inventory' or
                tags.get('source') != 'resolved_execution_plan' or
                tags.get('backend') != device.split(':')[0]):
            raise ValueError('GPU identity does not come from the admitted inventory')
        if re.fullmatch(r'CPU(?::[0-9]+)?', device):
            continue
        uuid, pci = tags['physical_id'], tags['pci_bus_address']
        if not re.fullmatch(r'[0-9a-f]{32}', uuid) or uuid == '0' * 32:
            raise ValueError('GPU inventory has no usable native UUID')
        if not re.fullmatch(r'[0-9a-f]{4,8}:[0-9a-f]{2}:[01][0-9a-f]\.[0-7]', pci):
            raise ValueError('GPU inventory has no complete canonical PCI endpoint')
        if device.startswith('CUDA:'):
            physical = 'GPU-' + '-'.join((uuid[:8], uuid[8:12], uuid[12:16], uuid[16:20], uuid[20:]))
            inventory = [d for d in observation['cuda']['devices'] if d['uuid'] == physical]
            if len(inventory) != 1:
                raise ValueError('Native UUID is absent from NVML inventory')
            # NVML pads the domain to eight digits; Linux normally uses four.
            address = inventory[0]['pci.bus_id'].lower()
            domain, endpoint = address.split(':', 1)
            address = f'{int(domain, 16):04x}:{endpoint}'
            if address != pci:
                raise ValueError('Native UUID and NVML PCI endpoint disagree')
        else:
            physical = pci
        participants.setdefault(rank, []).append(DeviceBinding(device, physical))
    if set(identities) != set(range(size)) or set(identities.values()) != set(namespace) or len(namespace) != size:
        raise ValueError('Native process identity does not cover the authenticated ranks')
    return tuple(RankBinding(rank, namespace[identities[rank]]['pid'],
        namespace[identities[rank]]['start_ticks'],
        tuple(sorted(participants.get(rank, []), key=lambda d: d.device))) for rank in range(size))


@dataclass(frozen=True)
class WorkloadWindow:
    """Explicit workload frontiers; telemetry cadence never limits generation time."""

    baseline: float
    started: float
    finished: float
    maximum_sample_gap_seconds: float
    maximum_stats_age_seconds: float
    page_size_bytes: int

    def __post_init__(self) -> None:
        """Require a baseline fixed before client work and an observed final frontier."""
        for name in ('baseline', 'started', 'finished', 'maximum_sample_gap_seconds', 'maximum_stats_age_seconds'):
            timestamp(getattr(self, name), name)
        if not self.baseline <= self.started < self.finished:
            raise ValueError('Baseline must precede the complete workload')
        if not self.maximum_sample_gap_seconds or not self.maximum_stats_age_seconds:
            raise ValueError('Observation freshness must be positive')
        if not natural(self.page_size_bytes, 'page size') or self.page_size_bytes & (self.page_size_bytes - 1):
            raise ValueError('Resource observation requires the native page size')


def observation_metrics(observation: dict, cell: str, ranks: tuple[RankBinding, ...]) -> tuple[dict, dict]:
    """Read exact process/GPU measurements without summing shared host mappings."""
    row = observation['cells'][cell]
    if observation.get('schema_version') != 2 or observation.get('errors') or row.get('errors'):
        raise ValueError('OS resource observation failed or has an unknown schema')
    processes = [p for p in row['processes'] if p['comm'] == 'llaminar2']
    if ({p['pid'] for p in processes} != {r.pid for r in ranks}
            or len(processes) != len(ranks)):
        raise ValueError('Missing, replaced or duplicate inference process')
    metrics, diagnostic = {}, {'processes': {}, 'archive': row.get('archive'), 'cgroup': row.get('ram')}
    for rank in ranks:
        process = next(p for p in processes if p['pid'] == rank.pid)
        if process['start_ticks'] != rank.start_ticks:
            raise ValueError('Inference PID was reused')
        status = process['status']
        # Swapping pages out must not make a growing anonymous heap look bounded.
        metrics[f'rank/{rank.rank}/anonymous_and_swap'] = (
            natural(status['RssAnon'], 'anonymous RSS') + natural(status['VmSwap'], 'process swap'))
        diagnostic['processes'][str(rank.rank)] = {
            'status': status, 'drm_mappings': process.get('drm_clients', [])}
        for device in rank.devices:
            key = f'rank/{rank.rank}/{device.device}'
            if device.device.startswith('CUDA:'):
                cuda = observation['cuda']
                if cuda['resolution_bytes'] != 1024**2:
                    raise ValueError('Unexpected NVML measurement resolution')
                inventory = [d for d in cuda['devices'] if d['uuid'] == device.physical_id]
                owners = [p for p in cuda['processes']
                          if p['pid'] == rank.pid and p['gpu_uuid'] == device.physical_id]
                if len(inventory) != 1 or len(owners) != 1:
                    raise ValueError('Missing or duplicated native CUDA process/device observation')
                metrics[key] = natural(owners[0]['used_gpu_memory_bytes'], 'NVML process bytes')
            else:
                clients = [c for c in process['drm_clients'] if c['drm-pdev'] == device.physical_id]
                identities = {c['drm-client-id'] for c in clients}
                if not clients or len(identities) != len(clients):
                    raise ValueError('Missing or duplicated native ROCm DRM client')
                metrics[key] = sum(natural(c['drm-total-vram_bytes'], 'DRM VRAM bytes') for c in clients)
    ram = row.get('ram')
    if ram:
        events = ram['events']
        if natural(events['oom'], 'cgroup OOM events') or natural(events['oom_kill'], 'cgroup OOM kills'):
            raise ValueError('Inference cgroup reported an OOM event')
    return metrics, diagnostic


def assess_observations(*, observations: Iterable[dict], perf: dict, cell: str,
                        container_id: str, boot_id: str,
                        window: WorkloadWindow, cache_capacity: dict[str, int]) -> dict:
    """Replay a complete workload without changing its baseline or deleting failures.

    Native rank/PID/device binding comes from server startup and OS namespaces,
    never a caller's device map. This receipt covers process/GPU growth
    and owner-reported cache capacity. Physical archive ownership still requires
    the independent metadata-only audit after all native owners have retired.
    """
    failures, gate, previous, covered_finish = [], None, None, False
    ranks: tuple[RankBinding, ...] = ()
    diagnostic_peaks: dict[str, int] = {}
    try:
        allowances = initialized_allowances(perf)
        if not re.fullmatch(r'[0-9a-f]{64}', container_id) or not boot_id:
            raise ValueError('Missing immutable container or host boot identity')
        host_allowances = {}
        for rank in range(perf['world_size']):
            host = [value for (owner_rank, device), value in allowances.items()
                    if owner_rank == rank and re.fullmatch(r'CPU(?::[0-9]+)?', device)]
            if not host:
                raise ValueError('Missing CPU allocator authority for an inference rank')
            host_allowances[rank] = sum(host)
        for sample in observations:
            if covered_finish:
                break
            observed = timestamp(sample['observed'], 'resource observation')
            if observed < window.baseline:
                if previous is not None:
                    failures.append('OS observation timestamp regressed before the sealed baseline')
                continue
            if previous is None and observed != window.baseline:
                raise ValueError('The sealed pre-workload baseline is missing')
            if previous is not None and not 0 < observed - previous <= window.maximum_sample_gap_seconds:
                failures.append('Missing, reordered or duplicate OS resource observations')
            previous = observed
            try:
                row = sample['cells'][cell]
                if sample['boot_id'] != boot_id or row['container_id'] != container_id:
                    raise ValueError('Host boot or inference container changed')
                if gate is None:
                    ranks = bind_native_ranks(perf, sample, cell)
                    expected = {(r.rank, d.device) for r in ranks for d in r.devices}
                    if {key for key in allowances if not re.fullmatch(r'CPU(?::[0-9]+)?', key[1])} != expected:
                        raise ValueError('Native GPU binding differs from admitted allocator inventory')
                metrics, diagnostic = observation_metrics(sample, cell, ranks)
                if gate is None:
                    limits = {}
                    for rank in ranks:
                        key = f'rank/{rank.rank}/anonymous_and_swap'
                        limits[key] = GrowthLimit(metrics[key], host_allowances[rank.rank], window.page_size_bytes,
                            f'initialized PhysicalMemoryAuthority rank {rank.rank} CPU owner rows')
                        for device in rank.devices:
                            key = f'rank/{rank.rank}/{device.device}'
                            limits[key] = GrowthLimit(metrics[key], allowances[(rank.rank, device.device)],
                                1024**2 if device.device.startswith('CUDA:') else window.page_size_bytes,
                                f'initialized PhysicalMemoryAuthority rank {rank.rank} allocator {device.device}')
                    gate = ResourceGrowthAssertions(identity={'boot': boot_id, 'container': container_id},
                        limits=limits, cache_capacity=cache_capacity,
                        maximum_sample_age_seconds=window.maximum_stats_age_seconds)
                gate.observe(observed=observed + timestamp(sample['collection_seconds'], 'probe duration'),
                    identity={'boot': sample['boot_id'], 'container': row['container_id']},
                    metrics=metrics, stats=row['stats_observation']['stats'], errors=[])
                archive = diagnostic['archive']
                if archive:
                    for field in ('logical_bytes', 'allocated_bytes', 'inodes'):
                        key = 'archive_' + field
                        diagnostic_peaks[key] = max(diagnostic_peaks.get(key, 0), natural(archive[field], key))
                for rank, details in diagnostic['processes'].items():
                    for field in ('VmRSS', 'VmHWM', 'RssAnon', 'RssFile', 'RssShmem', 'VmSwap'):
                        key = f'rank/{rank}/{field}'
                        diagnostic_peaks[key] = max(diagnostic_peaks.get(key, 0), natural(details['status'][field], field))
            except (KeyError, ValueError, TypeError) as error:
                failures.append(f'observation {observed}: {error}')
                # The first failed sample is not permission to pick a later,
                # larger baseline and make the original failure disappear.
                if gate is None:
                    break
            covered_finish = observed >= window.finished
        if not covered_finish:
            failures.append('Resource observations do not cover the complete workload')
    except (KeyError, ValueError, TypeError) as error:
        failures.append(str(error))
    result = gate.result() if gate else {'passed': False, 'samples': 0}
    return {'schema_version': 1, 'scope': 'process_gpu_growth_and_cache_capacity',
            'passed': result['passed'] and not failures, 'cell': cell,
            'container_id': container_id, 'boot_id': boot_id,
            'window': asdict(window), 'ranks': [asdict(rank) for rank in ranks],
            'growth': result, 'observation_failures': failures,
            'diagnostic_peaks': diagnostic_peaks,
            'physical_archive_ownership_audit_required': True}


def main() -> int:
    """Assess the lifecycle owner's fixed binding and journal after server teardown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--perf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    binding = json.loads(args.binding.read_text())
    window = WorkloadWindow(**binding.pop('window'))
    perf = collect_ranked_perf_stats(args.perf)
    with args.observations.open() as stream:
        result = assess_observations(observations=(json.loads(line) for line in stream), perf=perf,
                                     window=window, **binding)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
