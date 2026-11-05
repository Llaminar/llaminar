#!/usr/bin/env python3
"""@file CPUHostTopology.py
@brief Immutable Linux affinity/sysfs topology for test admission and CPU slot budgets.

Presentation-oriented lscpu fields/IDs are not physical topology authorities on
hybrid ARM. One observation retains original logical CPUs, package/core labels
and kernel thread-sibling sets. Physical cores are identified by their sibling
sets within packages, not by possibly repeated presentation labels. Unknown,
inconsistent or changing observations fail; no one-core geometry is invented.
This is a configure-time test-host snapshot, not the inference inventory or a
physical-memory ledger. It performs no affinity, package or host-policy changes.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Callable, Iterable


class TopologyError(ValueError):
    """Reject an incomplete or inconsistent test-host observation."""


@dataclass(frozen=True, slots=True)
class CPUIdSet:
    """Canonical disjoint inclusive intervals; large IDs/ranges never cause expansion."""

    intervals: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        """Make malformed/ambiguous interval identities unrepresentable."""
        previous = -2
        if type(self.intervals) is not tuple or not self.intervals or any(
                type(pair) is not tuple or len(pair) != 2 for pair in self.intervals):
            raise TopologyError('kernel CPU-list identity must be immutable nonempty interval pairs')
        for first, last in self.intervals:
            if type(first) is not int or type(last) is not int or first < 0 or first > last or first <= previous + 1:
                raise TopologyError('noncanonical kernel CPU-list identity')
            previous = last

    @classmethod
    def parse(cls, text: str) -> CPUIdSet:
        """Parse the kernel's CPU-list grammar, rejecting duplicate/overlapping IDs."""
        ranges = []
        for token in text.strip().split(','):
            if not re.fullmatch(r'[0-9]+(?:-[0-9]+)?', token):
                raise TopologyError(f'invalid kernel CPU-list token: {token!r}')
            bounds = token.split('-')
            first, last = int(bounds[0]), int(bounds[-1])
            if first > last:
                raise TopologyError(f'reversed kernel CPU range: {token}')
            ranges.append((first, last))
        merged: list[tuple[int, int]] = []
        for first, last in sorted(ranges):
            if merged and first <= merged[-1][1]:
                raise TopologyError('overlapping kernel CPU-list intervals')
            if merged and first == merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], last)
            else:
                merged.append((first, last))
        return cls(tuple(merged))

    def contains(self, cpu: int) -> bool:
        """Test one observed CPU without expanding a potentially sparse/large range."""
        return any(first <= cpu <= last for first, last in self.intervals)

    def text(self) -> str:
        """Serialize the canonical kernel-list representation without losing IDs."""
        return ','.join(str(a) if a == b else f'{a}-{b}' for a, b in self.intervals)


@dataclass(frozen=True, slots=True)
class LogicalCPU:
    """One original affinity-visible CPU and its genuine kernel topology labels."""

    cpu_id: int
    package_id: int
    core_id: int
    siblings: CPUIdSet


@dataclass(frozen=True, slots=True)
class TestHostTopology:
    """Complete immutable test-host evidence, retaining asymmetric packages/SMT as vectors."""

    cpus: tuple[LogicalCPU, ...]

    def __post_init__(self) -> None:
        """Authenticate membership/symmetry before admitting any derived count."""
        if type(self.cpus) is not tuple or not self.cpus:
            raise TopologyError('test-host affinity requires an immutable nonempty CPU tuple')
        if any(not isinstance(row, LogicalCPU) or not isinstance(row.siblings, CPUIdSet) for row in self.cpus):
            raise TopologyError('test-host topology contains an invalid CPU/sibling record')
        by_id = {row.cpu_id: row for row in self.cpus}
        if len(by_id) != len(self.cpus):
            raise TopologyError('duplicate affinity-visible CPU identity')
        for row in self.cpus:
            for value in (row.cpu_id, row.package_id, row.core_id):
                if type(value) is not int or value < 0:
                    raise TopologyError('kernel CPU/package/core identity is unknown or invalid')
            if not row.siblings.contains(row.cpu_id):
                raise TopologyError(f'CPU{row.cpu_id} is absent from its thread-sibling set')
            for sibling in self.cpus:
                if row.siblings.contains(sibling.cpu_id) and (
                    sibling.package_id != row.package_id or sibling.core_id != row.core_id or
                    sibling.siblings != row.siblings):
                    raise TopologyError(f'inconsistent kernel thread-sibling authority for CPU{row.cpu_id}')
        # Distinct sibling sets must be disjoint even outside current affinity;
        # otherwise two partial groups could claim the same physical core.
        sets = list({(row.package_id, row.siblings) for row in self.cpus})
        for index, (package, siblings) in enumerate(sets):
            for other_package, other in sets[index + 1:]:
                if any(max(a, c) <= min(b, d) for a, b in siblings.intervals for c, d in other.intervals):
                    raise TopologyError(f'overlapping physical-core sibling authorities in packages{package}/{other_package}')

    def document(self) -> dict:
        """Return exact observed topology; consumers must reject projections they cannot express."""
        packages: dict[int, dict[CPUIdSet, int]] = {}
        for row in self.cpus:
            cores = packages.setdefault(row.package_id, {})
            cores[row.siblings] = cores.get(row.siblings, 0) + 1
        widths = sorted((width for cores in packages.values() for width in cores.values()), reverse=True)
        return {
            'schema_version': 1,
            'cpus': [{'cpu_id': row.cpu_id, 'package_id': row.package_id,
                      'core_id': row.core_id, 'thread_siblings': row.siblings.text()}
                     for row in sorted(self.cpus, key=lambda item: item.cpu_id)],
            'packages': [{'package_id': package, 'physical_cores': len(cores),
                          'logical_cpus': sum(cores.values())} for package, cores in sorted(packages.items())],
            'package_count': len(packages),
            'physical_core_count': len(widths),
            'logical_cpu_count': len(self.cpus),
            'core_thread_widths': widths,
        }

    @classmethod
    def from_document(cls, document: dict) -> TestHostTopology:
        """Reject stale/forged derived fields rather than accepting parallel arithmetic."""
        try:
            rows = tuple(LogicalCPU(row['cpu_id'], row['package_id'], row['core_id'],
                                   CPUIdSet.parse(row['thread_siblings'])) for row in document['cpus'])
            topology = cls(rows)
            if json.dumps(topology.document(), sort_keys=True) != json.dumps(document, sort_keys=True):
                raise TopologyError('test-host topology document does not match its original CPU evidence')
            return topology
        except (KeyError, TypeError, AttributeError) as error:
            raise TopologyError(f'incomplete test-host topology document: {error}') from error


def observe_linux(sysfs_root: Path = Path('/sys/devices/system/cpu'),
                  affinity: Callable[[], Iterable[int]] | None = None) -> TestHostTopology:
    """Capture affinity/sysfs once; explicit dependencies are device-free fixture seams only."""
    if affinity is None:
        if platform.system() != 'Linux' or not hasattr(os, 'sched_getaffinity'):
            raise TopologyError('native test-host topology observation requires Linux affinity/sysfs')
        affinity = lambda: os.sched_getaffinity(0)
    allowed = tuple(sorted(affinity()))
    rows = []
    for cpu in allowed:
        if type(cpu) is not int or cpu < 0:
            raise TopologyError('invalid affinity CPU identity')
        directory = sysfs_root / f'cpu{cpu}' / 'topology'
        try:
            values = [(directory / name).read_text(encoding='ascii').strip()
                      for name in ('physical_package_id', 'core_id', 'thread_siblings_list')]
        except (OSError, UnicodeError) as error:
            raise TopologyError(f'incomplete kernel topology for CPU{cpu}: {error}') from error
        if not all(re.fullmatch(r'[0-9]+', value) for value in values[:2]):
            raise TopologyError(f'unknown kernel package/core identity for CPU{cpu}')
        rows.append(LogicalCPU(cpu, int(values[0]), int(values[1]), CPUIdSet.parse(values[2])))
    if tuple(sorted(affinity())) != allowed:
        raise TopologyError('affinity changed during test-host topology observation')
    return TestHostTopology(tuple(rows))


def main() -> int:
    """Publish only an authenticated native snapshot; errors are fatal and never a one-core default."""
    try:
        print(json.dumps(observe_linux().document(), sort_keys=True))
        return 0
    except TopologyError as error:
        print(f'CPU test-host topology admission failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
