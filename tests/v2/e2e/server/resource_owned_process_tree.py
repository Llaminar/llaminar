#!/usr/bin/env python3
"""Observe one launched process and its descendants through bounded proc metadata.

MPI launchers and session guardians do not own the engine's resident memory.
Follow their actual child links, pin each PID to its kernel start time, and
retain discovered descendants through parent retirement. Never scan the host
or select a process by name. Per-process RSS is diagnostic: shared mappings
make summing it into a physical-memory total incorrect.
"""
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

from resource_process_identity import read_process_status


def bounded_text(path: Path) -> str:
    """Read kernel metadata only, rejecting an incomplete oversized record."""
    with path.open('rb') as stream:
        value = stream.read(65537)
    if len(value) > 65536:
        raise ValueError('Process metadata exceeds its 64-KiB bound')
    return value.decode('utf-8')


@dataclass(frozen=True, order=True)
class ProcessIdentity:
    """One Linux process lifetime; a recycled PID never inherits observation."""

    pid: int
    start_ticks: int

    def __post_init__(self) -> None:
        """Reject booleans, missing start times, and non-process identifiers."""
        if any(type(value) is not int or value <= 0 for value in (self.pid, self.start_ticks)):
            raise ValueError('Process observation requires PID and kernel start ticks')


def process_stat(directory: Path) -> dict:
    """Parse identity and ancestry without opening environment or argv data."""
    value = bounded_text(directory / 'stat')
    prefix, separator, tail = value.partition(' (')
    comm, closing, tail = tail.rpartition(') ')
    fields = tail.split()
    if not separator or not closing or prefix != directory.name or len(fields) < 20:
        raise ValueError('Malformed process identity record')
    identity = ProcessIdentity(int(prefix), int(fields[19]))
    parent = int(fields[1])
    if parent < 0 or len(fields[0]) != 1:
        raise ValueError('Malformed process state or parent')
    return {**asdict(identity), 'ppid': parent, 'comm': comm, 'state': fields[0]}


class OwnedProcessTree:
    """Observe an original Popen lifetime and children discovered beneath it.

    Child discovery reads every owned thread's children because MPI helpers
    may launch workers from non-main threads. A child must still name that
    parent before any memory is read. Once admitted, its original identity
    remains owned even when its parent exits and it is reparented.
    """

    def __init__(self, pid: int, *, proc: Path = Path('/proc')) -> None:
        """Pin the live launch owner before the caller reaps its Popen object."""
        if type(pid) is not int or pid <= 0:
            raise ValueError('Local observation requires an owned positive PID')
        self.proc = proc
        first = process_stat(proc / str(pid))
        self.root = ProcessIdentity(first['pid'], first['start_ticks'])
        self._known = {self.root}

    def _current(self, identity: ProcessIdentity) -> dict | None:
        """Check lifetime before deeper reads; missing or reused means retired."""
        try:
            value = process_stat(self.proc / str(identity.pid))
        except (FileNotFoundError, ProcessLookupError):
            return None
        return value if value['start_ticks'] == identity.start_ticks else None

    def _children(self, identity: ProcessIdentity) -> list[int]:
        """Read only this owner's thread directories and recheck its lifetime."""
        directory = self.proc / str(identity.pid)
        children = set()
        try:
            for task in (directory / 'task').iterdir():
                if not task.name.isdecimal():
                    continue
                try:
                    values = bounded_text(task / 'children').split()
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if any(not value.isascii() or not value.isdecimal() or int(value) <= 0 for value in values):
                    raise ValueError('Malformed owned process children')
                children.update(map(int, values))
        except (FileNotFoundError, ProcessLookupError):
            return []
        return sorted(children) if self._current(identity) is not None else []

    def snapshot(self) -> dict:
        """Return individual memory rows, explicit zombies, and retired identities.

        Status and ancestry are bracketed by start-time checks. Exit races are
        explicit retirement; permission or malformed live metadata is an error.
        No missing counter is converted into zero memory consumption.
        """
        pending = deque(sorted(self._known))
        seen, retired, rows = set(), set(), []
        while pending:
            identity = pending.popleft()
            if identity in seen:
                continue
            seen.add(identity)
            before = self._current(identity)
            if before is None:
                retired.add(identity)
                continue
            directory = self.proc / str(identity.pid)
            for pid in self._children(identity):
                try:
                    child = process_stat(self.proc / str(pid))
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if child['ppid'] != identity.pid or self._current(identity) is None:
                    continue
                owner = ProcessIdentity(pid, child['start_ticks'])
                self._known.add(owner)
                pending.append(owner)
            if before['state'] in ('Z', 'X'):
                rows.append({**before, 'memory_available': False})
                continue
            try:
                memory = read_process_status(directory / 'status', identity.pid)
            except (FileNotFoundError, ProcessLookupError, ValueError):
                after = self._current(identity)
                if after is None:
                    retired.add(identity)
                    continue
                if after['state'] in ('Z', 'X'):
                    rows.append({**after, 'memory_available': False})
                    continue
                raise
            after = self._current(identity)
            if after is None:
                retired.add(identity)
                continue
            rows.append({**after, **memory, 'memory_available': True})
        self._known.difference_update(retired)
        return {'root': asdict(self.root), 'processes': sorted(rows, key=lambda row: row['pid']),
                'retired': [asdict(value) for value in sorted(retired)],
                'memory_scope': 'individual_processes_not_summed'}
