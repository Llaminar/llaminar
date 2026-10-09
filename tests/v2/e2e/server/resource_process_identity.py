#!/usr/bin/env python3
"""Read bounded Linux process metadata for rank and memory attribution.

The host observer owns the PID namespace; inference publishes its native PID in
startup evidence. NSpid joins those namespaces without reading environment data,
guessing MPI rank order or opening model/cache files. Units remain explicit.
"""
from pathlib import Path


def read_process_status(path: Path, host_pid: int) -> dict:
    """Read one complete status snapshot, retaining memory and namespace identity.

    A malformed or unavailable field is not zero usage. Linux presents process
    metadata in a small text file; a fixed bound also prevents a wrong mount or
    fixture from causing an unbounded read in the telemetry worker.
    """
    if type(host_pid) is not int or host_pid <= 0:
        raise ValueError('Process observation requires a positive host PID')
    with path.open('rb') as stream:
        payload = stream.read(65537)
    if len(payload) > 65536:
        raise ValueError('Process status exceeds its metadata bound')
    fields = {}
    wanted = {'VmRSS', 'VmHWM', 'RssAnon', 'RssFile', 'RssShmem', 'VmSwap', 'NSpid', 'Pid'}
    for line in payload.decode('ascii').splitlines():
        key, separator, value = line.partition(':')
        if key not in wanted:
            continue
        if not separator or key in fields:
            raise ValueError('Missing or duplicate process status field')
        fields[key] = value.split()
    if set(fields) != wanted or fields['Pid'] != [str(host_pid)]:
        raise ValueError('Process status lacks its host PID or memory fields')
    identifiers = fields.pop('NSpid')
    fields.pop('Pid')
    if (not identifiers or any(not value.isdecimal() or int(value) <= 0 for value in identifiers)
            or int(identifiers[0]) != host_pid):
        raise ValueError('Process status has an invalid PID namespace chain')
    memory = {}
    for name, value in fields.items():
        if len(value) != 2 or not value[0].isdecimal() or value[1] != 'kB':
            raise ValueError('Process memory counter has an unknown value or unit')
        memory[name] = int(value[0]) * 1024
    return {'status': memory, 'namespace_pids': list(map(int, identifiers))}
