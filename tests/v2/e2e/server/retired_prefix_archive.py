#!/usr/bin/env python3
"""Reclaim one test-owned prefix archive after its native owners have retired.

Canonical coding cells each get a fresh bounded cache. Retaining every cell's
32 GiB payload would make a full CI matrix consume terabytes despite bounded
serving growth. Completed metadata audits remain in the evidence directory;
this helper removes only the separately mounted archive and never opens a
payload. Retirement failure leaves storage intact for the still-live owners.
"""
import argparse
import json
from pathlib import Path
import shutil


def reclaim(directory: Path, retirement: dict) -> dict:
    """Require complete native retirement before unlinking any owned entry."""
    owners = retirement.get('retired')
    if (retirement.get('passed') is not True or retirement.get('errors') != {}
            or not isinstance(owners, dict) or not owners
            or any(state.get('Running') is not False or type(state.get('Pid')) is not int
                   or state['Pid'] != 0 or state.get('Status') not in {'exited', 'created', 'absent'}
                   for state in owners.values())):
        raise ValueError('Prefix archive reclamation requires proven native retirement')
    if directory.is_symlink() or not directory.is_dir() or directory.resolve() == Path('/'):
        raise ValueError('Prefix archive is not an isolated owned directory')
    entries = list(directory.iterdir())
    for entry in entries:
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    if list(directory.iterdir()):
        raise ValueError('Retired prefix archive changed during reclamation')
    return {'passed': True, 'removed_entries': len(entries), 'remaining_entries': 0,
            'payload_read_bytes': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--retirement', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(reclaim(args.directory, json.loads(args.retirement.read_text()))), flush=True)
