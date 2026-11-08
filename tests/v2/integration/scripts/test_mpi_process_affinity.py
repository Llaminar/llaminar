#!/usr/bin/env python3
"""Replay actual public owner/self-entry with externally authenticated CPU domains.

The observer does not load OpenMP. It chooses two admitted physical cores before
starting the exact production-linked probe. Bound, intentionally narrow, and
unbound policies must preserve only their original allowed partition; neither a
requested thread count nor the frontend's first-place mask certifies placement.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def cpu_list(text):
    """Parse the probe's explicit CPU identifiers, without inventing host placement."""
    return {int(item) for item in text.split(',') if item}


def physical(cpu):
    """Observe the kernel's physical identity independently of production policy."""
    base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
    return tuple(int((base / name).read_text()) for name in ('physical_package_id', 'core_id'))


def main():
    """Reject the original one-place child and verify exact restoration after retirement."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--helper', type=Path, required=True)
    args = parser.parse_args()
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        cores.setdefault(physical(cpu), cpu)
    if len(cores) < 2:
        raise RuntimeError('Process-affinity replay requires two admitted physical CPU cores')
    selected = list(cores.values())[:2]
    outer = set(selected)
    with tempfile.TemporaryDirectory(prefix='mpi-process-affinity-') as temporary:
        root = Path(temporary)
        for name, places, expected, bound in (
                ('bound', ','.join(f'{{{cpu}}}' for cpu in selected), outer, True),
                ('narrow', f'{{{selected[0]}}}', {selected[0]}, True),
                ('unbound', None, outer, False)):
            for route in ('self-entry', 'public-helper'):
                observation = root / f'{name}-{route}.txt'
                workers = len(expected)
                env = os.environ.copy()
                for key in ('OMP_PLACES', 'OMP_PROC_BIND', 'GOMP_CPU_AFFINITY', 'OMP_THREAD_LIMIT'):
                    env.pop(key, None)
                env.update(OMP_NUM_THREADS=str(workers), OMP_DYNAMIC='false',
                           OMP_PROC_BIND='close' if bound else 'false')
                if places:
                    env['OMP_PLACES'] = places
                command = [str(args.probe.resolve()), str(observation), str(workers)]
                if route == 'public-helper':
                    command = [str(args.helper.resolve()), '--session-base', str(root), '--'] + command
                command = ['taskset', '--cpu-list', ','.join(map(str, selected))] + command
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
                if result.returncode:
                    raise RuntimeError(f'{name}/{route} failed: {result.returncode}\n{result.stdout}{result.stderr}')
                lines = observation.read_text().splitlines()
                parent_receipt = (observation.with_suffix('.txt.parent').read_text()
                                  if route == 'self-entry' else None)
                print(json.dumps(dict(case=name, route=route, argv=command,
                                      expected_cpus=sorted(expected), actual_observation=lines,
                                      parent_restoration=parent_receipt, stdout=result.stdout,
                                      stderr=result.stderr)), flush=True)
                if lines[0] != f'received={workers}' or len(lines) != workers + 1:
                    raise AssertionError(f'Incomplete actual worker evidence: {lines}')
                observed = []
                for line in lines[1:]:
                    tid, cpu, mask = line.split()
                    allowed = cpu_list(mask)
                    if not allowed <= expected or int(cpu) not in allowed:
                        raise AssertionError(f'{name}/{route} escaped or lost its admitted partition: {line}')
                    if (bound and len(allowed) != 1) or (not bound and allowed != expected):
                        raise AssertionError(f'{name}/{route} actual worker mask differs from its policy: {line}')
                    observed.append(allowed)
                if set.union(*observed) != expected:
                    raise AssertionError(f'{name}/{route} lost original partition: {observed}, expected={expected}')
                if bound and len({physical(next(iter(mask))) for mask in observed}) != workers:
                    raise AssertionError('Requested workers did not receive distinct physical cores')
                # Public helper enters the probe as an already-owned process;
                # self-entry additionally returns through its waiting frontend.
                if route == 'self-entry':
                    parent = dict(line.split('=', 1) for line in observation.with_suffix('.txt.parent').read_text().splitlines())
                    if parent['original'] != parent['restored']:
                        raise AssertionError('Waiting frontend did not restore its exact original native mask')
                print(f'PASS {name}/{route}', flush=True)


if __name__ == '__main__':
    main()
