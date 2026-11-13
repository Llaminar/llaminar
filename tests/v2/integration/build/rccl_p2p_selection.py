"""Actual host-only native P2P selector extension of the existing selector proof owner.

The selected native build supplies both complete translation units and flags.
An exact reverse patch is the sole original-code control. Each policy runs in a
fresh process, retaining native static policy caching and actual parser logic.
"""
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile

PATCH = 'scripts/docker/patches/rccl-user-p2p-override.patch'
PROBE = 'tests/v2/integration/build/rccl_p2p_selection_probe.cpp'
INPUTS = (PATCH, PROBE, 'tests/v2/integration/build/rccl_p2p_selection.py')
NATIVE = ('src/graph/paths.cc', 'src/graph/topo.cc')
# name, explicit native environment, independently expected maximum path.
POLICIES = (
    ('unset', {}, -1),
    ('disable', {'NCCL_P2P_DISABLE': '1'}, 0),
    ('disable_over_level', {'NCCL_P2P_DISABLE': '1', 'NCCL_P2P_LEVEL': 'SYS'}, 0),
    ('enabled_default', {'NCCL_P2P_DISABLE': '0'}, -1),
    ('loc', {'NCCL_P2P_LEVEL': 'LOC'}, 0),
    ('pix', {'NCCL_P2P_LEVEL': 'PIX'}, 4),
    ('pxb', {'NCCL_P2P_LEVEL': 'PXB'}, 5),
    ('phb', {'NCCL_P2P_LEVEL': 'PHB'}, 8),
    ('sys', {'NCCL_P2P_LEVEL': 'SYS'}, 9),
    ('legacy_zero', {'NCCL_P2P_LEVEL': '0'}, 0),
    ('legacy_five', {'NCCL_P2P_LEVEL': '5'}, 9),
)
CHECKS = 240
# Three CPU defaults are PXB in the original; these explicit policies differ.
FAILURES = {'disable': 15, 'disable_over_level': 15, 'loc': 15,
            'pix': 3, 'phb': 9, 'sys': 12, 'legacy_zero': 15, 'legacy_five': 12}


def verify(report):
    """Reject omitted cases, stale input declarations and vacuous original controls."""
    if not isinstance(report, dict):
        raise RuntimeError('Missing actual P2P selection evidence')
    pins = report.get('native_sources', {})
    flags = report.get('native_compile_inputs', [])
    if (len(pins) != 2 or any(sum(k.endswith('/' + n) for k in pins) != 1 for n in NATIVE)
            or len(flags) != 2 or {r.get('source') for r in flags} != set(pins)
            or any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v) for v in pins.values())
            or any(not r.get('flags') or not all(isinstance(x, str) for x in r['flags']) for r in flags)):
        raise RuntimeError('Incomplete actual P2P native source/compiler inputs')
    rows = report.get('observations', [])
    expected = [(v, name) for v in ('native', 'original') for name, _, _ in POLICIES]
    if [(r.get('variant'), r.get('policy')) for r in rows] != expected:
        raise RuntimeError('Incomplete actual P2P selection policy matrix')
    for row in rows:
        failures = FAILURES.get(row['policy'], 0) if row['variant'] == 'original' else 0
        if (type(row.get('returncode')) is not int or row['returncode'] != int(failures != 0)
                or row.get('stderr') != ''
                or row.get('measured') != {'checks': CHECKS, 'failures': failures}
                or any(type(x) is not int for x in row['measured'].values())):
            raise RuntimeError('Actual P2P selector/default/negative-control outcome differs')


def compile_report(a, root, inputs, digest):
    """Execute only host selector work with the selected dependency's own compilation flags."""
    source_flags = [inputs(a.rccl_build, n) for n in NATIVE]
    report = dict(native_sources={str(s): digest(s) for s, _ in source_flags},
                  native_compile_inputs=[dict(source=str(s), flags=f) for s, f in source_flags],
                  observations=[])
    with tempfile.TemporaryDirectory(prefix='llaminar-rccl-p2p-selection-') as directory:
        tmp = Path(directory)
        original = tmp / 'original'
        dest = original / NATIVE[0]
        dest.parent.mkdir(parents=True)
        data = source_flags[0][0].read_bytes()
        preamble = b'#include "hip/hip_runtime.h"\n'
        prefix = preamble if data.startswith(preamble) else b''
        dest.write_bytes(data[len(prefix):])
        subprocess.run(['git', 'apply', '--reverse', str(root / PATCH)], cwd=original, check=True)
        dest.write_bytes(prefix + dest.read_bytes())
        symbols = tmp / 'probe.map'
        symbols.write_text('{ local: *; };\n')
        common = [a.compiler, '-x', 'hip', '--offload-host-only', '--offload-arch=' + a.architecture,
                  '--hip-path=' + str(a.sdk), '--rocm-path=' + str(a.sdk), '-std=c++17', '-O3',
                  '-ffunction-sections', '-fdata-sections', '-fvisibility=hidden']
        topo = tmp / 'topo.o'
        subprocess.run([*common, *source_flags[1][1], '-c', str(source_flags[1][0]), '-o', str(topo)],
                       cwd=a.rccl_build, check=True)
        for variant in ('native', 'original'):
            source = source_flags[0][0] if variant == 'native' else dest
            obj = tmp / (variant + '.o')
            subprocess.run([*common, *source_flags[0][1], '-c', str(source), '-o', str(obj)],
                           cwd=a.rccl_build, check=True)
            binary = tmp / (variant + '-probe')
            subprocess.run([*common, *source_flags[0][1], str(root / PROBE), '-x', 'none',
                            str(obj), str(topo), str(a.library), '-Wl,--gc-sections',
                            '-Wl,--version-script=' + str(symbols),
                            '-Wl,-rpath,' + str(a.library.resolve().parent),
                            '-Wl,-rpath,' + str(a.sdk / 'lib'), '-pthread', '-o', str(binary)],
                           cwd=a.rccl_build, check=True)
            for name, policy, expected in POLICIES:
                environment = dict(os.environ)
                environment.pop('NCCL_P2P_DISABLE', None)
                environment.pop('NCCL_P2P_LEVEL', None)
                environment.update(policy)
                run = subprocess.run([str(binary), str(expected)], env=environment,
                                     capture_output=True, text=True, timeout=30)
                report['observations'].append(dict(variant=variant, policy=name,
                    returncode=run.returncode, stderr=run.stderr, measured=json.loads(run.stdout)))
        for source, expected in report['native_sources'].items():
            if digest(source) != expected:
                raise RuntimeError('P2P selector source changed during proof')
    verify(report)
    return report
