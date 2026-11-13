#!/usr/bin/env python3
"""Authenticate real fused-fixture layouts and byte/count checks, without timing claims.

The existing performance executable owns preparation, production dispatch and
serial-byte replay. This narrow observer rejects old padded timing layouts and
requires both the actual contiguous output identity and completed guard checks.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


FAMILY = 'kernel.cpu_native_quantised_fused_verifier_rows_projection_launch'


def records(stdout, prefix):
    """Read fixture-owned key/value metadata, rejecting missing fields at assertion time."""
    return [dict(item.split('=', 1) for item in line.split(',')[1:])
            for line in stdout.splitlines() if line.startswith(prefix + ',')]


def main():
    """Exercise both production row strides through the actual existing fixture target."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--slice', choices=('gate', 'down'), required=True)
    parser.add_argument('--isa', choices=('avx2', 'avx512'), required=True)
    args = parser.parse_args()
    fmt, n, k, tiles, codebook = ('Q4_K', 320, 2560, 16, 5) if args.slice == 'gate' else ('Q5_1', 2560, 320, 10, 7)
    m, projections = 8, 4 if args.slice == 'gate' else 2
    role = 'gate_up' if args.slice == 'gate' else 'down'
    root_count = 2 if role == 'gate_up' else 1
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('LLAMINAR_CPU_FUSED_VERIFIER_', 'LLAMINAR_PERF_STATS_')) or key in ('GOMP_CPU_AFFINITY', 'OMP_THREAD_LIMIT'):
            env.pop(key, None)
    env.update(LLAMINAR_ISA_LEVEL=args.isa, OMP_NUM_THREADS='2', OMP_DYNAMIC='false',
               OMP_PROC_BIND='false', LLAMINAR_FORCE_CPU_ONLY_STARTUP='1',
               LLAMINAR_SKIP_CUDA_STARTUP='1', LLAMINAR_SKIP_ROCM_STARTUP='1')
    fields = dict(M=m, N_A=n, K=k, PROJECTIONS=projections, PURPOSE='prefill',
                  NUMERICAL_POLICY='GPUAlignedExpert', POLICY='Auto', POLICY_N=n,
                  EXPECTED_K_TILES=tiles, EXPECTED_CODEBOOK=codebook,
                  TEAM_LIFETIME='retained', OUTPUT_ROLE=role, WARMUP=1, SAMPLES=1, REPEATS=1)
    env.update({f'LLAMINAR_CPU_FUSED_VERIFIER_{key}': str(value) for key, value in fields.items()})
    with tempfile.TemporaryDirectory(prefix='cpu-fused-layout-') as directory:
        stats = Path(directory) / 'stats.json'
        env.update(LLAMINAR_PERF_STATS_JSON=str(stats), LLAMINAR_PERF_STATS_FILTER=FAMILY)
        argv = [str(args.binary.resolve()),
                f'--gtest_filter=AllFormats/CPUFusedVerifier.ProjectionCounterBundleReplay/{fmt}']
        result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=20)
        print(result.stdout, end='', flush=True)
        print(result.stderr, end='', flush=True)
        if result.returncode:
            raise RuntimeError(f'Actual production fixture failed: {result.returncode}')
        layouts = records(result.stdout, 'output_layout')
        if len(layouts) != 1:
            raise AssertionError('Missing actual aligned expert-major output layout identity')
        layout = layouts[0]
        expected = dict(revision='aligned_expert_major_v1', storage='AlignedVector',
                        base_alignment_bytes='64', ldc='logical_n', expert_order='contiguous_within_root',
                        output_role=role, root_count=str(root_count),
                        tail_padding_floats='0', guard_layout='separate_n_plus_7',
                        live_floats=str(projections * m * n), all_experts_aligned_64='1', all_rows_aligned_64='1')
        if layout != expected:
            raise AssertionError(f'Actual output layout differs: {layout}, expected={expected}')
        actual = records(result.stdout, 'output_projection')
        wanted = [dict(index=str(p), root=str(p % root_count),
                       offset_floats=str((p // root_count) * m * n), expert_floats=str(m * n),
                       ldc=str(n), base_mod_64='0', row_stride_bytes=str(n * 4), guard_ldc=str(n + 7))
                  for p in range(projections)]
        if actual != wanted:
            raise AssertionError(f'Actual descriptors lost contiguous aligned row/expert geometry: {actual}')
        verified = records(result.stdout, 'output_layout_verified')
        if verified != [dict(guard_replays='2', timed_serial_bytes='1', guard_bytes='1')]:
            raise AssertionError('Both padded guard replays and contiguous serial-byte comparison must complete')
        counter = re.search(r'^projection_counter_records,(\d+),observations,(\d+),value,([^\s]+)$', result.stdout, re.MULTILINE)
        expected_count = (2 + 1 + 1 * 1) * projections
        if not counter or (int(counter[1]), int(counter[2]), float(counter[3])) != (1, expected_count, float(expected_count)):
            raise AssertionError('Fixture layout change lost original projection accounting')
        exported = json.loads(stats.read_text())
        counters = [row for row in exported['records'] if row['domain'] == 'kernel'
                    and row['name'] == 'cpu_native_quantised_fused_verifier_rows_projection_launch']
        print(json.dumps(dict(argv=argv, slice=args.slice, isa=args.isa, output_layout=layout,
                              projection_counter=counters)), flush=True)
        if len(counters) != 1 or len(counters[0]['tags']) != 21 or counters[0]['count'] != expected_count or counters[0]['value'] != float(expected_count):
            raise AssertionError('Exported semantic identity or multiplicity changed')


if __name__ == '__main__':
    main()
