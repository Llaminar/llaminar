#!/usr/bin/env python3
"""Prepare or validate the complete post-benchmark coding matrix.

The native blessed inventory owns every job. Preparation authenticates both
preceding ISA gates before Actions receives a matrix; aggregation retains a
failure map for missing, duplicate, cancelled and independently failed jobs.
Neither operation starts inference, reads model/cache payloads or publishes
external state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from opencode_evidence import aggregate_jobs
from production_artifacts import write_json
from run_model_parity_opencode import admit_bundles, matrix_jobs


def main(argv: list[str] | None = None) -> int:
    """Emit a source-bound matrix or a replayable complete certification artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('prepare', 'aggregate'))
    parser.add_argument('--e2e-bundle', type=Path, required=True)
    parser.add_argument('--benchmark-bundle', type=Path, required=True)
    parser.add_argument('--expected-source-revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=Path)
    parser.add_argument('--jobs-result', choices=('success', 'failure', 'cancelled', 'skipped'))
    parser.add_argument('--github-output', type=Path)
    args = parser.parse_args(argv)
    if (args.operation == 'aggregate') != (args.jobs is not None):
        parser.error('--jobs is required only for aggregation')
    if args.operation == 'aggregate' and args.github_output:
        parser.error('--github-output applies only to matrix preparation')
    if (args.operation == 'aggregate') != (args.jobs_result is not None):
        parser.error('--jobs-result is required only for aggregation')
    pair, manifest, correctness, benchmarks = admit_bundles(args.e2e_bundle, args.benchmark_bundle)
    if pair['source']['revision'] != args.expected_source_revision:
        raise ValueError('Coding inputs differ from the exact PR head')
    if args.operation == 'prepare':
        jobs = matrix_jobs(manifest, pair['source']['revision'])
        if not jobs or len(jobs) > 256:
            raise ValueError('Complete canonical coding inventory exceeds the Actions matrix capacity')
        matrix = {'include': jobs}
        write_json(args.output, matrix)
        if args.github_output:
            with args.github_output.open('a') as output:
                output.write('matrix=' + json.dumps(matrix, separators=(',', ':')) + '\n')
        print('[opencode-matrix] Prepared ' + str(len(jobs)) + ' exact cell/ISA jobs', flush=True)
        return 0
    jobs, unreadable = [], []
    for path in sorted(args.jobs.rglob('cell-result.json')):
        try:
            jobs.append(json.loads(path.read_text()))
        except (OSError, ValueError) as error:
            unreadable.append({'path': str(path.relative_to(args.jobs)), 'error': repr(error)})
    report = aggregate_jobs(jobs, pair, manifest, correctness, benchmarks, jobs_result=args.jobs_result)
    if unreadable:
        report['passed'] = False
        report['unreadable_jobs'] = unreadable
    write_json(args.output, report)
    print('[opencode-matrix] ' + ('PASS ' if report['passed'] else 'FAIL ')
          + str(len(report['summaries'])) + '/' + str(report['expected_jobs'])
          + ' jobs; failures=' + repr(report['failures']), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
