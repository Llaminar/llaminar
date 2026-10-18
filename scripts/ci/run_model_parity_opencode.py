#!/usr/bin/env python3
"""Run one canonical coding cell after the complete same-image benchmark gate.

The full E2E inventory owns eligibility and production placement. CI shards
that inventory by exact case and shipping ISA; this runner cannot manufacture
or drop a model/topology axis. Every cell stages its complete declared model
artifact, admits context through PhysicalMemoryAuthority and owns one real
OpenCode app lifetime. It imposes no model-turn or session time limit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time

import run_model_parity_e2e as e2e
import run_published_image_suite as suites
from production_artifacts import digest, image_identity, validate_manifest, write_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/v2/e2e/server'))
from opencode_cell_runtime import CodingCell, run_app


def admit_bundles(e2e_bundle: Path, benchmark_bundle: Path) -> tuple[dict, dict, dict, dict]:
    """Require both complete ISA lanes before admitting even one coding job."""
    pair = json.loads((e2e_bundle / 'images.json').read_text())
    suites.validate_pair(pair)
    manifest, correctness = suites.admit_e2e(e2e_bundle, pair)
    _, benchmarks = suites.admit_benchmarks(benchmark_bundle, pair, manifest, correctness)
    return pair, manifest, correctness, benchmarks


def select_cell(manifest: dict, source_revision: str, exact: str) -> dict:
    """Select an existing exact case without regexes, name parsing or new axes."""
    rows = [row for row in validate_manifest(manifest, source_revision) if row['case'] == exact]
    if len(rows) != 1:
        raise ValueError('Requested coding cell is absent or duplicated in canonical E2E inventory')
    row = rows[0]
    CodingCell('sha256:' + '0' * 64, row['configuration'], row['backends'],
               Path(row['configuration']['model']), Path('/unused-coding-output'))
    return row


def matrix_jobs(manifest: dict, source_revision: str) -> list[dict]:
    """Project every blessed cell onto both shipping images, retaining full names."""
    result = []
    for row in validate_manifest(manifest, source_revision):
        select_cell(manifest, source_revision, row['case'])
        for isa in suites.ISAS:
            result.append({'case': row['case'], 'cpu_isa': isa,
                           'artifact_id': isa.lower() + '-' + digest(row['case'])})
    return result


def cell_evidence(directory: Path) -> dict:
    """Keep independent audit inputs with the result, including failed/absent audits."""
    paths = {'app': 'app/stress.json', 'native': 'native-audit/result.json',
             'runtime': 'runtime-policies.json', 'resources': 'resource-growth.json',
             'driver': 'driver-diagnostics.json', 'archive': 'archive-audit.json',
             'reclamation': 'archive-reclamation.json',
             'observations': 'stats-webapp-observations.result.json',
             'context': 'context-admission.json', 'model_metadata': 'model-metadata.json',
             'perf': 'perf/perf.json'}
    result = {name: json.loads((directory / suffix).read_text())
              for name, suffix in paths.items() if (directory / suffix).is_file()}
    result['admission_probes'] = [json.loads(path.read_text()) for path in sorted(directory.glob('admit-*.json'))]
    return result


def main(argv: list[str] | None = None) -> int:
    """Retain partial evidence on cancellation; only a full native lifetime may pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e2e-bundle', type=Path, required=True)
    parser.add_argument('--benchmark-bundle', type=Path, required=True)
    parser.add_argument('--cpu-isa', choices=suites.ISAS, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-ramdisk-root', type=Path, default=Path('/mnt/llaminar-production-parity'))
    args = parser.parse_args(argv)
    pair, manifest, correctness, benchmarks = admit_bundles(args.e2e_bundle, args.benchmark_bundle)
    row = select_cell(manifest, pair['source']['revision'], args.case)
    image = pair['images'][args.cpu_isa]
    identity = image_identity(image['id'])
    if identity['id'] != image['id'] or identity['labels'] != image['labels']:
        raise ValueError('Coding runtime differs from the benchmarked immutable image')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 1, 'scope': 'canonical_opencode_cell', 'complete': False, 'passed': False,
              'source_revision': pair['source']['revision'], 'image': image['id'],
              'cpu_isa': args.cpu_isa, 'manifest_digest': digest(manifest),
              'images_digest': digest(pair), 'case': row['case'], 'configuration': row['configuration'],
              'e2e_report_digest': digest(correctness[args.cpu_isa]),
              'benchmark_report_digest': digest(benchmarks[args.cpu_isa]), 'errors': [],
              'started': time.time()}
    output = args.output / 'opencode.json'
    write_json(output, report)
    staging = e2e.parity.CampaignCell(row['campaign'], e2e.parity.CampaignGroup(row['backends'], 'ALL'),
                                      model_files=tuple(row['model_files']))
    print('[opencode-cell] RUN ' + args.cpu_isa + ' ' + row['case'], flush=True)
    try:
        with e2e.parity.model_staging_workspace(args.model_ramdisk_root, Path('cache'), None) as workspace:
            staged, _ = e2e.parity.stage_models_in_ramdisk([staging], workspace.models, None,
                                                           persistent=workspace.persistent)
            workspace.protect_published_models()
            paths = {Path(item.source_path).resolve(): workspace.models / item.filename for item in staged}
            model = paths[Path(row['configuration']['model']).resolve()]
            cell = CodingCell(image['id'], row['configuration'], row['backends'], model, args.output / 'app-cell')
            report['controller'] = run_app(cell)
        report['complete'] = report['controller']['complete']
        report['passed'] = report['controller']['passed']
    except BaseException as error:
        report['errors'].append(repr(error))
        raise
    finally:
        report['finished'] = time.time()
        try:
            report['evidence'] = cell_evidence(args.output / 'app-cell')
            if report['passed']:
                from opencode_evidence import validate_cell
                validate_cell(report, pair, manifest, correctness, benchmarks)
        except Exception as error:
            report['errors'].append('Independent evidence validation: ' + repr(error))
            report['passed'] = False
        write_json(output, report)
        print('[opencode-cell] ' + ('PASS ' if report['passed'] else 'FAIL ') + args.case, flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    def cancel_cell(signum, frame):
        """Give SIGTERM the same explicit cleanup path as an operator interruption."""
        raise KeyboardInterrupt('Coding cell received signal ' + str(signum))
    signal.signal(signal.SIGTERM, cancel_cell)
    raise SystemExit(main())
