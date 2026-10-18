#!/usr/bin/env python3
"""Join complete coding lifetimes to the exact benchmarked canonical inventory.

The publisher replays application counts, PMA context decisions and native
runtime policy validators. Independent resource, driver and archive receipts
remain mandatory. A cached green outer flag cannot substitute for those
inputs, omit a cell/ISA or turn a cancelled session into certification.
"""
from __future__ import annotations

from pathlib import Path
import math
import re
import sys

from production_artifacts import digest, positive
from run_model_parity_opencode import matrix_jobs, select_cell
import run_published_image_suite as suites

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/v2/e2e/server'))
from opencode_cell_runtime import DISK_MIB, OUTPUT_TOKENS, runtime_evidence
from opencode_certification_evidence import validate_app_report, validate_native_coverage
from opencode_context_admission import AdmissionOutcome, ContextPolicy, select_context
from prefix_reuse_measurement import integer
from resource_growth_evidence import WorkloadWindow


def validate_prefix_summary(reuse: dict, requests: int) -> None:
    """Retain cache/TTFT measurements for every exchange without a stochastic hit floor."""
    if (reuse.get('schema') != 1 or reuse.get('complete') is not True
            or reuse.get('requests') != requests
            or reuse.get('scope') != 'exclusive_sequential_workload_request_boundary_observations'
            or set(reuse['groups']) != {'all', 'ordinary', 'compaction', 'post_compaction'}):
        raise ValueError('Coding prefix measurements do not cover the complete app session')
    groups = reuse['groups']
    if (groups['all']['requests'] != requests
            or any(groups['all'][field] != sum(groups[key][field]
                for key in ('ordinary', 'compaction', 'post_compaction'))
                for field in ('requests', 'prompt_tokens', 'restored_tokens'))):
        raise ValueError('Coding prefix group totals disagree')
    for group in groups.values():
        count = integer(group['requests'], 'prefix requests')
        prompt = integer(group['prompt_tokens'], 'prefix prompt tokens')
        restored = integer(group['restored_tokens'], 'prefix restored tokens')
        if restored > prompt or (count and not prompt):
            raise ValueError('Coding prefix token geometry is invalid')
        if group['token_reuse_rate'] != (restored / prompt if prompt else None):
            raise ValueError('Coding prefix reuse rate disagrees with measured tokens')
        for field in ('ttft_mean_seconds', 'ttft_median_seconds', 'ttft_p95_seconds'):
            value = group[field]
            if ((not count and value is not None) or (count and (
                    type(value) not in (int, float) or not math.isfinite(value) or value < 0))):
                raise ValueError('Coding prefix observations have invalid TTFT')
    for tier in ('ram', 'disk'):
        value = reuse['tiers'][tier]
        if (value['enabled'] is not True or not integer(value['capacity_bytes'], tier)
                or integer(value['boundary_peak_bytes'], tier) > value['capacity_bytes']):
            raise ValueError('Coding prefix cache exceeded its enabled tier bound')
    if reuse['tiers']['disk']['capacity_bytes'] != DISK_MIB * 1024**2:
        raise ValueError('Coding disk cache did not retain the requested capacity')
    for event, counters in reuse['traffic'].items():
        for field in ('operations', 'bytes'):
            integer(counters[field], event + '/' + field)


def validate_context(context: dict, model: dict, probes: list[dict]) -> None:
    """Replay the actual bounded search without inventing new capacity estimates."""
    if model.get('schema') != 1 or model.get('tool_format') != 'qwen_3_xml':
        raise ValueError('Coding model metadata lacks its qualified grammar/context')
    policy = ContextPolicy(model['context_tokens'], model['context_alignment_tokens'], OUTPUT_TOKENS)
    attempts = iter(context['attempts'])
    by_context = {row['context_tokens']: row for row in probes}
    if len(by_context) != len(probes) or len(probes) != len(context['attempts']):
        raise ValueError('Coding context proof omitted or duplicated a native admission probe')
    def replay(tokens):
        """Each search decision requires the corresponding completed native exit."""
        row = next(attempts, None)
        native = by_context.get(tokens)
        if row is None or native is None:
            raise ValueError('Coding context search lacks its next native admission decision')
        state = native['retired_container']
        outcome = AdmissionOutcome(row['outcome'])
        if (row['context_tokens'] != tokens or native['outcome'] != row['outcome']
                or state.get('Running') is not False or state.get('Pid') != 0
                or state.get('Status') != 'exited' or state.get('OOMKilled') is not False
                or state.get('Error') != '' or state.get('ExitCode') != (
                    0 if outcome is AdmissionOutcome.ACCEPTED else 1)):
            raise ValueError('Coding context admission disagrees with its native exit')
        return outcome
    if select_context(policy, replay) != context or next(attempts, None) is not None:
        raise ValueError('Coding context was not the maximum or adjacent PMA admission bound')


def validate_cell(report: dict, pair: dict, manifest: dict, correctness: dict, benchmarks: dict) -> dict:
    """Recompute a single canonical job's complete engine and lifecycle verdict."""
    isa = report.get('cpu_isa')
    if isa not in suites.ISAS:
        raise ValueError('Coding report omitted its shipping ISA')
    row = select_cell(manifest, pair['source']['revision'], report.get('case'))
    expected = {'case': row['case'], 'cpu_isa': isa, 'source_revision': pair['source']['revision'],
                'image': pair['images'][isa]['id'], 'images_digest': digest(pair),
                'manifest_digest': digest(manifest), 'e2e_report_digest': digest(correctness[isa]),
                'benchmark_report_digest': digest(benchmarks[isa])}
    if (report.get('schema') != 1 or report.get('scope') != 'canonical_opencode_cell'
            or report.get('complete') is not True or report.get('passed') is not True
            or report.get('errors') != [] or any(report.get(key) != value for key, value in expected.items())
            or report.get('configuration') != row['configuration']):
        raise ValueError('Coding report is incomplete or differs from its benchmarked inventory')
    control, evidence = report['controller'], report['evidence']
    if (control.get('complete') is not True or control.get('passed') is not True
            or control.get('errors') != [] or control.get('normal_shutdown') is not True
            or control.get('workload_returncode') != 0 or control.get('native_retired') is not True
            or control.get('observer_retired') is not True or control.get('image') != expected['image']
            or control.get('configuration') != row['configuration']
            or not re.fullmatch(r'[0-9a-f]{64}', control.get('container_id', ''))
            or positive(control['finished']) <= positive(control['started'])):
        raise ValueError('Coding report lacks a complete native lifetime')
    owners = control['owned_containers']
    if (owners.get('passed') is not True or owners.get('errors') != {} or not owners.get('retired')
            or any(state.get('Running') is not False or state.get('Pid') != 0
                   or state.get('Status') not in {'exited', 'created', 'absent'}
                   for state in owners['retired'].values())):
        raise ValueError('Coding report left native ownership unproven')
    native = control['retired_container']
    if (native.get('Status') != 'exited' or native.get('ExitCode') != 0
            or native.get('OOMKilled') is not False or native.get('Error') != ''):
        raise ValueError('Coding server did not retire normally')
    context = evidence['context']
    validate_context(context, evidence['model_metadata'], evidence['admission_probes'])
    if control['context_admission'] != context:
        raise ValueError('Coding workload differs from its context admission')
    app = validate_app_report(evidence['app'], model=control['model'], context_tokens=context['context_tokens'])
    if control['app'] != app:
        raise ValueError('Coding lifecycle changed the independently validated application verdict')
    validate_native_coverage(evidence['native'], app)
    reuse = evidence['app']['results'][0]['prefix_reuse']
    validate_prefix_summary(reuse, app['request_count'])
    policies = runtime_evidence(evidence['perf'], row['configuration'])
    if policies['passed'] is not True or digest(policies) != digest(evidence['runtime']):
        raise ValueError('Coding runtime evidence differs from native execution records')
    resources, driver = evidence['resources'], evidence['driver']
    growth = resources['growth']
    window = WorkloadWindow(**resources['window'])
    rank_count = integer(evidence['perf']['world_size'], 'native world size')
    expected_metrics = {f'rank/{rank}/anonymous_and_swap' for rank in range(rank_count)}
    expected_metrics.update(f"rank/{record['rank']}/{record['device']}"
        for record in evidence['perf']['records']
        if record.get('domain') == 'server' and record.get('name') == 'execution_participant'
        and not record['device'].startswith('CPU'))
    if (window.started != control['workload_started'] or window.finished != control['workload_finished']
            or set(growth['peak_bytes']) != expected_metrics
            or {row['rank'] for row in resources['ranks']} != set(range(rank_count))
            or len(resources['ranks']) != rank_count):
        raise ValueError('Coding resource observations omit a native rank/device or workload boundary')
    if (resources.get('schema_version') != 1 or resources.get('passed') is not True
            or resources.get('container_id') != control['container_id']
            or resources.get('boot_id') != driver.get('boot_id') or not resources.get('boot_id')
            or resources.get('observation_failures') != [] or growth.get('passed') is not True
            or growth.get('failures') != [] or growth.get('completed_progress') is not True
            or integer(growth.get('samples'), 'resource samples') < 2
            or not growth.get('peak_bytes') or growth['peak_bytes'].keys() != growth['maximum_bytes'].keys()
            or any(integer(value, name) > integer(growth['maximum_bytes'][name], name)
                   for name, value in growth['peak_bytes'].items())):
        raise ValueError('Coding resource observations are missing, stale or exceed admitted growth')
    if (driver.get('schema') != 2 or driver.get('complete') is not True or driver.get('passed') is not True
            or driver.get('error') is not None or driver.get('findings') != [] or driver.get('finding_count') != 0
            or driver.get('reader') not in {'direct', 'sudo', 'tools_container'}
            or not re.fullmatch(r'[0-9a-f]{32}', driver.get('window_id', ''))
            or integer(driver.get('snapshot_count'), 'driver snapshots') < 2
            or integer(driver.get('new_record_count'), 'driver records') < 0):
        raise ValueError('Coding driver window was incomplete or contained faults')
    observations = evidence['observations']
    if (observations.get('passed') is not True or observations.get('failed_polls') != 0
            or observations.get('error') is not None or integer(observations.get('snapshots'), 'stats snapshots') < 2
            or integer(observations.get('while_active'), 'active stats snapshots') == 0):
        raise ValueError('Coding statistics observation was incomplete')
    archive = evidence['archive']
    if archive.get('passed') is not True or not archive.get('archives'):
        raise ValueError('Coding cache archive ownership was not audited')
    for tier in archive['archives']:
        if (tier.get('passed') is not True or tier.get('format') != 3
                or tier.get('payload_read_bytes') != 0 or tier.get('orphan_payload_files') != 0
                or tier.get('budget_bytes') != DISK_MIB * 1024**2
                or integer(tier.get('active_payload_bytes'), 'disk bytes') > tier['budget_bytes']):
            raise ValueError('Coding cache archive ownership or metadata-only policy failed')
    reclaimed = evidence['reclamation']
    if (reclaimed.get('passed') is not True or reclaimed.get('remaining_entries') != 0
            or reclaimed.get('payload_read_bytes') != 0):
        raise ValueError('Coding matrix retained retired cache payloads')
    return {**expected, 'app': app, 'prefix_reuse': reuse, 'context_tokens': context['context_tokens'],
            'resource_samples': growth['samples'], 'driver_snapshots': driver['snapshot_count']}


def validate_job(job: dict, pair: dict, manifest: dict, correctness: dict, benchmarks: dict) -> dict:
    """Require both the inner lifetime and independent outer daemon retirement."""
    if (job.get('schema') != 1 or job.get('scope') != 'canonical_opencode_job'
            or job.get('complete') is not True or job.get('passed') is not True
            or job.get('errors') != []):
        raise ValueError('Coding job did not complete successfully')
    cell = job['cell']
    summary = validate_cell(cell, pair, manifest, correctness, benchmarks)
    for key in ('case', 'cpu_isa', 'image', 'images_digest', 'manifest_digest',
                'source_revision', 'e2e_report_digest', 'benchmark_report_digest'):
        if job.get(key) != summary[key]:
            raise ValueError('Coding job changed its independently validated cell identity: ' + key)
    cleanup = job['outer_cleanup']
    driver = cleanup['driver_retirement']
    if (cleanup.get('complete') is not True or cleanup.get('passed') is not True
            or cleanup.get('errors') != [] or cleanup.get('unexpected_live') != []
            or not re.fullmatch(r'llaminar-suite-driver-[0-9a-f]{32}', cleanup.get('owner', ''))
            or driver.get('Running') is not False or driver.get('Pid') != 0
            or driver.get('Status') != 'absent'
            or any(not re.fullmatch(r'[0-9a-f]{64}', identity)
                   or state.get('Running') is not False or state.get('Pid') != 0
                   or state.get('Status') not in {'exited', 'created'}
                   for identity, state in cleanup['retired'].items())):
        raise ValueError('Coding outer owner did not prove complete native retirement')
    return summary


def aggregate_jobs(jobs: list[dict], pair: dict, manifest: dict,
                   correctness: dict, benchmarks: dict, *, jobs_result: str = 'success') -> dict:
    """Retain every failure and require exactly the full canonical cell/ISA product.

    All supplied job bodies remain in the artifact so the release publisher
    can independently replay the same checks. The only digests are of small
    structured evidence; model weights and KV payloads are never inspected.
    """
    planned = {(row['case'], row['cpu_isa']): row['artifact_id']
               for row in matrix_jobs(manifest, pair['source']['revision'])}
    seen, summaries, failures = set(), {}, {}
    for index, job in enumerate(jobs):
        key = (job.get('case'), job.get('cpu_isa'))
        label = planned.get(key, 'unexpected-' + str(index))
        try:
            if key not in planned or key in seen:
                raise ValueError('Unexpected or duplicate canonical coding job')
            seen.add(key)
            summaries[label] = validate_job(job, pair, manifest, correctness, benchmarks)
        except (KeyError, ValueError, TypeError) as error:
            failures.setdefault(label, []).append(str(error))
    for key in planned.keys() - seen:
        failures.setdefault(planned[key], []).append('Missing canonical coding job')
    if jobs_result != 'success':
        failures['workflow'] = ['Matrix job outcome: ' + jobs_result]
    return {'schema': 1, 'scope': 'canonical_opencode_matrix',
            'complete': seen == planned.keys(), 'passed': not failures,
            'workflow_jobs_result': jobs_result,
            'source_revision': pair['source']['revision'], 'images_digest': digest(pair),
            'manifest_digest': digest(manifest), 'expected_jobs': len(planned),
            'e2e_reports': {isa: digest(correctness[isa]) for isa in suites.ISAS},
            'benchmark_reports': {isa: digest(benchmarks[isa]) for isa in suites.ISAS},
            'jobs': jobs, 'summaries': summaries, 'failures': failures}


def validate_matrix(report: dict, pair: dict, manifest: dict,
                    correctness: dict, benchmarks: dict) -> dict:
    """Recompute the complete release receipt, rejecting missing or edited evidence."""
    expected = aggregate_jobs(report['jobs'], pair, manifest, correctness, benchmarks)
    if expected['passed'] is not True or expected['complete'] is not True or report != expected:
        raise ValueError('Coding matrix is incomplete, failed or differs from its independently replayed proof')
    return expected
