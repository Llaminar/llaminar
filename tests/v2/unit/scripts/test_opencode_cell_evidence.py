#!/usr/bin/env python3
"""Reject missing or contradictory independent coding-cell evidence.

The application and context validators run unchanged. Native runtime policy
validation is stubbed at its separately tested boundary so these tests exercise
artifact joins without models, devices or a duplicate graph-policy oracle.
Every independently failed audit remains fatal even when summary flags stay
green; imperfect model-authored application code remains a separate result.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts/ci'))
sys.path.insert(0, str(Path(__file__).parent))
import opencode_evidence as evidence
import run_opencode_matrix as matrix_cli
import yaml
from production_artifacts import digest
from opencode_context_admission import AdmissionOutcome, ContextPolicy, select_context
from opencode_certification_evidence import validate_app_report
from test_opencode_certification_evidence import finished_report
import test_opencode_canonical_jobs as fixtures
from test_published_image_suite import image_pair


class CodingEvidenceTests(unittest.TestCase):
    """Admit only a complete native lifetime bound to both preceding gates."""

    def setUp(self):
        self.pair = image_pair()
        self.manifest = fixtures.CanonicalCodingJobsTests().manifest()
        self.row = self.manifest['cells'][0]
        self.correctness = {isa: {'phase': 'e2e', 'isa': isa} for isa in evidence.suites.ISAS}
        self.benchmarks = {isa: {'phase': 'benchmarks', 'isa': isa} for isa in evidence.suites.ISAS}
        isa = evidence.suites.ISAS[0]
        context = select_context(ContextPolicy(131072, 64), lambda _: AdmissionOutcome.ACCEPTED)
        native = {'Running': False, 'Pid': 0, 'Status': 'exited', 'OOMKilled': False,
                  'ExitCode': 0, 'Error': ''}
        app = finished_report(errors=1, task_errors=['generated app has a failing test'])
        group = {'requests': 30, 'prompt_tokens': 3000, 'restored_tokens': 2500,
                 'token_reuse_rate': 2500 / 3000, 'ttft_mean_seconds': 2,
                 'ttft_median_seconds': 1, 'ttft_p95_seconds': 4, 'prefill_seconds': 40}
        empty = {'requests': 0, 'prompt_tokens': 0, 'restored_tokens': 0,
                 'token_reuse_rate': None, 'ttft_mean_seconds': None,
                 'ttft_median_seconds': None, 'ttft_p95_seconds': None, 'prefill_seconds': 0}
        app['results'][0]['prefix_reuse'] = {'schema': 1, 'complete': True, 'requests': 30,
            'scope': 'exclusive_sequential_workload_request_boundary_observations',
            'groups': {'all': dict(group), 'ordinary': dict(group),
                       'compaction': dict(empty), 'post_compaction': dict(empty)},
            'tiers': {tier: {'enabled': True, 'capacity_bytes': evidence.DISK_MIB * 1024**2,
                            'boundary_peak_bytes': 1024} for tier in ('ram', 'disk')}, 'traffic': {}}
        app_summary = validate_app_report(app, model='Qwen-fixture', context_tokens=131072)
        self.runtime = {'passed': True, 'features': {'mtp': True}, 'checks': {}}
        control = {'complete': True, 'passed': True, 'errors': [], 'normal_shutdown': True,
                   'workload_returncode': 0, 'native_retired': True, 'observer_retired': True,
                   'image': self.pair['images'][isa]['id'], 'configuration': self.row['configuration'],
                   'container_id': 'f' * 64, 'started': 10, 'finished': 100,
                   'workload_started': 20, 'workload_finished': 90,
                   'owned_containers': {'passed': True, 'errors': {}, 'retired': {'server': native}},
                   'retired_container': native, 'context_admission': context,
                   'model': 'Qwen-fixture', 'app': app_summary}
        self.report = {'schema': 1, 'scope': 'canonical_opencode_cell', 'complete': True, 'passed': True,
                       'errors': [], 'cpu_isa': isa, 'case': self.row['case'],
                       'source_revision': self.pair['source']['revision'], 'image': control['image'],
                       'images_digest': digest(self.pair), 'manifest_digest': digest(self.manifest),
                       'e2e_report_digest': digest(self.correctness[isa]),
                       'benchmark_report_digest': digest(self.benchmarks[isa]),
                       'configuration': self.row['configuration'], 'controller': control,
                       'evidence': {'app': app, 'context': context,
                           'model_metadata': {'schema': 1, 'tool_format': 'qwen_3_xml',
                                              'context_tokens': 131072, 'context_alignment_tokens': 64},
                           'admission_probes': [{'context_tokens': 131072, 'outcome': 'accepted',
                                                 'retired_container': native}],
                           'native': {'complete': True, 'passed': True, 'joined_responses': 30,
                                      'verified_native_calls': 21, 'changed_responses': 0, 'changes': []},
                           'perf': {'world_size': 1, 'records': [{'domain': 'server',
                               'name': 'execution_participant', 'rank': 0, 'device': 'CUDA:0'}]},
                           'runtime': deepcopy(self.runtime),
                           'resources': {'schema_version': 1, 'passed': True, 'container_id': 'f' * 64,
                               'boot_id': 'fixture-boot', 'observation_failures': [],
                               'ranks': [{'rank': 0}],
                               'window': {'baseline': 19, 'started': 20, 'finished': 90,
                                          'maximum_sample_gap_seconds': 20,
                                          'maximum_stats_age_seconds': 10, 'page_size_bytes': 4096},
                               'growth': {'passed': True, 'failures': [], 'completed_progress': True,
                                   'samples': 50, 'peak_bytes': {'rank/0/anonymous_and_swap': 128, 'rank/0/CUDA:0': 128},
                                   'maximum_bytes': {'rank/0/anonymous_and_swap': 256, 'rank/0/CUDA:0': 256}}},
                           'driver': {'schema': 2, 'complete': True, 'passed': True, 'boot_id': 'fixture-boot',
                                      'error': None, 'findings': [], 'finding_count': 0, 'reader': 'direct',
                                      'window_id': 'a' * 32, 'snapshot_count': 100, 'new_record_count': 0},
                           'observations': {'passed': True, 'error': None, 'failed_polls': 0,
                                            'snapshots': 50, 'while_active': 40},
                           'archive': {'passed': True, 'archives': [{'passed': True, 'format': 3,
                               'payload_read_bytes': 0, 'orphan_payload_files': 0,
                               'budget_bytes': evidence.DISK_MIB * 1024**2, 'active_payload_bytes': 1234}]},
                           'reclamation': {'passed': True, 'remaining_entries': 0, 'payload_read_bytes': 0}}}
        patched = patch.object(evidence, 'runtime_evidence', return_value=self.runtime)
        self.runtime_validator = patched.start()
        self.addCleanup(patched.stop)

    def validate(self):
        """Exercise the production join using exact manifest and pair identities."""
        return evidence.validate_cell(self.report, self.pair, self.manifest, self.correctness, self.benchmarks)

    def jobs(self):
        """Build the complete canonical product with separately bound inner/outer proofs."""
        jobs = []
        for row in self.manifest['cells']:
            for isa in evidence.suites.ISAS:
                cell = deepcopy(self.report)
                cell.update(cpu_isa=isa, case=row['case'], configuration=row['configuration'],
                    image=self.pair['images'][isa]['id'],
                    e2e_report_digest=digest(self.correctness[isa]),
                    benchmark_report_digest=digest(self.benchmarks[isa]))
                cell['controller'].update(image=cell['image'], configuration=row['configuration'])
                job = {key: cell[key] for key in ('schema', 'complete', 'passed', 'errors', 'case',
                    'cpu_isa', 'image', 'images_digest', 'manifest_digest', 'source_revision',
                    'e2e_report_digest', 'benchmark_report_digest')}
                job.update(scope='canonical_opencode_job', cell=cell,
                    outer_cleanup={'complete': True, 'passed': True, 'errors': [], 'unexpected_live': [],
                        'owner': 'llaminar-suite-driver-' + 'b' * 32, 'retired': {},
                        'driver_retirement': {'Status': 'absent', 'Running': False, 'Pid': 0}})
                jobs.append(job)
        return jobs

    def aggregate(self, jobs):
        """Run the production aggregate without hiding individual validation failures."""
        return evidence.aggregate_jobs(jobs, self.pair, self.manifest, self.correctness, self.benchmarks)

    def test_complete_engine_evidence_admits_imperfect_model_application(self):
        summary = self.validate()
        self.assertTrue(summary['app']['gate_passed'])
        self.assertFalse(summary['app']['task_passed'])
        self.runtime_validator.assert_called_once_with(self.report['evidence']['perf'], self.row['configuration'])

    def test_every_independent_audit_is_required(self):
        original = deepcopy(self.report)
        for key in original['evidence']:
            self.report = deepcopy(original)
            del self.report['evidence'][key]
            with self.subTest(key=key), self.assertRaises((KeyError, ValueError)):
                self.validate()

    def test_stale_source_image_or_preceding_gate_cannot_be_relabelled_green(self):
        for key in ('source_revision', 'image', 'manifest_digest', 'images_digest',
                    'e2e_report_digest', 'benchmark_report_digest', 'case', 'cpu_isa'):
            original = self.report[key]
            self.report[key] = 'stale'
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()
            self.report[key] = original

    def test_native_and_observer_faults_cannot_hide_behind_outer_success(self):
        original = deepcopy(self.report)
        mutations = [lambda r: r['controller'].update(normal_shutdown=False),
            lambda r: r['controller']['retired_container'].update(OOMKilled=True),
            lambda r: r['controller'].update(observer_retired=False),
            lambda r: r['evidence']['native'].update(joined_responses=29),
            lambda r: r['evidence']['runtime']['features'].update(mtp=False),
            lambda r: r['evidence']['resources']['growth']['peak_bytes'].update({'rank/0/anonymous_and_swap': 257}),
            lambda r: r['evidence']['resources'].update(observation_failures=['gap']),
            lambda r: r['evidence']['driver'].update(findings=['GPU fault']),
            lambda r: r['evidence']['observations'].update(failed_polls=1),
            lambda r: r['evidence']['archive']['archives'][0].update(payload_read_bytes=1),
            lambda r: r['evidence']['reclamation'].update(remaining_entries=1)]
        for index, mutate in enumerate(mutations):
            self.report = deepcopy(original)
            mutate(self.report)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.validate()

    def test_missing_gpu_coverage_and_prefix_measurements_are_fatal(self):
        original = deepcopy(self.report)
        changes = [lambda r: r['evidence']['resources']['growth']['peak_bytes'].pop('rank/0/CUDA:0'),
                   lambda r: r['evidence']['resources']['window'].update(started=21),
                   lambda r: r['evidence']['app']['results'][0]['prefix_reuse'].update(requests=29),
                   lambda r: r['evidence']['app']['results'][0]['prefix_reuse']['groups']['all'].update(restored_tokens=2999)]
        for index, change in enumerate(changes):
            self.report = deepcopy(original)
            change(self.report)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.validate()

    def test_context_bounds_require_native_probes_and_the_original_search(self):
        original = deepcopy(self.report)
        changes = [lambda e: e['admission_probes'].clear(),
                   lambda e: e['admission_probes'][0]['retired_container'].update(ExitCode=139),
                   lambda e: e['context'].update(context_tokens=65536),
                   lambda e: e['model_metadata'].update(context_tokens=262144),
                   lambda e: e['context']['attempts'].append({'context_tokens': 64, 'outcome': 'accepted'})]
        for index, change in enumerate(changes):
            self.report = deepcopy(original)
            change(self.report['evidence'])
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.validate()

    def test_complete_matrix_replays_every_job_and_publisher_validation(self):
        report = self.aggregate(self.jobs())
        self.assertTrue(report['passed'])
        self.assertEqual(report['expected_jobs'], len(self.manifest['cells']) * len(evidence.suites.ISAS))
        self.assertEqual(evidence.validate_matrix(report, self.pair, self.manifest,
                                                  self.correctness, self.benchmarks), report)
        report['jobs'][0]['cell']['evidence']['native']['changed_responses'] = 1
        with self.assertRaises(ValueError):
            evidence.validate_matrix(report, self.pair, self.manifest, self.correctness, self.benchmarks)

    def test_missing_duplicate_and_failed_jobs_are_all_preserved(self):
        jobs = self.jobs()
        missing = jobs.pop()
        jobs.append(deepcopy(jobs[0]))
        jobs[1]['outer_cleanup']['unexpected_live'] = ['f' * 64]
        jobs[2]['cell']['evidence']['driver']['findings'] = ['GPU fault']
        report = self.aggregate(jobs)
        self.assertFalse(report['passed'])
        self.assertFalse(report['complete'])
        self.assertEqual(len(report['failures']), 4)
        self.assertIn('Missing canonical coding job', str(report['failures']))
        self.assertNotIn((missing['case'], missing['cpu_isa']),
                         {(row['case'], row['cpu_isa']) for row in report['jobs']})

    def test_outer_green_flags_cannot_hide_live_driver_or_stale_inner_cell(self):
        mutations = [lambda j: j['outer_cleanup']['driver_retirement'].update(Pid=321),
                     lambda j: j.update(image='sha256:' + '0' * 64),
                     lambda j: j['outer_cleanup'].update(retired={'f' * 64: {
                         'Status': 'running', 'Running': True, 'Pid': 456}})]
        for mutate in mutations:
            jobs = self.jobs()
            mutate(jobs[0])
            with self.subTest(mutate=mutate):
                self.assertFalse(self.aggregate(jobs)['passed'])

    def test_failed_rerun_cannot_reuse_an_earlier_green_artifact(self):
        for result in ('failure', 'cancelled', 'skipped'):
            with self.subTest(result=result):
                report = evidence.aggregate_jobs(self.jobs(), self.pair, self.manifest,
                    self.correctness, self.benchmarks, jobs_result=result)
                self.assertFalse(report['passed'])
                self.assertIn('workflow', report['failures'])

    def test_matrix_cli_preserves_selection_and_fails_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = ['--e2e-bundle', str(root / 'e2e'), '--benchmark-bundle', str(root / 'benchmarks'),
                    '--expected-source-revision', self.pair['source']['revision'],
                    '--output', str(root / 'matrix.json')]
            with patch.object(matrix_cli, 'admit_bundles',
                              return_value=(self.pair, self.manifest, self.correctness, self.benchmarks)):
                self.assertEqual(matrix_cli.main(['prepare', *args, '--github-output', str(root / 'gh-output')]), 0)
                matrix = json.loads((root / 'matrix.json').read_text())
                self.assertEqual(len(matrix['include']), 4)
                self.assertEqual(json.loads((root / 'gh-output').read_text().removeprefix('matrix=')), matrix)
                self.assertEqual(matrix_cli.main(['aggregate', *args, '--jobs', str(root / 'jobs'),
                                                  '--jobs-result', 'success']), 1)
                failed = json.loads((root / 'matrix.json').read_text())
                self.assertEqual(len(failed['failures']), 4)
                wrong = list(args)
                wrong[wrong.index('--expected-source-revision') + 1] = '0' * 40
                with self.assertRaises(ValueError):
                    matrix_cli.main(['prepare', *wrong])

    def test_workflow_runs_only_on_master_pr_after_benchmarks_and_keeps_all_failures(self):
        workflow = yaml.load((ROOT / '.github/workflows/master-pr-certification.yml').read_text(),
                             Loader=yaml.BaseLoader)
        self.assertEqual(set(workflow['on']), {'pull_request'})
        self.assertEqual(workflow['on']['pull_request']['branches'], ['master'])
        jobs = workflow['jobs']
        self.assertEqual(jobs['coding_matrix']['needs'], 'benchmarks')
        self.assertEqual(jobs['coding_cells']['needs'], 'coding_matrix')
        self.assertEqual(jobs['coding_cells']['strategy']['fail-fast'], 'false')
        self.assertIn('needs.coding_matrix.outputs.matrix', jobs['coding_cells']['strategy']['matrix'])
        self.assertEqual(jobs['coding_cells']['concurrency'], jobs['benchmarks']['concurrency'])
        self.assertEqual(set(jobs['coding']['needs']), {'coding_matrix', 'coding_cells'})
        self.assertEqual(jobs['coding']['if'], 'always()')
        from apply_master_ruleset import REQUIRED_CHECKS
        self.assertIn(jobs['coding']['name'], REQUIRED_CHECKS)
        command = '\n'.join(step.get('run', '') for step in jobs['coding_cells']['steps'])
        self.assertIn('--benchmark-bundle', command)
        self.assertIn('--expected-source-revision', command)
        self.assertNotIn('--turn-wall-time-limit', command)
        self.assertNotIn('--request-inactivity-timeout', command)
        aggregate = next(step for step in jobs['coding']['steps'] if 'MATRIX_RESULT' in step.get('env', {}))
        self.assertEqual(aggregate['env']['MATRIX_RESULT'], '${{ needs.coding_cells.result }}')
        self.assertIn('--jobs-result "$MATRIX_RESULT"', aggregate['run'])


if __name__ == '__main__':
    unittest.main()
