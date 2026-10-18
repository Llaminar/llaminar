#!/usr/bin/env python3
"""Prove exact canonical coding selection and isolated outer-driver retirement.

The matrix fixtures contain future case names rather than a model allowlist.
Daemon fixtures inject stale labels, lost ownership and live sibling processes;
only explicitly owned containers may be stopped, and recovered cleanup must
retain the original failed lifecycle instead of minting a passing receipt.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts/ci'))
sys.path.insert(0, str(Path(__file__).parent))
import docker_paths
import run_model_parity_opencode as coding
import run_published_image_suite as suite
from test_opencode_cell_runtime import configuration
from test_published_image_suite import full_inventory


class CanonicalCodingJobsTests(unittest.TestCase):
    """Every tagged row reaches both images without names becoming policy."""

    def manifest(self):
        """Use a producer-shaped tagged record and a second unrelated future name."""
        value = full_inventory()
        value['cells'] = value['cells'][:1]
        first = value['cells'][0]
        first['configuration'].update(configuration())
        second = deepcopy(first)
        second['case'] = 'Future/Cell[model+variant].ProductionParity/new-axis'
        second['configuration']['id'] = 'FutureCell'
        value['cells'].append(second)
        return value

    def test_all_rows_project_to_both_shipping_images(self):
        manifest = self.manifest()
        jobs = coding.matrix_jobs(manifest, manifest['source_revision'])
        self.assertEqual({(row['case'], row['cpu_isa']) for row in jobs}, {
            (row['case'], isa) for row in manifest['cells'] for isa in suite.ISAS})
        self.assertEqual(len({row['artifact_id'] for row in jobs}), len(jobs))
        self.assertEqual(coding.select_cell(manifest, manifest['source_revision'],
                                           manifest['cells'][1]['case']), manifest['cells'][1])

    def test_selector_is_exact_and_cannot_skip_invalid_inventory(self):
        manifest = self.manifest()
        for invalid in ('.*', 'unknown', manifest['cells'][0]['case'][:-1]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                coding.select_cell(manifest, manifest['source_revision'], invalid)
        manifest['cells'].append(deepcopy(manifest['cells'][0]))
        with self.assertRaises(ValueError):
            coding.matrix_jobs(manifest, manifest['source_revision'])

    def test_missing_production_policy_cannot_reuse_http_stress_arguments(self):
        for missing in ('benchmark', 'runtime'):
            manifest = self.manifest()
            del manifest['cells'][0]['configuration'][missing]
            with self.subTest(missing=missing), self.assertRaises((ValueError, KeyError)):
                coding.matrix_jobs(manifest, manifest['source_revision'])


class OuterCodingOwnershipTests(unittest.TestCase):
    """Clean up exact child ownership independently of unfinished inner reports."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / 'cleanup.json'
        self.owner = 'llaminar-suite-driver-' + '1' * 32
        self.identity = 'a' * 64
        self.child = {'Id': self.identity,
                      'Config': {'Labels': {docker_paths.CODING_OWNER_LABEL: self.owner}},
                      'State': {'Running': False, 'Pid': 0, 'Status': 'exited'}}
        retirement = patch.object(suite, 'retire_coding_driver', return_value={
            'Running': False, 'Pid': 0, 'Status': 'absent'})
        self.driver_retirement = retirement.start()
        self.addCleanup(retirement.stop)

    def inspect(self, command, **kwargs):
        """Return only the explicitly labelled child, preserving daemon state changes."""
        if command[1] == 'ps':
            self.assertIn('label=' + docker_paths.CODING_OWNER_LABEL + '=' + self.owner, command)
            return self.identity + '\n'
        self.assertEqual(command, ['docker', 'inspect', self.identity])
        return json.dumps([self.child])

    def test_explicit_driver_name_is_required_for_child_labels(self):
        with patch.dict(docker_paths.os.environ, {docker_paths.CODING_DRIVER_ENV: self.owner}):
            self.assertEqual(docker_paths.coding_owner_arguments(),
                             ['--label', docker_paths.CODING_OWNER_LABEL + '=' + self.owner])
        for owner in ('', 'other-container', 'llaminar-suite-driver-.*'):
            with self.subTest(owner=owner), patch.dict(docker_paths.os.environ,
                    {docker_paths.CODING_DRIVER_ENV: owner}), self.assertRaises(ValueError):
                docker_paths.coding_owner_arguments()

    def test_retired_child_can_be_removed(self):
        with patch.object(suite.subprocess, 'check_output', side_effect=self.inspect), \
             patch.object(suite.subprocess, 'run') as action:
            suite.retire_coding_children(self.owner, self.output)
        action.assert_called_once()
        self.assertEqual(action.call_args.args[0], ['docker', 'rm', self.identity])
        self.assertTrue(json.loads(self.output.read_text())['passed'])

    def test_live_child_is_stopped_but_original_failure_remains(self):
        self.child['State'].update(Running=True, Pid=123, Status='running')
        def action(command, **kwargs):
            if command[1] == 'stop':
                self.child['State'].update(Running=False, Pid=0, Status='exited')
        with patch.object(suite.subprocess, 'check_output', side_effect=self.inspect), \
             patch.object(suite.subprocess, 'run', side_effect=action) as commands:
            with self.assertRaises(RuntimeError):
                suite.retire_coding_children(self.owner, self.output)
        report = json.loads(self.output.read_text())
        self.assertTrue(report['complete'])
        self.assertFalse(report['passed'])
        self.assertEqual(report['unexpected_live'], [self.identity])
        self.assertEqual([row.args[0][1] for row in commands.call_args_list], ['stop', 'rm'])

    def test_foreign_child_is_never_stopped_or_removed(self):
        self.child['Config']['Labels'][docker_paths.CODING_OWNER_LABEL] = 'unrelated'
        with patch.object(suite.subprocess, 'check_output', side_effect=self.inspect), \
             patch.object(suite.subprocess, 'run') as action, self.assertRaises(RuntimeError):
            suite.retire_coding_children(self.owner, self.output)
        action.assert_not_called()
        self.assertFalse(json.loads(self.output.read_text())['complete'])

    def test_uncertain_native_exit_cannot_release_its_ownership(self):
        self.child['State'].update(Running=True, Pid=123, Status='running')
        with patch.object(suite.subprocess, 'check_output', side_effect=self.inspect), \
             patch.object(suite.subprocess, 'run') as action, self.assertRaises(RuntimeError):
            suite.retire_coding_children(self.owner, self.output)
        self.assertEqual([row.args[0][1] for row in action.call_args_list], ['stop'])
        report = json.loads(self.output.read_text())
        self.assertFalse(report['complete'])
        self.assertNotIn(self.identity, report['retired'])

    def test_live_or_unknown_driver_blocks_child_inventory_and_success(self):
        self.driver_retirement.side_effect = RuntimeError('driver still alive')
        with patch.object(suite.subprocess, 'check_output') as query, self.assertRaises(RuntimeError):
            suite.retire_coding_children(self.owner, self.output)
        query.assert_not_called()
        self.assertFalse(json.loads(self.output.read_text())['complete'])


class CodingDriverRetirementTests(unittest.TestCase):
    """Only a proved retired parent makes a closed child inventory possible."""

    def setUp(self):
        self.owner = 'llaminar-suite-driver-' + '1' * 32
        self.driver = {'Id': 'b' * 64,
            'Config': {'Labels': {docker_paths.CODING_OWNER_LABEL: self.owner}},
            'State': {'Running': True, 'Pid': 321, 'Status': 'running'}}
        self.removed = False
        inspection = patch.object(docker_paths, 'inspect_owned_container',
                                  side_effect=lambda _: None if self.removed else self.driver)
        self.inspect = inspection.start()
        self.addCleanup(inspection.stop)

    def daemon(self, command, **kwargs):
        """Model --rm and exact daemon absence after explicit driver retirement."""
        if command[1] == 'rm':
            self.removed = True
        return subprocess.CompletedProcess(command, 0, '', '')

    def test_parent_is_proved_absent_after_signal_wait_and_removal(self):
        with patch.object(suite.subprocess, 'run', side_effect=self.daemon) as run:
            state = suite.retire_coding_driver(self.owner)
        self.assertEqual([c.args[0][1] for c in run.call_args_list],
                         ['kill', 'wait', 'rm'])
        self.assertEqual(self.inspect.call_count, 2)
        self.assertEqual(state['Pid'], 0)
        self.assertEqual(state['Id'], self.driver['Id'])

    def test_unchecked_successful_removal_cannot_hide_a_surviving_driver(self):
        def daemon(command, **kwargs):
            result = self.daemon(command, **kwargs)
            if command[1] == 'rm':
                self.removed = False
            return result
        with patch.object(suite.subprocess, 'run', side_effect=daemon), self.assertRaises(RuntimeError):
            suite.retire_coding_driver(self.owner)

    def test_only_daemon_inventory_proves_automatic_removal(self):
        self.removed = True
        with patch.object(suite.subprocess, 'run', side_effect=self.daemon) as run:
            self.assertEqual(suite.retire_coding_driver(self.owner)['Status'], 'absent')
        run.assert_not_called()
        self.inspect.side_effect = OSError('daemon unavailable')
        with self.assertRaises(OSError):
            suite.retire_coding_driver(self.owner)

    def test_foreign_driver_is_never_signalled(self):
        self.driver['Config']['Labels'].clear()
        with patch.object(suite.subprocess, 'run', side_effect=self.daemon) as run, self.assertRaises(ValueError):
            suite.retire_coding_driver(self.owner)
        run.assert_not_called()


class OwnedContainerInspectionTests(unittest.TestCase):
    """A typed daemon inventory handles auto-removal without diagnostic spelling."""

    def test_empty_exact_inventory_proves_absence_without_inspecting(self):
        with patch.object(docker_paths.subprocess, 'check_output', return_value='') as query, \
             patch.object(docker_paths.subprocess, 'run') as inspect:
            self.assertIsNone(docker_paths.inspect_owned_container('owned'))
        self.assertIn('name=^/owned$', query.call_args.args[0])
        inspect.assert_not_called()

    def test_autoremoval_race_is_independent_of_docker_error_spelling(self):
        for diagnostic in ('error: no such object: owned', 'Error: No such object: owned',
                           'Error response from daemon: No such container: owned'):
            with self.subTest(diagnostic=diagnostic), \
                 patch.object(docker_paths.subprocess, 'check_output', side_effect=['a' * 64 + '\n', '']), \
                 patch.object(docker_paths.subprocess, 'run', return_value=
                              subprocess.CompletedProcess(['docker', 'inspect'], 1, '[]', diagnostic)):
                self.assertIsNone(docker_paths.inspect_owned_container('owned'))

    def test_failed_or_foreign_inspection_does_not_authorize_absence(self):
        with patch.object(docker_paths.subprocess, 'check_output', side_effect=OSError('daemon unavailable')):
            with self.assertRaises(OSError):
                docker_paths.inspect_owned_container('owned')
        with patch.object(docker_paths.subprocess, 'check_output', return_value='a' * 64 + '\n'), \
             patch.object(docker_paths.subprocess, 'run', return_value=
                          subprocess.CompletedProcess([], 1, '[]', 'failure')):
            with self.assertRaises(subprocess.CalledProcessError):
                docker_paths.inspect_owned_container('owned')
        with patch.object(docker_paths.subprocess, 'check_output', return_value='a' * 64 + '\n'), \
             patch.object(docker_paths.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0,
                          json.dumps([{'Id': 'a' * 64, 'Name': '/foreign'}]), '')):
            with self.assertRaises(ValueError):
                docker_paths.inspect_owned_container('owned')


class CodingNativeLeaseTests(unittest.TestCase):
    """A dead runner cannot admit another suite beside its surviving native owner."""

    def test_all_production_leases_require_a_successful_empty_native_inventory(self):
        from io import StringIO
        with patch.object(suite.pipeline, 'Path') as path, \
             patch.object(suite.pipeline.fcntl, 'flock'), \
             patch.object(docker_paths.subprocess, 'check_output', return_value='') as query:
            path.return_value.open.return_value.__enter__.return_value = StringIO()
            with suite.pipeline.device_lease():
                self.assertEqual(query.call_count, 1)
            self.assertIn('label=' + docker_paths.CODING_OWNER_LABEL, query.call_args.args[0])

    def test_live_or_unreadable_daemon_state_blocks_the_lease_body(self):
        for output in ('a' * 64 + '\n', 'malformed identity'):
            entered = False
            with self.subTest(output=output), patch.object(suite.pipeline, 'Path'), \
                 patch.object(suite.pipeline.fcntl, 'flock'), \
                 patch.object(docker_paths.subprocess, 'check_output', return_value=output), \
                 self.assertRaises((ValueError, RuntimeError)):
                with suite.pipeline.device_lease():
                    entered = True
            self.assertFalse(entered)
        with patch.object(suite.pipeline, 'Path'), patch.object(suite.pipeline.fcntl, 'flock'), \
             patch.object(docker_paths.subprocess, 'check_output', side_effect=OSError('daemon unreachable')):
            with self.assertRaises(OSError), suite.pipeline.device_lease():
                self.fail('An unknown native owner cannot release the device lease')


if __name__ == '__main__':
    unittest.main()
