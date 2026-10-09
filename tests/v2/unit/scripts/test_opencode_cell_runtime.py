#!/usr/bin/env python3
"""Exercise coding-cell ownership and failure frontiers without devices or models.

The daemon fixture can lose a successful create reply, reject a stop, or return
an unexpected admission result. The real lifecycle must retain those failures,
close independent observers and forbid successors until every native owner is
proven retired. Existing app validators authenticate the synthetic client report.
"""
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests/v2/e2e/server'))
sys.path.insert(0, str(Path(__file__).parent))
import opencode_cell_runtime as runtime
from retired_prefix_archive import reclaim
from test_opencode_certification_evidence import finished_report


def configuration(backend='ROCm', policy='dynamic'):
    """Small typed inventory fixture; no HTTP stress flags enter production argv."""
    return {'e2e': {'context_length': 8192, 'readiness_timeout_seconds': 30,
                    'cell_timeout_seconds': {'AVX512': 60, 'AVX2': 60},
                    'planning': {'mode': 'auto', 'strategy': 'tp',
                                 'device_counts': {backend.lower(): 2}, 'mpi_ranks': 1}},
            'benchmark': {'schema': 1, 'policy': 'production_defaults', 'context_length': 8192,
                          'args': ['--placement', 'auto', '--strategy', 'tp',
                                   '--devices', backend.lower() + ':2']},
            'runtime': {'generation': {'mtp_policy': policy, 'mtp_verify_mode': 'speculative-sampling'}}}


class ContainerOwnershipTests(unittest.TestCase):
    """A client failure cannot erase a native resource that may already exist."""

    def test_lost_create_reply_still_requires_native_retirement(self):
        owners = runtime.ContainerOwners()
        with patch.object(runtime, 'command', side_effect=subprocess.TimeoutExpired(['docker'], 30)):
            with self.assertRaises(subprocess.TimeoutExpired):
                owners.create('owned', ['image'])
        self.assertEqual(owners.identities, {'owned': None})
        self.assertIn('--init', owners.commands['owned'])
        with patch.object(runtime.docker_paths, 'inspect_owned_container', return_value={'Id': 'a' * 64}), \
             patch.object(runtime, 'retire_owned', return_value={'Running': False, 'Pid': 0}) as retire:
            self.assertTrue(owners.close()['passed'])
        retire.assert_called_once_with('owned')

    def test_only_daemon_proved_absent_unacknowledged_create_is_safe(self):
        owners = runtime.ContainerOwners({'owned': None})
        with patch.object(runtime.docker_paths, 'inspect_owned_container', return_value=None):
            self.assertTrue(owners.close()['passed'])
        for failure in (ValueError('foreign name'), OSError('daemon unavailable')):
            with self.subTest(failure=failure), patch.object(runtime.docker_paths,
                    'inspect_owned_container', side_effect=failure):
                self.assertFalse(owners.close()['passed'])
        owners.identities['owned'] = 'a' * 64
        failure = subprocess.CalledProcessError(1, ['docker', 'inspect', 'owned'],
                                               output='Error: No such object: owned')
        with patch.object(runtime, 'retire_owned', side_effect=failure):
            self.assertFalse(owners.close()['passed'])

    def test_failure_does_not_skip_other_owned_resources(self):
        owners = runtime.ContainerOwners({'first': 'a' * 64, 'second': 'b' * 64})
        def retire(name):
            if name == 'first':
                raise OSError('daemon unreachable')
            return {'Running': False, 'Pid': 0}
        with patch.object(runtime, 'retire_owned', side_effect=retire) as action:
            result = owners.close()
        self.assertFalse(result['passed'])
        self.assertEqual(set(result['retired']), {'second'})
        self.assertEqual(action.call_count, 2)
        with self.assertRaisesRegex(ValueError, 'unowned'):
            owners.retire('unrelated')

    def test_unknown_admission_never_becomes_a_smaller_context(self):
        self.assertEqual(runtime.admission_outcome(0, '--dry-run complete: preflight validation passed'),
                         runtime.AdmissionOutcome.ACCEPTED)
        self.assertEqual(runtime.admission_outcome(1, 'Memory plan validation failed'),
                         runtime.AdmissionOutcome.CAPACITY_REJECTED)
        for code, log in [(139, 'Memory plan validation failed'), (1, 'failed to initialize graph'),
                          (0, ''), (137, '--dry-run complete: preflight validation passed')]:
            with self.subTest(code=code, log=log), self.assertRaises(ValueError):
                runtime.admission_outcome(code, log)

    def test_failed_helper_retirement_cannot_replace_operator_cancellation(self):
        owners = runtime.ContainerOwners()
        interrupted = KeyboardInterrupt('cancelled')
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(runtime, 'command', return_value='a' * 64), \
             patch.object(runtime.subprocess, 'run', side_effect=interrupted), \
             patch.object(owners, 'remove', side_effect=OSError('daemon unavailable')):
            with self.assertRaises(KeyboardInterrupt) as caught:
                runtime.run_retired(owners, 'owned', ['image'], log=Path(temporary) / 'helper.log')
        self.assertIs(caught.exception, interrupted)
        self.assertEqual(owners.identities, {'owned': 'a' * 64})
        self.assertIn('retirement remains unproven', ' '.join(interrupted.__notes__))

    def test_fast_helper_uses_complete_daemon_logs_not_truncated_attach_output(self):
        owners = runtime.ContainerOwners()
        retired = {'Running': False, 'Pid': 0, 'Status': 'exited', 'ExitCode': 0,
                   'OOMKilled': False, 'Error': ''}
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(owners, 'create'), patch.object(owners, 'remove'), \
             patch.object(runtime, 'require_retired', return_value=retired), \
             patch.object(runtime.subprocess, 'run', side_effect=[
                 subprocess.CompletedProcess(['docker', 'start'], 0, 'truncated', ''),
                 subprocess.CompletedProcess(['docker', 'logs'], 0, '{"complete":true}\n', 'diagnostic\n')]) as run:
            log = Path(temporary) / 'helper.log'
            result, state = runtime.run_retired(owners, 'owned', ['image'], log=log, input_text='request')
            self.assertEqual(result.stdout, '{"complete":true}\n')
            self.assertEqual(result.stderr, 'diagnostic\n')
            self.assertEqual(log.read_text(), result.stdout + result.stderr)
        self.assertEqual(state, retired)
        self.assertIn('--interactive', run.call_args_list[0].args[0])
        self.assertEqual(run.call_args_list[0].kwargs['input'], 'request')
        self.assertEqual(run.call_args_list[1].args[0], ['docker', 'logs', 'owned'])

    def test_helper_cannot_retire_a_preexisting_owners_name(self):
        owners = runtime.ContainerOwners({'owned': 'a' * 64})
        with patch.object(owners, 'remove') as remove, self.assertRaises(ValueError):
            runtime.run_retired(owners, 'owned', ['image'], log=Path('/unused.log'))
        remove.assert_not_called()
        self.assertEqual(owners.identities, {'owned': 'a' * 64})

    def test_selected_policy_and_full_reasoning_budget_are_retained(self):
        for backend in ('CPU', 'CUDA', 'ROCm'):
            for policy in ('off', 'dynamic', 'depth_1', 'depth_15'):
                config = configuration(backend, policy)
                cell = runtime.CodingCell('sha256:' + 'a' * 64, config, backend,
                                          Path('/model/model.gguf'), Path('/results'))
                args = cell.server_arguments(131072)
                self.assertEqual(args[1:1 + len(config['benchmark']['args'])], config['benchmark']['args'])
                self.assertEqual(args[args.index('--context-length') + 1], '131072')
                self.assertNotIn('--mtp', args)
                with self.assertRaises(ValueError):
                    cell.server_arguments(32768)
        for policy in ('unknown', 'depth_0', 'depth_01', None):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                runtime.generation_policy(configuration(policy=policy))


class CellLifetimeTests(unittest.TestCase):
    """Drive the real lifetime through independent faults and retirement ordering."""

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        temporary = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.output = Path(temporary) / 'cell'
        self.cell = runtime.CodingCell('sha256:' + 'a' * 64, configuration(), 'ROCm',
                                      Path(temporary) / 'model.gguf', self.output)
        self.containers, self.events = {}, []
        self.client_error = self.stop_error = self.create_error = self.admission_error = None
        self.stats = {'model': 'Qwen-fixture',
                      'configuration': {'context_tokens': 131072, 'prefix_cache': {
                          'ram_budget_bytes_per_participant': runtime.RAM_MIB * 1024**2,
                          'disk_budget_bytes': runtime.DISK_MIB * 1024**2}},
                      'prefix_cache': {'storage': {'tiers': {
                          'ram': {'capacity_bytes': runtime.RAM_MIB * 2 * 1024**2},
                          'disk': {'capacity_bytes': runtime.DISK_MIB * 1024**2}}}}}
        for name, effect in [('command', self.command), ('admit_context', self.admit),
                             ('fresh_observation', self.observation), ('metadata', self.metadata),
                             ('runtime_evidence', lambda *_: {'passed': True}),
                             ('assess_observations', lambda **_: {'passed': True})]:
            self.stack.enter_context(patch.object(runtime, name, side_effect=effect))
        self.stack.enter_context(patch.object(runtime.docker_paths, 'device_args', return_value=[]))
        self.stack.enter_context(patch.object(runtime.docker_paths, 'mounts', return_value=[]))
        self.stack.enter_context(patch.object(runtime.docker_paths, 'inspect_owned_container',
            side_effect=lambda name: {'Id': 'a' * 64} if name in self.containers else None))
        self.stack.enter_context(patch.object(runtime, 'RuntimeStatsObserver', return_value=nullcontext()))
        self.stack.enter_context(patch.object(runtime, 'collect_and_publish_ranked_perf_stats', return_value={}))
        self.stack.enter_context(patch.object(runtime.subprocess, 'run', side_effect=self.process))
        self.stack.enter_context(patch.object(runtime.subprocess, 'Popen', side_effect=self.launch))
        self.stack.enter_context(patch.object(runtime, 'urlopen', side_effect=self.shutdown))

    def command(self, args, **kwargs):
        """Model native Docker state independently from the controller's flags."""
        self.events.append(tuple(args))
        if args[0] != 'docker':
            return ''  # Independent driver diagnostics are covered separately.
        action = args[1]
        if action == 'create':
            name = args[args.index('--name') + 1]
            self.containers[name] = {'Running': False, 'Pid': 0, 'Status': 'created',
                                     'ExitCode': 0, 'Error': '', 'OOMKilled': False}
            if self.create_error:
                raise self.create_error
            return ('b' if name.endswith('-resources') else 'a') * 64
        name = args[-1]
        if action == 'inspect':
            return json.dumps([{'State': self.containers[name]}])
        if action == 'start':
            self.containers[name].update(Running=True, Pid=123, Status='running')
            return name
        if action == 'wait':
            if name.endswith('-resources'):
                observer = json.loads((self.output / 'resource-controller.json').read_text())
                self.assertTrue(observer['complete'])
                self.assertFalse(json.loads((self.output / 'controller.json').read_text())['complete'])
                self.assertFalse(self.containers[name.removesuffix('-resources')]['Running'])
                self.containers[name].update(Running=False, Pid=0, Status='exited')
            return '0'
        if action == 'stop':
            if self.stop_error:
                raise self.stop_error
            self.containers[name].update(Running=False, Pid=0, Status='exited')
            return name
        if action == 'run':
            return json.dumps({'passed': True, 'archives': []})
        if action == 'rm':
            self.assertFalse(self.containers[name]['Running'])
            return name
        raise AssertionError(args)

    def admit(self, cell, name, devices, owners):
        """Publish a fixed admitted context, or leave one failed probe owned."""
        if self.admission_error:
            owners.create(name + '-admit', ['image'])
            self.containers[name + '-admit'].update(Running=True, Pid=123, Status='running')
            raise self.admission_error
        return {'context_tokens': 131072}

    def observation(self, output, identity, **kwargs):
        """Create bounded host metadata without /proc access or GPU probes."""
        resource = output / 'resources'
        resource.mkdir(exist_ok=True)
        (resource / 'observations.jsonl').write_text('{}\n')
        return {'observed': time.time(), 'boot_id': 'fixture', 'cells': {
            'app': {'stats_observation': {'stats': self.stats}}}}

    def metadata(self, url, route='/stats'):
        """The mock transport exposes the same admitted model/cache identity."""
        return {'status': 'ok'} if route == '/health' else self.stats

    def process(self, args, **kwargs):
        """Retain full authentic phase prompts through the existing app validator."""
        if args[0] == 'docker':
            self.containers[args[-1]].update(Running=False, Pid=0, Status='exited')
            return subprocess.CompletedProcess(args, 0, json.dumps({'passed': True,
                'payload_read_bytes': 0, 'remaining_entries': 0, 'archives': []}), '')
        if args[1].endswith('opencode_tool_stress.py'):
            if self.client_error:
                raise self.client_error
            directory = self.output / 'app'
            directory.mkdir()
            (directory / 'stress.json').write_text(json.dumps(finished_report()))
        else:
            self.assertTrue(args[1].endswith('opencode_native_audit.py'))
            directory = self.output / 'native-audit'
            directory.mkdir()
            (directory / 'result.json').write_text(json.dumps({
                'complete': True, 'passed': True, 'joined_responses': 30,
                'verified_native_calls': 21, 'changed_responses': 0, 'changes': []}))
        self.assertNotIn('timeout', kwargs)
        return SimpleNamespace(returncode=0)

    def launch(self, args, **kwargs):
        """Create a client or log follower without sharing their distinct lifetimes."""
        if args[0] != 'docker':
            self.assertTrue(kwargs['start_new_session'])
            self.process(args, **kwargs)
        return MagicMock(**{'wait.return_value': 0, 'poll.return_value': 0})

    def shutdown(self, request, **kwargs):
        """The server acknowledges shutdown before Docker observes native exit."""
        for name, state in self.containers.items():
            if not name.endswith('-resources'):
                state.update(Running=False, Pid=0, Status='exited')
        from io import BytesIO
        response = BytesIO(b'{"status":"shutting_down"}')
        response.status = 202
        return response

    def test_complete_only_after_native_and_observer_retirement(self):
        result = runtime.run_app(self.cell)
        self.assertTrue(result['passed'], result['errors'])
        self.assertTrue(result['complete'])
        self.assertEqual(result['app']['completed_phases'], 10)
        self.assertTrue(all(not state['Running'] for state in self.containers.values()))

    def test_failed_audit_preserved_while_other_audits_finish(self):
        with patch.object(runtime, 'runtime_evidence', side_effect=ValueError('graph mismatch')):
            result = runtime.run_app(self.cell)
        self.assertFalse(result['passed'])
        self.assertIn('graph mismatch', repr(result['errors']))
        self.assertIn('app', result)
        self.assertTrue(result['owned_containers']['passed'])
        self.assertTrue(any('finish' in event for event in self.events))

    def test_cancelled_client_closes_owned_resources_and_reraises(self):
        self.client_error = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            runtime.run_app(self.cell)
        result = json.loads((self.output / 'controller.json').read_text())
        self.assertFalse(result['passed'])
        self.assertTrue(result['owned_containers']['passed'])
        self.assertTrue(all(not state['Running'] for state in self.containers.values()))

    def test_failed_admission_owner_prevents_any_successor(self):
        self.admission_error = ValueError('admission failed')
        self.stop_error = OSError('daemon did not retire probe')
        with self.assertRaisesRegex(RuntimeError, 'no successor'):
            runtime.run_app(self.cell)
        result = json.loads((self.output / 'controller.json').read_text())
        self.assertFalse(result['passed'])
        self.assertFalse(result['owned_containers']['passed'])

    def test_create_reply_loss_keeps_cleanup_ownership(self):
        self.create_error = subprocess.TimeoutExpired(['docker', 'create'], 30)
        result = runtime.run_app(self.cell)
        self.assertFalse(result['passed'])
        self.assertTrue(result['owned_containers']['passed'])
        self.assertEqual(len(result['owned_containers']['retired']), 1)


class RetiredArchiveTests(unittest.TestCase):
    """A complete matrix keeps only metadata; cache bytes are never read to reclaim them."""

    def test_payload_unlinked_without_reading_or_following_external_symlinks(self):
        retirement = {'passed': True, 'errors': {}, 'retired': {
            'native': {'Running': False, 'Pid': 0, 'Status': 'exited'}}}
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            archive = parent / 'archive'
            archive.mkdir()
            outside = parent / 'outside'
            outside.write_text('keep')
            blocks = archive / 'model.kvcache.blocks'
            blocks.mkdir()
            (blocks / 'payload.kvblock').write_bytes(b'payload')
            (archive / 'alias').symlink_to(outside)
            (archive / 'model.kvcache').write_bytes(b'journal')
            with patch.object(Path, 'open', side_effect=AssertionError('cache payload opened')):
                result = reclaim(archive, retirement)
            self.assertEqual(result['payload_read_bytes'], 0)
            self.assertEqual(result['remaining_entries'], 0)
            self.assertEqual(outside.read_text(), 'keep')

    def test_missing_or_live_retirement_leaves_every_byte_owned(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary)
            payload = archive / 'payload.kvblock'
            payload.write_bytes(b'live')
            states = [{}, {'passed': True, 'errors': {}, 'retired': {}},
                      {'passed': True, 'errors': {}, 'retired': {'native': {
                          'Running': True, 'Pid': 123, 'Status': 'running'}}},
                      {'passed': False, 'errors': {'native': 'daemon error'}, 'retired': {}}]
            for state in states:
                with self.subTest(state=state), self.assertRaises(ValueError):
                    reclaim(archive, state)
                self.assertEqual(payload.read_bytes(), b'live')


if __name__ == '__main__':
    unittest.main()
