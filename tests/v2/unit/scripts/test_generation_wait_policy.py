#!/usr/bin/env python3
"""Device-free lifetime regressions for the real canonical HTTP probe client.

Requests, models and processes are authored mocks. The tests prove finite CI
semantics survive while an explicit local diagnostic uses a blocking socket,
retains all canonical validations, and preserves failure observations.
"""
from contextlib import ExitStack
import json
import os
import subprocess
import time
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts/ci'))
import generation_regression_http as client


class GenerationWaitPolicyTests(unittest.TestCase):
    """Explicit lifetime selection never changes token/prefix/MTP validators."""
    def test_ci_default_is_still_exactly600_and_expiry_fails(self):
        deadline=client.GenerationWaitPolicy.CI_BOUNDED.deadline(10)
        self.assertEqual(deadline,610)
        with patch.object(client.time,'monotonic',return_value=40):
            self.assertEqual(client.generation_wait_remaining(deadline),570)
        with patch.object(client.time,'monotonic',return_value=610):
            with self.assertRaises(TimeoutError):client.generation_wait_remaining(deadline)

    def test_local_wait_has_no_deadline_or_clock_probe(self):
        self.assertIsNone(client.GenerationWaitPolicy.LOCAL_UNBOUNDED.deadline(10))
        with patch.object(client.time,'monotonic',side_effect=AssertionError('No deadline clock')):
            self.assertIsNone(client.generation_wait_remaining(None))

    def exercise(self,policy,interrupt=False):
        """Exercise the unchanged real request loop with an authored four-prefix workload."""
        requests=[dict(id=str(i),body=dict(prompt='same',seed=4242)) for i in range(4)]
        profile=dict(mtp_policy='off',requests=requests)
        traces=[SimpleNamespace(completion=tuple(range(384))) for _ in requests]
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            for name in ('generation_profile','validate_partial_prompt','validate_prefix_outcome','validate_mtp_outcome','write_json'):
                stack.enter_context(patch.object(client,name,return_value=profile if name=='generation_profile' else None))
            workload=stack.enter_context(patch.object(client.GenerationWorkload,'from_record')).return_value
            workload.observe.side_effect=traces
            stack.enter_context(patch.object(client,'MovementRequirement'))
            movement=stack.enter_context(patch.object(client,'MovementLedgerObserver')).return_value
            stack.enter_context(patch.object(client.time,'monotonic',side_effect=(lambda:100) if policy is client.GenerationWaitPolicy.LOCAL_UNBOUNDED or interrupt else iter([100,110,111,112,113,120,121,122,123,130,131,132,133,140,141,142,143,144]).__next__))
            compare=stack.enter_context(patch.object(client,'compare_tokens',return_value=None))
            post=stack.enter_context(patch.object(client,'post_completion',side_effect=([{},KeyboardInterrupt('operator')] if interrupt else [{}, {}, {}, {}])))
            if interrupt:
                with self.assertRaises(KeyboardInterrupt):
                    client.run_probes({'runtime':{'movement_evidence':{}}},'http://fixture',Path(folder)/'result',wait_policy=policy)
                self.assertTrue(client.write_json.called)
                report=client.write_json.call_args.args[1]
                self.assertFalse(report['complete']);self.assertEqual(len(report['requests']),1)
                self.assertEqual(report['error'],'operator')
            else:
                report=client.run_probes({'runtime':{'movement_evidence':{}}},'http://fixture',Path(folder)/'result',wait_policy=policy)
                self.assertTrue(report['complete']);self.assertEqual(len(report['requests']),4)
                self.assertEqual(client.validate_mtp_outcome.call_count,4)
                self.assertEqual(client.validate_prefix_outcome.call_count,4)
                self.assertEqual(client.validate_partial_prompt.call_count,4)
                self.assertEqual(compare.call_count,3);movement.finish.assert_called_once()
            return post.call_args_list

    def test_all_four_canonical_requests_keep_validations_with_none_socket_timeout(self):
        calls=self.exercise(client.GenerationWaitPolicy.LOCAL_UNBOUNDED)
        self.assertEqual([call.args[2] for call in calls],[None]*4)

    def test_ci_requests_still_share_the_same_finite_sequence_budget(self):
        calls=self.exercise(client.GenerationWaitPolicy.CI_BOUNDED)
        self.assertEqual([call.args[2] for call in calls],[590,580,570,560])

    def test_operator_interruption_preserves_nonpassing_partial_observations(self):
        self.exercise(client.GenerationWaitPolicy.LOCAL_UNBOUNDED,True)

    def test_untyped_policy_rejected_before_request(self):
        with self.assertRaises(ValueError):client.run_probes({},'unused',Path('/unused'),wait_policy=None)

    def test_http_none_timeout_reaches_socket_api_without_global_override(self):
        from io import StringIO
        with patch.object(client.urllib.request,'urlopen') as open_request:
            open_request.return_value.__enter__.return_value=StringIO('{}')
            self.assertEqual(client.post_completion('http://fixture',{},None),{})
            self.assertIsNone(open_request.call_args.kwargs['timeout'])

    def test_cli_explicit_option_reaches_real_loop_default_remains_ci(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'config.json';p.write_text('{}')
            argv=['--configuration',str(p),'--base-url','http://fixture','--output',folder]
            with patch.object(client,'run_probes') as probes:
                self.assertEqual(client.main(argv),0)
                self.assertIs(probes.call_args.kwargs['wait_policy'],client.GenerationWaitPolicy.CI_BOUNDED)
                self.assertEqual(client.main(argv+['--local-unbounded-generation']),0)
                self.assertIs(probes.call_args.kwargs['wait_policy'],client.GenerationWaitPolicy.LOCAL_UNBOUNDED)

    def shell_helper(self):
        """Execute the actual production lifetime function in a device-free shell fixture."""
        text=(ROOT/'tests/v2/e2e/server/test_server_e2e.sh').read_text()
        body=text.split('wait_for_local_server_exit() {',1)[1].split('\n}\n',1)[0]
        return 'wait_for_local_server_exit() {'+body+'\n}\n'

    def test_harness_invalid_local_option_combinations_reject_before_startup(self):
        harness=ROOT/'tests/v2/e2e/server/test_server_e2e.sh'
        cases=[[],['--generation-configuration','/authored/config','--server-mode','docker'],
               ['--generation-configuration','/authored/config','--cross-host-case','authored']]
        for args in cases:
            environment={'PATH':os.environ['PATH'],'HOME':os.environ.get('HOME','/tmp')}
            result=subprocess.run(['bash',str(harness),'--local-unbounded-generation',*args],
                                  env=environment,text=True,capture_output=True,timeout=5)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertIn('Local unbounded generation requires',result.stderr)
            self.assertNotIn('Server started',result.stdout)

    def test_local_normal_wait_outlives_zero_ci_allowance_and_reaps_actual_child(self):
        script=self.shell_helper()+'''LOCAL_UNBOUNDED_GENERATION=1
SHUTDOWN_TIMEOUT=0
sleep .08 & child=$!
trap 'kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true' EXIT
wait_for_local_server_exit "$child" 0 || exit 91
wait "$child"
'''
        started=time.monotonic()
        result=subprocess.run(['bash','-c',script],text=True,capture_output=True,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertGreaterEqual(time.monotonic()-started,.07)

    def test_finite_ci_wait_does_not_kill_child_or_claim_normal_exit(self):
        script=self.shell_helper()+'''LOCAL_UNBOUNDED_GENERATION=0
SHUTDOWN_TIMEOUT=0
sleep 1 & child=$!
trap 'kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true' EXIT
if wait_for_local_server_exit "$child" 0; then exit 91; fi
kill -0 "$child" || exit 92
'''
        result=subprocess.run(['bash','-c',script],text=True,capture_output=True,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_failed_admin_uses30_second_exceptional_boundary_without_signalling_here(self):
        script=self.shell_helper()+'''LOCAL_UNBOUNDED_GENERATION=1
SHUTDOWN_TIMEOUT=0
SECONDS=0
calls=0
kill() { return 0; }
sleep() { calls=$((calls+1)); SECONDS=$((SECONDS+31)); }
if wait_for_local_server_exit 123 1; then exit 91; fi
[[ "$calls" == "1" ]] || exit 92
'''
        result=subprocess.run(['bash','-c',script],text=True,capture_output=True,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main(verbosity=2)
