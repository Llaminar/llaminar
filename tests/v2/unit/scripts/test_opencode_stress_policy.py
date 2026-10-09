#!/usr/bin/env python3
"""Adversarial proofs of release stress tolerance and exact client error attribution.

Five percent is a strict rational boundary. Sparse successful sessions cannot
hide a failed app, missing phases, unexercised tools or protocol corruption.
Observed NUL-command and binary-read errors are tested through the real saved
conversation validator on both JSON and SSE, with independent negative inputs.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests/v2/e2e/server'))
import opencode_tool_stress as stress
from opencode_stress_policy import stress_assessment, app_archive_compaction_assessment
from test_opencode_tool_stress import conversation, wire


def session(attempts=100, errors=0, **changes):
    """Supply independently authenticated counters and a complete phase record."""
    value = {'case': 'number', 'ordinal': 0, 'tool_attempts': attempts, 'tool_errors': errors,
             'protocol_passed': True, 'phase_progress': {'expected': 1, 'completed': 1, 'passed': True},
             'task_passed': errors == 0, 'successful_tools': ['write', 'read']}
    value.update(changes)
    return value


class OpenCodeStressPolicyTests(unittest.TestCase):
    """Exercise the user-selected tolerance without relying on random model behavior."""

    def test_strict_five_percent_boundary_uses_unique_tool_attempts(self):
        for attempts, errors, passed in [(0, 0, False), (19, 1, False), (20, 1, False),
                                         (21, 1, True), (100, 4, True), (100, 5, False), (142, 4, True)]:
            with self.subTest(attempts=attempts, errors=errors):
                self.assertEqual(stress_assessment([session(attempts, errors)])['passed'], passed)

    def test_large_counts_do_not_round_five_percent_into_success(self):
        self.assertFalse(stress_assessment([session(20 * 2**60, 2**60)])['passed'])
        self.assertTrue(stress_assessment([session(20 * 2**60 + 1, 2**60)])['passed'])

    def test_other_sessions_cannot_dilute_bad_coding_session(self):
        rows = [session(10000, 0), session(20, 1, case='webapp', ordinal=1)]
        self.assertFalse(stress_assessment(rows)['passed'])

    def test_one_bad_primitive_can_be_tolerated_in_a_complete_cell(self):
        self.assertTrue(stress_assessment([session(2, 1), session(100, 0)])['passed'])

    def test_protocol_failures_incomplete_phases_and_missing_evidence_are_fatal(self):
        for mutation in ({'protocol_passed': False}, {'phase_progress': {}},
                         {'phase_progress': {'expected': 10, 'completed': 9, 'passed': False}}):
            self.assertFalse(stress_assessment([session(**mutation)])['passed'])
        for attempts, errors in [(False, 0), (20, -1), (20, 21), (20.0, 0)]:
            with self.assertRaises(ValueError):
                stress_assessment([session(attempts, errors)])

    def test_stress_gate_retains_task_quality_and_requires_complete_coverage(self):
        row = session(100, 1)
        result = stress.campaign_assessment([row], ['number'], 1, 'stress')
        self.assertTrue(result['gate_passed'])
        self.assertFalse(result['task_passed'])
        row['successful_tools'] = ['write']
        self.assertFalse(stress.campaign_assessment([row], ['number'], 1, 'stress')['gate_passed'])

    def test_nul_and_binary_read_require_unchanged_wire_history_client_and_exact_error(self):
        for streamed in (False, True):
            for kind in ('nul', 'binary'):
                for mutation in (None, 'input', 'echo', 'tool', 'history', 'no_nul'):
                    with self.subTest(stream=streamed, kind=kind, mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                        directory = Path(temporary)
                        rows, events = conversation(directory, streamed)
                        index = 0 if kind == 'nul' else 1
                        identity = 'call_write' if kind == 'nul' else 'call_read'
                        name = 'bash' if kind == 'nul' else 'read'
                        command = 'echo "Unicode 🙂 and NUL ' + ('x' if mutation == 'no_nul' else '\x00') + '"'
                        args = {'command': command} if kind == 'nul' else {'filePath': 'result.txt'}
                        error = ("The argument 'file' must be a string without null bytes. Received "
                                 + '"' + command.replace('"', '\\"') + '"') if kind == 'nul' else (
                                 'Cannot read binary file: ' + str(directory / 'workspace/result.txt'))
                        for row in rows:
                            for definition in row['request']['tools']:
                                if definition['function']['name'] == ('write' if kind == 'nul' else 'read'):
                                    definition['function'] = {'name': name, 'parameters': {'type': 'object',
                                        'required': list(args), 'properties': {k: {'type': 'string'} for k in args}}}
                            for message in row['request']['messages']:
                                for call in message.get('tool_calls', []):
                                    if call['id'] == identity:
                                        call['function'] = {'name': name, 'arguments': json.dumps(args)}
                        message, _ = stress.response_message(rows[index]['response'], streamed)
                        message['tool_calls'][0]['function'] = {'name': name, 'arguments': json.dumps(args)}
                        rows[index]['response'] = wire(message, streamed)
                        part = events[index]['part']
                        part.update(tool=name, state={'status': 'error', 'input': deepcopy(args), 'error': error})
                        if mutation == 'input':
                            part['state']['input'][next(iter(args))] += 'changed'
                        elif mutation == 'echo':
                            part['state']['error'] += ' changed'
                        elif mutation == 'tool':
                            part['tool'] = 'glob'
                        elif mutation == 'history':
                            for previous in rows[index + 1]['request']['messages']:
                                for call in previous.get('tool_calls', []):
                                    if call['id'] == identity:
                                        call['function']['arguments'] = '{}'
                        (directory / 'client.jsonl').write_text('\n'.join(json.dumps(event) for event in events))
                        result = stress.check_session(directory, rows, 0)
                        recognized = mutation is None or (kind == 'binary' and mutation == 'no_nul')
                        self.assertEqual(result['protocol_passed'], recognized, result['errors'])
                        # Corrupt continuation remains a protocol failure, but
                        # cannot erase the separately authenticated client error.
                        self.assertEqual(result['tool_errors'], int(recognized or mutation == 'history'))
                        self.assertEqual(result['tool_attempts'], 2)
                        self.assertNotIn(name, result['successful_tools'])


    def test_default_cli_runs_one_complete_app_with_stress_policy(self):
        """An omitted case must never schedule the retired primitive sweep."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'app'
            row = session(case='webapp', phase_progress={'expected': 10, 'completed': 10, 'passed': True},
                          successful_tools=sorted(stress.required_tools('webapp')))
            command = ['opencode_tool_stress.py', '--base-url', 'http://127.0.0.1:1',
                       '--model', 'fixture', '--context-length', '262144', '--output', str(output)]
            with patch.object(sys, 'argv', command), patch.object(stress, 'require_ready'), \
                 patch.object(stress.subprocess, 'check_output', return_value='1.18.34'), \
                 patch.object(stress, 'run_session', return_value=row) as run:
                self.assertEqual(stress.main(), 0)
            run.assert_called_once()
            self.assertEqual(run.call_args.args[1:3], (0, 'webapp'))
            config = run.call_args.args[0]
            self.assertEqual(config.max_tokens, 32768)
            self.assertIsNone(config.request_timeout)
            self.assertIsNone(config.session_timeout)
            receipt = json.loads((output / 'stress.json').read_text())
            self.assertEqual(receipt['gate'], 'stress')
            self.assertEqual(receipt['iterations'], 1)
            self.assertTrue(receipt['gate_passed'])

    def test_app_does_not_require_compaction_but_rejects_bad_observed_work(self):
        """No maintenance activity is distinct from a valid compaction proof."""
        empty = app_archive_compaction_assessment([])
        self.assertTrue(empty['passed'])
        self.assertFalse(empty['exercised'])
        record = {'domain': 'prefix_archive', 'name': 'background_compactions', 'kind': 'counter',
                  'value': 1, 'tags': {'payload_bytes_copied': '0', 'authority': 'archive_worker'}}
        result = app_archive_compaction_assessment([record])
        self.assertTrue(result['passed'])
        self.assertTrue(result['exercised'])
        for mutation in ({'value': True}, {'value': 0}, {'value': -1}, {'value': float('nan')},
                         {'value': float('inf')}, {'value': 0.5}, {'kind': 'timer'},
                         {'tags': {'payload_bytes_copied': '1', 'authority': 'archive_worker'}},
                         {'tags': {'payload_bytes_copied': '0', 'authority': 'request'}}, {'tags': None}):
            with self.subTest(mutation=mutation):
                self.assertFalse(app_archive_compaction_assessment([record, record | mutation])['passed'])


if __name__ == '__main__':
    unittest.main()
