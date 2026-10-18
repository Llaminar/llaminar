#!/usr/bin/env python3
"""Adversarial, device-free tests of complete coding-workload certification.

Fixtures represent a finished client session with independently named calls
and phases. Tampering checks prove that cached green flags cannot drop turns,
restart the conversation, dilute tool errors or omit native response coverage.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests/v2/e2e/server'))
from opencode_certification_evidence import validate_app_report, validate_native_coverage, VERSION
from opencode_tool_stress import (assessment, campaign_assessment, fixture_cases,
                                  required_tools, session_prompts)


def finished_report(*, errors: int = 0, attempts: int = 21, task_errors: list | None = None) -> dict:
    """Retain declared real prompts with synthetic call and timing metadata."""
    prompts = session_prompts('webapp', fixture_cases()['webapp'])
    phases = []
    for index, prompt in enumerate(prompts):
        digest = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        phases.append({'phase': index, 'finished': True, 'returncode': 0,
                       'interrupted_by_harness': False, 'session_id': 'same-session',
                       'exchanges': 3, 'first_exchange': index * 3, 'elapsed_seconds': 600.0,
                       'prompt': {'passed': True, 'expected_sha256': digest,
                                  'observed_sha256': digest, 'error': None}})
    result = {**assessment([], task_errors or []), 'case': 'webapp', 'ordinal': 0,
              'timed_out': False, 'elapsed_seconds': 6000.0,
              'phase_progress': {'expected': len(prompts), 'completed': len(prompts),
                                 'passed': True, 'phases': phases},
              'request_count': len(phases) * 3,
              'tool_calls': [{'id': f'call-{index}'} for index in range(attempts)],
              'execution_errors': [{'call_id': f'call-{index}'} for index in range(errors)],
              'tool_attempts': attempts, 'tool_errors': errors,
              'successful_tools': sorted(required_tools('webapp'))}
    return {'schema': 2, 'complete': True, 'opencode_version': VERSION, 'gate': 'stress',
            'model': 'Qwen-fixture', 'context_length': 131072, 'max_tokens': 32768,
            'iterations': 1, 'concurrency': 1, 'measure_prefix_reuse': True,
            'turn_wall_time_limit_seconds': None, 'request_inactivity_timeout_seconds': None,
            'results': [result], **campaign_assessment([result], ['webapp'], 1, 'stress')}


def validate(report: dict) -> dict:
    """Supply the lifecycle owner's exact admitted client configuration."""
    return validate_app_report(report, model='Qwen-fixture', context_tokens=131072)


class OpenCodeCertificationEvidenceTests(unittest.TestCase):
    """Engine success requires complete evidence while app quality remains separate."""

    def test_long_turns_and_imperfect_app_remain_valid_below_error_allowance(self):
        report = finished_report(errors=1, task_errors=['generated app has a failing test'])
        result = validate(report)
        self.assertTrue(result['gate_passed'])
        self.assertFalse(result['task_passed'])
        self.assertEqual(result['stress_policy']['tool_errors'], 1)
        self.assertEqual(result['completed_phases'], 10)

    def test_five_percent_or_more_cannot_be_repaired_by_green_flags(self):
        for attempts in (20, 19):
            report = finished_report(errors=1, attempts=attempts)
            with self.subTest(attempts=attempts), self.assertRaises(ValueError):
                validate(report)
            report['gate_passed'] = True
            with self.assertRaises(ValueError):
                validate(report)

    def test_missing_duplicated_changed_or_interrupted_phase_is_fatal(self):
        mutations = [lambda p: p.pop(), lambda p: p.__setitem__(2, deepcopy(p[1])),
                     lambda p: p[2].update(session_id='restarted-session'),
                     lambda p: p[2].update(finished=False),
                     lambda p: p[2].update(interrupted_by_harness=True),
                     lambda p: p[2].update(returncode=1),
                     lambda p: p[2].update(first_exchange=0),
                     lambda p: p[2]['prompt'].update(observed_sha256='0' * 64),
                     lambda p: p[2].update(elapsed_seconds=float('nan'))]
        for index, mutate in enumerate(mutations):
            report = finished_report()
            mutate(report['results'][0]['phase_progress']['phases'])
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate(report)

    def test_protocol_fault_or_missing_tool_cannot_hide_behind_pass(self):
        for change in ('protocol', 'coverage'):
            report = finished_report()
            if change == 'protocol':
                report['results'][0]['protocol_errors'].append('changed tool argument')
            else:
                report['results'][0]['successful_tools'].remove('grep')
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(report)

    def test_wrong_workload_or_hidden_timeout_is_not_certification(self):
        for field, value in (('model', 'other-model'), ('context_length', 32768),
                             ('max_tokens', 4096), ('gate', 'protocol'),
                             ('iterations', 2), ('concurrency', 2),
                             ('opencode_version', 'unqualified'),
                             ('turn_wall_time_limit_seconds', 9000),
                             ('request_inactivity_timeout_seconds', 300)):
            report = finished_report()
            report[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(report)

    def test_tool_and_request_counts_require_actual_unique_identities(self):
        changes = [lambda r: r.update(request_count=r['request_count'] + 1),
                   lambda r: r.update(tool_attempts=1000),
                   lambda r: r.update(tool_errors=0),
                   lambda r: r['tool_calls'][1].update(id=r['tool_calls'][0]['id']),
                   lambda r: r['execution_errors'][0].update(call_id='missing')]
        for index, mutate in enumerate(changes):
            report = finished_report(errors=1)
            mutate(report['results'][0])
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate(report)

    def test_native_audit_must_cover_the_whole_session(self):
        app = validate(finished_report())
        audit = {'complete': True, 'passed': True, 'joined_responses': app['request_count'],
                 'verified_native_calls': app['stress_policy']['tool_attempts'],
                 'changed_responses': 0, 'changes': []}
        validate_native_coverage(audit, app)
        for field, value in (('joined_responses', audit['joined_responses'] - 1),
                             ('verified_native_calls', audit['verified_native_calls'] - 1),
                             ('complete', False), ('changed_responses', 1),
                             ('changes', ['changed native argument'])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_native_coverage({**audit, field: value}, app)


if __name__ == '__main__':
    unittest.main()
