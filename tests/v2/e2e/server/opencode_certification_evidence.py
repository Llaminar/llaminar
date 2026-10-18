"""Authenticate the complete app workload before it enters a CI receipt.

The live harness retains raw client/native evidence. This compact boundary
recomputes phase coverage, tool counts and the authorized error allowance; a
cached top-level pass cannot excuse a missing turn or a shortened workload.
Model-authored app quality stays visible independently of engine correctness.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys

from opencode_tool_stress import (assessment, campaign_assessment, fixture_cases,
                                  session_prompts)

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts/ci'))
from install_opencode_client import VERSION


def positive_seconds(value: object) -> bool:
    """Reject nonfinite timing metadata without bounding legitimate reasoning."""
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_app_report(report: dict, *, model: str, context_tokens: int,
                        output_tokens: int = 32768) -> dict:
    """Recompute one complete ten-phase session's qualification and counts.

    Context must be the caller's proven PMA admission. No elapsed-time or
    inactivity limit participates in certification. The prompt sequence comes
    from the same authored workload as the actual client, not a second list.
    """
    if (report.get('schema') != 2 or report.get('complete') is not True
            or report.get('opencode_version') != VERSION or report.get('gate') != 'stress'
            or report.get('model') != model or report.get('context_length') != context_tokens
            or report.get('max_tokens') != output_tokens
            or report.get('iterations') != 1 or report.get('concurrency') != 1
            or report.get('measure_prefix_reuse') is not True
            or 'turn_wall_time_limit_seconds' not in report
            or report['turn_wall_time_limit_seconds'] is not None
            or 'request_inactivity_timeout_seconds' not in report
            or report['request_inactivity_timeout_seconds'] is not None):
        raise ValueError('OpenCode workload identity, admission or completion differs from certification')
    results = report.get('results')
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError('OpenCode certification requires one complete app session')
    result = results[0]
    if (result.get('case') != 'webapp' or result.get('ordinal') != 0
            or result.get('timed_out') is not False
            or not positive_seconds(result.get('elapsed_seconds'))):
        raise ValueError('OpenCode app session was omitted, replaced or interrupted')
    for name in ('protocol_errors', 'task_errors'):
        errors = result.get(name)
        if not isinstance(errors, list) or any(not isinstance(error, str) for error in errors):
            raise ValueError('OpenCode session omitted its error evidence')
    verdict = assessment(result['protocol_errors'], result['task_errors'])
    if any(result.get(key) != value for key, value in verdict.items()):
        raise ValueError('OpenCode session verdict contradicts its original errors')
    prompts = session_prompts('webapp', fixture_cases()['webapp'])
    progress = result.get('phase_progress', {})
    phases = progress.get('phases')
    if (progress.get('expected') != len(prompts) or progress.get('completed') != len(prompts)
            or progress.get('passed') is not True or not isinstance(phases, list)
            or len(phases) != len(prompts)):
        raise ValueError('OpenCode app phase coverage is incomplete')
    sessions = set()
    exchanges = 0
    for index, (prompt, phase) in enumerate(zip(prompts, phases)):
        prompt_digest = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        evidence = phase.get('prompt', {})
        count, session = phase.get('exchanges'), phase.get('session_id')
        if (phase.get('phase') != index or phase.get('returncode') != 0
                or phase.get('finished') is not True or phase.get('interrupted_by_harness') is not False
                or type(count) is not int or count <= 0 or phase.get('first_exchange') != exchanges
                or not positive_seconds(phase.get('elapsed_seconds'))
                or not isinstance(session, str) or not session
                or evidence.get('passed') is not True or evidence.get('error') is not None
                or evidence.get('expected_sha256') != prompt_digest
                or evidence.get('observed_sha256') != prompt_digest):
            raise ValueError(f'OpenCode phase {index} lacks an uninterrupted authenticated prompt/response')
        sessions.add(session)
        exchanges += count
    if len(sessions) != 1 or result.get('request_count') != exchanges:
        raise ValueError('OpenCode phases do not cover one continuous conversation')
    calls, failures = result.get('tool_calls'), result.get('execution_errors')
    if not isinstance(calls, list) or not isinstance(failures, list):
        raise ValueError('OpenCode session omitted authenticated tool outcomes')
    call_ids = [call.get('id') for call in calls]
    failure_ids = [failure.get('call_id') for failure in failures]
    if (any(not isinstance(identity, str) or not identity for identity in call_ids + failure_ids)
            or len(set(call_ids)) != len(call_ids) or not set(failure_ids) <= set(call_ids)
            or result.get('tool_attempts') != len(call_ids)
            or result.get('tool_errors') != len(set(failure_ids))):
        raise ValueError('OpenCode tool counts disagree with their unique emitted call identities')
    summary = campaign_assessment(results, ['webapp'], 1, 'stress')
    if any(report.get(key) != value for key, value in summary.items()):
        raise ValueError('OpenCode campaign verdict differs from its complete session evidence')
    if not summary['gate_passed']:
        raise ValueError('OpenCode engine/protocol, phase, coverage or tool-error gate failed')
    return {**summary, 'request_count': exchanges, 'completed_phases': len(phases),
            'elapsed_seconds': result['elapsed_seconds'], 'model': model,
            'context_tokens': context_tokens, 'output_tokens': output_tokens,
            'opencode_version': VERSION}


def validate_native_coverage(native: dict, app: dict) -> None:
    """Require the independent native audit to cover every app response and call."""
    if (native.get('complete') is not True or native.get('passed') is not True
            or native.get('joined_responses') != app['request_count']
            or native.get('verified_native_calls') != app['stress_policy']['tool_attempts']
            or native.get('changed_responses') != 0 or native.get('changes') != []):
        raise ValueError('OpenCode native audit is incomplete or disagrees with the app workload')
