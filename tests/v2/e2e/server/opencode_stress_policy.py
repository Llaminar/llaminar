#!/usr/bin/env python3
"""Release stress policy separating rare model mistakes from engine defects.

The user-authorized tolerance is strictly below five percent of unique emitted
tool calls, measured both across the cell and per completed coding session.
Unknown, malformed, changed or incomplete protocol evidence is never excused by
that allowance. Phase completion is structural; elapsed generation time alone
is not a stall detector for legitimate long reasoning turns.
"""
from __future__ import annotations


def app_archive_compaction_assessment(records: list[dict]) -> dict:
    """Check any exercised archive maintenance without requiring unrelated churn.

    The coding workload is not obligated to cross the archive's compaction
    threshold. Its cache occupancy and ownership have separate mandatory checks;
    focused storage regressions exercise compaction and prove zero payload reads.
    Absence here is reported explicitly as unexercised, never as compaction proof.
    """
    rows = [row for row in records if row.get('domain') == 'prefix_archive'
            and row.get('name') == 'background_compactions']
    errors = []
    completed = 0
    for row in rows:
        value, tags = row.get('value'), row.get('tags', {})
        if (row.get('kind') != 'counter' or type(value) not in (float, int)
                or not 0 < value < 2**53 or int(value) != value
                or not isinstance(tags, dict) or tags.get('payload_bytes_copied') != '0'
                or tags.get('authority') != 'archive_worker'):
            errors.append('Malformed or payload-copying archive compaction')
        else:
            completed += int(value)
    return {'passed': not errors, 'exercised': bool(rows), 'completed_compactions': completed,
            'activity_required': False, 'errors': errors}


def stress_assessment(results: list[dict]) -> dict:
    """Require completed phases and a strict rational tool-error-rate threshold."""
    attempted = errors = 0
    phases_complete = bool(results)
    coding_sessions = []
    for result in results:
        count = result.get('tool_attempts')
        failed = result.get('tool_errors')
        if type(count) is not int or type(failed) is not int or not 0 <= failed <= count:
            raise ValueError('Stress policy requires authenticated tool outcome counts')
        attempted += count
        errors += failed
        phase = result.get('phase_progress', {})
        expected = phase.get('expected')
        complete = phase.get('completed')
        if (type(expected) is not int or expected <= 0 or type(complete) is not int
                or complete != expected or phase.get('passed') is not True):
            phases_complete = False
        if result.get('case') == 'webapp':
            coding_sessions.append({'ordinal': result['ordinal'], 'attempts': count, 'errors': failed,
                                    'passed': count > 0 and failed * 20 < count})
    error_rate_passed = attempted > 0 and errors * 20 < attempted and all(r['passed'] for r in coding_sessions)
    protocol_passed = bool(results) and all(row.get('protocol_passed') is True for row in results)
    return {'passed': protocol_passed and phases_complete and error_rate_passed,
            'protocol_passed': protocol_passed, 'phases_complete': phases_complete,
            'tool_attempts': attempted, 'tool_errors': errors,
            'tool_error_rate': errors / attempted if attempted else None,
            'tool_error_rate_limit': 0.05, 'limit_comparison': 'strictly_less_than',
            'tool_error_rate_passed': error_rate_passed, 'coding_sessions': coding_sessions}
