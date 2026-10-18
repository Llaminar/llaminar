#!/usr/bin/env python3
"""Device-free faults in complete native-token/wire joins and Unicode channels.

Fixtures keep native trace timing, token counts and HTTP/SSE channels independent.
The command test runs under Python optimization to prove qualification does not
depend on assert statements. Real tokenizer/parser replay is a separate native
metadata diagnostic; these tests never load models or initialize accelerators.
"""
from copy import deepcopy
import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests/v2/e2e/server'))
from opencode_native_audit import compare_decoded, join_responses, native_groups, thinking_override, wire_exchange

START = datetime.datetime(2026, 10, 7, tzinfo=datetime.timezone.utc).timestamp()
TOOLS = [{'type': 'function', 'function': {'name': 'read', 'parameters': {
    'type': 'object', 'required': ['filePath'], 'properties': {'filePath': {'type': 'string'}}}}}]
CALL = {'name': 'read', 'arguments': {'filePath': '🙂-👩🏽‍💻-🇬🇧-❤️-1️⃣-中文.txt'}}
NATIVE = '<tool_call><function=read><parameter=filePath>' + CALL['arguments']['filePath'] + '</parameter></function></tool_call>'


def trace(second=1):
    """Two native committed tokens, only the first of which becomes model text."""
    return [f'[00:00:{second:02d}.100] [ChatCompletion/token] path=stream index=0 token=10 forced=false disposition=text',
            f'[00:00:{second:02d}.101] [ChatCompletion/token] path=stream index=1 token=20 forced=false disposition=stop_token',
            'Prefix cache summary (stream): requested_tokens=100 matched_tokens=80']


def exchange(*, second=1, tools=True):
    """An exact complete SDK stream with either a tool call or plain compaction text."""
    delta = {'reasoning_content': 'Plan 🙂 👩🏽‍💻'}
    if tools:
        delta['tool_calls'] = [{'index': 0, 'function': {'name': CALL['name'], 'arguments': json.dumps(CALL['arguments'])}}]
    else:
        delta['content'] = 'Summary 🇬🇧 ❤️ 1️⃣ 中文 λ'
    def event(choices, usage):
        return {'id': 'response-' + str(second), 'model': 'fixture', 'created': 1, 'choices': choices, 'usage': usage}
    events = [event([{'delta': delta, 'finish_reason': None}], None),
              event([{'delta': {}, 'finish_reason': 'tool_calls' if tools else 'stop'}], None),
              event([], {'prompt_tokens': 100, 'completion_tokens': 2, 'total_tokens': 102})]
    request = {'stream': True, 'stream_options': {'include_usage': True}}
    if tools:
        request['tools'] = TOOLS
    return {'started': START + second, 'elapsed_seconds': .2, 'request': request,
        'response': {'status': 200, 'body': ''.join('data: ' + json.dumps(row) + '\n\n' for row in events) + 'data: [DONE]\n\n'}}


def replay(identity='app/0', *, tools=True):
    """Independent decoded bytes; escaping is exercised by the real JSON reader."""
    content = '' if tools else 'Summary 🇬🇧 ❤️ 1️⃣ 中文 λ'
    return {'id': identity, 'tool_format': 'qwen_3_xml', 'content': content, 'reasoning': 'Plan 🙂 👩🏽‍💻',
            'native_content': NATIVE if tools else content, 'calls': [CALL] if tools else []}


class NativeAuditTests(unittest.TestCase):
    """Reject plausible but unauthenticated evidence in every independent layer."""

    def joined(self):
        return join_responses(native_groups(trace(), START), [wire_exchange('app/0', exchange())])[0]

    def test_exact_emoji_channels_and_independent_arguments_pass(self):
        result = compare_decoded(self.joined(), [replay()])
        self.assertTrue(result['passed'])
        self.assertEqual(result['verified_native_calls'], 1)

    def test_compaction_without_tools_remains_in_the_complete_join(self):
        groups = native_groups(trace() + trace(2), START)
        wires = [wire_exchange('app/0', exchange()), wire_exchange('app/1', exchange(second=2, tools=False))]
        joined, records = join_responses(groups, wires)
        self.assertEqual(records[1]['tools'], [])
        result = compare_decoded(joined, [replay(), replay('app/1', tools=False)])
        self.assertTrue(result['passed'])
        self.assertEqual(result['joined_responses'], 2)

    def test_plain_content_reasoning_and_arguments_are_separately_checked(self):
        for field in ('content', 'reasoning', 'calls', 'native_content'):
            decoded = replay()
            if field == 'calls':
                decoded[field] = []
            else:
                decoded[field] = 'corrupted 🧑‍💻'
            with self.subTest(field=field):
                self.assertFalse(compare_decoded(self.joined(), [decoded])['passed'])

    def test_unknown_native_format_cannot_borrow_a_parser_replay_pass(self):
        value = replay()
        value['tool_format'] = 'generic'
        with self.assertRaisesRegex(ValueError, 'grammar'):
            compare_decoded(self.joined(), [value])

    def test_missing_extra_and_reordered_decoder_records_fail(self):
        for values in ([], [replay(), replay()], [replay('neighbor')]):
            with self.assertRaises(ValueError):
                compare_decoded(self.joined(), values)

    def test_native_index_gaps_duplicates_path_and_unknown_disposition_fail(self):
        for old, new in [('index=1', 'index=2'), ('index=1', 'index=0'), ('path=stream', 'path=unknown'),
                         ('forced=false', 'forced=true'), ('disposition=text', 'disposition=unknown'),
                         ('token=10', 'token=-1'), ('token=10', 'token=2147483648')]:
            lines = [line.replace(old, new) for line in trace()]
            with self.subTest(new=new), self.assertRaises(ValueError):
                native_groups(lines, START)

    def test_empty_unfinished_missing_initial_and_malformed_traces_fail(self):
        for lines in ([], trace()[:-1], trace()[1:], trace() + [trace()[0]], ['[ChatCompletion/token] truncated']):
            with self.assertRaises(ValueError):
                native_groups(lines, START)

    def test_text_or_a_second_terminal_token_cannot_follow_generation_end(self):
        for disposition in ('text', 'stop_token', 'runner_complete'):
            lines = trace()
            lines.insert(2, '[00:00:01.102] [ChatCompletion/token] path=stream index=2 token=10 forced=false disposition=' + disposition)
            with self.subTest(disposition=disposition), self.assertRaisesRegex(ValueError, 'after its terminal'):
                native_groups(lines, START)

    def test_midnight_rollover_preserves_time_and_small_clock_reversal_fails(self):
        lines = trace()
        lines[0] = lines[0].replace('00:00:01.100', '23:59:59.999')
        lines[1] = lines[1].replace('00:00:01.101', '00:00:00.000')
        result = native_groups(lines, START)
        self.assertAlmostEqual(result[0]['last'] - result[0]['first'], .001, places=5)
        lines = trace()
        lines[1] = lines[1].replace('00:00:01.101', '00:00:01.099')
        with self.assertRaisesRegex(ValueError, 'backwards'):
            native_groups(lines, START)

    def test_counts_timing_and_unrecorded_controls_are_independent_failures(self):
        groups = native_groups(trace(), START)
        wire = wire_exchange('app/0', exchange())
        for field, value in [('prompt_tokens', 101), ('completion_tokens', 3)]:
            bad = deepcopy(wire)
            bad['usage'][field] = value
            with self.assertRaisesRegex(ValueError, 'got 0'):
                join_responses(groups, [bad])
        bad = deepcopy(wire)
        bad['start'] += 1
        with self.assertRaisesRegex(ValueError, 'got 0'):
            join_responses(groups, [bad])
        with self.assertRaisesRegex(ValueError, 'Unaccounted'):
            join_responses(native_groups(trace() + trace(2), START), [wire])

    def test_ambiguous_neighbor_and_duplicate_wire_identity_fail(self):
        groups = native_groups(trace(), START)
        wire = wire_exchange('app/0', exchange())
        with self.assertRaisesRegex(ValueError, 'got 2'):
            join_responses(groups + deepcopy(groups), [wire])
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            join_responses(groups, [wire, wire])

    def test_missing_usage_and_failed_http_are_never_silently_skipped(self):
        for field in ('http', 'usage', 'stream', 'done'):
            value = exchange()
            if field == 'http':
                value['response']['status'] = 500
            elif field == 'usage':
                value['request']['stream_options'] = {}
            elif field == 'stream':
                value['request']['stream'] = False
            else:
                value['response']['body'] = value['response']['body'].replace('data: [DONE]', '')
            with self.subTest(field=field), self.assertRaises(ValueError):
                wire_exchange('app/0', value)

    def test_top_level_and_nested_thinking_controls_must_agree(self):
        self.assertIsNone(thinking_override({}))
        self.assertFalse(thinking_override({'chat_template_kwargs': {'enable_thinking': False}}))
        self.assertTrue(thinking_override({'enable_thinking': True, 'chat_template_kwargs': {'enable_thinking': True}}))
        for value in ({'enable_thinking': 1}, {'chat_template_kwargs': []},
                      {'enable_thinking': True, 'chat_template_kwargs': {'enable_thinking': False}}):
            with self.assertRaises(ValueError):
                thinking_override(value)

    def test_optimized_python_still_rejects_missing_native_tokens_and_retains_red(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / 'app/session-0'
            session.mkdir(parents=True)
            (session / 'exchange-0.json').write_text(json.dumps(exchange()))
            (root / 'server.log').write_text('\n'.join(trace()[1:]) + '\n')
            result = subprocess.run([sys.executable, '-O', str(ROOT / 'tests/v2/e2e/server/opencode_native_audit.py'),
                '--exchanges', str(root / 'app'), '--server-log', str(root / 'server.log'),
                '--server-start-unix', str(START), '--decoder', str(root / 'must-not-run'),
                '--model', str(root / 'fixture.gguf'), '--output', str(root / 'result')],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads((root / 'result/result.json').read_text())
            self.assertFalse(report['passed'])
            self.assertIn('index/path', report['error'])


if __name__ == '__main__':
    unittest.main()
