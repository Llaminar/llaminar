#!/usr/bin/env python3
"""Device-free proof of exact request joins and honest prefix-cache timing reports.

Fixtures model completed HTTP counters independently of the reporter. Tests
reject stale, reset and competing requests, check weighted reuse and retain
background traffic that finishes while the coding client executes tools.
"""
from copy import deepcopy
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'e2e/server'))
from prefix_reuse_measurement import idle_snapshot, measure_request, summarize
from opencode_tool_stress import recording_proxy


def snapshot(requests: int, *, matched: int = 4096, writes: int = 0) -> dict:
    """Return explicit native-style metadata for a sequential successful request."""
    return {'object': 'llaminar.stats', 'schema_version': 1, 'epoch': 3,
        'requests': {'started': requests, 'active': 0, 'completed': requests, 'failed': 0, 'disconnected': 0},
        'tokens': {'prompt': requests * 20000, 'completion': requests * 123},
        'last_request': None if requests == 0 else {'sequence': requests + 10,
            'handler_status': 200, 'outcome': 'completed', 'prompt_tokens': 20000, 'completion_tokens': 123,
            'prefix_cache': {'requested_tokens': 20000, 'matched_tokens': matched},
            'throughput': {'prefill_uncached_tokens': 20000 - matched},
            'timings': {'ttft_seconds': 8., 'prefill_seconds': 7.}},
        'prefix_cache': {'storage': {'churn': {'scope': 'committed_operations_since_last_reset',
                'disk_payload_writes': {'operations': writes, 'bytes': writes * 256}},
            'tiers': {'ram': {'enabled': True, 'capacity_bytes': 4096, 'used_bytes': 2048},
                      'disk': {'enabled': True, 'capacity_bytes': 32768, 'used_bytes': 8192}}}}}


class PrefixReuseMeasurementTests(unittest.TestCase):
    """Authenticate every source and distinguish work reduction from whole-session averages."""
    def measure(self, before=None, after=None):
        """Supply the independently known terminal wire counts."""
        return measure_request(before or snapshot(0), after or snapshot(1),
                               {'prompt_tokens': 20000, 'completion_tokens': 123})

    def test_exact_wire_join_preserves_request_and_bounded_tiers(self):
        """A post-reset absolute sequence is legal when exact epoch-local counts prove one request."""
        row = self.measure(after=snapshot(1, writes=2))
        self.assertEqual(row['sequence'], 11)
        self.assertEqual(row['request']['prefix_cache']['matched_tokens'], 4096)
        self.assertEqual(row['traffic']['disk_payload_writes'], {'operations': 2, 'bytes': 512})
        self.assertEqual(row['tiers']['ram']['after_bytes'], 2048)

    def test_stale_reset_failed_competing_and_incomplete_requests_fail(self):
        """Similar prompt lengths cannot rescue a broken lifecycle or terminal identity."""
        for mutate in (
            lambda s: s.update(epoch=4),
            lambda s: s['requests'].update(active=1),
            lambda s: s['requests'].update(started=3),
            lambda s: s['requests'].update(started=3, completed=3),
            lambda s: s['requests'].update(completed=0, failed=1),
            lambda s: s['last_request'].update(sequence=9),
            lambda s: s['last_request'].update(prompt_tokens=19999),
            lambda s: s['last_request'].update(completion_tokens=122),
            lambda s: s['tokens'].update(prompt=123),
            lambda s: s['last_request'].update(outcome='disconnected'),
        ):
            with self.subTest(mutation=mutate):
                after = snapshot(2)
                mutate(after)
                with self.assertRaises(ValueError):
                    self.measure(snapshot(1), after)
        with self.assertRaises(ValueError):
            self.measure(snapshot(1), snapshot(1))

    def test_invalid_timing_geometry_and_counter_values_fail(self):
        """NaN, negative work, drifting capacities and over-budget occupancy are errors."""
        for mutate in (
            lambda s: s['requests'].update(started=True),
            lambda s: s['last_request']['timings'].update(ttft_seconds=float('nan')),
            lambda s: s['last_request']['timings'].update(prefill_seconds=None),
            lambda s: s['last_request']['prefix_cache'].update(matched_tokens=20001),
            lambda s: s['last_request']['prefix_cache'].update(matched_tokens=-1),
            lambda s: s['last_request']['throughput'].update(prefill_uncached_tokens=20000),
            lambda s: s['prefix_cache']['storage']['tiers']['ram'].update(used_bytes=4097),
            lambda s: s['prefix_cache']['storage']['tiers']['disk'].update(capacity_bytes=65536),
            lambda s: s['prefix_cache']['storage']['churn']['disk_payload_writes'].update(bytes=-1),
        ):
            with self.subTest(mutation=mutate):
                after = snapshot(1)
                mutate(after)
                with self.assertRaises(ValueError):
                    self.measure(after=after)

    def test_disabled_tier_and_idle_boundary_require_real_zero_occupancy(self):
        """Disabled tiers stay visible without inventing a utilization percentage."""
        before, after = snapshot(0), snapshot(1)
        for value in (before, after):
            value['prefix_cache']['storage']['tiers']['disk'].update(enabled=False, capacity_bytes=0, used_bytes=0)
        self.assertFalse(self.measure(before, after)['tiers']['disk']['enabled'])
        for change in ({'active': 1}, {'started': 1}, {'failed': -1}):
            value = snapshot(0)
            value['requests'].update(change)
            with self.assertRaises(ValueError):
                idle_snapshot(value)

    def test_summary_separates_compaction_and_includes_between_turn_write_traffic(self):
        """Ratio-of-sums and native TTFT expose both reuse benefits and cold-compaction cost."""
        rows = []
        for index, (kind, matched, ttft) in enumerate((('ordinary', 19000, 1.),
                ('compaction', 0, 100.), ('post_compaction', 4096, 8.))):
            before, after = snapshot(index, writes=index * 4), snapshot(index + 1, matched=matched, writes=index * 4 + 1)
            after['last_request']['timings']['ttft_seconds'] = ttft
            rows.append({**self.measure(before, after), 'kind': kind})
        report = summarize(rows)
        self.assertEqual(report['groups']['all']['restored_tokens'], 23096)
        self.assertAlmostEqual(report['groups']['all']['token_reuse_rate'], 23096 / 60000)
        self.assertEqual(report['groups']['all']['ttft_p95_seconds'], 100.)
        self.assertEqual(report['groups']['ordinary']['ttft_mean_seconds'], 1.)
        self.assertEqual(report['groups']['post_compaction']['restored_tokens'], 4096)
        self.assertEqual(report['traffic']['disk_payload_writes']['operations'], 9)
        self.assertEqual(report['tiers']['ram']['boundary_peak_bytes'], 2048)

    def test_summary_rejects_gaps_resets_and_counter_or_capacity_drift(self):
        """An incomplete report cannot look like a successful reuse improvement."""
        rows = [{**self.measure(snapshot(i), snapshot(i + 1)), 'kind': 'ordinary'} for i in range(2)]
        for mutate in (
            lambda r: r[1].update(sequence=14),
            lambda r: r[1].update(epoch=4),
            lambda r: r[1].update(kind='unknown'),
            lambda r: r[1]['tiers']['ram'].update(capacity_bytes=8192),
            lambda r: r[0]['traffic_after']['disk_payload_writes'].update(bytes=1),
        ):
            changed = deepcopy(rows)
            mutate(changed)
            with self.assertRaises(ValueError):
                summarize(changed)
        with self.assertRaises(ValueError):
            summarize([])
        self.assertIsNone(summarize(rows)['groups']['compaction']['ttft_mean_seconds'])

    def test_http_recorder_preserves_stream_and_joins_all_compaction_phases(self):
        """Real HTTP/SSE relay measurements leave client bodies intact and reject stale stats."""
        for stale in (False, True):
            with self.subTest(stale=stale):
                received, stats_reads, completed = [], [], [0]
                identity = {'id': 'response', 'model': 'model', 'created': 1}
                chunks = [
                    {**identity, 'choices': [{'index': 0, 'delta': {'content': '🙂'}, 'finish_reason': None}], 'usage': None},
                    {**identity, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': None},
                    {**identity, 'choices': [], 'usage': {'prompt_tokens': 20000, 'completion_tokens': 123, 'total_tokens': 20123}},
                ]
                wire = (''.join('data: ' + json.dumps(row, ensure_ascii=False) + '\n\n' for row in chunks)
                        + 'data: [DONE]\n\n').encode()

                class Endpoint(BaseHTTPRequestHandler):
                    """Native-style terminal stats are published independently of the relay."""
                    def do_GET(self):
                        """Supply small stats metadata; GET never changes generation counters."""
                        stats_reads.append(self.path)
                        body = json.dumps(snapshot(0 if stale else completed[0])).encode()
                        self.send_response(200)
                        self.send_header('Content-Length', str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)

                    def do_POST(self):
                        """Return the same fragmented stream for each unmodified request."""
                        received.append(self.rfile.read(int(self.headers['Content-Length'])))
                        completed[0] += 1
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/event-stream')
                        self.send_header('Content-Length', str(len(wire)))
                        self.end_headers()
                        for fragment in (wire[:17], wire[17:]):
                            self.wfile.write(fragment)
                            self.wfile.flush()

                    def log_message(self, *_):
                        """Keep fixture HTTP logs out of regression output."""

                with ThreadingHTTPServer(('127.0.0.1', 0), Endpoint) as server:
                    worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
                    worker.start()
                    try:
                        with tempfile.TemporaryDirectory() as temporary:
                            directory = Path(temporary)
                            ordinary = {'stream': True, 'stream_options': {'include_usage': True},
                                        'messages': [{'role': 'user', 'content': 'Build a Python app 🙂'}]}
                            compaction = {**ordinary, 'messages': [
                                {'role': 'system', 'content': 'You are a context summarization agent.'},
                                {'role': 'user', 'content': 'Summarize\n<conversation>\nwork\n</conversation>\n'}]}
                            requests = [ordinary] if stale else [ordinary, compaction, ordinary]
                            bodies = [json.dumps(row, ensure_ascii=False).encode() for row in requests]
                            with recording_proxy(f'http://127.0.0.1:{server.server_port}', directory,
                                                 measure_prefix_reuse=True) as (url, rows):
                                target = urlsplit(url)
                                for body in bodies:
                                    connection = http.client.HTTPConnection(target.hostname, target.port, timeout=2)
                                    connection.request('POST', '/v1/chat/completions', body)
                                    self.assertEqual(connection.getresponse().read(), wire)
                                    connection.close()
                            self.assertEqual(received, bodies)
                            self.assertEqual(stats_reads, ['/stats'] * (2 * len(bodies)))
                            if stale:
                                self.assertIn('competing requests', rows[0]['error'])
                                self.assertNotIn('prefix_measurement', rows[0])
                            else:
                                measurements = [row['prefix_measurement'] for row in rows]
                                self.assertEqual([row['kind'] for row in measurements],
                                                 ['ordinary', 'compaction', 'post_compaction'])
                                self.assertEqual(summarize(measurements)['requests'], 3)
                    finally:
                        server.shutdown()
                        worker.join()


if __name__ == '__main__':
    unittest.main()
