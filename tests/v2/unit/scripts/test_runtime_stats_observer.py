#!/usr/bin/env python3
"""Device-free regressions for stats observation failure and continuation.

Controlled readers prove that an observation timeout cannot cancel a workload,
erase its failed poll, or discard subsequent snapshots. Invalid counters still
fail immediately, and workload exceptions retain their original ownership.
The tests do not load a model or relax the endpoint's accuracy requirements.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests/v2/e2e/server'))
from runtime_stats_observer import RuntimeStatsObserver


def snapshot(operations=1, *, epoch=0, active=1):
    """Make a minimal reader-validated snapshot with independently chosen churn."""
    return {'epoch': epoch, 'requests': {'active': active}, 'prefix_cache': {'storage': {
        'churn': {'scope': 'committed_operations_since_last_reset',
                  'ram_to_disk': {'operations': operations, 'bytes': operations * 32}}
    }}}


class RuntimeStatsObserverTests(unittest.TestCase):
    """Exercise observer lifecycle boundaries without wall-clock generation limits."""

    def setUp(self):
        """Give each observation an exclusively owned temporary journal."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'observations.jsonl'

    def read_json(self, suffix):
        """Read a completed report from the observation's retained sidecar."""
        return json.loads(self.output.with_suffix(suffix).read_text())

    def rows(self, suffix='.jsonl'):
        """Read flushed evidence without interpreting failed polls as snapshots."""
        return [json.loads(line) for line in self.output.with_suffix(suffix).read_text().splitlines()]

    def run_sequence(self, values, *, expected_failure=False, before_read=None):
        """Stop after the last scripted read while allowing the real thread to publish it."""
        pending = iter(enumerate(values))
        calls = []

        def read():
            index, value = next(pending)
            calls.append(index)
            if before_read:
                before_read(index)
            if index == len(values) - 1:
                observer.stopped.set()
            if isinstance(value, BaseException):
                raise value
            return value

        observer = RuntimeStatsObserver(read, self.output, interval_seconds=0)
        try:
            with observer:
                # A fatal validation error may stop before the final scripted
                # value. Join this observer rather than waiting for generation.
                observer.thread.join(2)
                self.assertFalse(observer.thread.is_alive())
        except AssertionError:
            if not expected_failure:
                raise
        else:
            self.assertFalse(expected_failure, 'Observer incorrectly certified failed evidence')
        return self.read_json('.result.json'), calls

    def test_timeout_is_journaled_before_next_poll_and_never_becomes_a_pass(self):
        """The original failure remains red even after a valid observation resumes."""
        def before(index):
            if index == 2:
                errors = self.rows('.errors.jsonl')
                self.assertEqual(len(errors), 1)
                self.assertEqual(errors[0]['attempt'], 2)
                self.assertEqual(errors[0]['kind'], 'transport')

        report, calls = self.run_sequence(
            [snapshot(), TimeoutError('stats read expired'), snapshot(2)],
            expected_failure=True, before_read=before)
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(report['snapshots'], 2)
        self.assertEqual(report['while_active'], 2)
        self.assertEqual(report['failed_polls'], 1)
        self.assertIn('stats read expired', report['error'])
        self.assertFalse(report['passed'])
        self.assertEqual(len(self.rows()), 2)

    def test_repeated_network_failures_preserve_each_attempt_and_original_error(self):
        """An outage has a complete failure journal and cannot fabricate zero counters."""
        report, calls = self.run_sequence([
            TimeoutError('first failure'), URLError('connection reset'), snapshot(3)],
            expected_failure=True)
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(report['failed_polls'], 2)
        self.assertEqual(report['snapshots'], 1)
        self.assertIn('first failure', report['error'])
        self.assertEqual([row['attempt'] for row in self.rows('.errors.jsonl')], [1, 2])

    def test_live_socket_timeout_does_not_end_later_observation(self):
        """A stalled HTTP response is recorded before the same endpoint is observed again."""
        release_first = threading.Event()
        paths = []

        class Peer(BaseHTTPRequestHandler):
            """Hold the first response until a later independent poll has completed."""

            def do_GET(self):
                """Return a real public-style snapshot after the controlled first stall."""
                paths.append(self.path)
                if len(paths) == 1:
                    release_first.wait(2)
                    return
                body = json.dumps(snapshot(2)).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                """Keep the fixture's access records in the asserted path list."""

        server = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
        worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01))
        worker.start()
        endpoint = f'http://127.0.0.1:{server.server_port}/stats'

        def read():
            with urlopen(endpoint, timeout=.05) as response:
                value = json.load(response)
            observer.stopped.set()
            return value

        observer = RuntimeStatsObserver(read, self.output, interval_seconds=0)
        try:
            with self.assertRaises(AssertionError):
                with observer:
                    observer.thread.join(2)
                    self.assertFalse(observer.thread.is_alive())
            report = self.read_json('.result.json')
            self.assertFalse(report['passed'])
            self.assertEqual(report['snapshots'], 1)
            self.assertEqual(report['failed_polls'], 1)
            self.assertGreaterEqual(report['maximum_attempt_seconds'], .05)
            self.assertEqual(paths, ['/stats', '/stats'])
            self.assertEqual(len(self.rows('.errors.jsonl')), 1)
        finally:
            release_first.set()
            server.shutdown()
            worker.join()
            server.server_close()

    def test_invalid_response_stops_observation_and_retains_immediate_error(self):
        """Schema failures are never retried into a passing observation."""
        report, calls = self.run_sequence([
            ValueError('invalid statistics schema'), snapshot()], expected_failure=True)
        self.assertEqual(calls, [0])
        self.assertEqual(report['snapshots'], 0)
        self.assertEqual(self.rows('.errors.jsonl')[0]['kind'], 'validation')

    def test_counter_regression_after_timeout_is_still_rejected(self):
        """A missing sample cannot erase the previous successful epoch's frontier."""
        report, calls = self.run_sequence([
            snapshot(3), TimeoutError('read failed'), snapshot(2), snapshot(4)],
            expected_failure=True)
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(report['snapshots'], 1)
        self.assertEqual(report['failed_polls'], 2)
        self.assertIn('churn regressed', self.rows('.errors.jsonl')[-1]['error'])

    def test_reset_epoch_allows_counter_reset(self):
        """PUT reset changes the counter epoch without making the observation stale."""
        report, calls = self.run_sequence([snapshot(3), snapshot(0, epoch=1), snapshot(1, epoch=1)])
        self.assertTrue(report['passed'])
        self.assertEqual(report['snapshots'], 3)
        self.assertEqual(report['failed_polls'], 0)
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(self.rows('.errors.jsonl'), [])

    def test_inactive_only_observation_cannot_certify_workload_coverage(self):
        """Healthy idle reads do not prove responsiveness during inference."""
        report, _ = self.run_sequence([snapshot(active=0)], expected_failure=True)
        self.assertFalse(report['passed'])
        self.assertEqual(report['while_active'], 0)

    def test_workload_exception_is_preserved_with_observation_failure(self):
        """Observation cleanup cannot replace the workload owner's actual exception."""
        observed = threading.Event()

        def read():
            observed.set()
            raise ValueError('bad stats')

        with self.assertRaisesRegex(RuntimeError, 'workload failed'):
            with RuntimeStatsObserver(read, self.output, interval_seconds=0):
                self.assertTrue(observed.wait(2))
                raise RuntimeError('workload failed')
        report = self.read_json('.result.json')
        self.assertIn('bad stats', report['error'])
        self.assertFalse(report['passed'])

    def test_existing_journal_rejects_before_any_read(self):
        """A resumed diagnostic may not truncate or replace an earlier observation."""
        self.output.write_text('retained evidence\n')
        calls = []
        with self.assertRaises(FileExistsError):
            with RuntimeStatsObserver(lambda: calls.append(True), self.output):
                self.fail('Existing journal was admitted')
        self.assertEqual(calls, [])
        self.assertEqual(self.output.read_text(), 'retained evidence\n')

    def test_invalid_cadence_is_rejected_before_ownership(self):
        """Non-finite and negative scheduling intervals are invalid policy."""
        for value in (-1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RuntimeStatsObserver(lambda: snapshot(), self.output, interval_seconds=value)


if __name__ == '__main__':
    unittest.main()
