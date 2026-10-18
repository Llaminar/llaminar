#!/usr/bin/env python3
"""Exercise failed driver-observer closure with real device-free child processes.

The synthetic observer publishes a ready handshake and then waits. Control
delivery and its small lifecycle deadline are injected independently, so a
failed close must retire its exact process before publishing closed state.
No kernel log is read and no running inference process is inspected or altered.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests/v2/e2e/server'))
import gpu_driver_diagnostics as driver


class DriverObserverRetirementTests(unittest.TestCase):
    """Require physical retirement even when the observer cannot close normally."""

    def begin_worker(self, root, ignore_term):
        """Publish readiness only after the independent process installs its signal policy."""
        script = '''import signal,sys,time
if sys.argv[2] == 'ignore': signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(sys.argv[1], 'w') as stream: stream.write('ready')
while True: time.sleep(1)
'''
        ready = root / 'ready'
        worker = subprocess.Popen([sys.executable, '-c', script, str(ready),
                                   'ignore' if ignore_term else 'normal'])

        def cleanup():
            """Keep a failing regression incapable of leaving its fixture alive."""
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=5)

        self.addCleanup(cleanup)
        deadline = time.monotonic() + 5
        while not ready.exists():
            if worker.poll() is not None or time.monotonic() >= deadline:
                self.fail('Fixture worker failed its independent ready handshake')
            time.sleep(.001)
        identity = driver.process_identity(worker.pid)
        self.assertIsNotNone(identity)
        state = root / 'cell.driver-state.json'
        driver.publish(state, {'schema': 2, 'state': 'armed', 'window_id': 'd' * 32,
                               'collector': identity})
        return worker, state, identity

    def test_failed_close_retires_exact_worker_before_publishing_closed(self):
        """Both undelivered control and an unresponsive final read remain failed and retire."""
        for delivery_failure, ignore_term in ((False, False), (False, True), (True, False)):
            with self.subTest(delivery_failure=delivery_failure, ignore_term=ignore_term), \
                    tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                worker, state, identity = self.begin_worker(root, ignore_term)
                report = root / 'cell.driver-diagnostics.json'
                with patch.object(driver, '_CONTROL_SECONDS', .03), \
                        patch.object(driver.socket, 'socket') as socket:
                    if delivery_failure:
                        socket.return_value.__enter__.return_value.connect.side_effect = OSError('control unavailable')
                    self.assertFalse(driver.finish(state, report))
                evidence = json.loads(report.read_text())
                self.assertFalse(evidence['complete'])
                self.assertIn('control unavailable' if delivery_failure else 'did not retire', evidence['error'])
                self.assertFalse(driver.process_is_live(identity), 'Closed receipt left its observer process alive')
                worker.wait(timeout=1)
                self.assertEqual(json.loads(state.read_text())['state'], 'closed')

    def test_cleanup_failure_keeps_live_admission_and_original_error(self):
        """An unavailable cleanup operation must not manufacture a closed lifetime."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker, state, identity = self.begin_worker(root, False)
            report = root / 'cell.driver-diagnostics.json'
            with patch.object(driver, '_CONTROL_SECONDS', .03), \
                    patch.object(driver.socket, 'socket'), \
                    patch.object(driver, 'retire_failed_collector', side_effect=PermissionError('fixture denied')):
                self.assertFalse(driver.finish(state, report))
            self.assertTrue(driver.process_is_live(identity))
            self.assertEqual(json.loads(state.read_text())['state'], 'armed')
            evidence = json.loads(report.read_text())
            self.assertIn('did not retire', evidence['error'])
            self.assertIn('observer retirement failed: fixture denied', evidence['error'])
            self.assertFalse(evidence['complete'])

    def test_pid_reuse_during_handle_open_never_receives_a_signal(self):
        """Start ticks are checked again after a native handle has been obtained."""
        identity = {'pid': 99999, 'start_ticks': 123}
        with patch.object(driver, 'process_is_live', side_effect=[True, False]), \
                patch.object(driver.os, 'pidfd_open', return_value=17) as opened, \
                patch.object(driver.os, 'close') as closed, \
                patch.object(driver.signal, 'pidfd_send_signal') as sent:
            driver.retire_failed_collector(identity)
        opened.assert_called_once_with(identity['pid'])
        closed.assert_called_once_with(17)
        sent.assert_not_called()

    def test_dead_owner_needs_no_native_handle(self):
        """Already retired or reused native identities cannot select a new process."""
        with patch.object(driver, 'process_is_live', return_value=False), \
                patch.object(driver.os, 'pidfd_open') as opened:
            driver.retire_failed_collector({'pid': 99999, 'start_ticks': 123})
        opened.assert_not_called()

    def test_invalid_exit_handle_is_not_retirement_evidence(self):
        """Only native process exit events can prove the failed owner is gone."""
        with patch.object(driver, 'process_is_live', return_value=True), \
                patch.object(driver.os, 'pidfd_open', return_value=17), \
                patch.object(driver.os, 'close') as closed, \
                patch.object(driver.signal, 'pidfd_send_signal'), \
                patch.object(driver.select, 'poll') as poll:
            poll.return_value.poll.return_value = [(17, driver.select.POLLNVAL)]
            with self.assertRaisesRegex(RuntimeError, 'lost its pidfd'):
                driver.retire_failed_collector({'pid': 99999, 'start_ticks': 123})
        closed.assert_called_once_with(17)


if __name__ == '__main__':
    unittest.main()
