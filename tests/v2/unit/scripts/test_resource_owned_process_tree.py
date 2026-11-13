#!/usr/bin/env python3
"""Replay local wrapper/engine ownership, exit races and PID reuse without GPUs.

Fixtures poison unrelated process metadata. A small real child process checks
Linux proc traversal; no inference, payload hashing or resource allocation policy
is involved. RSS stays per process and never becomes a shared-memory total.
"""
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests/v2/e2e/server'))
sys.path.insert(0, str(ROOT / 'scripts/ci'))
import resource_owned_process_tree as owned


class OwnedProcessTreeTests(unittest.TestCase):
    """Only original child links can grant a process lifetime observation."""

    def setUp(self):
        """Create an isolated proc tree with no access to host process metadata."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.proc = Path(temporary.name)

    def process(self, pid, parent, *, start=None, rss=8, children=(), state='S', flags=0):
        """Build original stat/status/children records with explicit byte units."""
        directory = self.proc / str(pid)
        directory.mkdir(exist_ok=True)
        fields = [state, str(parent), *(['0'] * 17), str(start or pid * 10), '0']
        fields[6] = str(flags)
        (directory / 'stat').write_text(f'{pid} (worker ) name) ' + ' '.join(fields))
        (directory / 'status').write_text(f'Pid:\t{pid}\nNSpid:\t{pid}\n' + ''.join(
            f'{field}:\t{rss} kB\n' for field in
            ('VmRSS', 'VmHWM', 'RssAnon', 'RssFile', 'RssShmem', 'VmSwap')))
        task = directory / 'task' / str(pid)
        task.mkdir(parents=True, exist_ok=True)
        (task / 'children').write_text(' '.join(map(str, children)))
        return directory

    def test_launcher_guardian_and_engine_have_individual_memory(self):
        """The wrapper's small RSS cannot substitute for the engine's RSS."""
        self.process(10, 1, rss=1, children=(20,))
        self.process(20, 10, rss=2, children=(30,))
        self.process(30, 20, rss=65536)
        tree = owned.OwnedProcessTree(10, proc=self.proc)
        result = tree.snapshot()
        self.assertEqual([row['pid'] for row in result['processes']], [10, 20, 30])
        self.assertEqual([row['status']['VmRSS'] for row in result['processes']],
                         [1024, 2048, 64 * 1024**2])
        self.assertEqual(result['root'], {'pid': 10, 'start_ticks': 100})
        self.assertEqual(result['memory_scope'], 'individual_processes_not_summed')
        self.assertNotIn('total_rss', result)

    def test_non_main_thread_children_are_included_and_foreign_processes_untouched(self):
        """MPI workers started by a helper thread retain the same ownership."""
        root = self.process(10, 1)
        thread = root / 'task' / '11'
        thread.mkdir(); (thread / 'children').write_text('20 20')
        self.process(20, 10)
        self.process(99, 1)
        original = Path.iterdir

        def directories(path):
            if path == self.proc or path == self.proc / '99' / 'task':
                raise AssertionError('Unrelated process enumeration')
            return original(path)

        tree = owned.OwnedProcessTree(10, proc=self.proc)
        with patch.object(Path, 'iterdir', directories):
            rows = tree.snapshot()['processes']
        self.assertEqual([row['pid'] for row in rows], [10, 20])

    def test_pid_reuse_retires_original_without_opening_replacement_memory(self):
        """A recycled child PID is not the descendant previously observed."""
        root = self.process(10, 1, children=(20,))
        self.process(20, 10)
        tree = owned.OwnedProcessTree(10, proc=self.proc)
        tree.snapshot()
        (root / 'task/10/children').write_text('')
        replacement = self.process(20, 99, start=999)
        (replacement / 'status').unlink()
        result = tree.snapshot()
        self.assertEqual([row['pid'] for row in result['processes']], [10])
        self.assertEqual(result['retired'], [{'pid': 20, 'start_ticks': 200}])
        self.assertEqual(tree.snapshot()['retired'], [])

    def test_discovered_child_remains_observable_after_parent_exit(self):
        """Retirement does not hide an already owned descendant's memory."""
        self.process(10, 1, children=(20,))
        self.process(20, 10, rss=123)
        tree = owned.OwnedProcessTree(10, proc=self.proc)
        tree.snapshot()
        shutil.rmtree(self.proc / '10')
        self.process(20, 1, rss=456)
        result = tree.snapshot()
        self.assertEqual([row['pid'] for row in result['processes']], [20])
        self.assertEqual(result['processes'][0]['status']['VmRSS'], 456 * 1024)
        self.assertEqual(result['retired'], [{'pid': 10, 'start_ticks': 100}])

    def test_stale_child_link_cannot_admit_foreign_process(self):
        """A PID listed before its exit must still have the original parent."""
        self.process(10, 1, children=(20,))
        foreign = self.process(20, 77)
        (foreign / 'status').unlink()
        tree = owned.OwnedProcessTree(10, proc=self.proc)
        self.assertEqual([row['pid'] for row in tree.snapshot()['processes']], [10])

    def test_root_reuse_is_retirement_without_following_new_children(self):
        """No new ancestry can be inferred from a replacement root process."""
        self.process(10, 1)
        tree = owned.OwnedProcessTree(10, proc=self.proc)
        root = self.process(10, 1, start=999, children=(99,))
        (root / 'status').unlink()
        result = tree.snapshot()
        self.assertEqual(result['processes'], [])
        self.assertEqual(result['retired'], [{'pid': 10, 'start_ticks': 100}])

    def test_exit_during_memory_read_is_explicit_retirement(self):
        """A missing status caused by exit cannot turn into zero RSS."""
        self.process(10, 1)
        tree = owned.OwnedProcessTree(10, proc=self.proc)

        def exit_before_read(path, pid):
            shutil.rmtree(path.parent)
            raise FileNotFoundError('process retired')

        with patch.object(owned, 'read_process_status', exit_before_read):
            result = tree.snapshot()
        self.assertEqual(result['processes'], [])
        self.assertEqual(result['retired'], [{'pid': 10, 'start_ticks': 100}])

    def test_pid_reuse_during_memory_read_discards_unbound_counters(self):
        """The trailing identity check rejects memory from a recycled PID."""
        self.process(10, 1)
        tree = owned.OwnedProcessTree(10, proc=self.proc)
        read = owned.read_process_status

        def recycled(path, pid):
            self.process(10, 99, start=999, rss=999999)
            return read(path, pid)

        with patch.object(owned, 'read_process_status', recycled):
            result = tree.snapshot()
        self.assertEqual(result['processes'], [])
        self.assertEqual(result['retired'], [{'pid': 10, 'start_ticks': 100}])

    def test_new_child_on_later_observation_and_repeated_empty_retirement(self):
        """Late launch, retirement and further samples cannot retain stale RSS."""
        self.process(10, 1)
        tree = owned.OwnedProcessTree(10, proc=self.proc)
        self.assertEqual(len(tree.snapshot()['processes']), 1)
        self.process(10, 1, children=(20,))
        self.process(20, 10, rss=123)
        self.assertEqual(len(tree.snapshot()['processes']), 2)
        shutil.rmtree(self.proc / '10'); shutil.rmtree(self.proc / '20')
        self.assertEqual(len(tree.snapshot()['retired']), 2)
        for _ in range(3):
            result = tree.snapshot()
            self.assertEqual(result['processes'], [])
            self.assertEqual(result['retired'], [])

    def test_zombie_reports_unavailable_memory_without_fabricated_zero(self):
        """An unreaped original Popen lifetime need not have resident counters."""
        root = self.process(10, 1, state='Z')
        (root / 'status').unlink()
        row = owned.OwnedProcessTree(10, proc=self.proc).snapshot()['processes'][0]
        self.assertFalse(row['memory_available'])
        self.assertNotIn('status', row)

    def test_exiting_mm_without_zombie_state_never_fabricates_memory(self):
        """PF_EXITING can precede both exit_mm memory removal and zombie state."""
        for state in ('R', 'S', 'D'):
            with self.subTest(state=state):
                root = self.process(10, 1, state=state, flags=0x00000004)
                (root / 'status').write_text('Pid:\t10\nNSpid:\t10\n')
                tree = owned.OwnedProcessTree(10, proc=self.proc)
                with patch.object(owned, 'read_process_status', side_effect=AssertionError('exiting mm read')):
                    result = tree.snapshot()
                row = result['processes'][0]
                self.assertFalse(row['memory_available'])
                self.assertEqual(row['memory_unavailable_reason'], 'observed_process_exit')
                self.assertNotIn('status', row)
                self.assertEqual(result['retired'], [])
                shutil.rmtree(root)
                self.assertEqual(tree.snapshot()['retired'], [{'pid': 10, 'start_ticks': 100}])

    def test_exit_mm_during_status_read_requires_original_exit_flag(self):
        """A racing missing mm is tolerated only with an actual trailing exit witness."""
        root = self.process(10, 1)
        tree = owned.OwnedProcessTree(10, proc=self.proc)
        read = owned.read_process_status
        def release_mm(path, pid):
            self.process(10, 1, state='R', flags=0x00000004)
            path.write_text('Pid:\t10\nNSpid:\t10\n')
            return read(path, pid)
        with patch.object(owned, 'read_process_status', release_mm):
            row = tree.snapshot()['processes'][0]
        self.assertEqual(row['state'], 'R')
        self.assertFalse(row['memory_available'])
        self.assertNotIn('status', row)

    def test_missing_mm_without_exit_witness_remains_fatal(self):
        """An unrelated task flag or malformed flag cannot excuse missing counters."""
        root = self.process(10, 1, state='R', flags=0x400000)
        (root / 'status').write_text('Pid:\t10\nNSpid:\t10\n')
        with self.assertRaisesRegex(ValueError, 'host PID or memory fields'):
            owned.OwnedProcessTree(10, proc=self.proc).snapshot()
        fields = (root / 'stat').read_text().rsplit(') ', 1)
        tail = fields[1].split(); tail[6] = '-4'
        (root / 'stat').write_text(fields[0] + ') ' + ' '.join(tail))
        with self.assertRaisesRegex(ValueError, 'task flags'):
            owned.OwnedProcessTree(10, proc=self.proc)

    def test_live_permission_and_malformed_memory_errors_are_not_hidden(self):
        """A live unreadable owner is an observation failure, never retirement."""
        self.process(10, 1)
        tree = owned.OwnedProcessTree(10, proc=self.proc)
        for error in (PermissionError('denied'), ValueError('missing counter')):
            with self.subTest(error=error), patch.object(owned, 'read_process_status', side_effect=error):
                with self.assertRaises(type(error)):
                    tree.snapshot()

    def test_bad_identity_children_and_oversized_metadata_are_rejected(self):
        """Malformed records cannot expand the observation boundary."""
        for value in (0, -1, True, '10'):
            with self.assertRaises(ValueError):
                owned.OwnedProcessTree(value, proc=self.proc)
        root = self.process(10, 1)
        tree = owned.OwnedProcessTree(10, proc=self.proc)
        for value in ('-1', '0', '20/../99', '1' * 65537):
            (root / 'task/10/children').write_text(value)
            with self.assertRaises(ValueError):
                tree.snapshot()

    def test_actual_spawned_child_memory_is_observed_before_retirement(self):
        """Real proc traversal reaches an engine-like child behind a launcher."""
        child = ('import os,signal; payload=bytearray(16*1024*1024); '
                 'print(os.getpid(),flush=True); signal.pause()')
        wrapper = ('import subprocess,sys; p=subprocess.Popen([sys.executable,"-c",'
                   + repr(child) + ']); sys.stdin.readline(); p.terminate(); p.wait()')
        process = subprocess.Popen([sys.executable, '-c', wrapper],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            tree = owned.OwnedProcessTree(process.pid)
            with selectors.DefaultSelector() as ready:
                ready.register(process.stdout, selectors.EVENT_READ)
                self.assertTrue(ready.select(5), 'Child did not publish its PID')
            child_pid = int(process.stdout.readline())
            result = tree.snapshot()
            rows = {row['pid']: row for row in result['processes']}
            self.assertIn(child_pid, rows)
            self.assertGreater(rows[child_pid]['status']['VmRSS'], 16 * 1024**2)
            process.stdin.write('\n'); process.stdin.flush()
            self.assertEqual(process.wait(timeout=5), 0)
            self.assertEqual(tree.snapshot()['processes'], [])
        finally:
            if process.poll() is None:
                process.stdin.write('\n'); process.stdin.flush()
                process.wait(timeout=5)
            process.stdin.close(); process.stdout.close()

    def test_actual_canary_collection_uses_tree_and_keeps_image_scope(self):
        """The production sampling entrypoint retains local engine rows."""
        import run_model_kld_canaries as runner
        self.process(10, 1, children=(20,))
        self.process(20, 10, rss=65536)
        tree = owned.OwnedProcessTree(10, proc=self.proc)
        with patch.object(runner.resources, 'collect', side_effect=lambda _: {'errors': []}):
            sample = runner.observe_native_resources({}, tree)
            self.assertEqual(sample['local_process_tree']['processes'][1]['status']['VmRSS'], 64 * 1024**2)
            self.assertNotIn('local_process', sample)
            self.assertEqual(runner.observe_native_resources({}, None), {'errors': []})


if __name__ == '__main__':
    unittest.main()
