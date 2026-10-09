#!/usr/bin/env python3
"""Device-free proof that resource probes stay inside owned process cgroups.

Unselected process details are poisoned independently of the collector. Exact
Docker identities, startup/retirement gaps and single-read controller snapshots
must prevent unrelated host probes while preserving real ownership failures.
No GPU context, host process inspection or cache payload read is performed.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests/v2/e2e/server'))
import resource_usage_observer as observer


class ResourceObserverScopeTests(unittest.TestCase):
    """Observe selected owners while treating every unrelated detail read as fatal."""

    def process(self, proc, pid, cgroup):
        """Create one independent proc fixture with an exact kernel start tick."""
        directory = proc / str(pid)
        directory.mkdir()
        (directory / 'cgroup').write_text('0::' + cgroup + '\n')
        (directory / 'comm').write_text('mpirun\n')
        fields = ['S', *(['0'] * 18), '424242', '0']
        (directory / 'stat').write_text(str(pid) + ' (mpirun) ' + ' '.join(fields))
        return directory

    def test_empty_scope_does_not_enumerate_processes(self):
        """A controller waiting for admission has no authority to inspect any PID."""
        with patch.object(Path, 'iterdir', side_effect=AssertionError('Unexpected process scan')):
            self.assertEqual(observer.process_inventory(frozenset()), {})

    def test_unselected_and_substring_cgroups_never_open_process_details(self):
        """Cgroup membership alone excludes foreign owners and misleading path names."""
        identity = 'a' * 64
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            owned = self.process(proc, 101, '/system.slice/docker-' + identity + '.scope')
            foreign = [self.process(proc, 102, '/system.slice/host.service'),
                       self.process(proc, 103, '/docker/' + identity + '-neighbor'),
                       self.process(proc, 104, '/docker/' + 'b' * 64)]
            read = Path.read_text
            seen = []

            def guarded(path, *args, **kwargs):
                """A forbidden detail read fails before returning any synthetic bytes."""
                seen.append(path)
                if path.parent in foreign and path.name != 'cgroup':
                    raise AssertionError('Unrelated process detail was opened: ' + str(path))
                return read(path, *args, **kwargs)

            with patch.object(Path, 'read_text', guarded):
                result = observer.process_inventory(frozenset({identity}), proc=proc)
            self.assertEqual(list(result), ['/system.slice/docker-' + identity + '.scope'])
            row = next(iter(result.values()))
            self.assertEqual(row['container_id'], identity)
            self.assertEqual(row['processes'], [{'pid': 101, 'comm': 'mpirun', 'start_ticks': 424242}])
            self.assertIn(owned / 'stat', seen)
            self.assertTrue(all(directory / 'cgroup' in seen for directory in foreign))

    def test_cgroupfs_and_systemd_owners_keep_distinct_process_identities(self):
        """Both native Docker cgroup layouts admit the same typed full-ID contract."""
        first, second = 'a' * 64, 'b' * 64
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            self.process(proc, 101, '/docker/' + first)
            self.process(proc, 102, '/user.slice/docker-' + second + '.scope')
            result = observer.process_inventory(frozenset({first, second}), proc=proc)
            self.assertEqual({row['container_id']: row['processes'][0]['pid'] for row in result.values()},
                             {first: 101, second: 102})

    def test_invalid_or_mutable_scope_fails_before_process_enumeration(self):
        """A short name, malformed identity or mutable selection cannot select host data."""
        for identities in (frozenset({'a' * 12}), frozenset({'A' * 64}),
                           frozenset({False}), {'a' * 64}, frozenset({'a' * 64 + '/child'})):
            with self.subTest(identities=identities), \
                    patch.object(Path, 'iterdir', side_effect=AssertionError('Unexpected scan')):
                with self.assertRaises(ValueError):
                    observer.process_inventory(identities)

    def test_ambiguous_nested_owned_scopes_fail(self):
        """One process cannot silently satisfy two selected container owners."""
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            self.process(proc, 101, '/docker/' + 'a' * 64 + '/docker/' + 'b' * 64)
            with self.assertRaisesRegex(ValueError, 'multiple selected'):
                observer.process_inventory(frozenset({'a' * 64, 'b' * 64}), proc=proc)

    def test_owned_process_permission_failure_is_not_hidden(self):
        """Narrowing scope never changes an owned unreadable process into a missing PID."""
        identity = 'a' * 64
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            owned = self.process(proc, 101, '/docker/' + identity)
            read = Path.read_text

            def denied(path, *args, **kwargs):
                """Deny only the selected process's protected identity record."""
                if path == owned / 'stat':
                    raise PermissionError('owned metadata denied')
                return read(path, *args, **kwargs)

            with patch.object(Path, 'read_text', denied), self.assertRaises(PermissionError):
                observer.process_inventory(frozenset({identity}), proc=proc)

    def test_collection_reads_controller_identity_once_before_process_probe(self):
        """An atomic controller replacement cannot relabel an earlier PID inventory."""
        identity = 'a' * 64
        with tempfile.TemporaryDirectory() as temporary:
            controller = Path(temporary) / 'controller.json'
            controller.write_text(json.dumps({'container_id': identity, 'complete': False, 'phase': 'app'}))
            read = Path.read_text
            seen = []

            def snapshot(path, *args, **kwargs):
                """A second controller read would expose a different lifecycle owner."""
                if path == Path('/proc/sys/kernel/random/boot_id'):
                    return 'fixture-boot'
                if path == controller:
                    seen.append(path)
                    if len(seen) > 1:
                        return json.dumps({'container_id': 'b' * 64})
                return read(path, *args, **kwargs)

            with patch.object(Path, 'read_text', snapshot), patch.object(Path, 'glob', return_value=iter(())), \
                    patch.object(Path, 'is_dir', return_value=False), \
                    patch.object(observer, 'numeric_fields', return_value={}), \
                    patch.object(observer, 'latest_stats', return_value={'fixture': True}), \
                    patch.object(observer, 'cuda_usage', return_value={}), \
                    patch.object(observer, 'process_inventory', return_value={}) as inventory:
                result = observer.collect({'backends': ['cuda', 'rocm'],
                    'cells': {'cell': {'controller': str(controller)}}, 'filesystems': {}})
            inventory.assert_called_once_with(frozenset({identity}))
            self.assertEqual(seen, [controller])
            self.assertEqual(result['cells']['cell']['container_id'], identity)
            self.assertEqual(result['cells']['cell']['errors'], [])

    def test_cpu_and_rocm_observers_do_not_require_nvml(self):
        """Unselected vendor probes cannot make an otherwise supported host fail."""
        for backends in (['cpu'], ['rocm'], ['cpu', 'rocm']):
            with self.subTest(backends=backends), \
                    patch.object(Path, 'read_text', return_value='fixture-boot'), \
                    patch.object(Path, 'glob', return_value=iter(())) as drm, \
                    patch.object(observer, 'numeric_fields', return_value={}), \
                    patch.object(observer, 'process_inventory', return_value={}), \
                    patch.object(observer, 'cuda_usage', side_effect=AssertionError('Unselected NVML probe')):
                value = observer.collect({'backends': backends, 'cells': {}, 'filesystems': {}})
                self.assertEqual(value['errors'], [])
                self.assertNotIn('cuda', value)
                self.assertEqual(drm.call_count, int('rocm' in backends))

    def test_selected_cuda_probe_failure_remains_a_failed_observation(self):
        """Vendor selection narrows scope without treating missing evidence as success."""
        with patch.object(Path, 'read_text', return_value='fixture-boot'), \
                patch.object(Path, 'glob', side_effect=AssertionError('Unselected DRM probe')), \
                patch.object(observer, 'numeric_fields', return_value={}), \
                patch.object(observer, 'process_inventory', return_value={}), \
                patch.object(observer, 'cuda_usage', side_effect=RuntimeError('NVML unavailable')):
            value = observer.collect({'backends': ['cuda'], 'cells': {}, 'filesystems': {}})
            self.assertEqual(value['errors'], ["NVML: RuntimeError('NVML unavailable')"])

    def test_backend_scope_is_mandatory_before_any_host_read(self):
        """Missing or ambiguous inventory cannot trigger a best-effort host-wide probe."""
        for backends in (None, [], ['cuda', 'cuda'], ['CUDA'], ['tpu'], [True], 'cuda'):
            with self.subTest(backends=backends), \
                    patch.object(Path, 'read_text', side_effect=AssertionError('Unexpected host read')), \
                    self.assertRaises(ValueError):
                observer.collect({'backends': backends, 'cells': {}, 'filesystems': {}})


if __name__ == '__main__':
    unittest.main()
