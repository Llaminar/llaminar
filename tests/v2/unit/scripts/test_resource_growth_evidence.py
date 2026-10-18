#!/usr/bin/env python3
"""Exercise complete OS/admission joins with adversarial multi-rank evidence.

The fixtures intentionally reuse device ordinal zero in different ranks, change
NVML enumeration order, and interleave cache churn with statistics resets. No
GPU or model is loaded. Each negative changes evidence which a live lifecycle
owner can lose independently; a later successful sample cannot erase that loss.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests/v2/e2e/server'))
from resource_growth_evidence import (
    DeviceBinding, RankBinding, WorkloadWindow, assess_observations, initialized_allowances, bind_native_ranks)
from resource_process_identity import read_process_status

UUID = 'GPU-12345678-1234-1234-1234-123456789abc'
CONTAINER = 'a' * 64


def proof() -> dict:
    """Give each of two native ranks one independent CPU and GPU allocator."""
    records = []
    for rank, device in [(0, 'CUDA:0'), (1, 'ROCm:0')]:
        records.append({'domain': 'server', 'name': 'rank_membership', 'kind': 'counter',
            'rank': rank, 'value': 1, 'tags': {'rank': str(rank), 'world_size': '2', 'authority_rank': '0'}})
        records.append({'domain': 'server', 'name': 'process_identity', 'kind': 'counter',
            'rank': rank, 'value': 1, 'phase': 'startup',
            'tags': {'schema': '1', 'identity_source': 'native_pid_namespace', 'pid': str(7 + rank)}})
        records.append({'domain': 'server', 'name': 'execution_participant', 'kind': 'counter',
            'rank': rank, 'value': 1, 'device': device, 'phase': 'startup',
            'tags': {'schema': '1', 'identity_source': 'communicator_cluster_inventory',
                     'source': 'resolved_execution_plan', 'node_id': '0',
                     'backend': device.split(':')[0],
                     'physical_id': UUID[4:].replace('-', '') if rank == 0 else 'b' * 32,
                     'pci_bus_address': '0000:02:00.0' if rank == 0 else '0000:01:00.0'}})
        for allocator in ('CPU', device):
            records.append({'domain': 'physical_memory', 'name': 'resource_admission', 'kind': 'counter',
                'rank': rank, 'device': allocator, 'phase': 'initialized', 'value': 1,
                'tags': {'rank': str(rank), 'owner_count': '1', 'incremental_bytes': '10000', 'available_bytes': '20000'}})
            records.append({'domain': 'physical_memory', 'name': 'owner_attestation', 'kind': 'counter',
                'rank': rank, 'device': allocator, 'phase': 'initialized', 'value': 1,
                'tags': {'rank': str(rank), 'owner': 'workspace', 'planned_new_bytes': '10000',
                         'planned_resident_bytes': '100', 'materialized_new_bytes': '4000',
                         'committed_new_bytes': '10000', 'adopted_resident_bytes': '100'}})
    return {'schema': 'llaminar.perf_stats.v1', 'world_size': 2, 'authority_rank': 0, 'records': records}


def sample(observed: int) -> dict:
    """Make process and GPU counters independent from each cache occupancy gauge."""
    def process(pid, start, drm):
        return {'pid': pid, 'comm': 'llaminar2', 'start_ticks': start, 'drm_clients': drm,
            'namespace_pids': [pid, 7 if pid == 100 else 8],
            'status': {'VmRSS': 22000, 'VmHWM': 24000, 'RssAnon': 10000, 'RssFile': 8000,
                       'RssShmem': 4000, 'VmSwap': 0}}

    def tier(name, capacity, used, identity):
        instance = {'capacity_bytes': capacity, 'used_bytes': used, 'enabled': True, 'initialized': True, **identity}
        return {name: {**{key: instance[key] for key in ('capacity_bytes', 'used_bytes', 'enabled', 'initialized')},
                       'instances': [instance]}}

    stats = {'snapshot_unix_seconds': observed, 'requests': {'completed': observed - 9}, 'epoch': 0,
        'prefix_cache': {'storage': {'tiers': {
            **tier('ram', 200, 120, {'participant': 'CUDA:0'}),
            **tier('disk', 300, 200, {'participants': ['CUDA:0']})}}}}
    return {'schema_version': 2, 'observed': observed, 'collection_seconds': 0.1, 'boot_id': 'boot-A', 'errors': [],
        'cells': {'coding': {'container_id': CONTAINER, 'errors': [], 'stats_observation': {'stats': stats},
            'ram': {'events': {'oom': 0, 'oom_kill': 0}, 'stat': {'anon': 20000, 'file': 8000, 'shmem': 4000}},
            'archive': {'logical_bytes': 200, 'allocated_bytes': 4096, 'inodes': 1},
            'processes': [process(100, 50, []), process(200, 51, [
                {'drm-client-id': '7', 'drm-pdev': '0000:01:00.0', 'drm-total-vram_bytes': 100000,
                 'drm-total-gtt_bytes': 2**35}])] }},
        'cuda': {'resolution_bytes': 1024**2,
                 'devices': [{'index': 2, 'uuid': UUID, 'pci.bus_id': '00000000:02:00.0'}],
                 'processes': [{'pid': 100, 'gpu_uuid': UUID, 'used_gpu_memory_bytes': 2**20}]}}


def arguments() -> dict:
    """Seal identities before the coding window; physical binding is explicit."""
    return {'perf': proof(), 'cell': 'coding', 'container_id': CONTAINER, 'boot_id': 'boot-A',
        'window': WorkloadWindow(9, 10, 11, 2, 1, 4096), 'cache_capacity': {'ram': 200, 'disk': 300}}


class ResourceEvidenceTests(unittest.TestCase):
    """Keep authority, telemetry continuity and workload coverage independently fatal."""

    def assess(self, samples=None, **changes):
        values = arguments()
        values.update(changes)
        return assess_observations(observations=samples or [sample(i) for i in (9, 10, 11)], **values)

    def assert_failure(self, result, fragment):
        self.assertFalse(result['passed'], result)
        self.assertIn(fragment, json.dumps(result))

    def test_native_rank_join_and_distinct_shared_host_observations_pass(self):
        result = self.assess()
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['growth']['samples'], 3)
        self.assertEqual(result['growth']['maximum_bytes']['rank/1/ROCm:0'], 110095)
        # Two clients mapping a 32-GiB pool must never become 64 GiB of RSS.
        self.assertEqual(result['diagnostic_peaks']['rank/1/VmRSS'], 22000)
        self.assertTrue(result['physical_archive_ownership_audit_required'])

    def test_all_observed_resources_fail_independently_and_failure_is_sticky(self):
        for resource in ('ram', 'cuda', 'rocm'):
            with self.subTest(resource=resource):
                values = [sample(i) for i in (9, 10, 11)]
                row = values[1]['cells']['coding']
                if resource == 'ram':
                    row['processes'][0]['status']['RssAnon'] = 20096
                elif resource == 'cuda':
                    values[1]['cuda']['processes'][0]['used_gpu_memory_bytes'] = 2**21 + 6000
                else:
                    row['processes'][1]['drm_clients'][0]['drm-total-vram_bytes'] = 110096
                self.assert_failure(self.assess(values), 'exceeded fixed resource envelope')

    def test_swap_cannot_hide_anonymous_growth_and_file_page_cache_is_not_added(self):
        values = [sample(i) for i in (9, 10, 11)]
        status = values[1]['cells']['coding']['processes'][0]['status']
        status.update(RssAnon=0, VmSwap=20096)
        self.assert_failure(self.assess(values), 'exceeded fixed resource envelope')
        status.update(RssAnon=10000, VmSwap=0, RssFile=2**35)
        values[1]['cells']['coding']['ram']['stat']['file'] = 2**36
        self.assertTrue(self.assess(values)['passed'])

    def test_nvml_ordinal_change_does_not_reassign_physical_gpu(self):
        values = [sample(i) for i in (9, 10, 11)]
        values[1]['cuda']['devices'][0]['index'] = 99
        self.assertTrue(self.assess(values)['passed'])
        values[1]['cuda']['processes'][0]['gpu_uuid'] = 'GPU-aaaaaaaa-1234-1234-1234-123456789abc'
        self.assert_failure(self.assess(values), 'CUDA process/device')

    def test_pid_reuse_and_missing_rank_cannot_borrow_neighbor_metrics(self):
        for mode in ('reuse', 'missing', 'duplicate'):
            values = [sample(i) for i in (9, 10, 11)]
            processes = values[1]['cells']['coding']['processes']
            if mode == 'reuse':
                processes[0]['start_ticks'] += 1
            elif mode == 'missing':
                processes.pop()
            else:
                processes.append(deepcopy(processes[0]))
            with self.subTest(mode=mode):
                self.assert_failure(self.assess(values), 'process' if mode != 'reuse' else 'PID was reused')

    def test_boot_container_oom_and_observer_errors_stay_fatal_after_recovery(self):
        for mode in ('boot', 'container', 'oom', 'observer'):
            values = [sample(i) for i in (9, 10, 11)]
            if mode == 'boot':
                values[1]['boot_id'] = 'boot-B'
            elif mode == 'container':
                values[1]['cells']['coding']['container_id'] = 'b' * 64
            elif mode == 'oom':
                values[1]['cells']['coding']['ram']['events']['oom'] = 1
            else:
                values[1]['errors'] = ['NVML failed']
            with self.subTest(mode=mode):
                self.assertFalse(self.assess(values)['passed'])

    def test_missing_baseline_never_selects_a_later_larger_sample(self):
        self.assert_failure(self.assess([sample(10), sample(11)]), 'baseline is missing')
        values = [sample(i) for i in (9, 10, 11)]
        values[0]['errors'] = ['unreadable process']
        result = self.assess(values)
        self.assert_failure(result, 'OS resource observation failed')
        self.assertEqual(result['growth']['samples'], 0)

    def test_gaps_partial_lifetime_and_nonmonotonic_journals_cannot_certify(self):
        cases = [[sample(9), sample(10)], [sample(9), sample(12)],
                 [sample(9), sample(9), sample(11)], [sample(9), sample(10), sample(9), sample(11)],
                 [sample(9), sample(10), sample(8), sample(11)]]
        for values in cases:
            with self.subTest(times=[x['observed'] for x in values]):
                self.assertFalse(self.assess(values)['passed'])
        self.assertTrue(self.assess([sample(8), sample(9), sample(10), sample(11), sample(12)])['passed'])
        self.assertTrue(self.assess([sample(9), sample(10), sample(11), {'errors': ['retired']}])['passed'])

    def test_missing_and_late_authority_rows_cannot_invent_a_growth_allowance(self):
        for mode in ('missing', 'terminal', 'timer', 'rank', 'boolean'):
            perf = proof()
            if mode == 'missing':
                perf['records'].pop()
            elif mode == 'terminal':
                perf['records'][-1]['phase'] = 'shutdown'
            elif mode == 'timer':
                perf['records'][-1]['kind'] = 'timer'
            elif mode == 'rank':
                perf['records'][-1]['rank'] = 0
            else:
                perf['records'][-1]['value'] = True
            with self.subTest(mode=mode):
                self.assertFalse(self.assess(perf=perf)['passed'])
        self.assertEqual(initialized_allowances(proof()), {
            (0, 'CPU'): 6000, (0, 'CUDA:0'): 6000, (1, 'CPU'): 6000, (1, 'ROCm:0'): 6000})

    def test_binding_must_cover_every_rank_and_every_admitted_gpu(self):
        for kind in ('process_identity', 'execution_participant'):
            perf = proof()
            perf['records'] = [row for row in perf['records'] if not (row['rank'] == 1 and row['name'] == kind)]
            self.assertFalse(self.assess(perf=perf)['passed'])

    def test_native_binding_uses_namespace_pid_not_process_or_gpu_order(self):
        observation = sample(9)
        observation['cells']['coding']['processes'].reverse()
        ranks = bind_native_ranks(proof(), observation, 'coding')
        self.assertEqual([(r.rank, r.pid, r.start_ticks) for r in ranks], [(0, 100, 50), (1, 200, 51)])
        self.assertEqual(ranks[0].devices, (DeviceBinding('CUDA:0', UUID),))
        self.assertEqual(ranks[1].devices, (DeviceBinding('ROCm:0', '0000:01:00.0'),))

    def test_native_pid_namespace_missing_alias_or_wrong_host_is_fatal(self):
        for namespace in ([], [101, 7], [100, 8], [100, True], [100, 0]):
            observation = sample(9)
            observation['cells']['coding']['processes'][0]['namespace_pids'] = namespace
            with self.subTest(namespace=namespace), self.assertRaises(ValueError):
                bind_native_ranks(proof(), observation, 'coding')

    def test_native_process_missing_duplicate_or_late_evidence_is_fatal(self):
        for mode in ('missing', 'duplicate', 'late', 'boolean', 'unknown_pid'):
            perf = proof()
            row = next(r for r in perf['records'] if r['name'] == 'process_identity')
            if mode == 'missing':
                perf['records'].remove(row)
            elif mode == 'duplicate':
                perf['records'].append(deepcopy(row))
            elif mode == 'late':
                row['phase'] = 'shutdown'
            elif mode == 'boolean':
                row['value'] = True
            else:
                row['tags']['pid'] = '10'
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                bind_native_ranks(perf, sample(9), 'coding')

    def test_native_gpu_uuid_and_pci_endpoint_must_both_match(self):
        for mode in ('bridge', 'uuid', 'ordinal', 'source', 'missing_pci', 'function'):
            perf = proof()
            row = next(r for r in perf['records'] if r['name'] == 'execution_participant')
            if mode == 'bridge':
                row['tags']['pci_bus_address'] = '0000:03:00.0'
            elif mode == 'uuid':
                row['tags']['physical_id'] = 'a' * 32
            elif mode == 'ordinal':
                row['device'] = 'CUDA:9'
            elif mode == 'source':
                row['tags']['identity_source'] = 'guessed_ordinal'
            elif mode == 'missing_pci':
                row['tags']['pci_bus_address'] = ''
            else:
                row['tags']['pci_bus_address'] = '0000:02:00.1'
            with self.subTest(mode=mode):
                self.assertFalse(self.assess(perf=perf)['passed'])

    def test_bounded_process_status_retains_inner_pid_and_exact_memory_units(self):
        status = ('Name:\tllaminar2\nPid:\t100\nNSpid:\t100\t50\t7\n' +
                  ''.join(f'{key}:\t{value} kB\n' for key, value in
                          zip(('VmRSS', 'VmHWM', 'RssAnon', 'RssFile', 'RssShmem', 'VmSwap'), range(6))))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status'
            path.write_text(status)
            result = read_process_status(path, 100)
            self.assertEqual(result['namespace_pids'], [100, 50, 7])
            self.assertEqual(result['status']['RssAnon'], 2048)
            for bad in (status.replace('100\t50\t7', '101\t50\t7'),
                        status.replace('NSpid:', 'Unknown:'), status + 'NSpid: 100 7\n',
                        status.replace(' kB', ' MiB'), status.replace('5 kB', '-5 kB'),
                        status.replace('Pid:\t100', 'Pid:\t101'), status + 'x' * 65537):
                path.write_text(bad)
                with self.assertRaises(ValueError):
                    read_process_status(path, 100)

    def test_cache_budget_and_participant_cannot_change_after_stats_reset(self):
        values = [sample(i) for i in (9, 10, 11)]
        stats = values[1]['cells']['coding']['stats_observation']['stats']
        stats['epoch'] = 1
        stats['requests']['completed'] = 0
        stats['prefix_cache']['storage']['tiers']['ram']['instances'][0]['participant'] = 'CUDA:1'
        self.assert_failure(self.assess(values), 'Cache ownership')

    def test_exact_measurement_schemas_reject_boolean_bytes_and_drm_aliases(self):
        values = [sample(i) for i in (9, 10, 11)]
        values[1]['cells']['coding']['processes'][0]['status']['RssAnon'] = True
        self.assert_failure(self.assess(values), 'nonnegative integer')
        values[1] = sample(10)
        clients = values[1]['cells']['coding']['processes'][1]['drm_clients']
        clients.append(deepcopy(clients[0]))
        self.assert_failure(self.assess(values), 'duplicated native ROCm')

    def test_invalid_device_window_and_process_bindings_are_rejected(self):
        for device, physical in [('CUDA:0', '0'), ('ROCm:0', '01:00.0'), ('CPU', UUID)]:
            with self.assertRaises(ValueError):
                DeviceBinding(device, physical)
        for window in [(10, 9, 11, 2, 1, 4096), (9, 10, 10, 2, 1, 4096),
                       (9, 10, 11, 0, 1, 4096), (9, 10, 11, 2, 1, 1000)]:
            with self.assertRaises(ValueError):
                WorkloadWindow(*window)
        with self.assertRaises(ValueError):
            RankBinding(0, True, 10, ())

    def test_command_consumes_rank_files_and_writes_a_complete_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = arguments()
            perf = config.pop('perf')
            from dataclasses import asdict
            config['window'] = asdict(config['window'])
            (root / 'binding.json').write_text(json.dumps(config))
            (root / 'observations.jsonl').write_text(''.join(json.dumps(sample(i)) + '\n' for i in (9, 10, 11)))
            for rank in (0, 1):
                (root / f'perf.rank-{rank}.json').write_text(json.dumps({
                    'schema': perf['schema'], 'records': [row for row in perf['records'] if row['rank'] == rank]}))
            result = subprocess.run([sys.executable, str(ROOT / 'tests/v2/e2e/server/resource_growth_evidence.py'),
                '--observations', str(root / 'observations.jsonl'), '--binding', str(root / 'binding.json'),
                '--perf', str(root / 'perf.json'), '--output', str(root / 'result.json')],
                text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads((root / 'result.json').read_text())['passed'])


if __name__ == '__main__':
    unittest.main()
