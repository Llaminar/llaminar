#!/usr/bin/env python3
"""Device-free adversarial tests for bounded resource evidence and OS accounting.

Real sparse files, open-unlinked inodes and aliased descriptors exercise metadata
observation. Scripted independent resource series prove that cache churn, stats
resets, warm-up allowance and observation failures cannot hide monotonically
growing memory. No model, accelerator or prefix payload read is needed.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests/v2/e2e/server'))
from resource_growth_assertions import GrowthLimit, ResourceGrowthAssertions, cache_limits, remaining_materialization
from resource_usage_observer import drm_clients, inode_usage, latest_stats, numeric_fields


def stats(*, observed=10, completed=1, epoch=0):
    """Provide independently sized participant caches and one shared archive."""
    def instance(size, used, **identity):
        return dict(enabled=True, initialized=True, capacity_bytes=size, used_bytes=used, **identity)
    return {'snapshot_unix_seconds': observed, 'epoch': epoch, 'requests': {'completed': completed},
            'prefix_cache': {'storage': {'tiers': {
                'ram': dict(enabled=True, capacity_bytes=200, used_bytes=110,
                            instances=[instance(100, 70, participant='ROCm:0'), instance(100, 40, participant='CUDA:0')]),
                'disk': dict(enabled=True, capacity_bytes=300, used_bytes=280,
                             instances=[instance(300, 280, participants=['ROCm:0', 'CUDA:0'])])}}}}


def owner(**changes):
    """An authority row with known lazy capacity, separate from observed RSS."""
    return dict(allocator='0/CPU:0', owner='prefix_host_tier', planned_new_bytes=200,
                materialized_new_bytes=140, committed_new_bytes=200,
                planned_resident_bytes=20, adopted_resident_bytes=20, **changes)


class GrowthAssertionsTests(unittest.TestCase):
    """Validate fixed envelopes and deliberately corrupt otherwise plausible runs."""

    def gate(self):
        """Seal one live identity and independently sourced memory/disk limits."""
        return ResourceGrowthAssertions(identity={'boot': 'A', 'pid': (10, 200)},
            limits={'vram': GrowthLimit(1000, 0, 1, 'complete captured serving graph'),
                    'ram': GrowthLimit(100, 60, 1, 'ready PMA remaining capacity'),
                    'archive': GrowthLimit(0, 320, 1, 'disk capacity + one admitted in-flight payload')},
            cache_capacity={'ram': 200, 'disk': 300}, maximum_sample_age_seconds=5)

    def observe(self, gate, index, *, changes=None, **keywords):
        """Supply a timed completed request with independently mutable measurements."""
        value = dict(observed=10 + index, identity={'boot': 'A', 'pid': (10, 200)},
                     metrics={'vram': 1000, 'ram': 140, 'archive': 300},
                     stats=stats(observed=10 + index, completed=index + 1), errors=[])
        value.update(keywords)
        if changes:
            value['metrics'].update(changes)
        gate.observe(**value)

    def test_admitted_lazy_growth_is_bounded_and_churn_can_shrink(self):
        gate = self.gate()
        for i, ram in enumerate([100, 120, 160, 90, 158]):
            self.observe(gate, i, changes={'ram': ram})
        self.assertTrue(gate.result()['passed'])
        self.assertEqual(gate.result()['peak_bytes']['ram'], 160)

    def test_every_resource_overflow_remains_failed_after_recovery(self):
        for metric, value in [('vram', 1001), ('ram', 161), ('archive', 321)]:
            with self.subTest(metric=metric):
                gate = self.gate()
                self.observe(gate, 0)
                self.observe(gate, 1, changes={metric: value})
                self.observe(gate, 2)
                self.assertFalse(gate.result()['passed'])
                self.assertIn(metric, gate.result()['failures'][0])

    def test_slow_growth_cannot_ratchet_baseline(self):
        gate = self.gate()
        for i in range(65):
            self.observe(gate, i, changes={'ram': 100 + i})
        self.assertEqual(len(gate.result()['failures']), 4)
        self.assertEqual(gate.result()['maximum_bytes']['ram'], 160)

    def test_put_stats_reset_keeps_all_resource_limits(self):
        gate = self.gate()
        self.observe(gate, 0)
        self.observe(gate, 1, stats=stats(observed=11, epoch=1, completed=0))
        self.observe(gate, 2, stats=stats(observed=12, epoch=1, completed=1), changes={'ram': 161})
        self.assertFalse(gate.result()['passed'])

    def test_valid_epoch_reset_preserves_progress_and_passes(self):
        gate = self.gate()
        self.observe(gate, 0)
        for i in (1, 2):
            self.observe(gate, i, stats=stats(observed=10 + i, epoch=1, completed=i - 1))
        self.assertTrue(gate.result()['passed'])

    def test_missing_telemetry_process_reuse_and_staleness_are_not_green(self):
        corruptions = [dict(errors=['NVML unavailable']), dict(metrics={'vram': 1000}),
                       dict(identity={'boot': 'A', 'pid': (10, 201)}), dict(identity={'boot': 'B', 'pid': (10, 200)}),
                       dict(stats=stats(observed=1)), dict(stats=stats(observed=99)), dict(observed=float('nan'))]
        for corruption in corruptions:
            with self.subTest(corruption=corruption):
                gate = self.gate()
                self.observe(gate, 0)
                self.observe(gate, 1, **corruption)
                self.observe(gate, 2)
                self.assertFalse(gate.result()['passed'])

    def test_no_inference_progress_or_duplicate_time_cannot_certify(self):
        gate = self.gate()
        self.observe(gate, 0)
        self.assertFalse(gate.result()['passed'])
        self.observe(gate, 1, stats=stats(observed=11, completed=1))
        self.assertFalse(gate.result()['passed'])
        self.observe(gate, 1)
        self.assertIn('did not advance', gate.result()['failures'][0])

    def test_counter_regression_without_reset_is_rejected(self):
        gate = self.gate()
        self.observe(gate, 0)
        self.observe(gate, 1, stats=stats(observed=11, completed=0))
        self.assertIn('regressed', gate.result()['failures'][0])

    def test_cache_participant_overflow_cannot_hide_in_aggregate_headroom(self):
        value = stats()
        ram = value['prefix_cache']['storage']['tiers']['ram']
        ram['instances'][0]['used_bytes'], ram['instances'][1]['used_bytes'] = 101, 9
        with self.assertRaisesRegex(ValueError, 'instance exceeded'):
            cache_limits(value, {'ram': 200, 'disk': 300})

    def test_shared_disk_is_counted_once_and_must_match_configured_capacity(self):
        self.assertEqual(cache_limits(stats(), {'ram': 200, 'disk': 300})['disk'], 280)
        value = stats()
        disk = value['prefix_cache']['storage']['tiers']['disk']
        disk['instances'] *= 2
        disk['capacity_bytes'], disk['used_bytes'] = 600, 560
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            cache_limits(value, {'ram': 200, 'disk': 600})

    def test_disabled_tiers_have_zero_occupancy_and_no_capacity(self):
        value = stats()
        for tier in value['prefix_cache']['storage']['tiers'].values():
            tier.update(enabled=False, capacity_bytes=0, used_bytes=0)
            for row in tier['instances']:
                row.update(enabled=False, initialized=False, capacity_bytes=0, used_bytes=0)
        self.assertEqual(cache_limits(value, {'ram': 0, 'disk': 0}), {'ram': 0, 'disk': 0})
        value['prefix_cache']['storage']['tiers']['ram']['instances'][0]['used_bytes'] = 1
        with self.assertRaises(ValueError):
            cache_limits(value, {'ram': 0, 'disk': 0})

    def test_authority_reservations_are_not_counted_as_already_materialized(self):
        self.assertEqual(remaining_materialization([owner()]), {'0/CPU:0': 60})
        row = owner()
        row['owner'] = 'workspace'
        self.assertEqual(remaining_materialization([owner(), row]), {'0/CPU:0': 120})

    def test_bad_or_missing_authority_proofs_cannot_invent_slack(self):
        mutations = [('materialized_new_bytes', 201), ('committed_new_bytes', 139),
                     ('adopted_resident_bytes', 21), ('planned_new_bytes', -1), ('planned_new_bytes', True)]
        for field, value in mutations:
            row = owner()
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                remaining_materialization([row])
        for values in ([], [owner(), owner()]):
            with self.assertRaises(ValueError):
                remaining_materialization(values)

    def test_limits_require_provenance_and_exact_units(self):
        for args in ((0, -1, 1, 'PMA'), (0, 1, 0, 'PMA'), (True, 1, 1, 'PMA'), (0, 1, 1, '')):
            with self.assertRaises(ValueError):
                GrowthLimit(*args)
        self.assertEqual(GrowthLimit(100, 20, 4, 'PMA').maximum_bytes, 123)


class MetadataObservationTests(unittest.TestCase):
    """Exercise actual inode lifetimes while preventing payload reads."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_sparse_allocated_extent_and_hardlink_are_independent(self):
        blocks = self.root / 'model.kvcache.blocks'
        blocks.mkdir()
        payload = blocks / 'a.kvblock'
        with payload.open('wb') as file:
            file.truncate(1024**3)
        os.link(payload, blocks / 'alias.kvblock')
        with patch.object(Path, 'open', side_effect=AssertionError('Payload read')):
            result = inode_usage(self.root, [])
        self.assertEqual(result['inodes'], 1)
        self.assertEqual(result['logical_bytes'], 1024**3)
        self.assertLess(result['allocated_bytes'], result['logical_bytes'])

    def test_deleted_open_payload_is_still_disk_usage_until_final_fd_retires(self):
        blocks = self.root / 'model.kvcache.blocks'
        blocks.mkdir()
        payload = blocks / 'a.kvblock'
        with payload.open('wb') as file:
            file.write(b'x' * 4096)
            file.flush()
            duplicate = os.dup(file.fileno())
            try:
                payload.unlink()
                result = inode_usage(self.root, [os.getpid()])
                self.assertEqual(result['groups']['retired_open']['logical_bytes'], 4096)
                self.assertEqual(result['inodes'], 1)
            finally:
                os.close(duplicate)
        self.assertEqual(inode_usage(self.root, [os.getpid()])['inodes'], 0)

    def test_archive_namespaces_are_separate_and_symlinks_are_rejected(self):
        for name in ('.', 'ordinary-control'):
            directory = self.root / name / 'model.kvcache.blocks'
            directory.mkdir(parents=True)
            (directory / 'a.kvblock').write_bytes(b'fixture')
        self.assertEqual(set(inode_usage(self.root, [])['namespaces']), {'.', 'ordinary-control'})
        (self.root / 'alias').symlink_to(self.root / 'ordinary-control', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            inode_usage(self.root, [])

    def test_drm_descriptor_aliases_are_deduplicated_but_device_aliases_not_summed(self):
        for fd, client, device in [('1', '10', '0000:01:00.0'), ('2', '10', '0000:01:00.0'), ('3', '11', '0000:02:00.0')]:
            (self.root / fd).write_text(f'drm-client-id: {client}\ndrm-pdev: {device}\ndrm-resident-gtt: 42 KiB\n')
        rows = drm_clients(self.root)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['drm-resident-gtt_bytes'] == 42 * 1024 for row in rows))

    def test_drm_unknown_units_fail_and_meminfo_retains_bytes(self):
        path = self.root / '1'
        path.write_text('drm-client-id: 1\ndrm-pdev: 0000:01:00.0\ndrm-resident-gtt: 42 MB\n')
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            drm_clients(self.root)
        path.write_text('MemAvailable: 42 kB\nThreads: 6\n')
        self.assertEqual(numeric_fields(path, kilobytes=True), {'MemAvailable': 43008, 'Threads': 6})

    def test_drm_binary_memory_units_are_exact_for_every_memory_field(self):
        """Kernel fdinfo scales large mappings to MiB without changing ownership."""
        fields = ('drm-total-vram', 'drm-shared-gtt', 'drm-resident-vram', 'drm-memory-vram')
        for text, expected in (('0', 0), ('17', 17), ('42 B', 42), ('42 KiB', 42 * 1024),
                               ('33315 MiB', 33315 * 1024**2), ('2 GiB', 2 * 1024**3),
                               ('1 TiB', 1024**4), ('0 MiB', 0)):
            with self.subTest(value=text):
                (self.root / '1').write_text('drm-client-id: 1\ndrm-pdev: 0000:01:00.0\n'
                    + ''.join(f'{field}: {text}\n' for field in fields))
                rows = drm_clients(self.root)
                self.assertEqual(len(rows), 1)
                for field in fields:
                    self.assertEqual(rows[0][field + '_bytes'], expected)
                    self.assertNotIn(field, rows[0])

    def test_drm_memory_rejects_unknown_units_and_malformed_counts(self):
        """Malformed metadata must not turn a large mapping into a small byte count."""
        for text in ('', '-1 MiB', '+1 KiB', '1.5 MiB', 'NaN MiB', '42 MB', '42 kB',
                     '42 KiB extra', '42 MiB extra', '42 XB', '１２ MiB'):
            with self.subTest(value=text), self.assertRaises(ValueError):
                (self.root / '1').write_text('drm-client-id: 1\ndrm-pdev: 0000:01:00.0\n'
                                           f'drm-total-vram: {text}\n')
                drm_clients(self.root)

    def test_drm_scaled_descriptor_aliases_remain_one_mapping(self):
        """Different units on aliased descriptors never create duplicate physical bytes."""
        for fd, value in (('1', '33315 MiB'), ('2', str(33315 * 1024) + ' KiB')):
            (self.root / fd).write_text('drm-client-id: 7\ndrm-pdev: 0000:01:00.0\n'
                                      f'drm-total-vram: {value}\n')
        self.assertEqual(drm_clients(self.root), [{'drm-client-id': '7',
            'drm-pdev': '0000:01:00.0', 'drm-total-vram_bytes': 33315 * 1024**2}])

    def test_stats_partial_append_retains_final_complete_line(self):
        path = self.root / 'stats-webapp-observations.jsonl'
        first = {'observed': 1, 'stats': stats()}
        path.write_text(json.dumps(first) + '\n{"observed":2')
        self.assertEqual(latest_stats(self.root), first)
        path.write_text(json.dumps(first) + '\n{broken}\n')
        with self.assertRaises(ValueError):
            latest_stats(self.root)

    def test_stats_missing_and_empty_journals_are_explicit(self):
        self.assertIsNone(latest_stats(self.root))
        (self.root / 'stats-tools-observations.jsonl').write_bytes(b'')
        self.assertIsNone(latest_stats(self.root))


if __name__ == '__main__':
    unittest.main()
