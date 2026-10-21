#!/usr/bin/env python3
"""Device-free explicit-history projection guards; never execute a model."""
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts/ci'))
import model_kld_history_replay as policy


class FrozenHistoryReplayTests(unittest.TestCase):
    """Small owned metadata proves rejection without model/native work."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = []
        shards = []
        for i in range(4):
            path = self.root / f'part{i}.gguf'
            path.write_bytes(bytes([i]))
            self.models.append(path)
            value = path.stat()
            identity = dict(path=str(path), device=value.st_dev, inode=value.st_ino,
                            bytes=1, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)
            shards.append(dict(name=path.name, path=str(path), bytes=1,
                               object_identity=json.dumps(identity, sort_keys=True, separators=(',', ':'))))
        reference = self.root / 'reference'
        reference.mkdir()
        self.metadata = reference / 'metadata.txt'
        self.metadata.write_text('token_ids: 1,2\ndecode_tokens: 3,4,5\n')
        source = dict(architecture='qwen4exp', main_layers=48, expert_count=512,
                      vocabulary=248320, reference_arithmetic='cpu_fp32', reference_execution='incremental_cached', shards=shards)
        self.cell = 'Original/QFN.ProductionParity/SyntheticMetadata'
        self.value = dict(schema='llaminar.model-kld-history-replay.v1', canonical_cell=self.cell,
            source=source, source_digest=policy._digest(source), corpus_manifest_digest='a'*64,
            reference_report_digest='b'*64,
            full_stage_metadata_digest=hashlib.sha256(self.metadata.read_bytes()).hexdigest(),
            history_id='original', context_group='short', prompt='metadata only', prompt_ids=[1, 2],
            continuation_ids=[3, 4, 5], decode_rows=2, reference_scope='full_stages',
            reference_directory=str(reference), expected_config={'configuration': {'n_threads': 4, 'mpi_procs': 1}},
            arithmetic=dict(gdn_projection='fp16_moe_alpha_beta', moe_router='fp16', hc_up='single_q8',
                            gdn_output='single_q8', fa_output='single_q8', gdn_qkv='single_q8'))
        self.path = self.root / 'replay.json'
        self.publish()

    def publish(self):
        """Close a diagnostic metadata file, not a numerical or source receipt."""
        self.path.write_text(json.dumps(self.value))

    def prepare(self):
        """Call the exact driver's production helper with explicit original inputs."""
        return policy.prepare_history_replay(self.path, self.cell, self.models)

    def test_complete_handoff_and_unchanged_sources(self):
        selection = self.prepare()
        artifacts = self.root / 'artifacts'
        artifacts.mkdir()
        handoff = selection.bind(artifacts)
        self.assertEqual(set(handoff), set(policy.HANDOFF_FIELDS))
        self.assertEqual(handoff[policy.HANDOFF_FIELDS[2]], self.cell)
        self.assertEqual(Path(handoff[policy.HANDOFF_FIELDS[0]]).read_bytes(), self.path.read_bytes())
        selection.verify_unchanged()
        replacement = self.root / 'replacement'
        replacement.write_bytes(b'x')
        os.replace(replacement, self.models[0])
        with self.assertRaisesRegex(ValueError, 'source stat identity changed'):
            selection.verify_unchanged()

    def test_foreign_cell_lm_only_and_incomplete_original_sources_reject(self):
        with self.assertRaises(ValueError):
            policy.prepare_history_replay(self.path, 'other', self.models)
        with self.assertRaises(ValueError):
            policy.prepare_history_replay(self.path, self.cell, self.models[:1])
        self.value['reference_scope'] = 'final_logits_only'
        self.publish()
        with self.assertRaisesRegex(ValueError, 'full-stage regeneration'):
            self.prepare()

    def test_pending_teachers_and_original_reference_digest_are_mandatory(self):
        self.value['continuation_ids'] = [3, 4]
        self.publish()
        with self.assertRaisesRegex(ValueError, r'D\+1'):
            self.prepare()
        self.value['continuation_ids'] = [3, 9, 5]
        self.publish()
        with self.assertRaisesRegex(ValueError, 'teacher IDs differ'):
            self.prepare()
        self.value['continuation_ids'] = [3, 4, 5]
        self.publish()
        selection = self.prepare()
        self.metadata.write_text('token_ids: 1,2\ndecode_tokens: 3,4,5\nchanged\n')
        with self.assertRaisesRegex(ValueError, 'metadata changed'):
            selection.verify_unchanged()

    @patch("os.sched_getaffinity", return_value={0, 1, 2, 3})
    @patch("model_kld_corpus.physical_workers", return_value=4)
    def test_actual_fixture_binding_cannot_be_omitted_or_borrowed(self, _physical, _affinity):
        selection = self.prepare()
        directory = self.root / 'native-artifacts'
        directory.mkdir()
        with self.assertRaises(OSError):
            selection.require_binding([str(directory)])
        binding = dict(schema='llaminar.model-kld-history-binding.v1', descriptor_sha256=selection.sha256,
            **{key:self.value[key] for key in ('canonical_cell', 'history_id', 'source_digest',
                'full_stage_metadata_digest', 'prompt', 'prompt_ids', 'continuation_ids',
                'decode_rows', 'reference_directory')})
        path = directory / 'history_replay_binding.json'
        path.write_text(json.dumps(binding))
        geometry = dict(schema='llaminar.model-kld-history-prefill-geometry.v1',
            descriptor_sha256=selection.sha256, canonical_cell=self.cell, history_id='original',
            capture_mode='production_serving_ladder', bucket_rows=[64, 128], minimum_rows=64,
            logical_rows=2, physical_rows=64, native_replay_required=True, native_replay_observed=True)
        from model_kld_corpus import physical_workers
        geometry['cpu_execution'] = dict(requested_workers=4, max_workers=4,
            allowed_cpus=sorted(os.sched_getaffinity(0)), physical_cores=physical_workers(),
            dynamic=False, unbound=True, places='cores', cpu_compute_required=True,
            parallel_work_observed=True)
        geometry_path = directory / 'history_replay_prefill_geometry.json'
        geometry_path.write_text(json.dumps(geometry))
        selection.require_binding([str(directory)])
        geometry['physical_rows'] = 2
        geometry_path.write_text(json.dumps(geometry))
        with self.assertRaisesRegex(ValueError, 'different physical bucket'):
            selection.require_binding([str(directory)])
        geometry['physical_rows'] = 64
        geometry_path.write_text(json.dumps(geometry))
        binding['prompt_ids'] = [1, 9]
        path.write_text(json.dumps(binding))
        with self.assertRaisesRegex(ValueError, 'did not bind'):
            selection.require_binding([str(directory)])

    def test_six_steps_require_seven_original_reference_teachers(self):
        self.value['decode_rows'] = 6
        self.value['continuation_ids'] = [3, 4, 5, 6, 7, 8, 9]
        self.publish()
        with self.assertRaisesRegex(ValueError, 'teacher IDs differ'):
            self.prepare()
        self.metadata.write_text('token_ids: 1,2\ndecode_tokens: 3,4,5,6,7,8,9\n')
        self.value['full_stage_metadata_digest'] = hashlib.sha256(self.metadata.read_bytes()).hexdigest()
        self.publish()
        self.assertEqual(self.prepare().descriptor['decode_rows'], 6)

    def test_explicit_worker_launch_preserves_native_identity_and_transport(self):
        """Only the selected history gets the unbound physical-core projection."""
        from unittest.mock import patch
        command = ('/usr/bin/mpirun', '-np', '1', '--bind-to', 'socket', '--map-by', 'socket',
                   '--mca', 'mpi_leave_pinned', '1',
                   '--mca', 'btl_vader_single_copy_mechanism', 'none',
                   '--mca', 'orte_allowed_exit_without_sync', '1',
                   '-x', 'OMP_NUM_THREADS=2', '-x', 'OMP_PLACES=sockets', '-x', 'OMP_PROC_BIND=close',
                   '-x', 'LLAMINAR_PERF_STATS_FILTER=forward_graph,immutable_lookup',
                   '/native/exact-target', '--gtest_filter=' + self.cell)
        environment = {'LLAMINAR_PERF_STATS_FILTER': 'forward_graph,immutable_lookup',
                       'OMP_NUM_THREADS': '2', 'OMP_PLACES': 'sockets', 'OMP_PROC_BIND': 'close'}
        selection = self.prepare()
        with patch('model_kld_corpus.physical_workers', return_value=4):
            actual, values = selection.launch(command, environment)
        self.assertEqual(actual[:7], ('/usr/bin/mpirun', '-np', '1', '--bind-to', 'none', '--map-by', 'slot'))
        self.assertEqual(actual[7:16], command[7:16])
        self.assertEqual(actual[-2:], command[-2:])
        exported = dict(item.split('=', 1) for index, item in enumerate(actual)
                        if index > 0 and actual[index - 1] == '-x')
        self.assertEqual(exported['OMP_NUM_THREADS'], '4')
        self.assertEqual(exported['OMP_PLACES'], 'cores')
        self.assertIn('kernel.cpu_moe_router_q8_worker_rows', exported['LLAMINAR_PERF_STATS_FILTER'])
        self.assertEqual(environment['OMP_NUM_THREADS'], '2')
        self.assertEqual(values['OMP_NUM_THREADS'], '4')
        self.assertEqual(values['OMP_PLACES'], 'cores')
        self.assertEqual(values['OMP_PROC_BIND'], 'false')
        self.assertEqual(values['KMP_AFFINITY'], 'disabled')
        self.assertEqual(values['LLAMINAR_PERF_STATS_FILTER'],
                         'forward_graph,immutable_lookup,kernel.cpu_moe_router_q8_worker_rows')
        with patch('model_kld_corpus.physical_workers', return_value=3):
            with self.assertRaisesRegex(ValueError, 'physical CPU resources'):
                selection.launch(command, environment)
        for bad in (command[:16] + ('-x', 'OMP_NUM_THREADS=99') + command[16:],
                    command[:16] + ('-x', 'OMP_NUM_THREADS=2') + command[16:],
                    command[:2] + ('2',) + command[3:],
                    command[:4] + ('none',) + command[5:],
                    command[:-2] + ('--host', 'foreign') + command[-2:]):
            with patch('model_kld_corpus.physical_workers', return_value=4):
                with self.assertRaises(ValueError):
                    selection.launch(bad, environment)
        for name in ('GOMP_CPU_AFFINITY', 'KMP_HW_SUBSET', 'KMP_PLACE_THREADS', 'OMP_THREAD_LIMIT'):
            with patch.dict(os.environ, {name: 'restricted'}):
                with self.assertRaisesRegex(ValueError, 'additional CPU resource control'):
                    selection.launch(command, environment)
        self.value['expected_config']['configuration']['n_threads'] = -1
        self.publish()
        with self.assertRaisesRegex(ValueError, 'physical CPU resources'):
            self.prepare().launch(command, environment)

    @patch("os.sched_getaffinity", return_value={0, 1, 2, 3})
    @patch("model_kld_corpus.physical_workers", return_value=4)
    def test_actual_worker_or_narrowed_affinity_receipt_rejects(self, _physical, _affinity):
        """A geometry-only pass cannot disguise the original socket/28-worker defect."""
        # Reuse the real original binding fixture, then replace only its passive
        # CPU observation; no worker or model is created by this test.
        self.test_actual_fixture_binding_cannot_be_omitted_or_borrowed()
        selection = self.prepare()
        directory = self.root / 'native-artifacts'
        binding = json.loads((directory / 'history_replay_binding.json').read_text())
        binding['prompt_ids'] = self.value['prompt_ids']
        (directory / 'history_replay_binding.json').write_text(json.dumps(binding))
        path = directory / 'history_replay_prefill_geometry.json'
        baseline = json.loads(path.read_text())
        for key, value in [('max_workers', 2), ('allowed_cpus', []),
                           ('parallel_work_observed', False), ('unbound', False)]:
            changed = json.loads(json.dumps(baseline))
            changed['cpu_execution'][key] = value
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'declared workers'):
                selection.require_binding([str(directory)])

    def test_driver_requires_explicit_exact_cell_and_rejects_ambient_handoff(self):
        from unittest.mock import patch
        import run_production_parity_campaigns as driver
        with self.assertRaises(SystemExit):
            driver.parse_args(['--build-dir', str(self.root), '--history-replay', str(self.path)])
        with patch.dict(os.environ, {policy.HANDOFF_FIELDS[0]: str(self.path)}):
            with self.assertRaises(SystemExit):
                driver.parse_args(['--build-dir', str(self.root), '--exact-cell', self.cell])


if __name__ == '__main__':
    unittest.main()
