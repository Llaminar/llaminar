#!/usr/bin/env python3
"""Device-free fake-reference tests for the actual shared cached panel writer.

No checkpoint, tokenizer download or model is loaded. Tiny authored arrays and
fake file owners exercise optional capture, one admission, cache isolation and
failure retirement. Fake original-source metadata is explicitly injected, never
claimed to authenticate a real GGUF or runtime backend.
"""
from pathlib import Path
import contextlib
import io
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts/ci'))
import numpy as np
from python.reference import PipelineStage
from python.reference import qwen4exp_accuracy_panel as panel
from python.reference.generate_qwen4exp_pipeline_snapshots import parse_args, load_frozen_history, generate_frozen_history
from python.reference.generate_qwen35_pipeline_snapshots import run_prefill_and_decode


class _Tokens:
    """Minimal tokenizer index result retaining the existing helper's API."""
    def __init__(self, ids):
        """Retain a small exact authored token-ID row."""
        self.ids = ids

    def __getitem__(self, index):
        """Expose batch0 in the same structure as return_tensors='pt'."""
        if index != 0:
            raise IndexError(index)
        return self

    def tolist(self):
        """Return a detached ordinary list of original prompt IDs."""
        return list(self.ids)


class FakeReference:
    """Track the actual helper's cached forwards without implementing a model."""
    def __init__(self, prompts=None, fail_forward=None, close_failure=False, vocabulary=8):
        """Bind authored tokenization and optional deterministic failure frontiers."""
        self.prompts = prompts or {'fixture': [1, 2]}
        self.tokenizer = self.tokenize
        self.calls, self.snapshots = [], {}
        self.fail_forward, self.close_failure = fail_forward, close_failure
        self.closed = 0
        self.vocabulary = vocabulary
        self.hf_model = SimpleNamespace(config=SimpleNamespace(vocab_size=vocabulary))

    def tokenize(self, prompt, return_tensors):
        """Return exact fixture IDs; never invoke a real tokenizer."""
        if return_tensors != 'pt':
            raise ValueError('wrong tokenizer contract')
        return {'input_ids': _Tokens(self.prompts[prompt])}

    def forward(self, ids, *, clear_snapshots, use_cache, capture_stages, past_key_values=None):
        """Publish identifiable small logits and an owned cache-chain token."""
        self.calls.append(dict(ids=list(ids), past=past_key_values, stages=capture_stages))
        if len(self.calls) == self.fail_forward:
            raise RuntimeError('injected forward failure')
        if not clear_snapshots or not use_cache:
            raise ValueError('cached canonical helper required')
        self.clear_snapshots()
        value = np.zeros((1, len(ids), self.vocabulary), dtype=np.float32)
        value[:, :, (len(self.calls) % 6) + 1] = 1.
        for stage in [PipelineStage.LM_HEAD, PipelineStage.EMBEDDING]:
            if capture_stages is None or stage in capture_stages:
                self.snapshots[(stage, -1)] = value.copy()
        return {'logits': value, 'past_key_values': object()}

    def get_snapshots(self):
        """Return the actual fixture snapshot bank as a separate mapping."""
        return self.snapshots.copy()

    def clear_snapshots(self):
        """Reset only captured observation data between actual forwards."""
        self.snapshots.clear()

    def close(self):
        """Observe sole retirement and optionally fail before complete publication."""
        self.closed += 1
        if self.close_failure:
            raise RuntimeError('injected close failure')


class QFNAccuracyReferenceTests(unittest.TestCase):
    """Exercise production helper/writer control flow with no model payloads."""
    def setUp(self):
        """Copy the single authored source panel into a temporary owned directory."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.authored = json.loads((ROOT/'scripts/ci/qfn_accuracy_prompts.json').read_text())
        # Keep two-step compatibility fixtures independent of the authored default.
        self.authored['decode_steps'] = 2
        self.panel_file = self.directory/'prompts.json'
        self.panel_file.write_text(json.dumps(self.authored))
        self.args = SimpleNamespace(accuracy_panel=self.panel_file, output=self.directory/'out',
                                    model='fixture-not-model', cache_bytes=0, tokenizer=None, decode_steps=2)

    def fake_model(self, **kwargs):
        """Map the mandatory anchor faithfully, all other fixtures to unique IDs."""
        mapping = {c['prompt']: (panel.ORIGINAL_IDS if i == 0 else [i, i+1])
                   for i, c in enumerate(self.authored['cases'])}
        return FakeReference(mapping, vocabulary=248320, **kwargs)

    def collect(self, model, source=None):
        """Invoke the actual writer with explicitly fake source metadata only."""
        if source is None:
            source = dict(vocabulary=248320, source_provenance='unit-fake-not-GGUF')
        factory = unittest.mock.Mock(return_value=model)
        with patch.object(panel, 'original_source', return_value=source), \
                contextlib.redirect_stdout(io.StringIO()):
            result = panel.generate_accuracy_panel(self.args, tokenizer=None, model_factory=factory)
        factory.assert_called_once()
        return result

    def test_complete_panel_one_admission_and_independent_cache_origins(self):
        """All16 actual helper calls publish48 full rows and D+1 teacher IDs."""
        model = self.fake_model()
        result = self.collect(model)
        self.assertEqual(model.closed, 1)
        self.assertEqual(len(model.calls), 48)
        self.assertEqual(len(result['histories']), 16)
        report = json.loads((self.args.output/'reference.json').read_text())
        self.assertTrue(report['complete'])
        self.assertEqual(len(report['cases']), 16)
        self.assertEqual(result['approval'], 'not_approved')
        for index, (history, case) in enumerate(zip(result['histories'], report['cases'])):
            self.assertEqual(len(history['continuation_ids']), 3)
            self.assertEqual(history['reference_scope'], 'final_logits_only')
            self.assertEqual(len(case['rows']), 3)
            self.assertIsNone(model.calls[index*3]['past'])
            self.assertTrue(all(c['past'] is not None for c in model.calls[index*3+1:index*3+3]))
            self.assertEqual((self.args.output/case['logits_file']).stat().st_size, 3*248320*4)
            first = len(history['prompt_ids'])
            self.assertEqual([r['consumed_tokens'] for r in case['rows']], [first, first+1, first+2])
            self.assertEqual([r['input_token'] for r in case['rows']], [None, *history['continuation_ids'][:2]])
        self.assertEqual(result['histories'][0]['prompt_ids'], panel.ORIGINAL_IDS)

    def test_authored_six_depth_and_exact_d_plus_one_publication(self):
        """The same16 histories gain six cached forwards with exact owned rows/files."""
        authored=json.loads((ROOT/'scripts/ci/qfn_accuracy_prompts.json').read_text())
        self.assertEqual(authored['decode_steps'],6)
        authored['decode_steps']=6;self.panel_file.write_text(json.dumps(authored))
        self.args.decode_steps=6
        # Tiny fake vocabulary admits the unchanged anchor IDs; no real model.
        model=self.fake_model();result=self.collect(model)
        self.assertEqual(len(model.calls),16*7);self.assertEqual(model.closed,1)
        report=json.loads((self.args.output/'reference.json').read_text())
        for index,(history,case) in enumerate(zip(result['histories'],report['cases'])):
            self.assertEqual(history['decode_rows'],6);self.assertEqual(len(history['continuation_ids']),7)
            self.assertEqual(len(case['rows']),7)
            self.assertIsNone(model.calls[index*7]['past'])
            self.assertTrue(all(c['past'] is not None for c in model.calls[index*7+1:index*7+7]))
            self.assertEqual((self.args.output/case['logits_file']).stat().st_size,7*248320*4)
            first=len(history['prompt_ids'])
            self.assertEqual([r['consumed_tokens'] for r in case['rows']],list(range(first,first+7)))
            self.assertEqual([r['input_token'] for r in case['rows']],[None,*history['continuation_ids'][:6]])

    def test_invalid_or_mismatched_depth_rejects_before_admission(self):
        """No invalid or ignored CLI count can reach factory or create output."""
        base=['--model','fixture-main','--output',str(self.args.output),'--accuracy-panel',str(self.panel_file)]
        factory=unittest.mock.Mock()
        for authored_count in [0,-1,256,257,True,2.5,'6']:
            authored=json.loads(json.dumps(self.authored));authored['decode_steps']=authored_count
            self.panel_file.write_text(json.dumps(authored))
            self.args.decode_steps=authored_count
            with self.subTest(authored=authored_count),self.assertRaises(ValueError):
                panel.generate_accuracy_panel(self.args,tokenizer=None,model_factory=factory)
            if authored_count == 256:
                # Per-history256 is legal, but16x257 exceeds the existing4096
                # total. An exactly matching CLI must still fail preadmission.
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    parse_args(base+['--decode-steps','256'])
            self.assertFalse(self.args.output.exists())
        for authored_count in [1,2,6,255]:
            authored=json.loads(json.dumps(self.authored));authored['decode_steps']=authored_count
            self.panel_file.write_text(json.dumps(authored))
            parsed=parse_args(base+['--decode-steps',str(authored_count)])
            self.assertEqual(parsed.decode_steps,authored_count)
            self.args.decode_steps=authored_count+1
            with self.subTest(authored=authored_count),self.assertRaisesRegex(ValueError,'authored panel depth'):
                panel.generate_accuracy_panel(self.args,tokenizer=None,model_factory=factory)
            self.assertFalse(self.args.output.exists())
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                parse_args(base+['--decode-steps',str(authored_count+1)])
        factory.assert_not_called()
        self.panel_file.write_text(json.dumps(self.authored))
        for cli in ['0','-1','257','6']:
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                parse_args(base+['--decode-steps',cli])

    def test_shared_helper_default_full_capture_is_unchanged(self):
        """DefaultNone still captures every supplied stage and full prefill rows."""
        model = FakeReference()
        with contextlib.redirect_stdout(io.StringIO()):
            count, ids, teachers = run_prefill_and_decode(model, 'fixture', 2, self.directory/'all')
        self.assertEqual(count, 6)
        self.assertEqual(ids, [1, 2])
        self.assertEqual(len(teachers), 3)
        self.assertEqual(teachers, [2, 3, 4])  # Original greedy argmax sequence is unchanged.
        self.assertIsNone(model.calls[0]['past'])
        self.assertTrue(all(c['stages'] is None for c in model.calls))
        self.assertTrue(all(c['past'] is not None for c in model.calls[1:]))
        self.assertEqual(np.load(self.directory/'all/LM_HEAD.npy').shape, (1, 2, 8))

    def test_explicit_lm_head_final_row_and_cache_chain(self):
        """The optional path changes only observation extent/stages, not forwards."""
        model = FakeReference()
        with contextlib.redirect_stdout(io.StringIO()):
            count, _, _ = run_prefill_and_decode(model, 'fixture', 2, self.directory/'lm',
                capture_stages=[PipelineStage.LM_HEAD], snapshot_final_row_only=True)
        self.assertEqual(count, 3)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(np.load(self.directory/'lm/LM_HEAD.npy').shape, (1, 1, 8))
        self.assertEqual(len(list((self.directory/'lm').glob('*.npy'))), 3)
        self.assertIsNone(model.calls[0]['past'])
        self.assertTrue(all(c['past'] is not None for c in model.calls[1:]))

    def test_invalid_capture_or_anchor_stops_before_forward(self):
        """No implicit stage choice or changed original tokenization is accepted."""
        for kwargs in [dict(snapshot_final_row_only=True),
                       dict(capture_stages=['LM_HEAD']),
                       dict(capture_stages=[PipelineStage.LM_HEAD]*2),
                       dict(capture_stages=[PipelineStage.EMBEDDING], snapshot_final_row_only=True),
                       dict(expected_token_ids=[7])]:
            model = FakeReference()
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_prefill_and_decode(model, 'fixture', 2, self.directory/'bad', **kwargs)
            self.assertEqual(model.calls, [])

    def test_invalid_original_snapshot_planes_fail_without_trimming(self):
        """Missing/extra stages, malformed types and nonfinite planes are fatal."""
        for bank in [{}, {(PipelineStage.LM_HEAD, -1): np.zeros((1, 2, 8), dtype=np.float64)},
                     {(PipelineStage.LM_HEAD, -1): np.zeros((2, 8), dtype=np.float32)},
                     {(PipelineStage.LM_HEAD, -1): np.full((1, 2, 8), np.nan, dtype=np.float32)},
                     {(PipelineStage.LM_HEAD, -1): np.zeros((1, 2, 8), dtype=np.float32),
                      (PipelineStage.EMBEDDING, -1): np.zeros((1, 2, 8), dtype=np.float32)}]:
            model = FakeReference()
            with patch.object(model, 'get_snapshots', return_value=bank), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                run_prefill_and_decode(model, 'fixture', 2, self.directory/'bad-plane',
                    capture_stages=[PipelineStage.LM_HEAD], snapshot_final_row_only=True)
            self.assertEqual(len(model.calls), 1)

    def test_authored_panel_inventory_and_malformed_inputs(self):
        """The original failure and all distinct16 text histories are mandatory."""
        self.assertEqual(panel.load_prompt_panel(self.panel_file), self.authored)
        for mutation in ['short', 'duplicate-id', 'duplicate-text', 'unsafe', 'anchor']:
            value = json.loads(json.dumps(self.authored))
            if mutation == 'short': value['cases'].pop()
            if mutation == 'duplicate-id': value['cases'][1]['name'] = value['cases'][0]['name']
            if mutation == 'duplicate-text': value['cases'][1]['prompt'] = value['cases'][0]['prompt']
            if mutation == 'unsafe': value['cases'][1]['name'] = '../escape'
            if mutation == 'anchor': value['cases'][0]['expected_prompt_ids'][0] = 0
            self.panel_file.write_text(json.dumps(value))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                panel.load_prompt_panel(self.panel_file)

    def test_forward_failure_retires_once_without_complete_report(self):
        """A failed first case cannot masquerade as a completed panel."""
        model = self.fake_model(fail_forward=2)
        with self.assertRaisesRegex(RuntimeError, 'forward failure'):
            self.collect(model)
        self.assertEqual(model.closed, 1)
        self.assertFalse((self.args.output/'reference.json').exists())

    def test_close_failure_never_publishes_complete_report(self):
        """Completion is published after the sole model actually retires."""
        model = self.fake_model(close_failure=True)
        # Stop after one complete fake history so this test is about retirement,
        # not requiring a complete panel's synthetic token diversity.
        with patch.object(panel, 'load_prompt_panel', return_value=dict(decode_steps=2, cases=self.authored['cases'][:1])):
            with self.assertRaisesRegex(RuntimeError, 'close failure'):
                self.collect(model)
        self.assertEqual(model.closed, 1)
        self.assertFalse((self.args.output/'reference.json').exists())

    def test_source_owner_change_prevents_complete_publication(self):
        """A changed original metadata owner is rejected before report completion."""
        model = self.fake_model()
        sources = [dict(vocabulary=248320, owner='fixture-original'),
                   dict(vocabulary=248320, owner='fixture-replaced')]
        with patch.object(panel, 'load_prompt_panel', return_value=dict(decode_steps=2, cases=self.authored['cases'][:1])), \
                patch.object(panel, 'original_source', side_effect=sources), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'source owner changed'):
            panel.generate_accuracy_panel(self.args, tokenizer=None, model_factory=lambda *a, **k: model)
        self.assertEqual(model.closed, 1)
        self.assertFalse((self.args.output/'reference.json').exists())

    def test_panel_cli_inactive_predictor_and_ordinary_mtp_rules(self):
        """Panel inactive-source handling and ordinary MTP selection remain unchanged."""
        base = ['--model', 'fixture-main', '--output', str(self.args.output)]
        selected = ['--shared-predictor', 'fixture-predictor.gguf']
        panel_args = ['--accuracy-panel', str(self.panel_file), '--decode-steps', '2']
        inactive = parse_args(base + panel_args + selected)
        self.assertEqual(inactive.shared_predictor, Path('fixture-predictor.gguf'))
        self.assertFalse(inactive.mtp_sidecar_snapshots)
        self.assertIsNone(parse_args(base).shared_predictor)
        active = parse_args(base + selected + ['--mtp-sidecar-snapshots'])
        self.assertTrue(active.mtp_sidecar_snapshots)
        for arguments in [base + selected, base + ['--mtp-sidecar-snapshots'],
                          base + panel_args + selected + ['--mtp-sidecar-snapshots'],
                          base + panel_args + selected + ['--mtp-sidecar-only']]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(arguments)

    def test_dot_and_parent_component_rejected_for_names_and_groups(self):
        """ASCII punctuation support does not permit directory components."""
        for field in ['name', 'group']:
            for component in ['.', '..']:
                value = json.loads(json.dumps(self.authored))
                value['cases'][1][field] = component
                self.panel_file.write_text(json.dumps(value))
                with self.subTest(field=field, component=component), self.assertRaises(ValueError):
                    panel.load_prompt_panel(self.panel_file)

    def bind_fake_original_source(self, model):
        """Create four tiny fake file owners, never real model or predictor data."""
        paths = []
        for index in range(4):
            path = self.directory/f'fixture-main-{index}.gguf'
            path.write_bytes(b'fake source metadata fixture')
            paths.append(path)
        parser = SimpleNamespace(file_path=paths[0],
            _split_parsers=[SimpleNamespace(file_path=p) for p in paths[1:]],
            metadata={'general.architecture': 'qwen4exp'})
        model.weights = SimpleNamespace(parser=parser)
        model.hf_model = SimpleNamespace(config=SimpleNamespace(vocab_size=248320))
        predictor = self.directory/'fixture-inactive-predictor.gguf'
        predictor.write_bytes(b'fake inactive descriptor fixture')
        return predictor

    def test_inactive_predictor_stat_only_descriptor_and_source_digest(self):
        """No predictor payload open/parser is needed to bind its original owner."""
        from model_kld_corpus import metadata_digest
        model = self.fake_model()
        predictor = self.bind_fake_original_source(model)
        with patch.object(Path, 'open', side_effect=AssertionError('No payload opens')), \
                patch('python.reference.qwen4exp.NativeSliceWeights', side_effect=AssertionError('No predictor loader')):
            main = panel.original_source(model, None)
            paired = panel.original_source(model, None, predictor)
        self.assertNotIn('shared_predictor', main)
        descriptor = paired['shared_predictor']
        self.assertEqual(set(descriptor), {'name', 'path', 'bytes', 'object_identity'})
        self.assertEqual(descriptor['name'], predictor.name)
        self.assertEqual(descriptor['path'], str(predictor.resolve()))
        self.assertEqual(descriptor['bytes'], predictor.stat().st_size)
        self.assertEqual(json.loads(descriptor['object_identity']), panel._file_identity(predictor))
        self.assertNotEqual(metadata_digest(main), metadata_digest(paired))
        self.assertEqual(paired, panel.original_source(model, None, predictor))
        for absent in [self.directory/'missing-predictor', self.directory]:
            with self.assertRaises((ValueError, FileNotFoundError)):
                panel.original_source(model, None, absent)

    def test_selected_predictor_rechecked_before_complete_publication(self):
        """Replacing an inactive original owner cannot leave a complete reference."""
        model = self.fake_model()
        predictor = self.bind_fake_original_source(model)
        self.args.shared_predictor = predictor
        original_forward = model.forward
        def replace_selected_owner(*args, **kwargs):
            """Mutate only the tiny fake owner at a deterministic lifecycle cut."""
            result = original_forward(*args, **kwargs)
            if len(model.calls) == 1:
                predictor.write_bytes(b'replaced inactive source fixture with a different byte extent')
            return result
        with patch.object(panel, 'load_prompt_panel', return_value=dict(decode_steps=2, cases=self.authored['cases'][:1])), \
                patch.object(model, 'forward', side_effect=replace_selected_owner), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'source owner changed'):
            panel.generate_accuracy_panel(self.args, tokenizer=None, model_factory=lambda *a, **k: model)
        self.assertEqual(model.closed, 1)
        self.assertFalse((self.args.output/'reference.json').exists())

    def test_selected_predictor_bound_in_actual_completed_report(self):
        """The selected descriptor enters the report source digest before close."""
        from model_kld_corpus import metadata_digest
        model = self.fake_model()
        predictor = self.bind_fake_original_source(model)
        self.args.shared_predictor = predictor
        with patch.object(panel, 'load_prompt_panel', return_value=dict(decode_steps=2, cases=self.authored['cases'][:1])), \
                contextlib.redirect_stdout(io.StringIO()):
            result = panel.generate_accuracy_panel(self.args, tokenizer=None, model_factory=lambda *a, **k: model)
        self.assertEqual(model.closed, 1)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(result['source']['shared_predictor']['path'], str(predictor.resolve()))
        report = json.loads((self.args.output/'reference.json').read_text())
        self.assertEqual(report['source_digest'], metadata_digest(result['source']))
        self.assertFalse(self.args.__dict__.get('mtp_sidecar_snapshots', False))

    def test_fresh_output_guard_precedes_admission(self):
        """Existing reference artifacts are never overwritten by a panel attempt."""
        self.args.output.mkdir()
        factory = unittest.mock.Mock()
        with self.assertRaisesRegex(ValueError, 'fresh'):
            panel.generate_accuracy_panel(self.args, tokenizer=None, model_factory=factory)
        factory.assert_not_called()

    def frozen_fixture(self):
        """Reuse the shared corpus schema fixture with real tiny stat-only source owners."""
        from test_model_kld_corpus_v2 import manifest as corpus_fixture
        manifest=corpus_fixture()
        source=manifest['source']
        source['tokenizer_path']=str(self.directory/'tokenizer-fixture')
        for index,descriptor in enumerate(source['shards']):
            path=self.directory/f'main-{index}.gguf'
            path.write_bytes(b'synthetic-unit-not-model')
            actual=panel._file_identity(path)
            descriptor.update(path=str(path),name=path.name,bytes=actual['bytes'],
                object_identity=json.dumps(actual,sort_keys=True,separators=(',',':')))
        history=manifest['histories'][0]
        history.update(decode_rows=6,continuation_ids=[5,0,5,0,5,0,5])
        path=self.directory/'frozen-manifest.json'
        path.write_text(json.dumps(manifest))
        args=SimpleNamespace(frozen_history=path,history_id=history['id'],model=source['shards'][0]['path'],
            tokenizer=source['tokenizer_path'],prompt='The quick brown fox jumps over the lazy dog',
            decode_steps=6,shared_predictor=None,cache_bytes=0,output=self.directory/'frozen-out')
        return args,manifest,history

    def test_frozen_teachers_are_actual_inputs_all_stages_cache_and_pending(self):
        """Frozen non-greedy teachers drive six cached forwards; the seventh stays Pending."""
        model=FakeReference()
        teachers=[5,0,5,0,5,0,5]
        with contextlib.redirect_stdout(io.StringIO()):
            count,ids,actual=run_prefill_and_decode(model,'fixture',6,self.directory/'forced',
                expected_token_ids=[1,2],forced_decode_tokens=teachers)
        self.assertEqual(ids,[1,2]);self.assertEqual(actual,teachers);self.assertEqual(count,14)
        self.assertEqual([call['ids'] for call in model.calls],[[1,2]]+[[token] for token in teachers[:6]])
        self.assertEqual(len(model.calls),7)
        self.assertIsNone(model.calls[0]['past'])
        self.assertTrue(all(call['past'] is not None for call in model.calls[1:]))
        self.assertTrue(all(call['stages'] is None for call in model.calls))
        self.assertEqual(np.load(self.directory/'forced/LM_HEAD.npy').shape,(1,2,8))
        for offset in range(6):
            self.assertTrue((self.directory/f'forced/decode_step{offset}_EMBEDDING.npy').is_file())

    def test_frozen_teacher_geometry_rejects_before_prefill(self):
        """Missing, extra, malformed and out-of-vocabulary teacher IDs cannot execute."""
        for teachers in ([1]*6,[1]*8,[1]*6+[8],[1]*6+[-1],[1]*6+[True],tuple([1]*7)):
            model=FakeReference()
            with self.subTest(teachers=teachers),self.assertRaisesRegex(ValueError,r'D\+1'):
                run_prefill_and_decode(model,'fixture',6,self.directory/'invalid',forced_decode_tokens=teachers)
            self.assertFalse(model.calls);self.assertFalse((self.directory/'invalid').exists())

    def test_frozen_cli_selects_only_original_history_and_exact_work(self):
        """One actual manifest selects its exact source/depth, without a second panel inventory."""
        args,manifest,history=self.frozen_fixture()
        base=['--model',args.model,'--output',str(args.output),'--tokenizer',args.tokenizer,
              '--decode-steps','6','--frozen-history',str(args.frozen_history),'--history-id',history['id']]
        parsed=parse_args(base)
        self.assertEqual(load_frozen_history(parsed),(manifest,history))
        for suffix in (['--history-id','missing'],['--decode-steps','2'],['--model',manifest['source']['shards'][1]['path']],
                       ['--tokenizer','foreign'],['--mtp-sidecar-snapshots'],['--accuracy-panel',str(self.panel_file)]):
            with self.subTest(suffix=suffix),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                parse_args(base+suffix)
        args.output.mkdir()
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parse_args(base)

    def test_frozen_full_stage_writer_closes_once_before_metadata(self):
        """Actual cached helper writes all fixture stages; only retired source can publish metadata."""
        args,manifest,history=self.frozen_fixture()
        model=FakeReference(prompts={history['prompt']:history['prompt_ids']})
        def publish(*values,**kwargs):
            self.assertEqual(model.closed,1)
            self.assertEqual(values[4],history['prompt_ids'])
            self.assertEqual(values[6],history['continuation_ids'])
            self.assertIn('qwen4exp_teacher_policy: explicit_frozen_continuation_ids',kwargs['extra_metadata_lines'])
            (args.output/'metadata.txt').write_text('unit-only')
        with patch.object(panel,'original_source',return_value=manifest['source']), \
             patch('python.reference.generate_qwen4exp_pipeline_snapshots.write_metadata',side_effect=publish), \
             contextlib.redirect_stdout(io.StringIO()):
            count,ids,teachers=generate_frozen_history(args,tokenizer=None,model_factory=lambda *a,**k:model)
        self.assertEqual(count,14);self.assertEqual(teachers,history['continuation_ids'])
        self.assertEqual(model.closed,1);self.assertTrue((args.output/'metadata.txt').is_file())
        self.assertEqual([c['ids'] for c in model.calls[1:]],[[x] for x in history['continuation_ids'][:6]])
        self.assertTrue(all(c['stages'] is None for c in model.calls))

    def test_frozen_close_failure_never_publishes_complete_metadata(self):
        """Original retirement failure cannot leave a consumable FullStages reference pack."""
        args,manifest,history=self.frozen_fixture()
        model=FakeReference(prompts={history['prompt']:history['prompt_ids']},close_failure=True)
        with patch.object(panel,'original_source',return_value=manifest['source']), \
             patch('python.reference.generate_qwen4exp_pipeline_snapshots.write_metadata') as metadata, \
             contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'close failure'):
            generate_frozen_history(args,tokenizer=None,model_factory=lambda *a,**k:model)
        self.assertEqual(model.closed,1);metadata.assert_not_called()
        self.assertFalse((args.output/'metadata.txt').exists())

    def test_frozen_original_stat_replacement_rejects_before_model(self):
        """Same-size file replacement is checked through the original frozen stat fields."""
        args,manifest,history=self.frozen_fixture()
        path=Path(manifest['source']['shards'][0]['path'])
        replacement=path.with_suffix('.replacement');replacement.write_bytes(path.read_bytes());replacement.replace(path)
        factory=unittest.mock.Mock()
        with self.assertRaisesRegex(ValueError,'stat identity'):
            generate_frozen_history(args,tokenizer=None,model_factory=factory)
        factory.assert_not_called();self.assertFalse(args.output.exists())

    def test_frozen_manifest_change_withholds_metadata_after_retirement(self):
        """A changed terminal teacher cannot relabel checkpoints captured under prior inputs."""
        args,manifest,history=self.frozen_fixture()
        model=FakeReference(prompts={history['prompt']:history['prompt_ids']})
        original=model.forward
        def changed(*values,**kwargs):
            result=original(*values,**kwargs)
            if len(model.calls)==7:
                after=json.loads(args.frozen_history.read_text())
                after['histories'][0]['continuation_ids'][-1]=4
                args.frozen_history.write_text(json.dumps(after))
            return result
        model.forward=changed
        with patch.object(panel,'original_source',return_value=manifest['source']), \
             patch('python.reference.generate_qwen4exp_pipeline_snapshots.write_metadata') as metadata, \
             contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'changed during'):
            generate_frozen_history(args,tokenizer=None,model_factory=lambda *a,**k:model)
        self.assertEqual(model.closed,1);metadata.assert_not_called()



class QFNAccuracySubsetTests(unittest.TestCase):
    """Exact subset control tests; all tokenizers/models/source owners are fake.

    No N2047 logits bank or real model is allocated: long geometry is checked as
    metadata, while tiny prompts exercise the actual cached writer and files.
    """
    setUp = QFNAccuracyReferenceTests.setUp
    collect = QFNAccuracyReferenceTests.collect

    def subset(self, cases=None):
        """Build a deliberately synthetic exact-ID request using the actual pin."""
        from python.reference.qwen4exp import CHECKPOINT_REVISION
        return dict(schema=panel.SUBSET_SCHEMA, tokenizer_revision=CHECKPOINT_REVISION,
                    cases=cases if cases is not None else [
                        dict(name='first', group='boundary', prompt='fixture-a',
                             expected_prompt_ids=[1,2], decode_steps=1),
                        dict(name='second', group='boundary', prompt='fixture-b',
                             expected_prompt_ids=[2,3,4], decode_steps=3)])

    def select(self, request):
        """Replace only this temporary authored input, never a canonical panel."""
        self.panel_file.write_text(json.dumps(request))
        self.args.decode_steps = None

    def model(self, request, **kwargs):
        """Supply the exact fixture tokenizer and ordinary cached FakeReference."""
        return FakeReference({c['prompt']: c['expected_prompt_ids'] for c in request['cases']},
                             vocabulary=248320, **kwargs)

    def test_subset_per_history_depth_exact_rows_one_admission(self):
        """Two actual cached histories retain independent starts and D+1 teachers."""
        request=self.subset(); self.select(request)
        model=self.model(request); result=self.collect(model)
        self.assertEqual(model.closed,1); self.assertEqual(len(model.calls),6)
        self.assertIsNone(model.calls[0]['past']); self.assertIsNone(model.calls[2]['past'])
        self.assertTrue(all(model.calls[i]['past'] is not None for i in (1,3,4,5)))
        self.assertEqual([h['decode_rows'] for h in result['histories']],[1,3])
        report=json.loads((self.args.output/'reference.json').read_text())
        for authored,history,case in zip(request['cases'],result['histories'],report['cases']):
            depth=authored['decode_steps']; start=len(authored['expected_prompt_ids'])
            self.assertEqual(history['prompt_ids'],authored['expected_prompt_ids'])
            self.assertEqual(len(history['continuation_ids']),depth+1)
            self.assertEqual([r['consumed_tokens'] for r in case['rows']],list(range(start,start+depth+1)))
            self.assertEqual((self.args.output/case['logits_file']).stat().st_size,(depth+1)*248320*4)
            self.assertEqual(history['reference_scope'],'final_logits_only')
        self.assertEqual(result['authored_panel_digest'],panel.metadata_digest(request))
        self.assertEqual(result['approval'],'not_approved')

    def test_subset_2047_nine_decode_metadata_and_256_single_case_bounds(self):
        """One exact long input observes2047..2056; no tensor/model is allocated."""
        request=self.subset([dict(name='qsa-2047',group='qsa-threshold',prompt='synthetic-only',
            expected_prompt_ids=[1]*2047,decode_steps=9)])
        self.select(request); loaded=panel.load_prompt_panel(self.panel_file)
        from model_kld_corpus_v2 import expected_rows
        history=dict(id='qsa-2047',context_group='qsa-threshold',prompt_ids=loaded['cases'][0]['expected_prompt_ids'],
                     decode_rows=9,continuation_ids=[2]*10)
        self.assertEqual([r['consumed_tokens'] for r in expected_rows(dict(histories=[history]))],list(range(2047,2057)))
        request['cases'][0]['decode_steps']=256; self.select(request)
        self.assertEqual(panel.load_prompt_panel(self.panel_file)['cases'][0]['decode_steps'],256)

    def test_subset_pin_types_duplicates_and_total_bounds_before_factory(self):
        """Malformed metadata cannot admit a model or create an output owner."""
        base=self.subset()
        mutations=[]
        def changed(fn):
            """Collect one independent malformed request."""
            value=json.loads(json.dumps(base));fn(value);mutations.append(value)
        changed(lambda v:v.update(tokenizer_revision='foreign'))
        changed(lambda v:v.update(decode_steps=9))
        changed(lambda v:v.update(cases=[]))
        changed(lambda v:v['cases'][0].pop('expected_prompt_ids'))
        for value in [0,257,True,'9']:
            changed(lambda v,value=value:v['cases'][0].update(decode_steps=value))
        for value in [[],[True],[248320],[-1],[1.5]]:
            changed(lambda v,value=value:v['cases'][0].update(expected_prompt_ids=value))
        changed(lambda v:v['cases'][1].update(name=v['cases'][0]['name']))
        changed(lambda v:v['cases'][1].update(prompt=v['cases'][0]['prompt']))
        changed(lambda v:v['cases'][1].update(expected_prompt_ids=v['cases'][0]['expected_prompt_ids']))
        changed(lambda v:v['cases'][0].update(name='../outside'))
        changed(lambda v:v['cases'][0].update(group='x'*129))
        for count,depth in [(129,1),(16,256)]:
            changed(lambda v,count=count,depth=depth:v.update(cases=[dict(name=f'case-{i}',group='g',
                prompt=f'text-{i}',expected_prompt_ids=[i],decode_steps=depth) for i in range(count)]))
        factory=unittest.mock.Mock()
        for value in mutations:
            self.select(value)
            with self.subTest(request=value),self.assertRaises(ValueError):
                panel.generate_accuracy_panel(self.args,tokenizer=None,model_factory=factory)
            self.assertFalse(self.args.output.exists())
        factory.assert_not_called()

    def test_all_tokenizations_checked_before_any_forward_and_always_close(self):
        """A later case's wrong exact IDs fail before the first case's work."""
        request=self.subset(); self.select(request); model=self.model(request)
        model.prompts[request['cases'][1]['prompt']]=[7]
        with self.assertRaisesRegex(ValueError,'second'):
            self.collect(model)
        self.assertEqual(model.calls,[]); self.assertEqual(model.closed,1)
        self.assertFalse((self.args.output/'reference.json').exists())
        self.assertFalse((self.args.output/'reference-candidate.json').exists())

    def test_actual_helper_rechecks_tokenization_at_use_boundary(self):
        """Token drift after the prepass cannot reach a cached forward."""
        request=self.subset(); self.select(request); model=self.model(request)
        original=model.tokenizer; calls=0
        def drifting(prompt,return_tensors):
            """Return a different fixture row only after both prepass checks."""
            nonlocal calls
            calls+=1
            if calls==3:return {'input_ids':_Tokens([7])}
            return original(prompt,return_tensors)
        model.tokenizer=drifting
        with self.assertRaisesRegex(ValueError,'Authored prompt tokenization'):
            self.collect(model)
        self.assertEqual(model.calls,[]); self.assertEqual(model.closed,1)
        self.assertFalse((self.args.output/'reference.json').exists())

    def test_subset_failure_and_close_failure_withhold_complete_report(self):
        """Late forward and final retirement failures cannot publish completion."""
        request=self.subset()
        for failure in ['forward','close']:
            self.select(request)
            self.args.output=self.directory/f'failure-{failure}'
            model=self.model(request,fail_forward=4 if failure=='forward' else None,
                             close_failure=failure=='close')
            with self.subTest(failure=failure),self.assertRaisesRegex(RuntimeError,failure+' failure'):
                self.collect(model)
            self.assertEqual(model.closed,1)
            self.assertFalse((self.args.output/'reference.json').exists())
            self.assertFalse((self.args.output/'reference-candidate.json').exists())

    def test_subset_cli_has_no_ignored_global_depth_legacy_default_stays_three(self):
        """Per-history counts are explicit; ordinary/v1 default3 remains identical."""
        request=self.subset(); self.select(request)
        base=['--model','fixture-main','--output',str(self.args.output)]
        selected=base+['--accuracy-panel',str(self.panel_file)]
        actual=parse_args(selected)
        self.assertIsNone(actual.decode_steps)
        for extra in [['--decode-steps','1'],['--decode-steps','3'],['--decode-steps','9'],
                      ['--mtp-sidecar-snapshots','--shared-predictor','fixture-predictor'],
                      ['--prompt','other']]:
            with self.subTest(extra=extra),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                parse_args(selected+extra)
        self.assertEqual(parse_args(base).decode_steps,3)
        self.authored['decode_steps']=3;self.panel_file.write_text(json.dumps(self.authored))
        self.assertEqual(parse_args(selected).decode_steps,3)
        self.authored['decode_steps']=2;self.panel_file.write_text(json.dumps(self.authored))
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parse_args(selected)

    def test_legacy_and_subset_same_inputs_publish_identical_payloads_and_rows(self):
        """Only authored-request digest changes; cached arithmetic/files stay exact.

        This is a byte comparison of the actual helper/writer under a fake
        reference, not a golden real-model or cross-batch numerical claim.
        """
        legacy=json.loads(json.dumps(self.authored));legacy['decode_steps']=1
        self.panel_file.write_text(json.dumps(legacy));self.args.decode_steps=1
        ids={c['prompt']:(panel.ORIGINAL_IDS if i==0 else [i,i+1]) for i,c in enumerate(legacy['cases'])}
        old=self.collect(FakeReference(ids,vocabulary=248320))
        old_dir=self.args.output
        request=self.subset([dict(c,expected_prompt_ids=ids[c['prompt']],decode_steps=1) for c in legacy['cases']])
        self.select(request);self.args.output=self.directory/'subset'
        new=self.collect(self.model(request))
        self.assertEqual(old['histories'],new['histories'])
        old_report=json.loads((old_dir/'reference.json').read_text())
        new_report=json.loads((self.args.output/'reference.json').read_text())
        self.assertEqual(old_report,new_report)
        for case in old_report['cases']:
            name=case['logits_file']
            self.assertEqual((old_dir/name).read_bytes(),(self.args.output/name).read_bytes())
        self.assertNotEqual(old['authored_panel_digest'],new['authored_panel_digest'])

if __name__ == '__main__':
    unittest.main()
