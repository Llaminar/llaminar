"""Device-free v2 corpus contracts using real shared metrics and tiny FP32 files.

All source/config/native identity objects here are explicitly fixture metadata,
not real model, path, runtime or qualification evidence. No model/device loads.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'scripts/ci'))
import model_kld_corpus_v2 as corpus
from full_vocabulary_metrics import full_vocabulary_metrics
from model_kld_corpus import metadata_digest


def manifest():
    """Authored fixed miniature histories deliberately have unequal decode lengths."""
    config = {'fixture_only': True, 'canonical_projection': 'not-a-runtime-config'}
    return dict(schema=corpus.SCHEMA, metric=corpus.METRIC, temperature=1., vocabulary=6,
        source=dict(architecture='qwen4exp', main_layers=48, expert_count=512, vocabulary=6,
            reference_arithmetic='cpu_fp32', reference_execution='incremental_cached',
            tokenizer_revision='fixture-tokenizer', shards=[dict(name=f'main-{i}.gguf',
                path=f'/fixture/main-{i}.gguf', bytes=1, object_identity=f'fixture-object-{i}') for i in range(4)]),
        histories=[dict(id=name, context_group='short', prompt='Fixture '+name, prompt_ids=[0,1] if name=='a' else [1,0],
            continuation_ids=[2]*(count+1), decode_rows=count,
            reference_directory=name+'/reference', reference_scope='final_logits_only')
            for name,count in [('a',1),('b',4)]],
        topologies=[dict(id='fixture-cpu', backend='cpu', canonical_cell='FIXTURE_NOT_NATIVE',
            expected_config=config, runtime_config_digest=metadata_digest(config),
            arithmetic=dict(gdn_projection='fp16_moe_alpha_beta', moe_router='fp16',
                hc_up='single_q8', gdn_output='single_q8', fa_output='single_q8', gdn_qkv='single_q8'))],
        reset_control=dict(history_id='a', replay_id='reset'),
        reference=dict(report='reference.json', report_digest='f'*64, format='float32_le_row_major'))


def observation(m, *, replacements=None):
    """Use actual full-vocabulary math for every fixture row, never assumed metric values."""
    reference = [6.,5.,4.,3.,2.,1.]
    topology = m['topologies'][0]
    rows = []
    for metadata in corpus.expected_rows(m):
        values = (replacements or {}).get(metadata['row_id'], reference)
        metrics = full_vocabulary_metrics(reference, values, vocabulary=m['vocabulary'])
        rows.append(dict(metadata, vocabulary=m['vocabulary'], metrics=metrics))
    return dict(schema='llaminar.model-kld-metrics.v2', complete=True, metric=corpus.METRIC,
        corpus_manifest_digest=metadata_digest(m), source_digest=metadata_digest(m['source']),
        reference_report_digest=m['reference']['report_digest'], topology_id=topology['id'],
        backend=topology['backend'], runtime_config_digest=topology['runtime_config_digest'],
        arithmetic=topology['arithmetic'], native_identity={'fixture': 'not-native-evidence'},
        proofs={key: True for key in corpus.PROOFS}, rows=rows)


def write_rank_reports(root, m, document):
    """Write explicit non-native fixture receipts for the actual passive consumer."""
    topology=m['topologies'][0];world=topology['expected_config']['configuration']['mpi_procs']
    document['native_identity']['actual_source_descriptor_sha256']='fixture-original-source'
    document['distributed_observer']=dict(world_size=world,authority_rank=0,rank_reports=[f'rank-{rank}.json' for rank in range(world)])
    for rank in range(world):
        records=[dict(domain='forward_graph',name='fixture-only',count=1,value=1.)]
        if rank >= 0:
            records=[dict(domain='moe_canonical_publication',name='cpu_packed_route_gather_calls',
                          phase=phase,tags=dict(expert_compute='tensor_sharded',participant=str(rank),degree=str(world),root='0'),count=1,value=1.) for phase in ('prefill','decode')]
        report=dict(schema='llaminar.qfn-observer-rank.v1',rank=rank,world_size=world,authority_rank=0,
                    role='authority' if rank==0 else 'follower',complete=True,native_retired=True,
                    runtime_config_digest=topology['runtime_config_digest'],corpus_manifest_digest=metadata_digest(m),
                    actual_owner=dict(original_pma_owner=True,source_file_set_identity='fixture-original-source'),path_records=records)
        (root/f'rank-{rank}.json').write_text(json.dumps(report))


def approval(m, baseline):
    """Pin fixture metadata explicitly; this is not a published model baseline."""
    return dict(schema='llaminar.model-kld-baseline-approval.v1',
        corpus_manifest_digest=metadata_digest(m), baseline_report_digest=metadata_digest(baseline))


def calibration(m, baseline, approved):
    """Explicit toy test limits are local fixture data; production has no source defaults."""
    topology = m['topologies'][0]
    groups = []
    for context in sorted({h['context_group'] for h in m['histories']}):
        for phase in ('prefill','decode'):
            groups.append(dict(backend=topology['backend'], topology_id=topology['id'], context_group=context,
                phase=phase, tail_threshold_kl_nats=1.,
                maximum=dict(mean_kl_nats=100., step_mean_kl_nats=100., row_kl_nats=5.,
                    paired_mean_kl_delta_nats=100., row_kl_delta_nats=100., mean_total_variation=1., tail_fraction=1.),
                minimum=dict(top1_agreement=0., mean_top5_overlap=0.)))
    return dict(schema='llaminar.model-kld-calibration.v1', corpus_manifest_digest=metadata_digest(m),
        approved_baseline_digest=metadata_digest(baseline), baseline_approval_digest=metadata_digest(approved), groups=groups)


def absolute_pilot(m, *, diagnostic_contexts=()):
    """Explicit synthetic limits, never a QFN policy or reviewed numerical evidence."""
    topology=m['topologies'][0]; groups=[]
    for context in sorted({h['context_group'] for h in m['histories']}):
        for phase in ('prefill','decode'):
            item=dict(backend=topology['backend'],topology_id=topology['id'],context_group=context,
                      phase=phase,role='diagnostic' if context in diagnostic_contexts else 'required')
            if item['role']=='required':
                item.update(maximum=dict(mean_kl_nats=100.,step_mean_kl_nats=100.,row_kl_nats=100.,
                    mean_total_variation=1.,tail_fraction=1.),
                    minimum=dict(top1_agreement=0.,mean_top5_overlap=0.),tail_threshold_kl_nats=0.)
            groups.append(item)
    return dict(schema='llaminar.model-kld-absolute-pilot.v1',corpus_manifest_digest=metadata_digest(m),groups=groups)


class QFNAbsolutePilotTest(unittest.TestCase):
    """Pure pilot aggregation/binding controls with actual tiny vocabulary metrics."""

    def test_required_only_scope_diagnostic_unavailable_and_no_baseline_claim(self):
        m=manifest();m['histories'][1]['context_group']='boundary'
        candidate=observation(m,replacements={'b:decode:3':[-100.,0.,0.,0.,0.,0.]})
        policy=absolute_pilot(m,diagnostic_contexts=('boundary',))
        result=corpus.summarize(m,candidate,absolute_pilot=policy)
        self.assertEqual(result['verdict'],'GREEN');self.assertTrue(result['passed'])
        self.assertEqual(result['scope'],'selected_topology_required_groups_only')
        scope=result['acceptance_scope']
        self.assertEqual(scope['policy'],'initial_absolute_pilot')
        self.assertEqual(len(scope['required_groups']),2);self.assertEqual(len(scope['diagnostic_groups']),2)
        self.assertEqual(scope['diagnostic_verdict'],'UNAVAILABLE');self.assertFalse(scope['whole_corpus_accepted'])
        self.assertFalse(result['calibrated']);self.assertFalse(result['certification_eligible'])
        self.assertTrue(result['numerical_verdict_only']);self.assertFalse(result['paired_baseline_available'])
        for key in ('approved_baseline_digest','baseline_approval_digest','calibration_digest'):self.assertIsNone(result[key])
        self.assertEqual(result['absolute_pilot_digest'],metadata_digest(policy))
        self.assertEqual(len(result['rows']),len(candidate['rows']))
        for group in result['groups']:
            self.assertFalse(group['calibrated'])
            self.assertIsNone(group['paired_mean_kl_delta_nats']);self.assertIsNone(group['row_kl_delta_nats'])
            if group['context_group']=='boundary':
                self.assertEqual(group['acceptance_role'],'diagnostic');self.assertIsNone(group['passed'])
                self.assertIsNone(group['tail_fraction']);self.assertIsNone(group['tail_count'])
        for row in result['rows']:
            self.assertIsNone(row['paired_kl_delta_nats']);self.assertIsNone(row['baseline_kl_nats'])
        replay=next(r for r in result['diagnostic_replays'] if r['history_id']=='b')
        self.assertEqual(replay['selected_row_ids'],[r['row_id'] for r in candidate['rows'] if r['history_id']=='b'])
        self.assertTrue(replay['require_all_layers']);self.assertTrue(replay['requires_full_stage_reference'])

    def test_every_absolute_mean_offset_row_tail_tv_and_agreement_constraint_is_enforced(self):
        m=manifest();candidate=observation(m,replacements={'b:decode:3':[1.,6.,5.,4.,3.,2.]})
        for family,names in (('maximum',corpus.ABSOLUTE_MAXIMUM_LIMITS),('minimum',corpus.MINIMUM_LIMITS)):
            for name in names:
                with self.subTest(constraint=name):
                    policy=absolute_pilot(m)
                    group=next(g for g in policy['groups'] if g['phase']=='decode')
                    group[family][name]=0. if family=='maximum' else 1.
                    result=corpus.summarize(m,candidate,absolute_pilot=policy)
                    self.assertEqual(result['verdict'],'RED');self.assertFalse(result['passed'])
                    self.assertIn(name,[f['criterion'] for f in result['failures']])
                    self.assertTrue(any('b:decode:3' in r['selected_row_ids'] for r in result['diagnostic_replays']))

    def test_bad_late_offset_cannot_hide_in_prompt_balanced_decode_mean(self):
        m=manifest();candidate=observation(m,replacements={'b:decode:3':[1.,6.,5.,4.,3.,2.]})
        row=next(r for r in candidate['rows'] if r['row_id']=='b:decode:3')['metrics']['kl_nats']
        policy=absolute_pilot(m);group=next(g for g in policy['groups'] if g['phase']=='decode')
        group['maximum'].update(mean_kl_nats=row/4,step_mean_kl_nats=row/2)
        result=corpus.summarize(m,candidate,absolute_pilot=policy)
        decode=next(g for g in result['groups'] if g['phase']=='decode')
        self.assertLess(decode['mean_kl_nats'],group['maximum']['mean_kl_nats'])
        self.assertEqual(decode['per_step'][-1]['decode_offset'],3)
        self.assertEqual(result['failures'],[dict(backend='cpu',topology_id='fixture-cpu',context_group='short',
                                                phase='decode',criterion='step_mean_kl_nats')])

    def test_exact_inclusive_limits_and_prompt_weighted_tail(self):
        m=manifest();candidate=observation(m,replacements={'a:decode:0':[1.,6.,5.,4.,3.,2.]})
        policy=absolute_pilot(m);first=corpus.summarize(m,candidate,absolute_pilot=policy)
        for actual,declared in zip(first['groups'],policy['groups']):
            declared['maximum']={name:actual[name] for name in corpus.ABSOLUTE_MAXIMUM_LIMITS}
            declared['minimum']={name:actual[name] for name in corpus.MINIMUM_LIMITS}
        result=corpus.summarize(m,candidate,absolute_pilot=policy)
        self.assertTrue(result['passed'])
        decode=next(g for g in result['groups'] if g['phase']=='decode')
        self.assertEqual(decode['tail_count'],1);self.assertEqual(decode['tail_fraction'],.5)
        self.assertNotEqual(decode['tail_fraction'],1/5)
        # A row exactly at the explicit tail threshold is not in its >threshold tail.
        for declared in policy['groups']:
            declared['tail_threshold_kl_nats']=declared['maximum']['row_kl_nats']
        equal=corpus.summarize(m,candidate,absolute_pilot=policy)
        self.assertTrue(all(g['tail_count']==0 for g in equal['groups']))

    def test_exact_corpus_backend_topology_context_phase_and_closed_fields(self):
        m=manifest();candidate=observation(m)
        mutations=(lambda p:p.__setitem__('schema','future'),
            lambda p:p.__setitem__('corpus_manifest_digest','0'*64),
            lambda p:p['groups'].pop(),lambda p:p['groups'].append(copy.deepcopy(p['groups'][0])),
            lambda p:p['groups'][0].__setitem__('backend','rocm'),
            lambda p:p['groups'][0].__setitem__('topology_id','foreign'),
            lambda p:p['groups'][0].__setitem__('context_group','foreign'),
            lambda p:p['groups'][0].__setitem__('phase','foreign'),
            lambda p:p['groups'][0].__setitem__('phase',[]),
            lambda p:p['groups'][0].__setitem__('role','expanded'),
            lambda p:p['groups'][0].__setitem__('extra',0),lambda p:p.__setitem__('extra',0))
        for mutate in mutations:
            policy=absolute_pilot(m);mutate(policy)
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):
                corpus.summarize(m,candidate,absolute_pilot=policy)

    def test_no_empty_required_set_no_diagnostic_thresholds_no_mixed_approval_modes(self):
        m=manifest();candidate=observation(m)
        with self.assertRaisesRegex(ValueError,'at least one required'):
            corpus.summarize(m,candidate,absolute_pilot=absolute_pilot(m,diagnostic_contexts=('short',)))
        policy=absolute_pilot(m);policy['groups'][0]['role']='diagnostic'
        with self.assertRaisesRegex(ValueError,'group fields'):corpus.summarize(m,candidate,absolute_pilot=policy)
        for kwargs in ({'baseline':candidate},{'baseline_approval':{}},{'calibration':{}},
                       {'baseline':candidate,'baseline_approval':approval(m,candidate)}):
            with self.subTest(kwargs=kwargs),self.assertRaisesRegex(ValueError,'cannot use'):
                corpus.summarize(m,candidate,absolute_pilot=absolute_pilot(m),**kwargs)

    def test_numeric_defaults_deltas_and_malformed_limits_cannot_enter_pilot(self):
        m=manifest();candidate=observation(m)
        mutations=(lambda g:g['maximum'].pop('row_kl_nats'),lambda g:g['minimum'].pop('top1_agreement'),
            lambda g:g['maximum'].__setitem__('row_kl_delta_nats',0.),
            lambda g:g['maximum'].__setitem__('mean_kl_nats',True),
            lambda g:g['maximum'].__setitem__('row_kl_nats',-1.),
            lambda g:g['maximum'].__setitem__('mean_kl_nats',float('nan')),
            lambda g:g['maximum'].__setitem__('mean_total_variation',1.1),
            lambda g:g['maximum'].__setitem__('tail_fraction',1.1),
            lambda g:g['minimum'].__setitem__('top1_agreement',-1.),
            lambda g:g['minimum'].__setitem__('mean_top5_overlap',float('inf')),
            lambda g:g.__setitem__('tail_threshold_kl_nats',101.))
        for mutate in mutations:
            policy=absolute_pilot(m);mutate(policy['groups'][0])
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):
                corpus.summarize(m,candidate,absolute_pilot=policy)

    def test_diagnostic_role_cannot_hide_missing_nonfinite_or_wrong_runtime_proof(self):
        m=manifest();m['histories'][1]['context_group']='boundary'
        policy=absolute_pilot(m,diagnostic_contexts=('boundary',))
        for mutate in (lambda o:o['rows'].pop(),lambda o:o['rows'][-1]['metrics'].__setitem__('kl_nats',float('nan')),
                       lambda o:o['proofs'].__setitem__('native_retired',False),
                       lambda o:o.__setitem__('runtime_config_digest','foreign')):
            candidate=observation(m);mutate(candidate)
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):
                corpus.summarize(m,candidate,absolute_pilot=policy)


class QFNBoundedRowOutlierTest(unittest.TestCase):
    """Count-bounded row exceptions never remove observations from other gates."""
    def fixture(self, count=1):
        m=manifest()
        replacements={f"b:decode:{i}":[2.5,5.,4.,3.,2.,1.] for i in range(count)}
        candidate=observation(m,replacements=replacements)
        policy=absolute_pilot(m)
        group=next(g for g in policy['groups'] if g['phase']=='decode')
        actual=full_vocabulary_metrics([6.,5.,4.,3.,2.,1.],[2.5,5.,4.,3.,2.,1.],vocabulary=6)['kl_nats']
        self.assertGreater(actual,1.2);self.assertLess(actual,1.5)
        group['maximum']['row_kl_nats']=1.2
        group['bounded_row_outliers']=dict(maximum_rows=1,maximum_kl_nats=1.5)
        return m,candidate,policy,group

    def test_zero_one_two_and_upper_cap(self):
        for count in (0,1,2):
            m,c,p,g=self.fixture(count)
            result=corpus.summarize(m,c,absolute_pilot=p)
            group=next(x for x in result['groups'] if x['phase']=='decode')
            self.assertEqual(group['passed'],count<=1)
            self.assertEqual(group['bounded_row_outliers']['observed_count'],count)
            self.assertEqual(group['bounded_row_outliers']['observed_rows'],[f'b:decode:{i}' for i in range(count)])
            self.assertEqual(group['mean_kl_nats'],corpus._prompt_mean(
                [r for r in c['rows'] if r['phase']=='decode'],lambda r:r['metrics']['kl_nats']))
        m,c,p,g=self.fixture()
        c=observation(m,replacements={'b:decode:0':[0.,5.,4.,3.,2.,1.]})
        g['maximum']['row_kl_nats']=1.2;g['bounded_row_outliers']['maximum_kl_nats']=1.5
        self.assertGreater(next(r for r in c['rows'] if r['row_id']=='b:decode:0')['metrics']['kl_nats'],1.5)
        self.assertFalse(next(x for x in corpus.summarize(m,c,absolute_pilot=p)['groups'] if x['phase']=='decode')['passed'])

    def test_other_gates_and_default_are_unchanged(self):
        for section,name,bound in [('maximum','mean_kl_nats',0.),('maximum','step_mean_kl_nats',0.),
                ('maximum','tail_fraction',0.),('maximum','mean_total_variation',0.),
                ('minimum','top1_agreement',1.),('minimum','mean_top5_overlap',1.)]:
            with self.subTest(name=name):
                m,c,p,g=self.fixture();g[section][name]=bound
                if name=='mean_top5_overlap':
                    c=observation(m,replacements={'b:decode:0':[2.5,5.,4.,3.,2.,2.1]})
                    row=next(r for r in c['rows'] if r['row_id']=='b:decode:0')['metrics']
                    self.assertGreater(row['kl_nats'],1.2);self.assertLess(row['kl_nats'],1.5)
                    self.assertLess(row['top5_overlap'],1.)
                result=corpus.summarize(m,c,absolute_pilot=p)
                self.assertFalse(next(x for x in result['groups'] if x['phase']=='decode')['passed'])
                self.assertIn(name,[failure['criterion'] for failure in result['failures']])
        m,c,p,g=self.fixture();del g['bounded_row_outliers']
        group=next(x for x in corpus.summarize(m,c,absolute_pilot=p)['groups'] if x['phase']=='decode')
        self.assertFalse(group['passed']);self.assertNotIn('bounded_row_outliers',group)

    def test_paired_calibration_preserves_mean_and_delta_gates(self):
        m,c,_,_=self.fixture()
        baseline=observation(m);approved=approval(m,baseline)
        policy=calibration(m,baseline,approved)
        group=next(g for g in policy['groups'] if g['phase']=='decode')
        group['maximum']['row_kl_nats']=1.2
        group['bounded_row_outliers']=dict(maximum_rows=1,maximum_kl_nats=1.5)
        result=corpus.summarize(m,c,baseline,approved,calibration=policy)
        summary=next(g for g in result['groups'] if g['phase']=='decode')
        self.assertTrue(summary['passed'])
        self.assertEqual(summary['bounded_row_outliers']['observed_rows'],['b:decode:0'])
        self.assertGreater(summary['mean_kl_nats'],0.)
        self.assertEqual(summary['mean_kl_nats'],summary['paired_mean_kl_delta_nats'])
        for criterion in ('mean_kl_nats','paired_mean_kl_delta_nats','row_kl_delta_nats'):
            with self.subTest(criterion=criterion):
                changed=copy.deepcopy(policy)
                next(g for g in changed['groups'] if g['phase']=='decode')['maximum'][criterion]=0.
                failed=corpus.summarize(m,c,baseline,approved,calibration=changed)
                self.assertIn(criterion,[f['criterion'] for f in failed['failures']])
                self.assertFalse(next(g for g in failed['groups'] if g['phase']=='decode')['passed'])

    def test_invalid_policy_and_nonfinite_rows_reject(self):
        invalid=[None,{},dict(maximum_rows=1),dict(maximum_kl_nats=1.5),
            dict(maximum_rows=True,maximum_kl_nats=10.),dict(maximum_rows=1.5,maximum_kl_nats=10.),
            dict(maximum_rows=0,maximum_kl_nats=10.),dict(maximum_rows=-1,maximum_kl_nats=10.),
            dict(maximum_rows=99,maximum_kl_nats=10.),dict(maximum_rows=1,maximum_kl_nats=float('inf')),
            dict(maximum_rows=1,maximum_kl_nats=float('nan')),dict(maximum_rows=1,maximum_kl_nats=-1.),
            dict(maximum_rows=1,maximum_kl_nats=0.),dict(maximum_rows=1,maximum_kl_nats=10.,extra=1)]
        for value in invalid:
            with self.subTest(value=value):
                m,c,p,g=self.fixture();g['bounded_row_outliers']=value
                with self.assertRaises(ValueError):corpus.summarize(m,c,absolute_pilot=p)
        m,c,p,g=self.fixture();c['rows'][-1]['metrics']['kl_nats']=float('nan')
        with self.assertRaises(ValueError):corpus.summarize(m,c,absolute_pilot=p)


class QFNCorpusV2Test(unittest.TestCase):
    """Actual pure source/schema/aggregation and payload-consumer regressions."""

    def test_exact_pending_inventory_and_uncalibrated_never_green(self):
        m=manifest(); baseline=observation(m); candidate=observation(m); approved=approval(m,baseline)
        result=corpus.summarize(m,candidate,baseline,approved)
        self.assertEqual(result['verdict'],'UNCALIBRATED'); self.assertIsNone(result['passed'])
        self.assertEqual(len(result['rows']),7)
        self.assertEqual([r['consumed_tokens'] for r in result['rows'][:2]],[2,3])
        self.assertEqual(result['rows'][1]['input_token'],2)
        replay=result['diagnostic_replays'][0]
        self.assertEqual(replay['prompt'],m['histories'][0]['prompt'])
        self.assertEqual(replay['reference_directory'],'a/reference')
        self.assertTrue(replay['require_all_layers']); self.assertTrue(replay['requires_full_stage_reference'])
        self.assertEqual(replay['continuation_ids'],m['histories'][0]['continuation_ids'])

    def test_first_collection_has_real_means_without_baseline_or_approval(self):
        m=manifest(); candidate=observation(m,replacements={'a:prefill':[1.,6.,5.,4.,3.,2.]})
        collection=corpus.summarize(m,candidate)
        self.assertEqual(collection['verdict'],'UNCALIBRATED'); self.assertIsNone(collection['passed'])
        self.assertFalse(collection['paired_baseline_available'])
        self.assertIsNone(collection['approved_baseline_digest']); self.assertIsNone(collection['baseline_approval_digest'])
        self.assertTrue(all(r['paired_kl_delta_nats'] is None and r['baseline_kl_nats'] is None for r in collection['rows']))
        paired=corpus.summarize(m,candidate,observation(m),approval(m,observation(m)))
        for actual, previous in zip(collection['groups'],paired['groups']):
            for key in ('mean_kl_nats','mean_total_variation','top1_agreement','mean_top5_overlap',
                        'median_row_kl_nats','p95_row_kl_nats','row_kl_nats','per_step'):
                self.assertEqual(actual[key],previous[key])
            self.assertIsNone(actual['paired_mean_kl_delta_nats']); self.assertIsNone(actual['row_kl_delta_nats'])
        for kwargs in ({'baseline':candidate},{'baseline_approval':{}},{'calibration':{}}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError): corpus.summarize(m,candidate,**kwargs)

    def test_prompt_weighting_and_exact_paired_deltas(self):
        m=manifest(); baseline=observation(m)
        changes={r['row_id']: [1.,6.,5.,4.,3.,2.] if r['history_id']=='a' else [5.,6.,4.,3.,2.,1.]
                 for r in baseline['rows']}
        candidate=observation(m,replacements=changes)
        result=corpus.summarize(m,candidate,baseline,approval(m,baseline))
        decode=next(g for g in result['groups'] if g['phase']=='decode')
        a=candidate['rows'][1]['metrics']['kl_nats']; b=candidate['rows'][3]['metrics']['kl_nats']
        self.assertAlmostEqual(decode['mean_kl_nats'],(a+b)/2)
        self.assertNotAlmostEqual(decode['mean_kl_nats'],(a+4*b)/5)
        self.assertAlmostEqual(decode['paired_mean_kl_delta_nats'],decode['mean_kl_nats'])
        self.assertEqual([s['prompt_count'] for s in decode['per_step']],[2,1,1,1])
        self.assertEqual(decode['row_count'],5)

    def test_calibration_required_complete_and_baseline_approved(self):
        m=manifest(); base=observation(m); approved=approval(m,base); policy=calibration(m,base,approved)
        self.assertEqual(corpus.summarize(m,base,base,approved,calibration=policy)['verdict'],'GREEN')
        for mutate in (lambda p:p['groups'].pop(), lambda p:p['groups'].append(copy.deepcopy(p['groups'][0])),
                       lambda p:p['groups'][0]['maximum'].pop('row_kl_nats'),
                       lambda p:p['groups'][0].__setitem__('backend','foreign')):
            bad=copy.deepcopy(policy); mutate(bad)
            with self.assertRaises(ValueError): corpus.summarize(m,base,base,approved,calibration=bad)
        wrong=dict(approved,baseline_report_digest='0'*64)
        with self.assertRaises(ValueError): corpus.summarize(m,base,base,wrong)

    def test_single_row_catastrophe_cannot_hide_in_mean(self):
        m=manifest(); base=observation(m); approved=approval(m,base)
        candidate=observation(m,replacements={'b:decode:3':[-100.,0.,0.,0.,0.,0.]})
        result=corpus.summarize(m,candidate,base,approved,calibration=calibration(m,base,approved))
        self.assertEqual(result['verdict'],'RED'); self.assertFalse(result['passed'])
        group=next(g for g in result['groups'] if g['phase']=='decode')
        self.assertLess(group['mean_kl_nats'],100.)
        self.assertEqual(group['tail_count'],1); self.assertGreater(group['mean_tail_excess_kl_nats'],0.)
        self.assertIn('row_kl_nats',[f['criterion'] for f in result['failures']])
        self.assertTrue(any('b:decode:3' in r['selected_row_ids'] for r in result['diagnostic_replays']))

    def test_top1_is_not_reference_top1_in_candidate_top5(self):
        m=manifest(); base=observation(m)
        candidate=observation(m,replacements={'a:decode:0':[4.,6.,5.,3.,2.,1.]})
        actual=candidate['rows'][1]['metrics']
        self.assertFalse(actual['top1_same']); self.assertEqual(actual['top5_overlap'],1.)
        result=corpus.summarize(m,candidate,base,approval(m,base))
        decode=next(g for g in result['groups'] if g['phase']=='decode')
        self.assertEqual(decode['top1_agreement'],.5); self.assertEqual(decode['mean_top5_overlap'],1.)

    def test_missing_duplicate_nonfinite_mismatched_and_wrong_identity_reject(self):
        m=manifest(); base=observation(m); approved=approval(m,base)
        mutations=(lambda r:r['rows'].pop(), lambda r:r['rows'].__setitem__(1,copy.deepcopy(r['rows'][0])),
            lambda r:r['rows'][1].__setitem__('consumed_tokens',999),
            lambda r:r['rows'][0].pop('decode_offset'), lambda r:r['rows'][0].__setitem__('vocabulary',5),
            lambda r:r['rows'][0]['metrics'].__setitem__('kl_nats',float('nan')),
            lambda r:r['rows'][0]['metrics'].__setitem__('reference_entropy_nats',float('inf')),
            lambda r:r.__setitem__('backend','foreign'), lambda r:r.__setitem__('runtime_config_digest','foreign'),
            lambda r:r['proofs'].__setitem__('publication_authenticated',False),
            lambda r:r['rows'][0]['metrics'].__setitem__('top5_overlap',.7))
        for mutate in mutations:
            bad=copy.deepcopy(base); mutate(bad)
            with self.subTest(mutation=mutate):
                with self.assertRaises(ValueError): corpus.summarize(m,bad,base,approved)
        for field in ('prompt_ids','continuation_ids'):
            bad=copy.deepcopy(m); bad['histories'][0][field][0]=True
            with self.assertRaises(ValueError): corpus.validate_manifest(bad)
        bad=copy.deepcopy(m); bad['histories'][0]['continuation_ids'].pop()
        with self.assertRaises(ValueError): corpus.validate_manifest(bad)

    def test_context_phase_and_backend_are_never_pooled(self):
        m=manifest(); m['histories'][1]['context_group']='long'
        base=observation(m); result=corpus.summarize(m,base,base,approval(m,base))
        self.assertEqual({(g['context_group'],g['phase']) for g in result['groups']},
                         {('short','prefill'),('short','decode'),('long','prefill'),('long','decode')})
        self.assertFalse(result['acceptance_phase_or_backend_pooling'])
        wrong=copy.deepcopy(base); wrong['topology_id']='foreign'
        with self.assertRaises(ValueError): corpus.summarize(m,wrong,base,approval(m,base))

    def test_bounded_ascii_identifiers_duplicate_prompts_and_inactive_paired_source(self):
        for name in ('.','..','nonascii-é','x'*129):
            m=manifest(); m['histories'][0]['id']=name
            with self.assertRaises(ValueError): corpus.validate_manifest(m)
        m=manifest(); m['histories'][1]['prompt_ids']=m['histories'][0]['prompt_ids']
        with self.assertRaisesRegex(ValueError,'Duplicate original prompt'): corpus.validate_manifest(m)
        m=manifest(); m['histories'][0]['decode_rows']=257
        m['histories'][0]['continuation_ids']=[2]*258
        with self.assertRaises(ValueError): corpus.validate_manifest(m)
        m=manifest(); m['source']['shared_predictor']=dict(path='/fixture/mtp.gguf',name='mtp.gguf',bytes=1,
                                                        object_identity='fixture-inactive-pair')
        with self.assertRaisesRegex(ValueError,'descriptor/configuration'): corpus.validate_manifest(m)
        config=m['topologies'][0]['expected_config']; config['configuration']={'shared_predictor':'/fixture/mtp.gguf'}
        m['topologies'][0]['runtime_config_digest']=metadata_digest(config)
        self.assertEqual(corpus.validate_manifest(m),m)
        m['source']['shared_predictor']['path']='/fixture/foreign.gguf'
        with self.assertRaises(ValueError): corpus.validate_manifest(m)

    def _replay_real_tiny_files_full_extent_reset_and_nonfinite_boundaries(self, distributed=False, external_manifest=False):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'reference';root.mkdir(); m=manifest(); refs=[]; native=[]
            if distributed:
                m['topologies'][0]['expected_config']['configuration']={'mpi_procs':2,
                    'moe_routed_expert_plan':{'domains':[{'routed_compute_policy':'TensorSharded',
                        'participants':[{'device_type':'CPU'},{'device_type':'CPU'}],'world_ranks':[0,1]}]}}
                m['topologies'][0]['runtime_config_digest']=metadata_digest(m['topologies'][0]['expected_config'])
            for h in m['histories']:
                rows=[dict(r,vocabulary=6) for r in corpus.expected_rows(m) if r['history_id']==h['id']]
                values=np.tile(np.array([6,5,4,3,2,1],dtype='<f4'),(len(rows),1))
                values.tofile(root/(h['id']+'-ref.f32')); values.tofile(root/(h['id']+'-native.f32'))
                refs.append(dict(history_id=h['id'],rows=rows,logits_file=h['id']+'-ref.f32'))
                native.append(dict(history_id=h['id'],rows=rows,logits_file=str(root/(h['id']+'-native.f32'))))
            reference=dict(schema='llaminar.model-kld-reference.v2',complete=True,
                source_digest=metadata_digest(m['source']),workload_digest=corpus._workload_digest(m),cases=refs)
            m['reference']['report_digest']=metadata_digest(reference)
            (root/'reference.json').write_text(json.dumps(reference)); (root/'manifest.json').write_text(json.dumps(m))
            catalog=root/'approval.json'; catalog.write_text(json.dumps(dict(schema=2,manifest_digest=metadata_digest(m))))
            explicit=None
            if external_manifest:
                explicit=Path(tmp)/'selected-manifest.json';explicit.write_text(json.dumps(m))
                baseline=corpus.load_corpus(root,catalog=catalog)
                with self.assertRaises(OSError):corpus.load_corpus(root,catalog=catalog,manifest_path=Path(tmp)/'absent.json')
                wrong=Path(tmp)/'wrong-catalog.json';wrong.write_text(json.dumps(dict(schema=2,manifest_digest='0'*64)))
                with self.assertRaisesRegex(ValueError,'Unapproved'):corpus.load_corpus(root,catalog=wrong,manifest_path=explicit)
                changed=copy.deepcopy(m);changed['histories'][0]['prompt_ids']=[2,1]
                explicit.write_text(json.dumps(changed));wrong.write_text(json.dumps(dict(schema=2,manifest_digest=metadata_digest(changed))))
                with self.assertRaisesRegex(ValueError,'reference'):corpus.load_corpus(root,catalog=wrong,manifest_path=explicit)
                changed=copy.deepcopy(m);changed['source']['tokenizer_revision']='foreign-reference'
                explicit.write_text(json.dumps(changed));wrong.write_text(json.dumps(dict(schema=2,manifest_digest=metadata_digest(changed))))
                with self.assertRaisesRegex(ValueError,'reference'):corpus.load_corpus(root,catalog=wrong,manifest_path=explicit)
                for escape in ('../reference.json',str(root/'reference.json')):
                    changed=copy.deepcopy(m);changed['reference']['report']=escape
                    explicit.write_text(json.dumps(changed));wrong.write_text(json.dumps(dict(schema=2,manifest_digest=metadata_digest(changed))))
                    with self.assertRaises(ValueError):corpus.load_corpus(root,catalog=wrong,manifest_path=explicit)
                explicit.write_text(json.dumps(m))
            loaded=corpus.load_corpus(root,catalog=catalog,manifest_path=explicit)
            if external_manifest:
                self.assertEqual(loaded['reference_payloads'],baseline['reference_payloads'])
                self.assertIn(str(explicit),loaded['inputs'])
            report=observation(m); report.pop('rows'); report.pop('metric'); report['schema']='llaminar.model-kld-observation.v2'
            reset_history=dict(m['histories'][0],id='reset')
            reset_rows=[dict(r,vocabulary=6) for r in corpus.expected_rows(dict(m,histories=[reset_history]))]
            (root/'reset.f32').write_bytes((root/'a-native.f32').read_bytes())
            report.update(cases=native,reset_case=dict(history_id='reset',rows=reset_rows,logits_file=str(root/'reset.f32')))
            if distributed: write_rank_reports(root,m,report)
            path=root/'native.json'; path.write_text(json.dumps(report))
            bad=copy.deepcopy(report); bad['source_digest']='foreign'; path.write_text(json.dumps(bad))
            with patch.object(corpus,'full_vocabulary_metrics',side_effect=AssertionError('invalid header reached payload math')):
                with self.assertRaisesRegex(ValueError,'identity'): corpus.score_observation(loaded,path)
            path.write_text(json.dumps(report))
            scored=corpus.score_observation(loaded,path)
            self.assertEqual(len(scored['rows']),7); self.assertNotIn('passed',scored)
            if external_manifest:
                explicit.write_text(json.dumps(m,indent=2)) # Even equivalent metadata must keep its current-run file identity.
                with self.assertRaises(ValueError):corpus.score_observation(loaded,path)
                loaded=corpus.load_corpus(root,catalog=catalog,manifest_path=explicit)
                self.assertEqual(len(corpus.score_observation(loaded,path)['rows']),7)
            if distributed:
                authority=root/'rank-0.json';authority_original=authority.read_text()
                incomplete=json.loads(authority_original);incomplete['path_records'].pop()
                authority.write_text(json.dumps(incomplete))
                with self.assertRaisesRegex(ValueError,'gather phase'):corpus.score_observation(loaded,path)
                authority.write_text(authority_original)
                receipt=root/'rank-1.json';original_receipt=receipt.read_text();receipt.unlink()
                with self.assertRaises((ValueError,OSError)):corpus.score_observation(loaded,path)
                receipt.write_text(original_receipt)
                for mutation in ('retired','source','phase','rank','config','role','participant','degree'):
                    bad=json.loads(original_receipt)
                    if mutation=='retired':bad['native_retired']=False
                    elif mutation=='source':bad['actual_owner']['source_file_set_identity']='foreign'
                    elif mutation=='phase':bad['path_records'].pop()
                    elif mutation=='rank':bad['rank']=0
                    elif mutation=='config':bad['runtime_config_digest']='foreign'
                    elif mutation=='role':bad['role']='authority'
                    elif mutation=='participant':bad['path_records'][0]['tags']['participant']='0'
                    elif mutation=='degree':bad['path_records'][0]['tags']['degree']='3'
                    receipt.write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):corpus.score_observation(loaded,path)
                receipt.write_text(original_receipt)

            original=(root/'b-native.f32').read_bytes()
            (root/'b-native.f32').write_bytes(original[:-4])
            with self.assertRaisesRegex(ValueError,'Truncated'): corpus.score_observation(loaded,path)
            (root/'b-native.f32').write_bytes(original)
            changed=np.frombuffer(original,dtype='<f4').copy(); changed[-1]=np.nan; changed.tofile(root/'b-native.f32')
            with self.assertRaises(ValueError): corpus.score_observation(loaded,path)
            (root/'b-native.f32').write_bytes(original)
            reset=np.fromfile(root/'reset.f32',dtype='<f4'); reset[0]+=1; reset.tofile(root/'reset.f32')
            with self.assertRaisesRegex(ValueError,'reset'): corpus.score_observation(loaded,path)
            (root/'reset.f32').write_bytes((root/'a-native.f32').read_bytes())
            ref=np.fromfile(root/'b-ref.f32',dtype='<f4'); ref[-1]=np.inf; ref.tofile(root/'b-ref.f32')
            refreshed=corpus.load_corpus(root,catalog=catalog,manifest_path=explicit)
            with self.assertRaises(ValueError): corpus.score_observation(refreshed,path)

    def test_real_tiny_files_full_extent_reset_and_nonfinite_boundaries(self):
        self._replay_real_tiny_files_full_extent_reset_and_nonfinite_boundaries()

    def test_distributed_rank_source_path_and_retirement_reports(self):
        self._replay_real_tiny_files_full_extent_reset_and_nonfinite_boundaries(distributed=True)

    def test_explicit_metadata_manifest_preserves_original_reference_root(self):
        self._replay_real_tiny_files_full_extent_reset_and_nonfinite_boundaries(distributed=True,external_manifest=True)

    def test_legacy_v1_policy_still_owns_its_original_inventory(self):
        from model_kld_canary import CANARIES,definition,evaluate_rows
        self.assertEqual(set(CANARIES),{('dense','cuda'),('dense','rocm'),('moe','cuda'),('moe','rocm')})
        p=definition('moe','rocm')
        rows=[dict(consumed_tokens=p.prompt_tokens+i,kl_nats=0.) for i in range(129)]
        self.assertTrue(evaluate_rows(rows,model='moe',backend='rocm')['passed'])
        self.assertEqual(corpus.SCHEMA,'llaminar.model-kld-corpus.v2')


if __name__=='__main__': unittest.main()
