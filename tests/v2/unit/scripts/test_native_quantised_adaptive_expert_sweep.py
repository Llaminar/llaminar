#!/usr/bin/env python3
"""Device-free admission of adaptive expert clocks and additive point scopes.

Synthetic observations prove timing grades and original receipt identities;
these tests make no claim of native timing, physical placement or model speed.
"""
from contextlib import redirect_stderr
from copy import deepcopy
import io
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT / "tests/v2/performance/kernels"))

from native_quantised_dispatch import cpu_exact_sweep as sweep
from native_quantised_dispatch.adaptive_tournament import validate_fused_tournament, is_promotion_grade
from native_quantised_dispatch.adapters.evidence import native_double_digest, summarize_sorted_timing
from native_quantised_dispatch.corpus import ObservationCorpus
import test_cpu_execution_exact as fixture


def target_plan(**changes):
    """Select roles through the shared model/TP catalog, with no numeric shapes."""
    values=dict(expert_only=True, expert_scope=dict(model_family='qwen4exp',tp_degree=2,
        layout='intermediate_shard',isa=('AVX512','AVX512'),purposes=('decode','mtp_verifier','prefill')),
        expert_role_formats=(('gate_up','Q4_K'),('down','Q5_1')),
        prefill_row_policy='two-anchor',tournament='adaptive_v1')
    values.update(changes)
    return sweep.build_plan(28,**values)


def adaptive_payload(candidate='Pairwise', *, screened=False):
    """Upgrade the existing native fixture to a complete synthetic v2 contest."""
    p=fixture.CPUExecutionNativeEvidenceTest.payload();key=list(p['exact_key']);p['exact_key']=key
    p['descriptor_policies']=[int(candidate=='WideRows')]*len(p['descriptor_keys'])
    p.update(schema='cpu_execution_exact_native_v2',candidate=candidate,holdout=False,
        timing_stage='screening' if screened else 'final',promotion_grade=not screened,
        finalist_reason='screened_out' if screened else 'screen_winner|installed_baseline',
        screen_samples_us=[10.,11.,12.],screen_repeat_count=1,
        configured_full_sample_count=30,configured_full_repeat_count=8)
    decisions=[]
    for name in ('Pairwise','WideRows'):
        final=name=='Pairwise'
        decisions.append(dict(candidate_id=name,correctness_pass=True,
            screen_samples_us=[10.,11.,12.],screen_sorted_native_digest=native_double_digest([10.,11.,12.]),
            timing_stage='final' if final else 'screening',promotion_grade=final,
            finalist_reason='screen_winner|installed_baseline' if final else 'screened_out',
            final_sample_count=30 if final else 0))
    p['tournament']=dict(schema='native_quantised_tournament_v1',backend='cpu',source_format=p['source_format'],
        shape=p['shape'],purpose=p['purpose'],m=key[9],n=key[10],k=key[12],threads=key[8],
        build_isa=key[6],runtime_isa=key[7],holdout=False,expert_rows=p['expert_rows'],
        target_descriptor=0,exact_key=key,candidate_ids=['Pairwise','WideRows'],finalist_candidate_ids=['Pairwise'],
        baseline_candidate_id='Pairwise',winner_candidate_id='Pairwise',baseline_auto_calls=1,
        baseline_auto_route_records=[dict(count=2,tags=dict(m=str(key[9]),n=str(key[10]),policy_n=str(key[11]),
            k=str(key[12]),codebook=str(key[2]),purpose=p['purpose'],isa='AVX512',
            requested_verifier_policy='Auto',effective_verifier_policy='Pairwise'))],candidates=decisions)
    samples=p['screen_samples_us'] if screened else [10.,11.,12.]*10
    summary=summarize_sorted_timing(tuple(sorted(samples)))
    p.update(raw_samples_us=samples,sorted_samples_us=sorted(samples))
    p['timing']=dict(sample_count=len(samples),repeat_count=1 if screened else 8,warmup_count=0 if screened else 5,
        min_us=summary.minimum,median_us=summary.median,p95_us=summary.p95,mad_us=summary.mad,cv=summary.cv,
        sorted_native_digest=summary.digest)
    p['correctness'].update(guard_replays=4,repeated_replay=True)
    p['counters'].update(projection_observations=0,expected_projection_observations=0,
        sample_observation_basis='timed_calls_collection_suppressed')
    return p


class AdaptiveExpertScopeTest(unittest.TestCase):
    """The target is additive and old broad receipts keep their original scope."""
    def test_two_actual_formats_use_catalog_only_and_publish_points(self):
        plan=target_plan();jobs=list(sweep.iter_jobs(plan))
        self.assertEqual(len(jobs),2);self.assertEqual(plan['broader_expert_job_count'],28)
        self.assertFalse(plan['broader_inventory_in_this_transaction']);self.assertFalse(plan['legacy_shapes'])
        self.assertEqual({(j['role'],j['source_format'],j['n'],j['k']) for j in jobs},
            {('gate_up','Q4_K',320,2560),('down','Q5_1',2560,320)})
        for job in jobs:
            self.assertEqual({r['purpose'] for r in job['requests']},{'decode','mtp_verifier','prefill'})
        self.assertTrue({128,256,512,1024}.issubset(plan['expert_prefill_m']))
        self.assertIn('no interval',plan['prefill_publication'])

    def test_new_choice_opt_in_preserves_historical_plan_and_scoped_native_requests(self):
        old=target_plan(tournament='full')
        self.assertNotIn('execution_exact_choice_abi',old)
        self.assertFalse(any(r['candidate']=='OrderedThreeRows' for r in sweep.iter_requests(old)))
        scope=dict(model_family='qwen4exp',tp_degree=2,layout='intermediate_shard',
            isa=('AVX512','AVX512'),purposes=('prefill',))
        fresh=target_plan(tournament='full',expert_scope=scope,expert_m=(64,),expert_ordered_three=True)
        self.assertEqual(fresh['execution_exact_choice_abi'],2)
        jobs=list(sweep.iter_jobs(fresh));self.assertEqual(len(jobs),2)
        for job in jobs:
            expected={'Pairwise','WideRows','OrderedThreeRows'} if job['role']=='gate_up' else {'Pairwise','WideRows'}
            self.assertEqual({r['candidate'] for r in job['requests']},expected)
            self.assertEqual({r['holdout'] for r in job['requests']},{False,True})
        with self.assertRaises(ValueError):target_plan(expert_ordered_three=True,
            expert_scope={**scope,'purposes':('mtp_verifier',)})
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            args=SimpleNamespace(fused_avx512_bin=directory/'build/tests/v2/trainer')
            sweep._command(args,fresh,jobs[0],directory)
            native=json.loads((directory/'requests.json').read_text())
            self.assertEqual(native['execution_exact_choice_abi'],2)
            self.assertEqual(native['schema'],'cpu_execution_exact_batch_v1')
        self.assertNotEqual(old['plan_digest'],fresh['plan_digest'])

    def test_actual_m_projection_keeps_native_train_and_mixed_holdouts(self):
        scope=dict(model_family='qwen4exp',tp_degree=2,layout='intermediate_shard',
            isa=('AVX512','AVX512'),purposes=('mtp_verifier',))
        plan=target_plan(expert_scope=scope,expert_m=(3,),tournament='full')
        jobs=list(sweep.iter_jobs(plan));requests=list(sweep.iter_requests(plan))
        self.assertEqual(plan['expert_m'],(3,));self.assertEqual(plan['total_jobs'],2)
        self.assertEqual(plan['total_logical_requests'],8);self.assertEqual(len(requests),8)
        self.assertEqual({r['m'] for r in requests},{3})
        self.assertEqual({r['purpose'] for r in requests},{'mtp_verifier'})
        for job in jobs:
            self.assertEqual({(r['candidate'],r['holdout']) for r in job['requests']},
                {('Pairwise',False),('Pairwise',True),('WideRows',False),('WideRows',True)})
            expected=[3,3,3] if job['role']=='gate_up' else [3,3]
            for request in job['requests']:
                self.assertEqual(request['expert_rows'],expected+([1] if request['holdout'] else []))
        broad=target_plan(expert_scope=scope,tournament='full')
        self.assertNotIn('expert_m',broad)
        self.assertTrue({sweep.job_id(j) for j in jobs}.isdisjoint(sweep.job_id(j) for j in sweep.iter_jobs(broad)))
        # Request identities remain reusable when their actual execution inputs
        # match; narrowing a prepared batch never relabels the old completion.
        self.assertTrue({sweep.job_id(r) for r in requests}.issubset(
            sweep.job_id(r) for r in sweep.iter_requests(broad)))

    def test_actual_m_projection_rejects_unadmitted_or_conflicting_scope(self):
        for values in ((0,),(1,),(-3,),(3,3),(17,),(3.0,),(True,)):
            with self.subTest(values=values),self.assertRaises(ValueError):target_plan(expert_m=values)
        with self.assertRaises(ValueError):sweep.build_plan(28,expert_m=(3,))
        with self.assertRaises(ValueError):target_plan(expert_m=(128,))
        prefill_scope=dict(model_family='qwen4exp',tp_degree=2,layout='intermediate_shard',
            isa=('AVX512','AVX512'),purposes=('prefill',))
        self.assertEqual(target_plan(expert_scope=prefill_scope,expert_m=(512,128))['expert_m'],(128,512))

    def test_actual_m_cli_resume_requires_unchanged_saved_scope_without_native(self):
        with tempfile.TemporaryDirectory() as directory:
            argv=['--output-dir',directory,'--threads','28','--plan-only','--stage','expert-only',
                '--expert-model-family','qwen4exp','--expert-tp-degree','2',
                '--expert-layout','intermediate_shard','--expert-isa','AVX512:AVX512',
                '--expert-purpose','mtp_verifier','--expert-role-format','gate_up:Q4_K',
                '--expert-role-format','down:Q5_1','--expert-m','3']
            with mock.patch.object(sweep.subprocess,'run',side_effect=AssertionError('unexpected native launch')):
                self.assertEqual(sweep.main(argv),0)
                saved=(Path(directory)/sweep.PLAN_NAME).read_bytes()
                self.assertEqual(sweep.main(argv+['--resume']),0)
                with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                    sweep.main(argv[:-1]+['4','--resume'])
                self.assertEqual((Path(directory)/sweep.PLAN_NAME).read_bytes(),saved)

    def test_complement_retains_twenty_six_admissible_batches(self):
        plan=target_plan(expert_role_formats=(),exclude_expert_role_formats=(('gate_up','Q4_K'),('down','Q5_1')))
        self.assertEqual(plan['total_jobs'],26)
        self.assertEqual(len(plan['source_unadmitted']),14)

    def test_default_plan_preserves_original_broad_identity(self):
        plan=sweep.build_plan(28)
        self.assertEqual(plan['schema'],'cpu-exact-sweep-v1');self.assertEqual(plan['total_jobs'],1269)
        self.assertNotIn('tournament',plan);self.assertNotIn('prefill_row_policy',plan);self.assertNotIn('expert_m',plan)

    def test_scope_rejects_invalid_source_slice_and_legacy_adaptive(self):
        with self.assertRaises(ValueError):target_plan(expert_role_formats=(('down','Q4_K'),))
        with self.assertRaises(ValueError):sweep.build_plan(28,tournament='adaptive_v1')

    def test_completed_same_producer_jobs_launch_nothing(self):
        from test_native_quantised_cpu_sweep_reuse import original_completed
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'output';output.mkdir()
            job,producer,plan,contract=original_completed(Path(directory)/'provisional')
            cell=output/'cells'/job['kind']/sweep.job_id(job)
            job,producer,plan,contract=original_completed(cell)
            args=SimpleNamespace(output_dir=output,emit_only=False,batch_limit=0)
            with mock.patch.object(sweep,'iter_jobs',return_value=iter([job])),mock.patch.object(sweep.subprocess,'run',side_effect=AssertionError('unexpected native launch')):
                self.assertEqual(sweep._collect_stage(args,plan,producer,'rows'),0)
            foreign=deepcopy(producer);foreign['programs']['avx512_bin']['binary_sha256']='foreign-used-owner'
            with mock.patch.object(sweep,'iter_jobs',return_value=iter([job])),mock.patch.object(sweep.subprocess,'run',side_effect=AssertionError('unexpected native launch')):
                with self.assertRaises(ValueError):sweep._collect_stage(args,plan,foreign,'rows')

    def test_incomplete_scope_stays_pending_and_no_automatic_relabel(self):
        plan=target_plan()
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(output_dir=Path(directory),emit_only=True,batch_limit=0)
            with mock.patch.object(sweep.subprocess,'run',side_effect=AssertionError('unexpected native launch')):
                self.assertEqual(sweep._collect_stage(args,plan,{'digest':'producer'},'expert-only'),2)
        other=target_plan(expert_scope={**plan['expert_scope'],'isa':('AVX512','AVX2')})
        self.assertTrue({sweep.job_id(j) for j in sweep.iter_jobs(plan)}.isdisjoint(sweep.job_id(j) for j in sweep.iter_jobs(other)))


class AdaptiveExpertEvidenceTest(unittest.TestCase):
    """All byte proofs remain mandatory; screen clocks cannot win an overlay."""
    def test_v2_final_and_screening_cannot_import_timed_counter_collection(self):
        evidence = fixture.CPUExecutionNativeEvidenceTest()
        for screened in (False, True):
            original = adaptive_payload('WideRows' if screened else 'Pairwise', screened=screened)
            evidence.adapt(original)
            for field, value in (('timed_collection_disabled', None), ('timed_collection_disabled', False),
                                 ('timed_projection_observations', None), ('timed_projection_observations', 1)):
                bad = deepcopy(original)
                if value is None: bad['counters'].pop(field)
                else: bad['counters'][field] = value
                with self.subTest(screened=screened, field=field, value=value), self.assertRaisesRegex(ValueError, 'disabled timed collection'):
                    evidence.adapt(bad)
            bad = deepcopy(original)
            bad['counters'].update(projection_observations=6, expected_projection_observations=6)
            with self.subTest(screened=screened), self.assertRaisesRegex(ValueError, 'untimed guards'):
                evidence.adapt(bad)

    def test_native_final_and_screening_have_distinct_timing_grade(self):
        final=validate_fused_tournament(adaptive_payload(),minimum_samples=30)
        screen=validate_fused_tournament(adaptive_payload('WideRows',screened=True),minimum_samples=30)
        self.assertTrue(final.promotion_grade);self.assertFalse(screen.promotion_grade)

    def test_missing_candidate_failed_bytes_or_underfloor_final_is_rejected(self):
        for mutation in ('missing','bytes','floor','baseline','actual_m'):
            p=adaptive_payload()
            if mutation=='missing':p['tournament']['candidate_ids'].pop()
            if mutation=='bytes':p['tournament']['candidates'][1]['correctness_pass']=False
            if mutation=='floor':p['tournament']['candidates'][0]['final_sample_count']=3
            if mutation=='baseline':p['tournament']['baseline_auto_route_records'][0]['tags']['requested_verifier_policy']='Pairwise'
            if mutation=='actual_m':p['tournament']['m']+=1
            with self.assertRaises(ValueError,msg=mutation):validate_fused_tournament(p,minimum_samples=30)

    def test_final_credible_runner_up_is_not_an_extra_profile_obligation(self):
        evidence=fixture.CPUExecutionNativeEvidenceTest()
        payloads=[adaptive_payload(),adaptive_payload('WideRows')]
        for p in payloads:
            p['tournament']['finalist_candidate_ids']=['Pairwise','WideRows']
            decision=p['tournament']['candidates'][1]
            decision.update(timing_stage='final',promotion_grade=True,final_sample_count=30,
                finalist_reason='credible_runner_up')
        payloads[1]['finalist_reason']='credible_runner_up'
        rows=[evidence.adapt(p) for p in payloads]
        receipt={'entries':[dict(key=rows[0].physical_launch_evidence['key'],candidate_id=rows[0].candidate_id)]}
        selected=sweep.expert_profile_observations(ObservationCorpus(rows),receipt)
        self.assertEqual([r.candidate_id for r in selected],[rows[0].candidate_id])

    def test_new_declared_adaptive_roster_is_complete_and_old_two_choice_still_valid(self):
        old=adaptive_payload()
        validate_fused_tournament(old,minimum_samples=30)
        fresh=deepcopy(old);fresh['execution_exact_choice_abi']=2
        with self.assertRaisesRegex(ValueError,'roster'):validate_fused_tournament(fresh,minimum_samples=30)
        fresh['tournament']['candidate_ids'].append('OrderedThreeRows')
        decision=deepcopy(fresh['tournament']['candidates'][1]);decision['candidate_id']='OrderedThreeRows'
        fresh['tournament']['candidates'].append(decision)
        validate_fused_tournament(fresh,minimum_samples=30)
        fresh.pop('execution_exact_choice_abi')
        with self.assertRaisesRegex(ValueError,'roster'):validate_fused_tournament(fresh,minimum_samples=30)

    def test_new_screened_choice_keeps_correctness_but_cannot_publish(self):
        from native_quantised_dispatch.exact_oracle import candidate_is_eligible
        p=adaptive_payload('WideRows',screened=True)
        p.update(candidate='OrderedThreeRows',execution_exact_choice_abi=2,
                 promotion_eligible=True,descriptor_policies=[2,2])
        p['tournament']['candidate_ids'].append('OrderedThreeRows')
        decision=deepcopy(p['tournament']['candidates'][1]);decision['candidate_id']='OrderedThreeRows'
        p['tournament']['candidates'].append(decision)
        p['counters']['route_records']=[dict(count=2,tags=dict(m='64',n='320',policy_n='320',
            k='2560',codebook='5',k_tiles='16',purpose='prefill',isa='AVX512',
            requested_verifier_policy='OrderedThreeRows',effective_verifier_policy='OrderedThreeRows',physical_row_tile='3'))]
        observation=fixture.CPUExecutionNativeEvidenceTest().adapt(p)
        self.assertTrue(observation.bitwise_equal)
        self.assertFalse(is_promotion_grade(observation))
        self.assertFalse(candidate_is_eligible(observation))

    def test_fresh_full_choice_profiles_only_winner_and_actual_baseline(self):
        rows=[fixture.observe(fixture.row(name,choice_abi=2))
              for name in ('Pairwise','WideRows','OrderedThreeRows')]
        rows=[replace(row,physical_launch_evidence={**row.physical_launch_evidence,'target_descriptor':0})
              for row in rows]
        receipt={'entries':[dict(key=rows[0].physical_launch_evidence['key'],candidate_id=rows[2].candidate_id)]}
        profiles=sweep.expert_profile_observations(ObservationCorpus(rows),receipt)
        self.assertEqual({row.candidate_id for row in profiles},{rows[0].candidate_id,rows[2].candidate_id})

    def test_old_native_v1_behavior_remains_unchanged(self):
        evidence=fixture.CPUExecutionNativeEvidenceTest()
        self.assertEqual(evidence.adapt(evidence.payload()).median_us,11.)

    def test_v2_screen_round_trip_is_ineligible_without_losing_correctness(self):
        evidence=fixture.CPUExecutionNativeEvidenceTest()
        final=evidence.adapt(adaptive_payload());screen=evidence.adapt(adaptive_payload('WideRows',screened=True))
        self.assertTrue(is_promotion_grade(final));self.assertFalse(is_promotion_grade(screen))
        self.assertTrue(screen.numerical_correctness);self.assertTrue(screen.repeat_equal)
        receipt={'entries':[dict(key=final.physical_launch_evidence['key'],candidate_id=final.candidate_id)]}
        profiles=sweep.expert_profile_observations(ObservationCorpus([final,screen]),receipt)
        self.assertEqual([r.candidate_id for r in profiles],[final.candidate_id])

if __name__=='__main__':unittest.main()
