#!/usr/bin/env python3
"""Device-free reuse of original completed CPU receipts and modern selection.

All clocks/producers are authored synthetic metadata. No native executable,
model, binary inspection or profiler is invoked by this suite.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tests/v2/performance/kernels'))
from native_quantised_dispatch import cpu_exact_sweep as sweep
from native_quantised_dispatch.cpu_completed_sweep import inspect_completed_job,relevant_producer_matches,retain_installed_decode_stage
from native_quantised_dispatch.candidate_observation import write_observation_csv
from native_quantised_dispatch.corpus import ObservationCorpus
import test_cpu_execution_exact as fixture


def original_completed(directory):
    """Build a small byte-bound synthetic original receipt, not a fresh producer."""
    job=dict(kind='legacy_grouped',build_isa='AVX512',runtime_isa='AVX512',source_formats=['Q4_K'],
        shape='ProtocolFixture',n=320,k=2560,m_values=[64],measurement_basis='prepared_execution')
    programs={'avx512_bin':{'binary_sha256':'synthetic-original','core_sha256':'synthetic-core'},
        'fused_avx512_bin':{'binary_sha256':'synthetic-unused-owner'}}
    producer=dict(programs=programs,digest=sweep.canonical_digest(programs))
    contract=json.loads((sweep.KERNELS/'native_quantised_dispatch/manifests/cpu_native_quantised_serial_m1_arithmetic_v1.json').read_text())['contract_hash']
    row=replace(fixture.observe(fixture.row()),build_id=producer['digest']+'|cpu_isa=AVX512',
        compiler_id='cmake-release-closure:'+producer['digest'],serial_m1_policy_hash=contract,
        threading_or_stream_mode='openmp:build=AVX512:requested=AVX512:effective=AVX512:threads=28',
        warmup_count=5,sample_count=30)
    directory.mkdir(parents=True);observations=directory/'observations.csv'
    write_observation_csv(observations,ObservationCorpus([row]))
    raw=directory/'timing.csv';raw.write_text('synthetic original native bytes, not real measured clocks')
    (directory/'session-base').mkdir()
    sweep._atomic(directory/'request.json',dict(job=job,producer=producer,argv=['synthetic-original-owner']))
    sweep._atomic(directory/'complete.json',dict(job_digest=sweep.canonical_digest(job),producer_digest=producer['digest'],
        native_exit=0,session_retired=True,artifacts={str(p.resolve()):sweep.file_digest(p) for p in (observations,raw)}))
    return job,producer,dict(threads=28,warmups=5,samples=30),contract


def original_completed_expert(directory):
    """Bind synthetic clean batch timing to the EXISTING completion format."""
    from native_quantised_dispatch.adapters.cpu_execution_exact import adapt_cpu_execution_exact_native
    from native_quantised_dispatch.adapters.cpu_verifier import CPUVerifierAdapterContext
    from native_quantised_dispatch.adapters.evidence import summarize_sorted_timing
    job, producer, plan, contract = original_completed(directory)
    job.update(kind='expert_rows', requests=[dict(candidate='Pairwise', holdout=False, m=64)])
    plan['repeats'] = 8
    evidence = directory / 'evidence'; evidence.mkdir()
    raw = evidence / (sweep.canonical_digest(job['requests'][0]).split(':')[1] + '.json')
    payload = fixture.CPUExecutionNativeEvidenceTest.payload()
    samples = [10., 11., 13.] * 10; summary = summarize_sorted_timing(tuple(sorted(samples)))
    payload.update(raw_samples_us=samples, sorted_samples_us=sorted(samples))
    payload['timing'].update(sample_count=30, repeat_count=8, warmup_count=5,
        min_us=summary.minimum, median_us=summary.median, p95_us=summary.p95,
        mad_us=summary.mad, cv=summary.cv, sorted_native_digest=summary.digest)
    raw.write_text(json.dumps(payload))
    row = adapt_cpu_execution_exact_native(raw, CPUVerifierAdapterContext.workflow_smoke(
        corpus_id='sha256:' + '2' * 64), holdout=False)
    row = replace(row, build_id=producer['digest']+'|cpu_isa=AVX512',
        compiler_id='cmake-release-closure:'+producer['digest'], serial_m1_policy_hash=contract,
        threading_or_stream_mode='openmp:build=AVX512:requested=AVX512:effective=AVX512:threads=28')
    observations = directory / 'observations.csv'
    write_observation_csv(observations, ObservationCorpus([row]))
    sweep._atomic(directory/'request.json',dict(job=job,producer=producer,argv=['synthetic-original-owner']))
    sweep._atomic(directory/'complete.json',dict(job_digest=sweep.canonical_digest(job),producer_digest=producer['digest'],
        native_exit=0,session_retired=True,artifacts={str(v.resolve()):sweep.file_digest(v) for v in (observations,raw)}))
    return job, producer, plan, contract, raw


class CPUCompletedSweepReuseTest(unittest.TestCase):
    """Reuse checks actual used owners and never changes original observations."""
    def test_expert_reuse_rejects_dirty_timing_even_with_rebound_completion(self):
        for field, value in (('timed_collection_disabled', None), ('timed_collection_disabled', False),
                             ('timed_projection_observations', None), ('timed_projection_observations', 1)):
            with self.subTest(field=field,value=value), tempfile.TemporaryDirectory() as root:
                directory = Path(root) / 'original'
                job, producer, plan, contract, raw = original_completed_expert(directory)
                self.assertIsNotNone(inspect_completed_job(directory,job,producer,plan,contract_hash=contract)[0])
                payload=json.loads(raw.read_text())
                if value is None: payload['counters'].pop(field)
                else: payload['counters'][field]=value
                raw.write_text(json.dumps(payload))
                receipt=json.loads((directory/'complete.json').read_text())
                receipt['artifacts'][str(raw.resolve())]=sweep.file_digest(raw)
                sweep._atomic(directory/'complete.json',receipt)
                admitted,reason=inspect_completed_job(directory,job,producer,plan,contract_hash=contract)
                self.assertIsNone(admitted)
                self.assertIn('disabled timed collection',reason)
                self.assertTrue(raw.exists())

    def test_clean_expert_completion_reuses_original_and_launches_zero(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);job,producer,plan,contract,_=original_completed_expert(root/'provisional')
            paid=root/'paid';cell=paid/'cells'/job['kind']/sweep.job_id(job)
            cell.parent.mkdir(parents=True)
            job,producer,plan,contract,raw=original_completed_expert(cell)
            output=root/'new';output.mkdir()
            args=SimpleNamespace(output_dir=output,reuse_completed_root=paid,emit_only=False,batch_limit=0)
            original=raw.read_bytes()
            with mock.patch.object(sweep,'iter_jobs',return_value=iter([job])), \
                    mock.patch.object(sweep.subprocess,'run',side_effect=AssertionError('native launch')):
                self.assertEqual(sweep._collect_stage(args,plan,producer,'expert-only'),0)
            self.assertEqual(raw.read_bytes(),original)
            completion=output/'cells'/job['kind']/sweep.job_id(job)/'complete.json'
            self.assertIn('retained_original_receipt',json.loads(completion.read_text()))

    def test_install_rebuilds_every_selected_owner_build_once(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);output=root/'output';output.mkdir()
            args=SimpleNamespace(stage='all',output_dir=output,
                avx2_bin=root/'avx2/tests/v2/v2_perf_cpu_native_quantised_gemv',
                avx512_bin=root/'legacy512/tests/v2/v2_perf_cpu_native_quantised_gemv',
                fused_avx2_bin=root/'avx2/tests/v2/v2_perf_cpu_fused_verifier',
                fused_avx512_bin=root/'fresh-fused/tests/v2/v2_perf_cpu_fused_verifier')
            with mock.patch.object(sweep.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as build:
                sweep._rebuild_after_decode(args)
            commands=[call.args[0] for call in build.call_args_list]
            self.assertEqual(len(commands),3)
            by_build={Path(command[2]).name:command for command in commands}
            self.assertIn('v2_perf_cpu_native_quantised_gemv',by_build['avx2'])
            self.assertIn('v2_perf_cpu_fused_verifier',by_build['avx2'])
            self.assertNotIn('v2_perf_cpu_fused_verifier',by_build['legacy512'])
            self.assertNotIn('v2_perf_cpu_native_quantised_gemv',by_build['fresh-fused'])
            self.assertIn('v2_perf_cpu_fused_verifier',by_build['fresh-fused'])
            self.assertTrue(all('--parallel' in command for command in commands))

    def test_local_failure_is_fatal_before_external_adoption_or_launch(self):
        for emit_only in (False,True):
            with self.subTest(emit_only=emit_only),tempfile.TemporaryDirectory() as root:
                root=Path(root);paid=root/'paid';output=root/'new'
                job,producer,plan,contract=original_completed(root/'provisional')
                original=paid/'cells'/job['kind']/sweep.job_id(job)
                job,producer,plan,contract=original_completed(original)
                local=output/'cells'/job['kind']/sweep.job_id(job);local.mkdir(parents=True)
                failure=local/'failure.json';failure.write_text('{"returncode":17,"original_failure":"preserve"}')
                original_failure=failure.read_bytes()
                args=SimpleNamespace(output_dir=output,reuse_completed_root=paid,emit_only=emit_only,batch_limit=0)
                with mock.patch.object(sweep,'iter_jobs',return_value=iter([job])),mock.patch.object(sweep.subprocess,'run',side_effect=AssertionError('native launch')),mock.patch('native_quantised_dispatch.cpu_completed_sweep.inspect_completed_job',side_effect=AssertionError('external adoption attempted')):
                    with self.assertRaisesRegex(ValueError,'local native batch failed.*fresh output'):
                        sweep._collect_stage(args,plan,producer,'rows')
                self.assertEqual(failure.read_bytes(),original_failure)
                self.assertFalse((local/'complete.json').exists())
                self.assertFalse((local/'observations.csv').exists())
                self.assertTrue((original/'complete.json').exists())

    def test_same_basename_build_owners_keep_distinct_logs(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);output=root/'logs';output.mkdir()
            a=root/'a/build';b=root/'b/build'
            args=SimpleNamespace(stage='all',output_dir=output,
                avx2_bin=a/'tests/v2/v2_perf_cpu_native_quantised_gemv',
                fused_avx2_bin=a/'tests/v2/v2_perf_cpu_fused_verifier',
                avx512_bin=b/'tests/v2/v2_perf_cpu_native_quantised_gemv',
                fused_avx512_bin=b/'tests/v2/v2_perf_cpu_fused_verifier')
            def build(command,*,stdout,**unused):
                stdout.write(str(command));return SimpleNamespace(returncode=0)
            with mock.patch.object(sweep.subprocess,'run',side_effect=build):sweep._rebuild_after_decode(args)
            logs=list(output.glob('decode-rebuild-*.log'))
            self.assertEqual(len(logs),2)
            self.assertNotEqual(logs[0].name,logs[1].name)
            self.assertTrue(any('avx2_bin' in log.name for log in logs))
            self.assertTrue(any('avx512_bin' in log.name for log in logs))
            contents=[log.read_text() for log in logs]
            self.assertTrue(any(str(a) in text for text in contents))
            self.assertTrue(any(str(b) in text for text in contents))

    def test_changed_unused_fused_owner_preserves_original_paid_legacy_job(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root)/'original';job,old,plan,contract=original_completed(directory)
            current=deepcopy(old);current['programs']['fused_avx512_bin']['binary_sha256']='new-unrelated-owner'
            current['digest']=sweep.canonical_digest(current['programs'])
            inspected,reason=inspect_completed_job(directory,job,current,plan,contract_hash=contract)
            self.assertIsNotNone(inspected,reason)
            self.assertTrue(relevant_producer_matches([job],old,current))
            self.assertEqual(inspected['complete']['producer_digest'],old['digest'])

    def test_changed_used_owner_format_actual_m_isa_and_contract_are_not_reused(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root)/'original';job,old,plan,contract=original_completed(directory)
            for field,value in (('m_values',[3]),('source_formats',['Q5_1']),('runtime_isa','AVX2')):
                inspected,reason=inspect_completed_job(directory,{**job,field:value},old,plan,contract_hash=contract)
                self.assertIsNone(inspected,field);self.assertTrue(reason)
            changed=deepcopy(old);changed['programs']['avx512_bin']['binary_sha256']='new-used-owner'
            self.assertIsNone(inspect_completed_job(directory,job,changed,plan,contract_hash=contract)[0])
            self.assertIsNone(inspect_completed_job(directory,job,old,plan,contract_hash='sha256:'+'f'*64)[0])
            self.assertIsNone(inspect_completed_job(directory,job,old,{**plan,'samples':31},contract_hash=contract)[0])

    def test_mutated_artifact_or_unretired_session_is_not_reused(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root)/'original';job,old,plan,contract=original_completed(directory)
            (directory/'session-base/live').write_text('not retired')
            self.assertIsNone(inspect_completed_job(directory,job,old,plan,contract_hash=contract)[0])
            (directory/'session-base/live').unlink();(directory/'timing.csv').write_text('changed')
            self.assertIsNone(inspect_completed_job(directory,job,old,plan,contract_hash=contract)[0])

    def test_external_original_completion_then_local_resume_launches_zero(self):
        with tempfile.TemporaryDirectory() as root:
            original=Path(root)/'paid';output=Path(root)/'new';output.mkdir()
            provisional=Path(root)/'fixture';job,old,plan,contract=original_completed(provisional)
            cell=original/'cells'/job['kind']/sweep.job_id(job);cell.parent.mkdir(parents=True)
            # Author the bound receipt at its final original path.
            provisional.rename(Path(root)/'unused-fixture')
            job,old,plan,contract=original_completed(cell)
            current=deepcopy(old);current['programs']['fused_avx512_bin']['binary_sha256']='new-unused'
            current['digest']=sweep.canonical_digest(current['programs'])
            args=SimpleNamespace(output_dir=output,reuse_completed_root=original,emit_only=False,batch_limit=0)
            before=(cell/'observations.csv').read_bytes()
            with mock.patch.object(sweep,'iter_jobs',return_value=iter([job])),mock.patch.object(sweep.subprocess,'run',side_effect=AssertionError('native launch')):
                self.assertEqual(sweep._collect_stage(args,plan,current,'rows'),0)
            with mock.patch.object(sweep,'iter_jobs',return_value=iter([job])),mock.patch.object(sweep.subprocess,'run',side_effect=AssertionError('native launch')):
                self.assertEqual(sweep._collect_stage(args,plan,current,'rows'),0)
            reference=json.loads((output/'cells'/job['kind']/sweep.job_id(job)/'complete.json').read_text())
            self.assertEqual(reference['producer_digest'],old['digest'])
            self.assertEqual((cell/'observations.csv').read_bytes(),before)
            self.assertIn('retained_original_receipt',reference)

    def test_incompatible_external_completion_remains_explicitly_pending(self):
        with tempfile.TemporaryDirectory() as root:
            paid=Path(root)/'paid';fixture_dir=Path(root)/'fixture';job,old,plan,contract=original_completed(fixture_dir)
            cell=paid/'cells'/job['kind']/sweep.job_id(job);cell.parent.mkdir(parents=True)
            fixture_dir.rename(Path(root)/'unused');job,old,plan,contract=original_completed(cell)
            current=deepcopy(old);current['programs']['avx512_bin']['binary_sha256']='different-used-owner'
            args=SimpleNamespace(output_dir=Path(root)/'new',reuse_completed_root=paid,emit_only=True,batch_limit=0)
            with mock.patch.object(sweep,'iter_jobs',return_value=iter([job])),mock.patch.object(sweep.subprocess,'run',side_effect=AssertionError('native launch')):
                self.assertEqual(sweep._collect_stage(args,plan,current,'rows'),1)
            self.assertTrue((args.output_dir/'cells'/job['kind']/sweep.job_id(job)/'reuse-not-admitted.json').exists())


class RetainedInstalledM1StageTest(unittest.TestCase):
    """Original M1 clocks precede rebuild; installed policy is retained honestly."""
    def fixture(self, root):
        root=Path(root);original=root/'paid';output=root/'new';owner=root/'owner'
        for path in (original,output,owner,original/'generated'):path.mkdir(parents=True,exist_ok=True)
        plan=dict(legacy_shapes=[dict(name='ProtocolFixture',n=320,k=2560)],expert_geometries=[],
            formats=['Q4_K'],regimes=[['AVX512','AVX512']],grouped_m=[2,3],expert_prefill_m=[])
        plan['plan_digest']=sweep.canonical_digest(plan)
        source=original/'generated/CPUNativeQuantisedDecodePolicyGenerated.inc';source.write_text('synthetic installed M1 policy')
        (owner/source.name).write_bytes(source.read_bytes())
        programs={'avx512_bin':{'binary_sha256':'synthetic-post-M1','core_library_sha256':'synthetic-core'},
            'fused_avx512_bin':{'binary_sha256':'synthetic-unused'}}
        current=dict(programs=programs,digest=sweep.canonical_digest(programs))
        before=deepcopy(current);before['programs']['avx512_bin']['binary_sha256']='synthetic-pre-M1'
        before['digest']=sweep.canonical_digest(before['programs'])
        sweep._atomic(original/'cpu-exact-plan.json',plan)
        sweep._atomic(source.with_suffix('.receipt.json'),dict(plan_digest=plan['plan_digest'],output_sha256=sweep.file_digest(source)))
        sweep._atomic(original/'decode-stage.json',dict(output_sha256=sweep.file_digest(source),installed=True,rebuilt=True,producer_digest=before['digest']))
        sweep._atomic(original/'decode-producer.json',before);sweep._atomic(original/'rows-producer.json',current)
        current=deepcopy(current);current['programs']['fused_avx512_bin']['binary_sha256']='new-unused-fused'
        current['digest']=sweep.canonical_digest(current['programs'])
        return original,output,owner,current,plan,before

    def test_completed_installed_stage_skips_remeasurement_and_preserves_original_producer(self):
        with tempfile.TemporaryDirectory() as root:
            original,output,owner,current,plan,before=self.fixture(root)
            with mock.patch.object(sweep.subprocess,'run',side_effect=AssertionError('native launch')):
                self.assertTrue(retain_installed_decode_stage(original,output,owner,current,plan))
            stage=json.loads((output/'decode-stage.json').read_text())
            self.assertEqual(stage['producer_digest'],before['digest'])
            self.assertEqual(json.loads((output/'decode-producer.json').read_text()),before)
            self.assertTrue(stage['retained_original_stage']['retained_policy_not_remeasured'])
            self.assertEqual(stage['retained_original_stage']['covered_decode_jobs'],1)
            self.assertEqual((output/'generated/CPUNativeQuantisedDecodePolicyGenerated.inc').read_bytes(),
                (owner/'CPUNativeQuantisedDecodePolicyGenerated.inc').read_bytes())

    def test_missing_scope_changed_used_owner_or_installed_output_rejects_retention(self):
        for mutation in ('scope','used_owner','installed','incomplete'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as root:
                original,output,owner,current,plan,before=self.fixture(root)
                if mutation=='scope':plan={**plan,'formats':['Q5_1']}
                if mutation=='used_owner':current['programs']['avx512_bin']['binary_sha256']='changed-used-owner'
                if mutation=='installed':(owner/'CPUNativeQuantisedDecodePolicyGenerated.inc').write_text('changed')
                if mutation=='incomplete':
                    stage=json.loads((original/'decode-stage.json').read_text());stage['rebuilt']=False
                    sweep._atomic(original/'decode-stage.json',stage)
                with self.assertRaises(ValueError):retain_installed_decode_stage(original,output,owner,current,plan)



if __name__=='__main__':unittest.main()
