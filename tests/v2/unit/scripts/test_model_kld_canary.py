#!/usr/bin/env python3
"""Device-free adversarial phase/identity/threshold regressions for KL canaries."""
from pathlib import Path
import math
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts/ci'))
import model_kld_canary as policy


def rows(model, prefill=0., decode=0.):
    """Create a complete tiny scalar metric fixture without model/GPU payloads."""
    first=policy.definition(model,'cuda').prompt_tokens
    return [dict(consumed_tokens=first+i,kl_nats=prefill if i==0 else decode) for i in range(129)]


class ModelKLDCanaryTests(unittest.TestCase):
    """Either phase must independently reject bad evidence on every backend/model."""
    def test_exact_policy_inventory_and_limits(self):
        self.assertEqual(set(policy.CANARIES),{(m,b) for m in ['dense','moe'] for b in ['cuda','rocm']})
        self.assertEqual(policy.DEFAULT_GDN_POLICY,'fp16_moe_alpha_beta')
        self.assertTrue(all(value.half_alpha_beta==(model=='moe') for (model,backend),value in policy.CANARIES.items()))
        self.assertEqual({key:(v.prefill_ceiling,v.decode_ceiling) for key,v in policy.CANARIES.items()},
                         {('dense','cuda'):(.001,.09),('dense','rocm'):(.001,.13),('moe','cuda'):(.01,.41),('moe','rocm'):(.05,.47)})
    def test_inclusive_boundaries_and_independent_phase_corruption(self):
        for (model,backend),p in policy.CANARIES.items():
            for phase,limit in [('prefill',p.prefill_ceiling),('decode',p.decode_ceiling)]:
                for value,expected in [(math.nextafter(limit,-math.inf),True),(limit,True),(math.nextafter(limit,math.inf),False)]:
                    arguments={phase:value};data=rows(model,**arguments)
                    result=policy.evaluate_rows(data,model=model,backend=backend)
                    with self.subTest(model=model,backend=backend,phase=phase,value=value):
                        self.assertEqual(result['passed'],expected)
                        self.assertEqual(result['phases'][phase]['passed'],expected)
                        other='decode' if phase=='prefill' else 'prefill'
                        self.assertTrue(result['phases'][other]['passed'])
                        self.assertEqual(result['phases']['prefill']['row_count'],1)
                        self.assertEqual(result['phases']['decode']['row_count'],128)
    def test_bad_prefill_cannot_hide_in_good_decode_average(self):
        for model,backend in policy.CANARIES:
            p=policy.definition(model,backend)
            result=policy.evaluate_rows(rows(model,prefill=p.prefill_ceiling*2),model=model,backend=backend)
            self.assertLess(result['mean_kl_nats'],p.decode_ceiling)
            self.assertFalse(result['passed'])
    def test_bad_decode_cannot_borrow_prefill_allowance(self):
        for model,backend in policy.CANARIES:
            p=policy.definition(model,backend)
            result=policy.evaluate_rows(rows(model,decode=p.decode_ceiling*1.001),model=model,backend=backend)
            self.assertLess(result['mean_kl_nats'],p.decode_ceiling)
            self.assertFalse(result['passed'])
    def test_complete_ordered_population_required(self):
        for model,backend in policy.CANARIES:
            original=rows(model)
            for bad in [original[:-1],original+original[-1:],list(reversed(original)),tuple(original),original[1:]+original[:1]]:
                with self.assertRaises(ValueError):policy.evaluate_rows(bad,model=model,backend=backend)
            for value in [True,float(original[0]['consumed_tokens']),original[1]['consumed_tokens']]:
                bad=rows(model);bad[0]['consumed_tokens']=value
                with self.assertRaises(ValueError):policy.evaluate_rows(bad,model=model,backend=backend)
    def test_invalid_values_fail_in_either_phase(self):
        for model,backend in policy.CANARIES:
            for index in [0,1,128]:
                for value in [True,False,'0',None,-1.,math.nan,math.inf,-math.inf]:
                    bad=rows(model);bad[index]['kl_nats']=value
                    with self.assertRaises(ValueError):policy.evaluate_rows(bad,model=model,backend=backend)
                bad=rows(model);del bad[index]['kl_nats']
                with self.assertRaises(ValueError):policy.evaluate_rows(bad,model=model,backend=backend)
    def test_no_backend_or_model_borrowing(self):
        for model,backend in [('dense','cpu'),('dense','CUDA'),('moe','rocm:0'),('unknown','cuda'),(True,'cuda'),('dense',None)]:
            with self.assertRaises(ValueError):policy.evaluate_rows(rows('dense'),model=model,backend=backend)
        self.assertFalse(policy.evaluate_rows(rows('dense',decode=.11),model='dense',backend='cuda')['passed'])
        self.assertTrue(policy.evaluate_rows(rows('dense',decode=.11),model='dense',backend='rocm')['passed'])
        with self.assertRaises(ValueError):policy.evaluate_rows(rows('dense'),model='moe',backend='rocm')
    def test_tail_spike_is_not_trimmed(self):
        for model,backend in policy.CANARIES:
            p=policy.definition(model,backend);data=rows(model);data[-1]['kl_nats']=129*p.decode_ceiling
            result=policy.evaluate_rows(data,model=model,backend=backend)
            self.assertFalse(result['passed']);self.assertEqual(result['phases']['decode']['max_kl_nats'],data[-1]['kl_nats'])



def canary_report(source_revision='a'*40,image='sha256:'+'1'*64,cpu_isa='AVX512',manifest_digest='d'*64):
    """Complete published-image fixture; no synthetic model execution is claimed."""
    result=dict(schema=1,complete=True,passed=True,source_revision=source_revision,image=image,
                cpu_isa=cpu_isa,corpus_manifest_digest=manifest_digest,diagnostic_only=False,cells=[])
    for model,backend in policy.CANARIES:
        data=rows(model);metrics=policy.evaluate_rows(data,model=model,backend=backend)
        metrics.update(rows=data,corpus_manifest_digest=manifest_digest,full_vocabulary=True)
        result['cells'].append(dict(model=model,backend=backend,complete=True,passed=True,native_retired=True,
            reset_byte_equal=True,driver_window_clean=True,raw_logits_removed=True,returncode=0,
            phases=metrics['phases'],metrics=metrics,production_path=dict(complete_graph_replay=True,
            prefill=True,decode=True,native_router=True,native_wide_projections=True,half_alpha_beta_weights=60 if model=='moe' else 0)))
    return result


class ModelKLDReportTests(unittest.TestCase):
    """Receipt acceptance recomputes every row and pins the exact published lane."""
    def check(self,report,**kwargs):
        return policy.validate_report(report,source_revision='a'*40,image='sha256:'+'1'*64,
            cpu_isa='AVX512',corpus_manifest_digest='d'*64,**kwargs)
    def test_complete_valid_report(self):
        self.check(canary_report())
    def test_all_run_identities_and_completed_lifetime_are_required(self):
        for key,value in [('source_revision','b'*40),('image','sha256:'+'2'*64),('cpu_isa','AVX2'),
                          ('corpus_manifest_digest','e'*64),('diagnostic_only',True),('complete',False),('schema',True)]:
            report=canary_report();report[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.check(report)
        for key in ['complete','native_retired','reset_byte_equal','driver_window_clean','raw_logits_removed']:
            report=canary_report();report['cells'][0][key]=False
            with self.subTest(key=key),self.assertRaises(ValueError):self.check(report)
        for code in [None,1,False,0.]:
            report=canary_report();report['cells'][0]['returncode']=code
            with self.assertRaises(ValueError):self.check(report)
    def test_missing_duplicate_phases_and_wrong_precision_fail(self):
        report=canary_report();report['cells'].pop()
        with self.assertRaises(ValueError):self.check(report)
        report=canary_report();report['cells'][1]=report['cells'][0]
        with self.assertRaises(ValueError):self.check(report)
        for key,value in [('prefill',False),('decode',False),('native_router',False),('half_alpha_beta_weights',60)]:
            report=canary_report();report['cells'][0]['production_path'][key]=value
            with self.assertRaises(ValueError):self.check(report)
    def test_forged_phase_summary_and_full_vocabulary_fail(self):
        report=canary_report();report['cells'][0]['metrics']['rows'][0]['kl_nats']=1
        with self.assertRaises(ValueError):self.check(report)
        report=canary_report();report['cells'][0]['metrics']['full_vocabulary']=False
        with self.assertRaises(ValueError):self.check(report)
        report=canary_report();report['cells'][0]['metrics']['model']='moe'
        with self.assertRaises(ValueError):self.check(report)
    def test_valid_red_can_collect_other_lane_but_cannot_admit(self):
        report=canary_report();cell=report['cells'][0];data=rows('dense',prefill=1)
        metrics=policy.evaluate_rows(data,model='dense',backend='cuda')
        metrics.update(rows=data,corpus_manifest_digest='d'*64,full_vocabulary=True)
        cell.update(metrics=metrics,phases=metrics['phases'],passed=False);report['passed']=False
        self.check(report,require_passed=False)
        with self.assertRaises(ValueError):self.check(report)
        report['passed']=True
        with self.assertRaises(ValueError):self.check(report,require_passed=False)


class ModelKLDCorpusTests(unittest.TestCase):
    """Sparse extent fixtures exercise admission without loading models or logits."""
    def setUp(self):
        import tempfile,json
        import model_kld_corpus as corpus
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.catalog=self.root/'approval.json'
        self.manifest=dict(schema='llaminar.model-kld-corpus.v1',temperature=1.,vocabulary=248320,models={})
        for model in ['dense','moe']:
            p=policy.definition(model,'cuda');directory=self.root/model;directory.mkdir()
            cases=[dict(name=name,prompt_ids=[1]*count,continuation_ids=[2]*steps) for name,count,steps in
                   [('short',512,5),('long',p.prompt_tokens,129),('reset',512,5)]]
            workload={'cases':cases};case=dict(name='long',prompt_tokens=p.prompt_tokens,continuation_ids=[2]*129,
                rows=[dict(consumed_tokens=p.prompt_tokens+i,vocabulary=p.vocabulary,argmax=2) for i in range(129)],logits_file='reference.f32')
            reference=dict(complete=True,passed=True,cases=[case])
            (directory/'workload.json').write_text(json.dumps(workload));(directory/'reference.json').write_text(json.dumps(reference))
            with (directory/'reference.f32').open('wb') as output:output.truncate(129*p.vocabulary*4)
            self.manifest['models'][model]=dict(model_file=p.model_file,model_bytes=123,oracle=dict(complete=True,passed=True,cached_prefill=False),
                workload=model+'/workload.json',reference_report=model+'/reference.json',
                workload_digest=corpus.metadata_digest(workload),reference_report_digest=corpus.metadata_digest(reference),
                alpha_beta_sources={str(i):dict(source_format='FP32',rows=1,columns=1) for i in range(96 if model=='dense' else 60)})
        self.seal()
    def seal(self):
        import json
        import model_kld_corpus as corpus
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))
        self.catalog.write_text(json.dumps(dict(schema=1,path='model-kld/fixture',manifest_digest=corpus.metadata_digest(self.manifest))))
    def load(self):
        import model_kld_corpus as corpus
        return corpus.load_corpus(self.root,catalog=self.catalog)
    def test_portable_metadata_only_admission_and_stat_freshness(self):
        import model_kld_corpus as corpus
        value=self.load();self.assertEqual(set(value['models']),{'dense','moe'})
        corpus.unchanged(value['inputs'])
        path=self.root/'moe/reference.f32'
        with path.open('r+b') as output:output.write(b'changed')
        with self.assertRaises(ValueError):corpus.unchanged(value['inputs'])
    def test_changed_frozen_token_cannot_borrow_shape_or_manifest(self):
        import json
        path=self.root/'dense/workload.json';document=json.loads(path.read_text());document['cases'][1]['prompt_ids'][0]=4
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError,'metadata changed'):self.load()
    def test_missing_lfs_extent_and_unapproved_manifest_fail(self):
        path=self.root/'dense/reference.f32';path.write_text('version https://git-lfs.github.com/spec/v1\n')
        with self.assertRaisesRegex(ValueError,'LFS'):self.load()
        (self.root/'manifest.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Unapproved'):self.load()
    def test_complete_oracle_and_original_control_inventory_required(self):
        self.manifest['models']['dense']['oracle']['cached_prefill']=True;self.seal()
        with self.assertRaisesRegex(ValueError,'oracle'):self.load()
        self.manifest['models']['dense']['oracle']['cached_prefill']=False
        self.manifest['models']['dense']['alpha_beta_sources'].pop('0');self.seal()
        with self.assertRaisesRegex(ValueError,'inventory'):self.load()
    def test_traversal_alias_and_symlink_escape_rejected(self):
        import model_kld_corpus as corpus
        for spelling in ['/tmp/file','../file','dense/../moe/reference.json','dense//reference.json','./dense/reference.json','dense\\reference.json']:
            with self.assertRaises(ValueError):corpus.relative_file(self.root,spelling)
        (self.root/'outside').symlink_to('/etc/hosts')
        with self.assertRaises(ValueError):corpus.relative_file(self.root,'outside')


class ModelKLDScoringTests(unittest.TestCase):
    """Tiny full rows verify KL direction, reset bytes and stale publications."""
    def test_reference_direction_shift_invariance_and_nonfinite(self):
        import numpy as np
        import model_kld_corpus as corpus
        p=np.array([0.,1.,2.]);q=np.array([2.,0.,1.])
        self.assertAlmostEqual(corpus.row_metrics(p,p)['kl_nats'],0.)
        self.assertAlmostEqual(corpus.row_metrics(p,q)['kl_nats'],corpus.row_metrics(p+1000,q-1000)['kl_nats'])
        probabilities=np.exp(p)/np.exp(p).sum();candidate=np.exp(q)/np.exp(q).sum()
        self.assertAlmostEqual(corpus.row_metrics(p,q)['kl_nats'],float(np.sum(probabilities*np.log(probabilities/candidate))))
        for bad in [[0.,float('nan'),1.],[0.,float('inf'),1.],[0.]]:
            with self.assertRaises(ValueError):corpus.row_metrics(p,bad)
    def test_full_score_reset_bytes_and_stale_rows(self):
        from unittest.mock import patch
        from dataclasses import replace
        import tempfile,json,numpy as np
        import model_kld_corpus as corpus
        tiny=replace(policy.definition('dense','cuda'),prompt_tokens=4,vocabulary=4)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);workload={'cases':[]};observations=[];arrays=[]
            for name,prompt,count in [('short',2,5),('long',4,129),('reset',2,5)]:
                wanted=dict(name=name,prompt_ids=[1]*prompt,continuation_ids=[2]*count);workload['cases'].append(wanted)
                values=np.array([[0.,1.,2.+i/100,3.] for i in range(count)],dtype=np.float32)
                path=root/(name+'.f32');values.tofile(path);arrays.append((path,values))
                observations.append(dict(name=name,prompt_tokens=prompt,continuation_ids=[2]*count,logits_file=str(path),
                    rows=[dict(consumed_tokens=prompt+i,vocabulary=4,argmax=int(np.argmax(v))) for i,v in enumerate(values)]))
            reference=root/'reference.f32';arrays[1][1].tofile(reference)
            state=dict(manifest_digest='d'*64,inputs={},models={'dense':dict(workload=workload,reference_payload=reference)})
            report=root/'logits.json';report.write_text(json.dumps(dict(complete=True,passed=True,cases=observations)))
            with patch.object(policy,'definition',return_value=tiny),patch.object(corpus,'definition',return_value=tiny),patch.object(corpus,'physical_workers',return_value=1):
                result=corpus.score_observation(state,report,model='dense',backend='cuda')
                self.assertTrue(result['passed']);self.assertTrue(result['reset_byte_equal']);self.assertEqual(result['row_count'],129)
                data=arrays[2][1].copy();data[0,0]=-0.;data.tofile(arrays[2][0])
                with self.assertRaisesRegex(ValueError,'logit bytes'):corpus.score_observation(state,report,model='dense',backend='cuda')
                arrays[2][1].tofile(arrays[2][0]);data=arrays[1][1].copy();data[1]=data[0];data.tofile(arrays[1][0])
                with self.assertRaisesRegex(ValueError,'stale'):corpus.score_observation(state,report,model='dense',backend='cuda')


class ModelKLDProductionPathTests(unittest.TestCase):
    """Counters must prove the scoped policy on the actual captured backend."""

    def fixture(self, model, backend):
        """Build a small native-counter record with original source metadata."""
        device = backend + ':0'
        entry = dict(alpha_beta_sources={
            'blk.0.ssm_alpha.weight': dict(source_format='FP32', rows=16, columns=64),
            'blk.0.ssm_beta.weight': dict(source_format='FP32', rows=16, columns=64),
        })
        report = dict(contract=dict(
            topology='single_device', device=device, physical_devices=1,
            local_tp_devices=0, prefill_max_bucket=512,
            activation_precision='fp32', kv_precision='fp16', mtp=False,
            prefix_cache=False, event_ordered_host_logits=True,
            gdn_arithmetic=policy.DEFAULT_GDN_POLICY, router_arithmetic='native'))
        records = [dict(domain='forward_graph', device=device, name=name,
                        count=1, tags=tags) for name, tags in [
            ('full_graph_replay_calls', {}),
            ('prefill_graph_phase', {'capture_phase': 'replay'}),
            ('decode_graph_phase', {'phase': 'replay'}),
        ]]
        records += [dict(domain='gdn', device=device, name='gdn_projection_route',
                         count=1, tags=dict(names=name, codebooks=codebooks, route=route))
                    for name, codebooks, route in [
            ('qkv+z', '7+24', 'native_subgroup'),
            ('alpha+beta', 'none+none', 'same_kernel_mixed_codebook_subgroup'),
        ]]
        if model == 'moe':
            records += [dict(domain='weights', device=device, count=1,
                             name='gdn_projection_arithmetic_prepared_weight', tags=dict(
                canonical_name=name, policy=policy.DEFAULT_GDN_POLICY,
                source_format=source['source_format'], prepared_format='FP16',
                source_path='source_decoded_half_rne',
                source_bytes=str(source['rows'] * source['columns'] * 4),
                prepared_bytes=str(source['rows'] * source['columns'] * 2)))
                for name, source in entry['alpha_beta_sources'].items()]
        return dict(records=records), report, entry

    def test_model_scope_is_backend_symmetric(self):
        import run_model_kld_canaries as runner
        for model, backend in policy.CANARIES:
            perf, report, entry = self.fixture(model, backend)
            with self.subTest(model=model, backend=backend):
                result = runner.production_path(perf, report, backend, entry, model=model)
                self.assertEqual(result['half_alpha_beta_weights'], 2 if model == 'moe' else 0)
                self.assertTrue(result['prefill'] and result['decode'])
                self.assertTrue(result['native_wide_projections'] and result['native_router'])

    def test_missing_or_wrong_capture_cannot_certify_either_model(self):
        from copy import deepcopy
        import run_model_kld_canaries as runner
        for model, backend in policy.CANARIES:
            perf, report, entry = self.fixture(model, backend)
            for index in range(3):
                bad = deepcopy(perf)
                del bad['records'][index]
                with self.subTest(model=model, backend=backend, missing=index), self.assertRaises(ValueError):
                    runner.production_path(bad, report, backend, entry, model=model)
            for name in ['eager_forward_calls', 'segmented_graph_replay_calls']:
                bad = deepcopy(perf)
                bad['records'].append(dict(domain='forward_graph', device=backend + ':0', name=name, count=1))
                with self.assertRaisesRegex(ValueError, 'capture'):
                    runner.production_path(bad, report, backend, entry, model=model)
            bad = deepcopy(perf)
            bad['records'][0]['device'] = backend + ':1'
            with self.assertRaisesRegex(ValueError, 'device'):
                runner.production_path(bad, report, backend, entry, model=model)

    def test_half_preparation_cannot_leak_to_dense_wide_or_router(self):
        from copy import deepcopy
        import run_model_kld_canaries as runner
        for backend in ['cuda', 'rocm']:
            half, _, _ = self.fixture('moe', backend)
            for model in ['dense', 'moe']:
                perf, report, entry = self.fixture(model, backend)
                rogue = deepcopy(half['records'][-1])
                if model == 'moe':
                    rogue['tags']['canonical_name'] = 'blk.0.attn_qkv.weight'
                perf['records'].append(rogue)
                with self.assertRaisesRegex(ValueError, 'precision conversion'):
                    runner.production_path(perf, report, backend, entry, model=model)
                perf, report, entry = self.fixture(model, backend)
                perf['records'].append(dict(name='moe_router_arithmetic_prepared_weight', count=1))
                with self.assertRaisesRegex(ValueError, 'Router policy'):
                    runner.production_path(perf, report, backend, entry, model=model)

    def test_source_extent_and_policy_are_authenticated(self):
        from copy import deepcopy
        import run_model_kld_canaries as runner
        for backend in ['cuda', 'rocm']:
            perf, report, entry = self.fixture('moe', backend)
            for key, value in [('policy', 'fp16_alpha_beta'), ('source_format', 'Q8_0'),
                               ('prepared_format', 'FP32'), ('source_bytes', '4'),
                               ('prepared_bytes', '2'), ('source_path', 'requantized')]:
                bad = deepcopy(perf)
                bad['records'][-1]['tags'][key] = value
                with self.subTest(backend=backend, key=key), self.assertRaisesRegex(ValueError, 'preparation'):
                    runner.production_path(bad, report, backend, entry, model='moe')
            bad = deepcopy(perf)
            bad['records'].pop()
            with self.assertRaisesRegex(ValueError, 'Incomplete alpha/beta'):
                runner.production_path(bad, report, backend, entry, model='moe')
            report['contract']['gdn_arithmetic'] = 'fp16_alpha_beta'
            with self.assertRaisesRegex(ValueError, 'default execution contract'):
                runner.production_path(perf, report, backend, entry, model='moe')


class ModelKLDLifecycleTests(unittest.TestCase):
    """Failed/unstarted runs retire owned native work before raw-output cleanup."""
    def test_created_exited_and_live_container_retirement(self):
        from unittest.mock import patch
        import json
        import run_model_kld_canaries as runner
        for status in ['created','exited','running']:
            initial=dict(Running=status=='running',Pid=8 if status=='running' else 0,Status=status)
            final=dict(Running=False,Pid=0,Status='created' if status=='created' else 'exited')
            with patch.object(runner.subprocess,'check_output',side_effect=[json.dumps([{'State':initial}]),json.dumps([{'State':final}])]),patch.object(runner.subprocess,'run') as execute:
                runner.retire(None,'owner')
                commands=[call.args[0] for call in execute.call_args_list]
                self.assertEqual(any(c[1]=='wait' for c in commands),status!='created')
                self.assertEqual(any(c[1]=='stop' for c in commands),status=='running')
                self.assertEqual(commands[-1],['docker','rm','owner'])
    def test_exception_and_interrupt_remove_raw_payloads(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import tempfile,json
        import run_model_kld_canaries as runner
        for error in [ValueError('bad workload'),KeyboardInterrupt('cancel')]:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);model=root/'fixture.gguf';model.write_bytes(b'abcd')
                corpus={'models':{'dense':{'entry':{'model_file':model.name,'model_bytes':4},'workload_path':root/'input.json'}}}
                args=SimpleNamespace(binary=root/'binary',keep_logits=False)
                with patch.object(runner.driver,'begin',side_effect=error),patch.object(runner,'retire') as retire:
                    with self.assertRaises(type(error)):runner.execute_cell(args,corpus,'dense','cuda',model,root/'cell',None)
                retire.assert_called_once_with(None,None)
                report=json.loads((root/'cell/controller.json').read_text())
                self.assertFalse(report['passed']);self.assertTrue(report['raw_logits_removed']);self.assertTrue(report['native_retired'])
                self.assertFalse((root/'cell/logits').exists())


class ModelKLDCITests(unittest.TestCase):
    """Master-only integration keeps both ISA evidence and native owner cleanup."""
    def test_complete_red_numerics_still_collect_both_isas(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import tempfile,json,subprocess
        import run_published_image_suite as suite
        from test_published_image_suite import image_pair,full_inventory
        from production_artifacts import write_json,digest
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pair=image_pair();manifest=suite.pipeline.e2e_projection(full_inventory(),pair['source']['revision'])
            approval=json.loads((suite.ROOT/'scripts/ci/approved_model_kld_corpus.json').read_text())
            visited=[]
            def execute(args,driver,command,lane,log):
                if lane.name=='kld':
                    isa=lane.parent.name.upper();visited.append(isa)
                    self.assertEqual(Path(command[command.index('--report')+1]).parent,lane)
                    report=canary_report(pair['source']['revision'],pair['images'][isa]['id'],isa,approval['manifest_digest'])
                    if isa=='AVX512':
                        cell=report['cells'][0];data=rows('dense',prefill=1.)
                        metrics=policy.evaluate_rows(data,model='dense',backend='cuda')
                        metrics.update(rows=data,corpus_manifest_digest=approval['manifest_digest'],full_vocabulary=True)
                        cell.update(metrics=metrics,phases=metrics['phases'],passed=False);report['passed']=False
                    write_json(lane/'kld-canaries.json',report)
                    if not report['passed']:raise subprocess.CalledProcessError(1,command)
                else:
                    isa=lane.name.upper()
                    write_json(lane/'e2e.json',dict(schema=1,image=pair['images'][isa]['id'],source_revision=pair['source']['revision'],
                        manifest_digest=digest(manifest),selected=len(manifest['cells']),correctness_passed=True,
                        cells=[dict(row,outcome='passed',return_code=0) for row in manifest['cells']]))
            with patch.object(suite,'build_driver',return_value='tools'),patch.object(suite,'run_in_driver',side_effect=execute):
                with self.assertRaisesRegex(ValueError,'KL phase failures'):
                    suite.run_e2e(SimpleNamespace(model_kld_canaries=True,models=root,model_ramdisk_root=root),pair,manifest,root)
            self.assertEqual(visited,list(suite.ISAS))
            receipt=json.loads((root/'e2e-receipt.json').read_text())
            self.assertFalse(receipt['complete']);self.assertEqual(set(receipt['model_kld_reports']),set(suite.ISAS))
    def test_master_cannot_omit_kl_or_accept_changed_report(self):
        import tempfile,json
        import run_published_image_suite as suite
        from test_published_image_suite import image_pair,write_e2e_bundle
        from production_artifacts import write_json
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pair=image_pair();write_e2e_bundle(root,pair)
            suite.admit_e2e(root,pair,require_model_kld=True)
            receipt=json.loads((root/'e2e-receipt.json').read_text());receipt['model_kld_required']=False
            write_json(root/'e2e-receipt.json',receipt)
            with self.assertRaisesRegex(ValueError,'requires GPU KL'):suite.admit_e2e(root,pair,require_model_kld=True)
            receipt['model_kld_required']=True;write_json(root/'e2e-receipt.json',receipt)
            report=root/'avx2/kld/kld-canaries.json';data=json.loads(report.read_text());data['cpu_isa']='AVX512';write_json(report,data)
            with self.assertRaises(ValueError):suite.admit_e2e(root,pair,require_model_kld=True)
    def test_outer_driver_labels_and_retires_numerical_children_on_cancel(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import tempfile
        import run_published_image_suite as suite
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'cache').mkdir();lane=root/'kld';lane.mkdir();socket=root/'docker.sock';socket.touch()
            args=SimpleNamespace(output=root,models=root,model_ramdisk_root=root,e2e_bundle=None,suite='e2e',model_kld_canaries=True)
            with patch.dict(suite.os.environ,{'DOCKER_HOST':'unix://'+str(socket)}),patch.object(suite.docker_paths,'device_args',return_value=[]),patch.object(suite.docker_paths,'mounts',return_value=[]),patch.object(suite.pipeline,'run',side_effect=KeyboardInterrupt) as run,patch.object(suite,'retire_coding_children') as retire,patch.object(suite,'restore_lane_ownership'):
                with self.assertRaises(KeyboardInterrupt):suite.run_in_driver(args,'tools',['python','canary.py'],lane,'kld.log')
            command=run.call_args.args[0];owner=command[command.index('--name')+1]
            self.assertIn(suite.docker_paths.CODING_DRIVER_ENV+'='+owner,command)
            self.assertIn(suite.docker_paths.CODING_OWNER_LABEL+'='+owner,command)
            retire.assert_called_once_with(owner,lane/'outer-cleanup.json')
            self.assertTrue(lane.stat().st_mode & 0o020)


class ModelKLDPackagingTests(unittest.TestCase):
    """The native observer and compact proofs survive installed-image boundaries."""
    def test_asset_names_and_download_do_not_require_prior_local_receipts(self):
        from unittest.mock import patch
        import tempfile
        import publish_master_release as release
        import build_public_docs as docs
        with patch.object(Path,'read_text',side_effect=AssertionError('Naming read local evidence')):
            names=release.release_asset_sources(Path('/missing/e2e'),Path('/missing/benchmarks'),model_kld=True)
            legacy=release.release_asset_sources(Path('/missing/e2e'),Path('/missing/benchmarks'))
        self.assertIn('kld-avx2.json',names);self.assertNotIn('kld-avx2.json',legacy)
        metadata=dict(tag_name='2026-10-18.1',assets=[dict(name=n) for n in names])
        with tempfile.TemporaryDirectory() as temp,patch.object(docs.subprocess,'run') as run:
            docs.download_assets('Llaminar/llaminar',metadata,Path(temp)/'download')
            self.assertIn('kld-avx512.json',run.call_args.args[0])
            metadata['assets']=[a for a in metadata['assets'] if a['name']!='kld-avx2.json']
            with self.assertRaises(ValueError):docs.download_assets('Llaminar/llaminar',metadata,Path(temp)/'other')
    def test_installed_observer_must_exist_and_is_sealed(self):
        from unittest.mock import patch
        import tempfile,json
        import prebuilt_test_image as installed
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);binary=root/'llaminar_model_kld_canary'
            registration={'tests':[dict(name='V2_Integration_ModelKLDCanaryCLI',command=[str(binary),'--help'],properties={'LABELS':['ProductionTestPreflight']})]}
            with patch.object(installed.subprocess,'check_output',return_value=json.dumps(registration)):
                with self.assertRaisesRegex(ValueError,'executable is missing'):installed.inventory(root)
                for name in ['CMakeCache.txt','compile_commands.json','build.ninja','CMakeFiles/rules.ninja','CTestTestfile.cmake']:
                    path=root/name;path.parent.mkdir(exist_ok=True);path.write_text('fixture')
                binary.write_bytes(b'placeholder native observer')
                inventory=installed.inventory(root);self.assertIn(binary.name,inventory['files'])
                receipt=root/'receipt.json';receipt.write_text(json.dumps(dict(schema=1,inventory=inventory)))
                installed.validate(receipt,root)
                binary.write_bytes(b'changed binary')
                with self.assertRaises(ValueError):installed.validate(receipt,root)

if __name__=='__main__':unittest.main()
