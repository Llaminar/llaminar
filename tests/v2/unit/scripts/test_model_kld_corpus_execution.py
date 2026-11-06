"""Device-free adapter tests; toy source bytes and mocked native processes only.

No fixture metadata below is a native model, build, driver or approval receipt.
The existing execute_cell lifecycle is exercised without launching a subprocess.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts/ci'))
import model_kld_corpus_execution as adapter
import run_model_kld_canaries as runner
from model_kld_corpus import metadata_digest
from test_model_kld_corpus_v2 import manifest, absolute_pilot, observation


def fixture(root):
    """Construct four honestly toy stat-pinned inputs, no original model reads."""
    m=manifest()
    for index,descriptor in enumerate(m['source']['shards']):
        path=root/descriptor['name'];path.write_bytes(bytes([index]))
        descriptor['path']=str(path)
        pin=adapter.identity(path)
        descriptor['object_identity']=json.dumps(dict(path=pin[0],device=pin[1],inode=pin[2],bytes=pin[3],mtime_ns=pin[4],ctime_ns=pin[5]))
    t=m['topologies'][0]
    t['expected_config']={'configuration':{'device_for_this_rank':dict(hostname='',numa_node=0,device_type='CPU',device_ordinal=0)}}
    t['runtime_config_digest']=metadata_digest(t['expected_config'])
    t['engine_argv']=['-m',m['source']['shards'][0]['path'],'--no-mtp']
    return dict(manifest=m,inputs={}),t


class CorpusExecutionTests(unittest.TestCase):
    """Actual adapter boundaries and the one original process lifecycle."""

    def test_physical_observation_includes_storage_only_without_changing_compute(self):
        for local, remote in (('CUDA', 'ROCm'), ('ROCm', 'CUDA')):
            configuration = {'device_for_this_rank': dict(device_type=local, device_ordinal=0),
                             'domain_definitions': [{'devices': [dict(device_type='CPU', device_ordinal=0)]}]}
            endpoint = lambda backend: dict(world_rank=0, backend=backend, ordinal=0)
            placement = dict(mode='DeclaredRankLocal', owners=[dict(participant=endpoint(local)),
                             dict(participant=endpoint(remote))], coordinator=endpoint(local))
            configuration['ngram_table_policy'] = dict(gpu_adaptive=dict(placement=placement))
            config = dict(configuration=configuration)
            before = copy.deepcopy(config)
            self.assertEqual(adapter.physical_devices(config), ['cpu:0', 'cuda:0', 'rocm:0'])
            self.assertEqual(config, before)
            # Repeated references across source/owner/coordinator roles observe
            # one physical device; they never turn storage into compute.
            placement['owners'].append(dict(participant=endpoint(remote)))
            self.assertEqual(adapter.physical_devices(config), ['cpu:0', 'cuda:0', 'rocm:0'])
            placement.update(mode='ConsumerLocal', owners=[], coordinator=None)
            self.assertEqual(adapter.physical_devices(config), ['cpu:0', local.lower()+':0'])

    def test_physical_storage_observation_rejects_malformed_or_nonlocal_endpoints(self):
        original = dict(configuration=dict(device_for_this_rank=dict(device_type='CUDA', device_ordinal=0),
            ngram_table_policy=dict(gpu_adaptive=dict(placement=dict(mode='DeclaredRankLocal',
                owners=[dict(participant=dict(world_rank=0, backend='ROCm', ordinal=0))],
                coordinator=dict(world_rank=0, backend='CUDA', ordinal=0))))))
        bad_endpoints = [None, {}, dict(world_rank=1, backend='ROCm', ordinal=0),
            dict(world_rank=-1, backend='ROCm', ordinal=0), dict(world_rank=False, backend='ROCm', ordinal=0),
            dict(world_rank=0, backend='CPU', ordinal=0), dict(world_rank=0, backend='Metal', ordinal=0),
            dict(world_rank=0, backend={}, ordinal=0),
            dict(world_rank=0, backend='ROCm', ordinal=-1), dict(world_rank=0, backend='ROCm', ordinal=True),
            dict(world_rank=0, backend='ROCm', ordinal=0, extra=1)]
        for endpoint in bad_endpoints:
            for role in ('owner', 'coordinator'):
                config = copy.deepcopy(original)
                placement = config['configuration']['ngram_table_policy']['gpu_adaptive']['placement']
                if role == 'owner': placement['owners'][0]['participant'] = endpoint
                else: placement['coordinator'] = endpoint
                with self.subTest(endpoint=endpoint, role=role), self.assertRaises(ValueError):
                    adapter.physical_devices(config)
        for owners in (None, {}, [None], [{}]):
            config = copy.deepcopy(original)
            config['configuration']['ngram_table_policy']['gpu_adaptive']['placement']['owners'] = owners
            with self.assertRaises(ValueError): adapter.physical_devices(config)

    def test_selection_is_corpus_owned_and_rejects_unknown_duplicate(self):
        with tempfile.TemporaryDirectory() as temp:
            c,t=fixture(Path(temp))
            self.assertEqual(adapter.select_topologies(c,None),[t])
            for selected in (['foreign'],[t['id'],t['id']]):
                with self.assertRaises(ValueError):adapter.select_topologies(c,selected)

    def test_actual_public_parser_projection_and_no_changed_arguments(self):
        with tempfile.TemporaryDirectory() as temp:
            c,t=fixture(Path(temp))
            described=dict(schema='llaminar.model-kld-described-config.v1',config=t['expected_config'],
                runtime_config_digest=t['runtime_config_digest'],engine_argv=t['engine_argv'])
            self.assertEqual(adapter.validate_described_config(t,described),t['engine_argv'])
            for key,value in [('config',{}),('runtime_config_digest','foreign'),('engine_argv',['--other']),('schema','old')]:
                bad=dict(described);bad[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):adapter.validate_described_config(t,bad)
            for argv in (None,[],['bad\0arg']):
                bad=dict(t,engine_argv=argv)
                with self.assertRaises(ValueError):adapter.validate_described_config(bad,described)

    def test_original_source_pin_and_inactive_predictor_are_both_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t=fixture(root)
            predictor=root/'inactive.gguf';predictor.write_bytes(b'x')
            pin=adapter.identity(predictor)
            c['manifest']['source']['shared_predictor']=dict(name=predictor.name,path=str(predictor),bytes=1,
                object_identity=json.dumps(dict(path=pin[0],device=pin[1],inode=pin[2],bytes=pin[3],mtime_ns=pin[4],ctime_ns=pin[5])))
            a=adapter.CorpusExecution(c,t)
            self.assertIn(predictor,a.sources)
            self.assertEqual(a.backends,['cpu']);self.assertEqual(a.devices,['cpu:0'])
            self.assertEqual(set(a.environment),set(adapter.SELECTORS.values()))
            predictor.write_bytes(b'xx')
            with self.assertRaises(ValueError):adapter.CorpusExecution(c,t)

    def test_same_size_pre_admission_replacement_cannot_claim_original_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t=fixture(root)
            original=Path(c['manifest']['source']['shards'][0]['path'])
            replacement=root/'replacement';replacement.write_bytes(b'z');replacement.replace(original)
            self.assertEqual(original.stat().st_size,c['manifest']['source']['shards'][0]['bytes'])
            with self.assertRaisesRegex(ValueError,'reference-issued metadata'):adapter.CorpusExecution(c,t)
        with tempfile.TemporaryDirectory() as temp:
            c,t=fixture(Path(temp))
            descriptor=c['manifest']['source']['shards'][0]
            descriptor['object_identity']='"not-an-original-stat-descriptor"'
            with self.assertRaises(ValueError):adapter.CorpusExecution(c,t)

    def test_reboot_device_renumbering_keeps_all_current_runtime_pins(self):
        """Only the frozen st_dev may differ; current pins remain completely strict."""
        from model_kld_corpus import unchanged
        with tempfile.TemporaryDirectory() as temp:
            c,t=fixture(Path(temp))
            for descriptor in c['manifest']['source']['shards']:
                frozen=json.loads(descriptor['object_identity']);frozen['device']+=1
                descriptor['object_identity']=json.dumps(frozen)
            a=adapter.CorpusExecution(c,t)
            for path in a.sources:self.assertEqual(a.pins[str(path)],adapter.identity(path))
            path=a.sources[0];current=adapter.identity(path)
            changed=list(current);changed[1]+=1
            with patch('model_kld_corpus.identity',return_value=tuple(changed)), self.assertRaisesRegex(ValueError,'changed during execution'):
                unchanged({str(path):current})

    def test_durable_file_fields_and_device_types_remain_strict(self):
        """Reference device renumbering cannot mask any independent file mutation."""
        with tempfile.TemporaryDirectory() as temp:
            c,t=fixture(Path(temp));descriptor=c['manifest']['source']['shards'][0]
            frozen=json.loads(descriptor['object_identity']);frozen['device']+=1
            for field in ('path','inode','bytes','mtime_ns','ctime_ns'):
                changed=dict(frozen);changed[field]=changed[field]+'other' if field=='path' else changed[field]+1
                descriptor['object_identity']=json.dumps(changed)
                with self.subTest(field=field), self.assertRaises(ValueError):adapter.CorpusExecution(c,t)
            for value in (-1,True,1.0,'other'):
                changed=dict(frozen);changed['device']=value;descriptor['object_identity']=json.dumps(changed)
                with self.subTest(device=value), self.assertRaises(ValueError):adapter.CorpusExecution(c,t)

    def test_prepare_uses_same_parser_binary_and_publishes_exact_workload(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t=fixture(root);a=adapter.CorpusExecution(c,t)
            binary=root/'observer';binary.write_bytes(b'fixture-not-executable')
            output=root/'output';output.mkdir()
            described=dict(schema='llaminar.model-kld-described-config.v1',config=t['expected_config'],
                runtime_config_digest=t['runtime_config_digest'],engine_argv=t['engine_argv'])
            with patch.object(runner.subprocess,'check_output',return_value=json.dumps(described)) as describe:
                path,argv=a.prepare(runner,SimpleNamespace(binary=binary),output,{'PATH':'fixture'})
            self.assertEqual(describe.call_args.args[0],[str(binary),'--describe-config','--',*t['engine_argv']])
            self.assertEqual(json.loads(path.read_text()),dict(schema='llaminar.model-kld-workload.v2',corpus=c['manifest'],topology_id=t['id']))
            self.assertEqual(argv,t['engine_argv']);self.assertFalse((output/'logits').exists())

    def test_score_rejects_absent_perf_and_wrong_actual_executable(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t=fixture(root);a=adapter.CorpusExecution(c,t)
            binary=root/'observer';binary.write_bytes(b'fixture');a.binary_pin=adapter.identity(binary);a.binary_sha256=adapter.file_sha256(binary)
            payload=root/'logits';payload.mkdir()
            native=dict(native_identity={'executable_sha256':a.binary_sha256},actual_owner={'fixture':True},path_records=[])
            (payload/'logits.json').write_text(json.dumps(native));(root/'perf-stats.json').write_text('{"records":[]}')
            with self.assertRaisesRegex(ValueError,'PerfStats'):a.score(runner,payload,root)
            (root/'perf-stats.json').write_text('{"records":[{"fixture":true}]}')
            native['native_identity']['executable_sha256']='foreign';(payload/'logits.json').write_text(json.dumps(native))
            with self.assertRaisesRegex(ValueError,'executable'):a.score(runner,payload,root)

    def _replay_original_lifecycle_retirement(self, distributed=False):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);output=root/'cell'
            args=SimpleNamespace(binary=root/'observer',keep_logits=False)
            a=SimpleNamespace(topology={'id':'fixture','expected_config':{'configuration':{'mpi_procs':2 if distributed else 1}}},devices=['cpu:0'],backends=['cpu'],environment={},
                prepare=Mock(return_value=(root/'workload',[])),score=Mock())
            if distributed:
                args.mpi_rankfile=root/'original.rankfile';args.mpi_rankfile.write_text('rank 0=localhost slot=0:0-27\nrank 1=localhost slot=1:0-27\n')
                for name in ('llaminar_mpi_session','mpirun'):
                    executable=root/name;executable.write_text('fixture-only-not-a-runtime');executable.chmod(0o700)
            child=Mock(pid=123);child.wait.return_value=1
            # This fixture owns a simulated launch; process traversal itself is
            # exercised with real proc children in OwnedProcessTreeResources.
            local_owner=Mock(spec=runner.OwnedProcessTree)
            local_owner.snapshot.return_value={'root':{'pid':123,'start_ticks':1},
                'processes':[],'retired':[],'memory_scope':'individual_processes_not_summed'}
            events=[]
            def launch(command,**kwargs):
                if distributed:
                    self.assertEqual(command[0],str(root/'llaminar_mpi_session'))
                    self.assertIn(str(args.mpi_rankfile),command);self.assertIn('core',command)
                    self.assertTrue((output/'mpi-session').is_dir())
                    self.assertIn(str(args.binary),command)
                else:self.assertEqual(command[0],str(args.binary))
                self.assertFalse((output/'logits').exists())
                (output/'logits').mkdir();(output/'logits/logits.json').write_text('{"complete":false,"fixture":true}')
                if distributed:
                    for rank in range(2):
                        (output/'logits'/f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank,fixture_only=True,complete=False)))
                return child
            def retire(*unused):events.append('retired');self.assertTrue((output/'logits/logits.json').exists())
            sample=dict(errors=[],cells={'canary':{'errors':[]}},filesystems={'output':{'available_bytes':10**9}})
            with patch.dict(runner.os.environ,{'PATH':str(root)} if distributed else {}), \
                 patch.object(runner,'retire',side_effect=retire),patch.object(runner.driver,'begin'),\
                 patch.object(runner.driver,'finish',return_value=True),patch.object(runner.driver,'validate_evidence'),\
                 patch.object(runner.resources,'collect',return_value=sample),patch.object(runner.subprocess,'Popen',side_effect=launch),\
                 patch.object(runner,'OwnedProcessTree',return_value=local_owner) as observe:
                with self.assertRaisesRegex(ValueError,'observer failed'):
                    runner.execute_cell(args,{'inputs':{}},'qwen4exp','cpu',None,output,None,adapter=a)
            observe.assert_called_once_with(child.pid);local_owner.snapshot.assert_called_once_with()
            self.assertEqual(json.loads((output/'resources.jsonl').read_text())['local_process_tree'],
                local_owner.snapshot.return_value)
            self.assertEqual(events,['retired']);a.score.assert_not_called()
            self.assertFalse((output/'logits').exists())
            self.assertEqual(json.loads((output/'native-observation.json').read_text())['complete'],False)
            self.assertTrue(json.loads((output/'controller.json').read_text())['native_retired'])
            if distributed:
                controller=json.loads((output/'controller.json').read_text())
                self.assertEqual(controller['native_launch']['overlap_policy'],'strict')
                for rank in range(2):self.assertEqual(json.loads((output/f'rank-{rank}.json').read_text())['rank'],rank)

    def test_one_existing_lifecycle_retires_before_failed_native_artifact_cleanup(self):
        self._replay_original_lifecycle_retirement()

    def test_rankfile_launch_keeps_original_process_retirement_and_rank_evidence(self):
        self._replay_original_lifecycle_retirement(distributed=True)

    def test_rankfile_original_process_tree_tracks_and_retires_real_host_children(self):
        """Host-only failing fixture processes exercise original ownership, never MPI/model readiness."""
        import time
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'output';rankfile=root/'rankfile';rankfile.write_text('fixture authored mapping, not executed by native MPI')
            ready=root/'ready'
            # The fixtures execute real child/join lifetimes and return failure.
            # No wrapper asserts model/native readiness or numerical success.
            wrapper='#!/usr/bin/python3\nimport subprocess,sys\np=subprocess.Popen(sys.argv[sys.argv.index("--")+1:]);sys.exit(p.wait())\n'
            for name in ('llaminar_mpi_session','mpirun'):
                executable=root/name;executable.write_text(wrapper);executable.chmod(0o700)
            native=root/'observer'
            native.write_text('#!/usr/bin/python3\nimport json,pathlib,subprocess,sys,time\n'
                'out=pathlib.Path(sys.argv[sys.argv.index("--output")+1]);out.mkdir();(out/"logits.json").write_text(json.dumps(dict(fixture_only=True,complete=False)))\n'
                'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(.6)"])\n'
                + 'pathlib.Path('+repr(str(ready))+').write_text("host children live")\n'
                + 'code=p.wait();sys.exit(1)\n')
            native.chmod(0o700)
            args=SimpleNamespace(binary=native,keep_logits=False,mpi_rankfile=rankfile)
            a=SimpleNamespace(topology={'id':'host-process-fixture','expected_config':{'configuration':{'mpi_procs':2}}},
                devices=['cpu:0'],backends=['cpu'],environment={},prepare=Mock(return_value=(root/'workload',['--mpi-procs','2'])),score=Mock())
            sample=dict(errors=[],cells={'canary':{'errors':[]}},filesystems={'output':{'available_bytes':10**9}})
            def observe_resources(unused):
                deadline=time.monotonic()+5
                while not ready.exists():
                    if time.monotonic()>deadline:raise AssertionError('host fixture did not publish child startup')
                    time.sleep(.005)
                return sample
            with patch.dict(runner.os.environ,{'PATH':str(root)}),patch.object(runner.driver,'begin'),\
                 patch.object(runner.driver,'finish',return_value=True),patch.object(runner.driver,'validate_evidence'),\
                 patch.object(runner.resources,'collect',side_effect=observe_resources):
                with self.assertRaisesRegex(ValueError,'observer failed'):
                    runner.execute_cell(args,{'inputs':{}},'qwen4exp','cpu',None,output,None,adapter=a)
            snapshot=json.loads((output/'resources.jsonl').read_text())['local_process_tree']
            self.assertGreaterEqual(len(snapshot['processes']),4)
            self.assertEqual(snapshot['memory_scope'],'individual_processes_not_summed')
            for original in snapshot['processes']:
                from resource_owned_process_tree import process_stat
                try:current=process_stat(Path('/proc')/str(original['pid']))
                except FileNotFoundError:continue
                self.assertNotEqual(current['start_ticks'],original['start_ticks'])
            state=json.loads((output/'controller.json').read_text())
            self.assertTrue((output/'mpi-session').is_dir())
            self.assertTrue(state['native_retired']);self.assertFalse(state['passed']);self.assertEqual(state['returncode'],1)
            self.assertFalse((output/'logits').exists());a.score.assert_not_called()

    def test_rankfile_command_is_explicit_pinned_and_old_single_rank_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rankfile=root/'rankfile';rankfile.write_text('fixture original authored MPI mapping')
            for name in ('llaminar_mpi_session','mpirun'):
                executable=root/name;executable.write_text('fixture-only-not-executed');executable.chmod(0o700)
            args=SimpleNamespace(binary=root/'observer',mpi_rankfile=rankfile)
            original=[str(args.binary),'--workload','fixture','--output','fixture-output','--','--mpi-procs','2']
            a=SimpleNamespace(topology={'expected_config':{'configuration':{'mpi_procs':2}}})
            output=root/'output';output.mkdir()
            command,pins,proof=runner.original_local_launch(args,a,None,output,original,{'PATH':str(root)})
            self.assertEqual(command[0],str(root/'llaminar_mpi_session'))
            self.assertEqual(command[-len(original):],original)
            self.assertEqual(command[command.index('-np')+1],'2')
            self.assertEqual(command[command.index('--rankfile')+1],str(rankfile))
            self.assertNotIn('--oversubscribe',command)
            self.assertEqual(command[command.index('--bind-to')+1],'core')
            self.assertTrue((output/'mpi-session').is_dir())
            self.assertTrue(proof['performance_comparison_eligible'])
            runner.unchanged(pins)
            with self.assertRaisesRegex(ValueError,'fresh'):runner.original_local_launch(args,a,None,output,original,{'PATH':str(root)})
            other=root/'symlink-output';other.mkdir();(other/'mpi-session').symlink_to(root,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'fresh'):runner.original_local_launch(args,a,None,other,original,{'PATH':str(root)})
            rankfile.write_text('changed original mapping')
            with self.assertRaises(ValueError):runner.unchanged(pins)
            args.mpi_rankfile=None
            one=SimpleNamespace(topology={'expected_config':{'configuration':{'mpi_procs':1}}})
            self.assertEqual(runner.original_local_launch(args,one,None,root,original,{'PATH':str(root)}),(original,{},None))
            self.assertEqual(runner.original_local_launch(args,None,{'fixture_image':True},root,original,{'PATH':str(root)}),(original,{},None))

    def test_rankfile_missing_scope_and_launcher_fail_before_native_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rankfile=root/'rankfile';rankfile.write_text('fixture')
            args=SimpleNamespace(binary=root/'observer',mpi_rankfile=None)
            a=SimpleNamespace(topology={'expected_config':{'configuration':{'mpi_procs':2}}})
            call=lambda image=None:runner.original_local_launch(args,a,image,root,['fixture-native'],{'PATH':str(root)})
            with self.assertRaisesRegex(ValueError,'requires an explicit'):call()
            args.mpi_rankfile=rankfile
            with self.assertRaisesRegex(ValueError,'local declared'):call({'fixture_image':True})
            a.topology['expected_config']['configuration']['mpi_procs']=1
            with self.assertRaisesRegex(ValueError,'local declared'):call()
            a.topology['expected_config']['configuration']['mpi_procs']=3
            with self.assertRaises(OSError):call() # No original session launcher; no alternate search.
            self.assertFalse((root/'mpi-session').exists())
            launcher=root/'llaminar_mpi_session';launcher.write_text('fixture');launcher.chmod(0o700)
            with self.assertRaisesRegex(ValueError,'mpirun'):call()
            args.mpi_rankfile=root/'absent'
            with self.assertRaises(OSError):call()

    def test_explicit_manifest_and_rankfile_cli_keep_v2_scope(self):
        with patch.object(runner,'run_corpus_v2',return_value=0) as execute:
            self.assertEqual(runner.main(['--binary','fixture','--report','report','--corpus-root','reference',
                '--corpus-catalog','catalog','--corpus-manifest','metadata.json','--mpi-rankfile','rankfile']),0)
            args=execute.call_args.args[0]
            self.assertEqual(args.corpus_manifest,Path('metadata.json'));self.assertEqual(args.mpi_rankfile,Path('rankfile'))
        for extra in (['--corpus-manifest','metadata'],['--mpi-rankfile','rankfile']):
            with self.subTest(extra=extra),self.assertRaises(ValueError):runner.main(['--binary','fixture','--report','report',*extra])
        with self.assertRaises(ValueError):runner.main(['--container-image','fixture','--report','report','--corpus-catalog','catalog','--mpi-rankfile','rankfile'])

    def test_first_local_collection_preserves_unqualified_uncalibrated_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t=fixture(root)
            binary=root/'observer';binary.write_bytes(b'fixture-not-executable')
            (root/'CMakeCache.txt').write_text('CMAKE_BUILD_TYPE:STRING=Integration\n')
            sources=('scripts/ci/run_model_kld_canaries.py','scripts/ci/model_kld_corpus_execution.py',
                'scripts/ci/model_kld_corpus_v2.py','scripts/ci/full_vocabulary_metrics.py',
                'src/v2/tools/ModelKLDCanary.cpp','src/v2/tools/ModelKLDQFNContract.h')
            for name in sources:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture source only')
            args=SimpleNamespace(binary=binary,container_image=None,corpus_root=root,corpus_catalog=root/'catalog',
                topology_id=None,baseline_report=None,baseline_approval=None,calibration=None,
                prerequisites_report=None,report=root/'run.json',source_revision='fixture',keep_logits=False)
            result=dict(complete=True,observations_complete=True,passed=None,verdict='UNCALIBRATED')
            with patch.object(runner,'ROOT',root),patch.object(adapter.corpus_v2,'load_corpus',return_value=c),                 patch.object(runner,'execute_cell',return_value=result) as execute:
                self.assertEqual(runner.run_corpus_v2(args),0)
            state=json.loads(args.report.read_text())
            self.assertTrue(state['complete']);self.assertIsNone(state['passed'])
            self.assertFalse(state['certification_eligible']);self.assertTrue(state['diagnostic_only'])
            self.assertEqual(state['verdict'],'UNCALIBRATED');self.assertFalse(state['prerequisites']['available'])
            self.assertEqual(state['qualification'],'local_diagnostic_only_prerequisites_unavailable')
            self.assertEqual(len(state['source_inputs']),6)
            self.assertEqual(state['binary']['sha256'],adapter.file_sha256(binary))
            self.assertIsNotNone(execute.call_args.kwargs['adapter'])

    def test_baseline_inputs_cannot_be_partial_and_v1_options_stay_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            c,t=fixture(Path(temp))
            for kwargs in ({'baseline':'missing'},{'approval':'missing'},{'calibration':'missing'}):
                with self.assertRaises(ValueError):adapter.CorpusExecution(c,t,**kwargs)
        with patch.object(runner,'run_corpus_v2',return_value=0) as v2:
            self.assertEqual(runner.main(['--binary','fixture','--report','report','--corpus-catalog','catalog']),0)
            v2.assert_called_once()
        with self.assertRaises(ValueError):runner.main(['--binary','fixture','--report','report','--topology-id','foreign'])


class AbsolutePilotExecutionTests(unittest.TestCase):
    """Authored toy metrics/policies only; no native admission or model proof."""

    def test_pre_admission_policy_parser_uses_exact_original_row_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            c,t=fixture(Path(temp));m=c['manifest'];policy=absolute_pilot(m)
            scope=adapter.validate_absolute_pilot(policy,m,t)
            self.assertFalse(scope['whole_corpus_accepted'])
            self.assertEqual(scope['diagnostic_verdict'],'UNAVAILABLE')
            for mutate in (lambda p:p.update(corpus_manifest_digest='foreign'),
                           lambda p:p['groups'].pop(),
                           lambda p:p['groups'].append(copy.deepcopy(p['groups'][0])),
                           lambda p:p['groups'][0].update(role='foreign'),
                           lambda p:p['groups'][0]['maximum'].update(mean_kl_nats=float('nan'))):
                bad=copy.deepcopy(policy);mutate(bad)
                with self.assertRaises(ValueError):adapter.validate_absolute_pilot(bad,m,t)

    def test_adapter_policy_pin_mixing_and_real_summary_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t=fixture(root);m=c['manifest']
            m['histories'][1]['context_group']='boundary'
            policy=absolute_pilot(m,diagnostic_contexts=('boundary',))
            path=root/'pilot.json';path.write_text(json.dumps(policy))
            for key in ('baseline','approval','calibration'):
                with self.assertRaisesRegex(ValueError,'Absolute pilot'):
                    adapter.CorpusExecution(c,t,absolute_pilot=path,**{key:root/'absent'})
            a=adapter.CorpusExecution(c,t,absolute_pilot=path)
            binary=root/'observer';binary.write_bytes(b'fixture')
            a.binary_pin=adapter.identity(binary);a.binary_sha256=adapter.file_sha256(binary)
            payload=root/'logits';payload.mkdir()
            raw=dict(native_identity=dict(executable_sha256=a.binary_sha256),actual_owner={'toy':True},
                     path_records=[],proofs={'toy':True})
            (payload/'logits.json').write_text(json.dumps(raw))
            (root/'perf-stats.json').write_text('{"records":[{"toy":true}]}')
            metrics=observation(m,replacements={'b:decode:3':[-100.,0.,0.,0.,0.,0.]})
            with patch.object(adapter.corpus_v2,'score_observation',return_value=metrics):
                result=a.score(runner,payload,root)
            self.assertTrue(result['passed'])
            self.assertFalse(result['whole_corpus_accepted']);self.assertFalse(result['calibrated'])
            self.assertFalse(result['certification_eligible'])
            self.assertEqual(result['scope'],'selected_topology_required_groups_only')
            self.assertEqual(result['diagnostic_verdict'],'UNAVAILABLE')
            self.assertEqual(len(result['acceptance_scope']['diagnostic_groups']),2)
            path.write_text(json.dumps(dict(policy,corpus_manifest_digest='foreign')))
            with self.assertRaises(ValueError):a.score(runner,payload,root)

    def args(self,root,policy):
        return SimpleNamespace(binary=root/'observer',container_image=None,corpus_root=root,
            corpus_catalog=root/'catalog',topology_id=None,baseline_report=None,baseline_approval=None,
            calibration=None,absolute_pilot=policy,prerequisites_report=None,report=root/'run.json',
            source_revision='authored-fixture',keep_logits=False)

    def test_runner_rejects_invalid_or_mixed_policy_before_execute(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t=fixture(root)
            (root/'observer').write_bytes(b'fixture');(root/'CMakeCache.txt').write_text('CMAKE_BUILD_TYPE:STRING=Integration\n')
            path=root/'pilot.json';path.write_text(json.dumps(absolute_pilot(c['manifest'])))
            for mode in ('multi','baseline_report','baseline_approval','calibration','foreign'):
                args=self.args(root,path);current=copy.deepcopy(c)
                if mode=='multi':current['manifest']['topologies'].append(dict(t,id='other'))
                elif mode=='foreign':path.write_text('{}')
                else:setattr(args,mode,root/'absent')
                with patch.object(adapter.corpus_v2,'load_corpus',return_value=current),patch.object(runner,'execute_cell') as execute:
                    with self.assertRaises(ValueError):runner.run_corpus_v2(args)
                    execute.assert_not_called()
                path.write_text(json.dumps(absolute_pilot(c['manifest'])))

    def test_runner_required_only_green_and_red_top_level_scope(self):
        for passed in (True,False):
            with self.subTest(passed=passed),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);c,t=fixture(root)
                (root/'observer').write_bytes(b'fixture');(root/'CMakeCache.txt').write_text('CMAKE_BUILD_TYPE:STRING=Integration\n')
                for name in ('scripts/ci/run_model_kld_canaries.py','scripts/ci/model_kld_corpus_execution.py',
                    'scripts/ci/model_kld_corpus_v2.py','scripts/ci/full_vocabulary_metrics.py',
                    'src/v2/tools/ModelKLDCanary.cpp','src/v2/tools/ModelKLDQFNContract.h'):
                    target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('fixture source')
                path=root/'pilot.json';path.write_text(json.dumps(absolute_pilot(c['manifest'])))
                args=self.args(root,path)
                result=dict(complete=True,observations_complete=True,passed=passed,verdict='GREEN' if passed else 'RED')
                with patch.object(runner,'ROOT',root),patch.object(adapter.corpus_v2,'load_corpus',return_value=c),patch.object(runner,'execute_cell',return_value=result):
                    self.assertEqual(runner.run_corpus_v2(args),0 if passed else 1)
                report=json.loads(args.report.read_text())
                self.assertEqual(report['verdict'],'REQUIRED_GROUPS_GREEN' if passed else 'REQUIRED_GROUPS_RED')
                self.assertFalse(report['whole_corpus_accepted']);self.assertFalse(report['calibrated'])
                self.assertFalse(report['certification_eligible']);self.assertTrue(report['diagnostic_only'])
                self.assertEqual(report['diagnostic_verdict'],'UNAVAILABLE')

    def test_v1_and_container_routes_cannot_consume_pilot(self):
        with self.assertRaises(ValueError):
            runner.main(['--binary','fixture','--report','report','--absolute-pilot','pilot'])
        args=self.args(Path('/fixture'),Path('/fixture/pilot'));args.container_image='fixture'
        with self.assertRaises(ValueError):runner.run_corpus_v2(args)



def paired_fixture(root):
    """Tiny private metadata only; no executable, native owner or approved result."""
    c,t=fixture(root)
    exact='Synthetic.ProductionParity/Fixture'
    t['canonical_cell']=exact
    t['expected_config']['configuration']['model_path']=c['manifest']['source']['shards'][0]['path']
    t['expected_config']['configuration']['shared_predictor']=None
    t['expected_config']['configuration']['max_seq_len']=4096
    t['runtime_config_digest']=metadata_digest(t['expected_config'])
    documents={'workload':c['manifest'], 'catalog':{'schema':2,'manifest_digest':metadata_digest(c['manifest'])},
               'pilot':absolute_pilot(c['manifest']), 'benchmark':{'scope':'synthetic-unexecuted-obligation'}}
    refs={}
    for name,value in documents.items():
        path=root/('manifest.json' if name=='workload' else name+'.json')
        path.write_text(json.dumps(value))
        refs[name]=dict(path=str(path),sha256=adapter.hashlib.sha256(path.read_bytes()).hexdigest())
    record=dict(model_parity_schema=1,id='Fixture',model=c['manifest']['source']['shards'][0]['path'],
        runtime=dict(generation=dict(mtp_policy='off')),
        diagnostic_benchmark=dict(schema=1,policy='diagnostic_declared',
            certification_eligible=False,context_length=4096,args=['--no-mtp']),
        paired_workload=dict(schema=2,policy='diagnostic_obligation',certification_eligible=False,
            role='required',case_id='Fixture',source_fields=['model'],arithmetic=copy.deepcopy(t['arithmetic']),
            accuracy=dict(arguments_from='diagnostic_benchmark',policy='initial_absolute_pilot',
                catalog=refs['catalog'],workload=refs['workload'],pilot=refs['pilot'],
                approved_baseline=False,calibrated=False,whole_corpus_accepted=False),
            benchmark=dict(policy='diagnostic_declared',arguments_from='diagnostic_benchmark',profile=refs['benchmark'])))
    return c,t,exact,record


class PairedCorpusExecutionTests(unittest.TestCase):
    """Join real APIs through synthetic documents; never launch a model process."""

    def test_focused_receipt_selects_only_explicit_local_diagnostic_inventory(self):
        """The opt-in admits no canonical/runtime eligibility for a diagnostic case."""
        import model_parity_inventory as inventory
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t,exact,record=paired_fixture(root)
            (root/'observer').write_text('not executable');(root/'matrix').write_text('not executable')
            record['inventory_role']='numerical_diagnostic'
            args=AbsolutePilotExecutionTests().args(root,None)
            args.corpus_catalog=None;args.paired_cell=exact
            args.diagnostic_focused_prerequisites_report=root/'focused-fixture-not-a-native-receipt'
            with patch.object(inventory,'discover',return_value=[(SimpleNamespace(name='original'),exact,record)]) as discover, \
                 patch.object(inventory,'matrix_binary',return_value=root/'matrix'), \
                 patch.object(adapter.corpus_v2,'load_corpus',return_value=c):
                runner.select_paired_corpus(args)
            self.assertIs(discover.call_args.args[1],inventory.InventoryScope.LOCAL_DIAGNOSTIC)
            args.prerequisites_report=root/'complete'
            with patch.object(inventory,'discover') as discover,self.assertRaises(ValueError):
                runner.select_paired_corpus(args)
            discover.assert_not_called()

    def test_focused_panel_keeps_accuracy_rows_and_never_publishes_certifiable_green(self):
        """All actual work remains in the original adapter; only prerequisite scope changes."""
        for passed in (True,False):
            with self.subTest(passed=passed),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);c,t,exact,record=paired_fixture(root)
                (root/'observer').write_text('not executable')
                (root/'CMakeCache.txt').write_text('CMAKE_BUILD_TYPE:STRING=Integration\n')
                selected=self.selection(root,c,exact,record)
                for name in ('scripts/ci/run_model_kld_canaries.py','scripts/ci/model_kld_corpus_execution.py',
                    'scripts/ci/model_kld_corpus_v2.py','scripts/ci/full_vocabulary_metrics.py',
                    'src/v2/tools/ModelKLDCanary.cpp','src/v2/tools/ModelKLDQFNContract.h'):
                    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('honestly synthetic test source')
                args=AbsolutePilotExecutionTests().args(root,selected.paths['pilot'])
                args._paired_selection=selected;args.corpus_catalog=selected.paths['catalog'];args.topology_id=[t['id']]
                args.diagnostic_focused_prerequisites_report=root/'focused.json'
                args.diagnostic_focused_prerequisites_report.write_text('fixture; validator mocked, never native evidence')
                result=dict(complete=True,observations_complete=True,passed=passed,verdict='GREEN' if passed else 'RED')
                scope=dict(focused_tests=['V2_Integration_FixtureDeviceFree'],focused_completed_ns=123)
                with patch.object(runner,'ROOT',root), \
                     patch('run_production_prerequisites.validate_focused_panel_prerequisites',return_value=scope) as focused, \
                     patch('run_production_prerequisites.validate_source_freshness') as full_source, \
                     patch('run_production_parity_campaigns.reuse_unchanged_production_test_preflight') as full, \
                     patch.object(runner,'execute_cell',return_value=result) as execute:
                    self.assertEqual(runner.run_corpus_v2(args),0 if passed else 1)
                self.assertEqual(focused.call_count,3);full.assert_not_called();full_source.assert_not_called()
                self.assertIs(execute.call_args.kwargs['adapter'].paired,selected)
                report=json.loads(args.report.read_text())
                self.assertEqual(report['mode'],'local_paired_focused_diagnostic')
                self.assertEqual(report['verdict'],'DIAGNOSTIC_REQUIRED_GROUPS_GREEN' if passed else 'DIAGNOSTIC_REQUIRED_GROUPS_RED')
                self.assertIsNone(report['passed']);self.assertEqual(report['diagnostic_passed'],passed)
                self.assertEqual(report['cells'],[result]);self.assertFalse(report['full_preflight'])
                self.assertFalse(report['certification_eligible']);self.assertTrue(report['diagnostic_only'])
                self.assertEqual(report['prerequisites']['original_completed_ns'],123)
                self.assertEqual(report['prerequisites']['native_backend_regressions'],'not_covered_by_DeviceFree_selection')

    def test_stale_focused_receipt_and_nonpaired_option_fail_before_native_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t,exact,record=paired_fixture(root)
            (root/'observer').write_text('not executable')
            (root/'CMakeCache.txt').write_text('CMAKE_BUILD_TYPE:STRING=Integration\n')
            selected=self.selection(root,c,exact,record)
            for name in ('scripts/ci/run_model_kld_canaries.py','scripts/ci/model_kld_corpus_execution.py',
                'scripts/ci/model_kld_corpus_v2.py','scripts/ci/full_vocabulary_metrics.py',
                'src/v2/tools/ModelKLDCanary.cpp','src/v2/tools/ModelKLDQFNContract.h'):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture source')
            args=AbsolutePilotExecutionTests().args(root,selected.paths['pilot'])
            args._paired_selection=selected;args.corpus_catalog=selected.paths['catalog'];args.topology_id=[t['id']]
            args.diagnostic_focused_prerequisites_report=root/'focused.json'
            args.diagnostic_focused_prerequisites_report.write_text('{}')
            with patch.object(runner,'ROOT',root), \
                 patch('run_production_prerequisites.validate_focused_panel_prerequisites',side_effect=ValueError('stale focused')),\
                 patch.object(runner,'execute_cell') as execute:
                with self.assertRaisesRegex(ValueError,'stale focused'):runner.run_corpus_v2(args)
                execute.assert_not_called()
        with self.assertRaises(ValueError):
            runner.main(['--binary','fixture','--report','report','--diagnostic-focused-prerequisites-report','focused'])
        with self.assertRaises(SystemExit):
            runner.main(['--binary','fixture','--report','report','--paired-cell','exact',
                '--prerequisites-report','complete','--diagnostic-focused-prerequisites-report','focused'])

    def test_external_metadata_requires_explicit_policy_and_original_contained_payloads(self):
        """Only the bound document changes location; no new reference alias is granted."""
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t,exact,record=paired_fixture(root)
            external=root/'metadata';external.mkdir()
            old=Path(record['paired_workload']['accuracy']['workload']['path'])
            target=external/'manifest.json';target.write_bytes(old.read_bytes())
            record['paired_workload']['accuracy']['workload']['path']=str(target)
            with patch.object(adapter.corpus_v2,'load_corpus',return_value=c) as load:
                with self.assertRaisesRegex(ValueError,'exact approved corpus manifest'):
                    adapter.PairedCorpusSelection.from_export(exact,record,root,root)
                load.assert_not_called()
                selected=adapter.PairedCorpusSelection.from_export(exact,record,root,root,
                    metadata_policy=adapter.PairedMetadataPolicy.SOURCE_BOUND_DIAGNOSTIC)
                self.assertEqual(load.call_args.kwargs['manifest_path'],target)
                self.assertEqual(load.call_args.args[0],root)
                self.assertEqual(selected.paths['workload'],target)
            with self.assertRaisesRegex(ValueError,'policy'):
                adapter.PairedCorpusSelection.from_export(exact,record,root,root,metadata_policy='untyped')

    def selection(self,root,c,exact,record):
        with patch.object(adapter.corpus_v2,'load_corpus',return_value=c):
            return adapter.PairedCorpusSelection.from_export(exact,record,root,root)

    def test_exact_export_selects_one_source_owned_topology_and_all_documents(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t,exact,record=paired_fixture(root)
            selected=self.selection(root,c,exact,record)
            self.assertEqual(selected.topology,t)
            self.assertEqual(set(selected.paths),{'workload','catalog','pilot','benchmark'})
            self.assertEqual(selected.observation()['benchmark_status'],'DECLARED_NOT_EXECUTED')
            self.assertFalse(selected.observation()['certification_eligible'])
            record['paired_workload']['arithmetic']['hc_up']='two_term_q8'
            self.assertEqual(selected.record['paired_workload']['arithmetic']['hc_up'],'single_q8')

    def test_no_profile_or_foreign_role_policy_or_source_is_inferred(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t,exact,record=paired_fixture(root)
            for mutate in (lambda r:r.pop('paired_workload'),
                           lambda r:r['paired_workload'].update(source_fields=['model','shared_predictor']),
                           lambda r:r['paired_workload'].update(role='foreign'),
                           lambda r:r['paired_workload'].update(case_id='Other'),
                           lambda r:r['runtime']['generation'].update(mtp_policy='dynamic'),
                           lambda r:r.update(model=str(root/'main-1.gguf')),
                           lambda r:r['paired_workload']['accuracy'].update(approved_baseline=True)):
                bad=copy.deepcopy(record);mutate(bad)
                with self.subTest(mutation=mutate),self.assertRaises(ValueError):self.selection(root,c,exact,bad)

    def test_predictor_presence_is_symmetric_and_exact_not_an_inactive_exception(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t,exact,record=paired_fixture(root)
            path=root/'inactive.gguf';path.write_bytes(b'x')
            pin=adapter.identity(path)
            descriptor=dict(name=path.name,path=str(path),bytes=1,
                object_identity=json.dumps(dict(path=pin[0],device=pin[1],inode=pin[2],bytes=pin[3],mtime_ns=pin[4],ctime_ns=pin[5])))
            missing=copy.deepcopy(record)
            missing['shared_predictor']=str(path)
            missing['paired_workload']['source_fields'].append('shared_predictor')
            with self.assertRaisesRegex(ValueError,'predictor source role'):self.selection(root,c,exact,missing)
            c['manifest']['source']['shared_predictor']=descriptor
            t['expected_config']['configuration']['shared_predictor']=str(path)
            t['runtime_config_digest']=metadata_digest(t['expected_config'])
            for name,document in (('workload',c['manifest']),('pilot',absolute_pilot(c['manifest'])),
                                  ('catalog',dict(schema=2,manifest_digest=metadata_digest(c['manifest'])))):
                location=root/('manifest.json' if name=='workload' else name+'.json')
                location.write_text(json.dumps(document))
                for current in (record,missing):
                    current['paired_workload']['accuracy'][name]['sha256']=adapter.file_sha256(location)
            with self.assertRaisesRegex(ValueError,'predictor source role'):self.selection(root,c,exact,record)
            self.assertIn('shared_predictor',self.selection(root,c,exact,missing).record)
            missing['shared_predictor']=str(root/'main-1.gguf')
            with self.assertRaisesRegex(ValueError,'another original predictor'):self.selection(root,c,exact,missing)

    def test_missing_duplicate_or_wrong_topology_never_selects_a_neighbor(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t,exact,record=paired_fixture(root)
            for values in ([],[dict(t,canonical_cell='Other.ProductionParity/Other')],[t,dict(t,id='duplicate')]):
                bad=copy.deepcopy(c);bad['manifest']['topologies']=values
                with self.assertRaises(ValueError):self.selection(root,bad,exact,record)

    def test_every_document_digest_and_stale_document_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t,exact,record=paired_fixture(root)
            for name in ('catalog','workload','pilot','benchmark'):
                bad=copy.deepcopy(record)
                reference=bad['paired_workload']['benchmark']['profile'] if name=='benchmark' else bad['paired_workload']['accuracy'][name]
                reference['sha256']='0'*64
                with self.subTest(document=name),self.assertRaises(ValueError):self.selection(root,c,exact,bad)
            selected=self.selection(root,c,exact,record)
            selected.paths['benchmark'].write_text('{}')
            with self.assertRaises(ValueError):adapter.unchanged(selected.pins)

    def test_corpus_cannot_choose_an_unbound_candidate_arithmetic(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t,exact,record=paired_fixture(root)
            for field in adapter.SELECTORS:
                bad=copy.deepcopy(record)
                bad['paired_workload']['arithmetic'][field]='foreign'
                with self.subTest(field=field),self.assertRaises(ValueError):self.selection(root,c,exact,bad)
            bad=copy.deepcopy(record);bad['paired_workload'].pop('arithmetic')
            with self.assertRaises(ValueError):self.selection(root,c,exact,bad)

    def test_public_parser_intent_rejects_placement_math_and_wrong_source_roles(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t,exact,record=paired_fixture(root)
            selected=self.selection(root,c,exact,record)
            observed=dict(schema='llaminar.model-kld-described-config.v1',engine_argv=t['engine_argv'],
                config=t['expected_config'],runtime_config_digest=t['runtime_config_digest'],
                qfn_canonical_intent={'synthetic_placement':'original'},qfn_arithmetic_intent=t['arithmetic'])
            for mutation in ('none','placement','arithmetic','source','missing'):
                canonical=copy.deepcopy(observed)
                canonical['engine_argv']=selected.canonical_arguments
                if mutation=='placement':canonical['qfn_canonical_intent']['synthetic_placement']='foreign'
                elif mutation=='arithmetic':canonical['qfn_arithmetic_intent']['hc_up']='two_term_q8'
                elif mutation=='source':
                    canonical['config']['configuration']['model_path']=str(root/'main-1.gguf')
                    canonical['runtime_config_digest']=metadata_digest(canonical['config'])
                elif mutation=='missing':canonical.pop('qfn_canonical_intent')
                with patch.object(runner.subprocess,'check_output',return_value=json.dumps(canonical)):
                    if mutation=='none':selected.require_public_projection(runner,SimpleNamespace(binary=root/'not-executed'),{},observed)
                    else:
                        with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                            selected.require_public_projection(runner,SimpleNamespace(binary=root/'not-executed'),{},observed)

    def test_public_parser_receives_separately_declared_model_and_context(self):
        """Actual exports contain policy-only argv; their source fields must reach the parser."""
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t,exact,record=paired_fixture(root)
            selected=self.selection(root,c,exact,record)
            expected=['--no-mtp','-m',record['model'],'--context-length','4096']
            self.assertEqual(selected.canonical_arguments,expected)
            self.assertEqual(record['diagnostic_benchmark']['args'],['--no-mtp'])
            observed=dict(schema='llaminar.model-kld-described-config.v1',engine_argv=t['engine_argv'],
                config=t['expected_config'],runtime_config_digest=t['runtime_config_digest'],
                qfn_canonical_intent={'synthetic_placement':'original'},qfn_arithmetic_intent=t['arithmetic'])
            canonical=dict(observed,engine_argv=expected)
            binary=root/'not-executed'
            with patch.object(runner.subprocess,'check_output',return_value=json.dumps(canonical)) as describe:
                selected.require_public_projection(runner,SimpleNamespace(binary=binary),{},observed)
            self.assertEqual(describe.call_args.args[0],
                [str(binary),'--describe-config','--',*expected])
            for change in ({'context_length':0},{'context_length':True},
                           {'policy':'production_defaults'},{'schema':True}):
                bad=copy.deepcopy(record);bad['diagnostic_benchmark'].update(change)
                with self.subTest(change=change),self.assertRaisesRegex(ValueError,'diagnostic profile'):
                    self.selection(root,c,exact,bad)

    def test_all_actual_policy_groups_are_required_or_diagnostic_before_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t,exact,record=paired_fixture(root)
            path=root/'pilot.json'
            policy=json.loads(path.read_text());policy['groups'].pop()
            path.write_text(json.dumps(policy))
            record['paired_workload']['accuracy']['pilot']['sha256']=adapter.file_sha256(path)
            with self.assertRaises(ValueError):self.selection(root,c,exact,record)

    def test_calibrated_documents_bind_actual_baseline_without_self_scoring(self):
        from test_model_kld_corpus_v2 import approval, calibration
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t,exact,record=paired_fixture(root)
            baseline=observation(c['manifest'])
            approved=approval(c['manifest'],baseline)
            calibrated=calibration(c['manifest'],baseline,approved)
            original=record['paired_workload']['accuracy']
            accuracy={key:original[key] for key in ('arguments_from','catalog','workload')}
            accuracy['policy']='calibrated_revision'
            for name,document in (('baseline',baseline),('approval',approved),('calibration',calibrated)):
                path=root/(name+'.json');path.write_text(json.dumps(document))
                accuracy[name]=dict(path=str(path),sha256=adapter.file_sha256(path))
            record['paired_workload']['accuracy']=accuracy
            selected=self.selection(root,c,exact,record)
            self.assertNotIn('pilot',selected.paths)
            approved['baseline_report_digest']='foreign'
            (root/'approval.json').write_text(json.dumps(approved))
            accuracy['approval']['sha256']=adapter.file_sha256(root/'approval.json')
            with self.assertRaises(ValueError):self.selection(root,c,exact,record)

    def test_runner_requires_actual_prerequisites_and_rejects_independent_overrides(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t,exact,record=paired_fixture(root)
            (root/'observer').write_text('not executable')
            for field in ('prerequisites_report','absolute_pilot','topology_id','corpus_catalog'):
                args=AbsolutePilotExecutionTests().args(root,None)
                args.corpus_catalog=None
                args.paired_cell=exact;args.prerequisites_report=root/'genuine-receipt-not-created'
                if field=='prerequisites_report':args.prerequisites_report=None
                else:setattr(args,field,['foreign'] if field=='topology_id' else root/'foreign')
                with patch('model_parity_inventory.discover') as discover:
                    with self.subTest(field=field),self.assertRaises(ValueError):runner.select_paired_corpus(args)
                    discover.assert_not_called()

    def test_stale_genuine_prerequisite_validators_stop_before_native_collection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t,exact,record=paired_fixture(root)
            (root/'observer').write_text('not executable')
            (root/'CMakeCache.txt').write_text('CMAKE_BUILD_TYPE:STRING=Integration\n')
            (root/'stale.json').write_text('{}')
            selected=self.selection(root,c,exact,record)
            for name in ('scripts/ci/run_model_kld_canaries.py','scripts/ci/model_kld_corpus_execution.py',
                'scripts/ci/model_kld_corpus_v2.py','scripts/ci/full_vocabulary_metrics.py',
                'src/v2/tools/ModelKLDCanary.cpp','src/v2/tools/ModelKLDQFNContract.h'):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic source')
            for rejection in ('source','inventory'):
                args=AbsolutePilotExecutionTests().args(root,None)
                args._paired_selection=selected;args.corpus_catalog=selected.paths['catalog']
                args.absolute_pilot=selected.paths['pilot'];args.topology_id=[t['id']]
                args.prerequisites_report=root/'stale.json'
                with patch.object(runner,'ROOT',root), \
                     patch('run_production_prerequisites.validate_source_freshness',side_effect=ValueError('stale') if rejection=='source' else None), \
                     patch('run_production_parity_campaigns.reuse_unchanged_production_test_preflight',side_effect=ValueError('incomplete') if rejection=='inventory' else None), \
                     patch.object(runner,'execute_cell') as execute:
                    with self.subTest(rejection=rejection),self.assertRaises(ValueError):runner.run_corpus_v2(args)
                    execute.assert_not_called()

    def test_actual_inventory_selector_only_projects_returned_exact_case(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c,t,exact,record=paired_fixture(root)
            (root/'observer').write_text('not executable');(root/'matrix').write_text('not executable')
            args=AbsolutePilotExecutionTests().args(root,None)
            args.corpus_catalog=None
            args.paired_cell=exact;args.prerequisites_report=root/'genuine-receipt-not-created'
            with patch('model_parity_inventory.discover',return_value=[(SimpleNamespace(name='original'),exact,record)]) as discover, \
                 patch('model_parity_inventory.matrix_binary',return_value=root/'matrix'), \
                 patch.object(adapter.corpus_v2,'load_corpus',return_value=c):
                runner.select_paired_corpus(args)
            self.assertIs(discover.call_args.args[1],__import__('model_parity_inventory').InventoryScope.ALL)
            self.assertEqual(args.topology_id,[t['id']])
            self.assertEqual(args.absolute_pilot,root/'pilot.json')
            self.assertIsNone(args.baseline_report)
            self.assertIn(str(root/'matrix'),args._paired_selection.pins)

class DeclaredBenchmarkRepresentationTests(unittest.TestCase):
    """Source document/argv spelling only, with no model or acceptance fabrication."""
    @staticmethod
    def document_record():
        return {'model':'/actual/source.gguf','diagnostic_benchmark':{'schema':1,'policy':'diagnostic_declared',
            'certification_eligible':False,'context_length':512,'configuration_policy':'lossless_declared_document_v1',
            'configuration_document':{'kind':'llaminar.orchestration-config','schema_version':1,'configuration':{'ngram_table_policy':{
                'storage':'AdaptiveVRAMRAMSSD','gpu_adaptive':{'local_hot_cache':{'consumer_overrides':[
                    {'participant':{'world_rank':0,'backend':'ROCm','ordinal':0},'maximum_cache_rows':0,'cache_physical_byte_limit':0},
                    {'participant':{'world_rank':0,'backend':'ROCm','ordinal':1},'maximum_cache_rows':512,'cache_physical_byte_limit':4096}]}}}}}}}

    def test_lossless_original_document_retains_unequal_caps_and_pinned_artifact(self):
        for replay in range(20):
            record=self.document_record()
            with tempfile.TemporaryDirectory() as directory:
                arguments,owner,pins=adapter.canonical_diagnostic_arguments(record,Path(directory))
                self.assertIsNone(owner)
                self.assertEqual(arguments[0],'--config')
                path=Path(arguments[1]);self.assertEqual(path.parent,Path(directory))
                self.assertEqual(json.loads(path.read_text()),record['diagnostic_benchmark']['configuration_document'])
                adapter.unchanged(pins)
                self.assertEqual(arguments[2:],['-m',record['model'],'--context-length','512'])
                with self.assertRaises(ValueError):adapter.canonical_diagnostic_arguments(record,Path(directory))
                path.write_text('{}')
                with self.assertRaises(ValueError):adapter.unchanged(pins)
            arguments,owner,pins=adapter.canonical_diagnostic_arguments(record)
            self.assertIsNotNone(owner);adapter.unchanged(pins)
            owner.cleanup()

    def test_document_and_argv_are_exclusive_and_incomplete_union_fails(self):
        original=self.document_record()
        invalid=[]
        record=copy.deepcopy(original);record['diagnostic_benchmark']['args']=['--no-mtp'];invalid.append(record)
        record=copy.deepcopy(original);del record['diagnostic_benchmark']['configuration_document'];invalid.append(record)
        for value in (None,{},[],{'configuration':None}):
            record=copy.deepcopy(original);record['diagnostic_benchmark']['configuration_document']=value;invalid.append(record)
        record=copy.deepcopy(original);record['diagnostic_benchmark']['configuration_policy']='invented';invalid.append(record)
        for record in invalid:
            with self.assertRaises(ValueError):adapter.declared_benchmark_profile(record)
        legacy={'model':'/actual/source.gguf','diagnostic_benchmark':{'schema':1,'policy':'diagnostic_declared',
            'certification_eligible':False,'context_length':512,'args':['--no-mtp']}}
        arguments,owner,pins=adapter.canonical_diagnostic_arguments(legacy)
        self.assertEqual(arguments,['--no-mtp','-m',legacy['model'],'--context-length','512'])
        self.assertIsNone(owner);self.assertEqual(pins,{})
        legacy['diagnostic_benchmark']['configuration_policy']='lossless_declared_document_v1'
        with self.assertRaises(ValueError):adapter.declared_benchmark_profile(legacy)

if __name__=='__main__':unittest.main()
