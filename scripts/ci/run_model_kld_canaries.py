#!/usr/bin/env python3
"""Run frozen dense/MoE phase canaries against an exact published GPU runtime.

Master PRs opt into this bounded numerical gate; Unit/preflight only test its
contracts. The native observer belongs to the runtime image and links its actual
serving core. Local Release qualification can select a binary explicitly. Each
model runs on one CUDA and one ROCm device, with native router/wide arithmetic,
model-scoped alpha/beta defaults, full captured graphs and no MTP or prefix reuse. Raw candidate
logits are removed after scoring unless a local diagnostic explicitly retains them.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid

# Avoid BLAS pools inside the physical-core row executor.
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/v2/e2e/server'))
import gpu_driver_diagnostics as driver
import resource_usage_observer as resources
import docker_paths
from model_kld_canary import DEFAULT_GDN_POLICY, definition
from model_kld_corpus import (CATALOG,identity,load_corpus,physical_workers,require,
                               score_observation,unchanged)
from production_artifacts import image_identity,image_cpu_isa,write_json

FILTER='kernel,forward_graph,gpu_graph_inventory,weight_loading,gemm,cuda_moe,rocm_moe,moe'


def engine_arguments(model,backend,threads):
    """One explicit production geometry; no test-only arithmetic or eager mode."""
    require(backend in ['cuda','rocm'],'Unsupported canary backend')
    return ['-m',str(model),'--mpi-procs','1','--device',backend+':0','-c','32768',
        '--no-mtp','--no-prefix-cache','--kv-cache-precision','fp16','--activation-precision','fp32',
        '--threads',str(threads),'--prefill-max-bucket-size','512','-n','256',
        '--temperature','0','--top-k','40','--top-p','.9','--seed','42']


def production_path(perf,report,backend,entry,*,model):
    """Require both captured phases and the model's declared default precision."""
    half_controls=definition(model,backend).half_alpha_beta
    expected=dict(topology='single_device',device=backend+':0',physical_devices=1,local_tp_devices=0,
        prefill_max_bucket=512,activation_precision='fp32',kv_precision='fp16',mtp=False,prefix_cache=False,
        event_ordered_host_logits=True,gdn_arithmetic=DEFAULT_GDN_POLICY,router_arithmetic='native')
    require(report.get('contract')==expected and all(type(report['contract'].get(k)) is type(v) for k,v in expected.items()),'Native observer changed the default execution contract')
    rows=perf.get('records');require(type(rows) is list and rows,'Missing native path counters')
    rows=[r for r in rows if r.get('value',0)>0 or r.get('count',0)>0]
    graphs=[r for r in rows if r.get('domain')=='forward_graph']
    require(not any(r['name'].startswith(('eager_','segmented_')) for r in graphs),'Incomplete homogeneous capture')
    require(all(not r.get('device') or r['device'].lower()==backend+':0' for r in graphs),'Unexpected graph device')
    require(any(r['name']=='full_graph_replay_calls' for r in graphs),'No complete captured replay')
    require(any(r['name']=='prefill_graph_phase' and r.get('tags',{}).get('capture_phase') in ['capture','replay'] for r in graphs),'No captured prefill')
    require(any(r['name']=='decode_graph_phase' and r.get('tags',{}).get('phase')=='replay' for r in graphs),'No captured decode')
    weights=[r for r in rows if r['name']=='gdn_projection_arithmetic_prepared_weight']
    expected_sources=entry['alpha_beta_sources'] if half_controls else {}
    require(bool(expected_sources)==half_controls,'Incorrect default alpha/beta inventory')
    found=set()
    for row in weights:
        tags=row['tags'];name=tags.get('canonical_name');require(name in expected_sources,'Unexpected projection precision conversion')
        source=expected_sources[name];require(row.get('device','').lower()==backend+':0','Prepared weight belongs to another device')
        require(tags.get('policy')==DEFAULT_GDN_POLICY and tags.get('source_format')==source['source_format'] and
                tags.get('prepared_format')=='FP16' and tags.get('source_path')=='source_decoded_half_rne' and
                tags.get('source_bytes')==str(source['rows']*source['columns']*4) and
                tags.get('prepared_bytes')==str(source['rows']*source['columns']*2),'Incorrect half operand preparation')
        found.add(name)
    require(found==set(expected_sources),'Incomplete alpha/beta preparation')
    require(not any(r['name']=='moe_router_arithmetic_prepared_weight' or 'hadamard32' in r['name'] for r in rows),'Router policy changed')
    groups=[r['tags'] for r in rows if r['name']=='gdn_projection_route']
    require(groups and {x.get('names') for x in groups}=={'qkv+z','alpha+beta'},'Unexpected GDN grouping')
    require(all(x.get('codebooks')=='none+none' if x['names']=='alpha+beta' else x.get('route')=='native_subgroup' for x in groups),'Wrong GDN arithmetic scope')
    return dict(complete_graph_replay=True,prefill=True,decode=True,half_alpha_beta_weights=len(found),native_router=True,native_wide_projections=True)


def retire(process,container):
    """Join the owned process/container, including exceptional teardown."""
    if container:
        state=json.loads(subprocess.check_output(['docker','inspect',container],text=True))[0]['State']
        if state['Running']:
            subprocess.run(['docker','stop','--time','10',container],check=True,stdout=subprocess.DEVNULL,timeout=20)
        # A created container has never executed and cannot publish an exit.
        # Waiting for it would leak the very owner this cleanup must retire.
        if state['Status']!='created':
            subprocess.run(['docker','wait',container],check=True,stdout=subprocess.DEVNULL,timeout=30)
        state=json.loads(subprocess.check_output(['docker','inspect',container],text=True))[0]['State']
        require(not state['Running'] and state['Pid']==0 and state['Status'] in ['created','exited'],'Native container did not retire')
    if process is not None and process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
    if container:subprocess.run(['docker','rm',container],check=True,stdout=subprocess.DEVNULL,timeout=30)


def execute_cell(args,corpus,model,backend,model_path,output,image):
    """Bracket admission, inference, teardown and scoring with owned evidence."""
    output.mkdir();payload=output/'logits';payload.mkdir()
    entry=corpus['models'][model];model_pin=identity(model_path)
    require(Path(model_path).name==entry['entry']['model_file'] and model_pin[3]==entry['entry']['model_bytes'],'GGUF differs from the frozen reference model')
    cell=dict(complete=False,passed=False,model=model,backend=backend,phase='admission',container_id=None,devices=[backend+':0'])
    controller=output/'controller.json';write_json(controller,cell)
    process=None;container=None;armed=False;retired=False;code=None
    try:
        env=dict(PATH=os.environ['PATH'],OMP_NUM_THREADS=str(physical_workers()),OMP_PLACES='cores',OMP_PROC_BIND='false',
                 OPENBLAS_NUM_THREADS='1',LLAMINAR_LOG_LEVEL='INFO',LLAMINAR_PERF_STATS_FILTER=FILTER)
        if image:
            candidate='llaminar-kld-'+uuid.uuid4().hex
            command=['docker','create','--name',candidate,'--init',*docker_paths.coding_owner_arguments(),
                *docker_paths.device_args(image['id'],'CUDA' if backend=='cuda' else 'ROCm',user=f'{os.getuid()}:{os.getgid()}'),
                *docker_paths.mounts([(model_path.parent,str(model_path.parent),True),(args.corpus_root,str(args.corpus_root),True),(output,str(output),False)]),
                '--entrypoint','/usr/local/bin/llaminar_model_kld_canary']
            env['LLAMINAR_PERF_STATS_JSON']=str(output/'perf-stats.json')
            for key,value in env.items():
                if key!='PATH':command+=['--env',key+'='+value]
            command += [image['id'],'--workload',str(entry['workload_path']),'--output',str(payload),'--',*engine_arguments(model_path,backend,physical_workers())]
            subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
            container=candidate
            cell['container_id']=subprocess.check_output(['docker','inspect','--format','{{.Id}}',container],text=True).strip()
            command=['docker','start','--attach',container]
        else:
            env['LD_LIBRARY_PATH']=str(args.binary.parent)
            env['LLAMINAR_PERF_STATS_JSON']=str(output/'perf-stats.json')
            command=[str(args.binary),'--workload',str(entry['workload_path']),'--output',str(payload),'--',*engine_arguments(model_path,backend,physical_workers())]
        cell.update(phase='native',command=command);write_json(controller,cell)
        driver.begin(output/'driver-state.json');armed=True
        manifest=dict(backends=[backend],cells={'canary':{'controller':str(controller)}},filesystems={'output':str(output)})
        with (output/'native.log').open('x') as log,(output/'resources.jsonl').open('x') as observations:
            process=subprocess.Popen(command,env=env if not image else None,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            started=time.monotonic()
            while True:
                sample=resources.collect(manifest)
                if not image and Path(f'/proc/{process.pid}/status').exists():
                    sample['local_process']=resources.numeric_fields(Path(f'/proc/{process.pid}/status'),kilobytes=True)
                observations.write(json.dumps(sample)+'\n');observations.flush()
                require(not sample['errors'] and not any(v['errors'] for v in sample['cells'].values()),'Resource observation failed')
                require(sample['filesystems']['output']['available_bytes']>256*1024*1024,'Canary artifact filesystem exhausted')
                require(time.monotonic()-started<900,'Canary exceeded the canonical 15-minute cell budget')
                try:code=process.wait(timeout=5);break
                except subprocess.TimeoutExpired:pass
        retire(process,container);container=None;retired=True
        driver_ok=driver.finish(output/'driver-state.json',output/'canary.driver-diagnostics.json');armed=False
        require(driver_ok,'GPU driver window failed')
        driver.validate_evidence(output)
        require(code==0,'Native canary observer failed')
        unchanged(corpus['inputs']);require(identity(model_path)==model_pin,'GGUF changed during native run')
        report=json.loads((payload/'logits.json').read_text())
        proof=production_path(json.loads((output/'perf-stats.json').read_text()),report,backend,entry['entry'],model=model)
        metrics=score_observation(corpus,payload/'logits.json',model=model,backend=backend)
        write_json(output/'metrics.json',metrics)
        cell.update(passed=metrics['passed'],verdict=metrics['verdict'],phases=metrics['phases'],mean_kl_nats=metrics['mean_kl_nats'],
                    production_path=proof,reset_byte_equal=True,driver_window_clean=True,metrics=metrics)
    except BaseException as error:
        cell['error']=repr(error);raise
    finally:
        try:
            if not retired:retire(process,container);retired=True
        finally:
            if armed:
                if not driver.finish(output/'driver-state.json',output/'canary.driver-diagnostics.json'):cell['passed']=False
            cell.update(complete=True,native_retired=retired,returncode=code,phase='retired' if retired else 'failed_retirement')
            if not args.keep_logits and retired:
                shutil.rmtree(payload);cell['raw_logits_removed']=True
            write_json(controller,cell)
    return cell


def main(argv=None):
    """Run every model/backend before reporting numerical failures; protocol errors fail hard."""
    parser=argparse.ArgumentParser(description=__doc__)
    execution=parser.add_mutually_exclusive_group(required=True)
    execution.add_argument('--container-image');execution.add_argument('--binary',type=Path)
    parser.add_argument('--source-revision');parser.add_argument('--models',type=Path,default=Path('/opt/llaminar-models'))
    parser.add_argument('--corpus-root',type=Path);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--keep-logits',action='store_true')
    args=parser.parse_args(argv)
    require(not args.container_image or not args.keep_logits,'CI always removes raw candidate logits')
    pin=json.loads(CATALOG.read_text());args.corpus_root=(args.corpus_root or ROOT/'corpora'/pin['path']).resolve(strict=True)
    corpus=load_corpus(args.corpus_root);args.models=args.models.resolve(strict=True)
    args.report=args.report.resolve();args.report.parent.mkdir(parents=True,exist_ok=True)
    if args.binary:
        args.binary=args.binary.resolve(strict=True)
        require('CMAKE_BUILD_TYPE:STRING=Release' in (args.binary.parent/'CMakeCache.txt').read_text(),'Local KL gate requires Release')
    image=image_identity(args.container_image) if args.container_image else None
    if image:require(image['labels'].get('org.opencontainers.image.revision')==args.source_revision,'Canary image/source mismatch')
    state=dict(schema=1,complete=False,passed=False,source_revision=args.source_revision,
               image=image['id'] if image else None,cpu_isa=image_cpu_isa(image).value if image else 'local',
               corpus_manifest_digest=corpus['manifest_digest'],diagnostic_only=image is None,cells=[])
    write_json(args.report,state)
    try:
        for model in ['dense','moe']:
            for backend in ['cuda','rocm']:
                name=model+'-'+backend;print('[model-kld] '+name,flush=True)
                result=execute_cell(args,corpus,model,backend,args.models/definition(model,backend).model_file,args.report.parent/name,image)
                state['cells'].append(result);write_json(args.report,state)
                print(json.dumps({k:result[k] for k in ['model','backend','passed','phases']}),flush=True)
        state['complete']=True;state['passed']=all(c['passed'] for c in state['cells'])
        return 0 if state['passed'] else 1
    except BaseException as error:state['error']=repr(error);raise
    finally:write_json(args.report,state)

if __name__=='__main__':
    for sig in [signal.SIGTERM,signal.SIGINT]:
        signal.signal(sig,lambda signum,frame:(_ for _ in ()).throw(KeyboardInterrupt('Canary interrupted')))
    raise SystemExit(main())
