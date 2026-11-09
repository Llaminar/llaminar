"""Canonical resumable CPU exact-only collection transaction.

This delegates arithmetic to existing native owners, admission to their common
adapters, and selection to the common exact oracle. Installed generic programs
are retained, never fit/CV/retrained. The factorized plan declares every legal
and source-unadmitted cell without serializing a massive Cartesian product.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .cpu_exact_overlay import canonical_digest, file_digest, RetainedCPUProgram, authenticate_retained_programs
from .format_registry import FORMAT_SPECS, format_spec, registry_digest
from .prefill_matrix import cpu_expert_prefill_exact_m_values
from .shape_manifest import load_shape_manifest, expert_projection_geometries

ROOT = Path(__file__).resolve().parents[5]
KERNELS = ROOT / "tests/v2/performance/kernels"
PLAN_NAME = "cpu-exact-plan.json"
REGIMES = (("AVX2", "AVX2"), ("AVX512", "AVX2"), ("AVX512", "AVX512"))
GROUPED_M = tuple(range(2, 17)) + (31,)


def _atomic(path: Path, value) -> None:
    """Publish one complete bounded receipt; partial evidence is never resumed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(path.name + ".inprogress")
    with stage.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(stage, path)


def build_plan(threads: int, *, warmups: int = 5, samples: int = 30, repeats: int = 8, prefill_row_policy: str = "exact-points",
               expert_only: bool = False, expert_scope: dict | None = None, tournament: str = "full", model_scope: str = "all-retained",
               expert_role_formats: tuple[tuple[str,str],...] = (), exclude_expert_role_formats: tuple[tuple[str,str],...] = (),
               expert_m: tuple[int,...] = ()) -> dict:
    """Project shared production geometry, source formats and actual M anchors."""
    if min(threads, warmups, samples, repeats) <= 0:
        raise ValueError("CPU exact collection requires positive team/timing settings")
    manifest = load_shape_manifest()
    admitted = set(manifest.cpu_measurement_names())
    shapes = [dict(name=s.name, n=s.n, k=s.k) for s in manifest.shapes if s.exact_overlay and s.name in admitted]
    excluded = [dict(name=s.name, n=s.n, k=s.k, reason="outside canonical CPU measurement envelope")
                for s in manifest.shapes if s.exact_overlay and s.name not in admitted]
    if tournament not in {"full","adaptive_v1"}:
        raise ValueError("unknown native tournament protocol")
    if expert_only and (not expert_scope or set(expert_scope)!={"model_family","tp_degree","layout","isa","purposes"}):
        raise ValueError("expert-only requires explicit shared model/TP/layout/ISA/purpose scope")
    if not expert_only and expert_scope is not None:
        raise ValueError("expert selection requires the explicit expert-only transaction")
    projections=expert_projection_geometries(manifest)
    if model_scope=='modern':
        from .modern_sweep_scope import modern_expert_projection_geometries
        projections=modern_expert_projection_geometries(projections)
    if expert_only:
        projections=tuple(projection for projection in projections
            if (projection.model_family,projection.tp_degree,projection.layout.value)==
               (expert_scope['model_family'],expert_scope['tp_degree'],expert_scope['layout']))
        if not projections:raise ValueError("expert selection is absent from the shared production catalog")
        if tuple(expert_scope['isa']) not in REGIMES:
            raise ValueError("expert-only selected an unavailable build/runtime ISA regime")
        if (not expert_scope['purposes'] or len(set(expert_scope['purposes']))!=len(expert_scope['purposes']) or
                set(expert_scope['purposes'])-{'decode','mtp_verifier','prefill'}):
            raise ValueError("expert-only requires explicit supported native purposes")
        shapes=[]
    if model_scope not in {'all-retained','modern'}:raise ValueError('unknown CPU model selection scope')
    if model_scope=='modern':
        from .modern_sweep_scope import modern_sweep_shape_names
        names=set(modern_sweep_shape_names(manifest))
        shapes=[shape for shape in shapes if shape['name'] in names]
    experts = {}
    for projection in projections:
        key = (projection.role.value, projection.physical_n, projection.physical_k, projection.policy_n)
        entry = experts.setdefault(key, dict(role=projection.role.value, n=projection.physical_n,
            k=projection.physical_k, policy_n=projection.policy_n, shape=projection.shape_name, sources=[]))
        source = asdict(projection)
        source['role'] = projection.role.value; source['layout'] = projection.layout.value
        entry['sources'].append(source)
    plan = dict(schema="cpu-exact-sweep-v1", inventory_digest=manifest.digest(), format_registry_digest=registry_digest(),
                threads=threads, warmups=warmups, samples=samples, repeats=repeats,
                regimes=REGIMES, legacy_shapes=shapes, expert_geometries=list(experts.values()),
                formats=[spec.label for spec in FORMAT_SPECS], grouped_m=GROUPED_M,
                expert_prefill_m=cpu_expert_prefill_exact_m_values(), excluded_geometry=excluded,
                ordinary_backend_native_prefill=dict(installed=False,
                    reason="No existing installable exact runtime boundary; this transaction does not claim that missing surface is installed."),
                vector_basis="smallest aligned homogeneous expert vector satisfying all-Wide filled-class bound, at least two experts; heldout appends an M1 expert",
                timing_basis="whole native operation/bundle; never divided per leaf",
                generic_fit=False, generic_cv=False, corpus_download=False)
    if model_scope=='modern':
        plan.update(schema='cpu-exact-modern-sweep-v1',model_scope='modern',
            excluded_legacy_geometry=[shape.name for shape in manifest.shapes if shape.exact_overlay and shape.name in admitted and shape.name not in names])
    if prefill_row_policy not in {"exact-points","two-anchor"}:
        raise ValueError("unknown prefill row measurement policy")
    if prefill_row_policy == "two-anchor":
        from .prefill_row_regimes import prefill_regime_measurement_m_values
        plan.update(schema="cpu-exact-sweep-v3-prefill-anchor-points",prefill_row_policy="two-anchor",
            prefill_publication="exact actual-M points only; no interval runtime qualification claimed")
        plan['expert_prefill_m']=prefill_regime_measurement_m_values(plan['expert_prefill_m'])
    if tournament=="adaptive_v1" and not expert_only:
        raise ValueError("adaptive_v1 first slice supports only the explicit expert-only owner")
    if tournament=="adaptive_v1":
        plan.update(schema="cpu-exact-sweep-v4-adaptive",tournament="adaptive_v1",screen_samples=3,
            promotion_scope="only final timing is promotion-grade; every valid candidate retains correctness and raw screening evidence")
    if expert_only:
        plan.update(schema="cpu-expert-only-exact-sweep-v1",expert_only=True,expert_scope=expert_scope,
            regimes=(tuple(expert_scope['isa']),),
            scope="additive shared-catalog expert-only projection; broad decode/grouped/ordinary campaign remains unqualified")
    if expert_m:
        if not expert_only:
            raise ValueError("expert actual-M projection requires the explicit expert-only transaction")
        if (any(type(m) is not int or m < 2 for m in expert_m) or
                len(set(expert_m)) != len(expert_m)):
            raise ValueError("expert actual-M values must be unique integers of at least two")
        for purpose in expert_scope['purposes']:
            admitted_m = plan['expert_prefill_m'] if purpose == 'prefill' else plan['grouped_m']
            if not set(expert_m).issubset(admitted_m):
                raise ValueError("expert actual-M projection is outside the existing " + purpose + " row inventory")
        # Actual M remains execution geometry. This finite projection changes
        # only collection obligations and their plan/job identities, never an
        # old receipt, row regime, generated choice or runtime arithmetic key.
        plan['expert_m'] = tuple(sorted(expert_m))
    if expert_role_formats or exclude_expert_role_formats:
        if not expert_only:raise ValueError("role/format selection requires explicit expert-only scope")
        selected=tuple(expert_role_formats);excluded_formats=tuple(exclude_expert_role_formats)
        if (len(set(selected))!=len(selected) or len(set(excluded_formats))!=len(excluded_formats) or
                set(selected)&set(excluded_formats)):
            raise ValueError("duplicate or contradictory expert role/format selection")
        known_roles={geometry['role'] for geometry in plan['expert_geometries']}
        for role,label in (*selected,*excluded_formats):
            spec=format_spec(label)
            if role not in known_roles:raise ValueError("selected expert role is absent from shared catalog slice")
            if selected and (role,label) in selected and not any(geometry['role']==role and
                    geometry['k']%(256 if spec.source_is_superblock else 32)==0 for geometry in plan['expert_geometries']):
                raise ValueError("explicit expert source format is not native-block admissible in this original slice")
        plan['expert_role_formats']=selected;plan['exclude_expert_role_formats']=excluded_formats
    unsupported = []
    source_geometries = (("legacy", shapes), ("expert", plan["expert_geometries"]))
    for kind, geometries in source_geometries:
        for geometry in geometries:
            for spec in FORMAT_SPECS:
                block = 256 if spec.source_is_superblock else 32
                if geometry['k'] % block:
                    unsupported.append(dict(operation=kind, shape=geometry.get('shape',geometry.get('name')),
                        role=geometry.get('role'), source_format=spec.label, physical_k=geometry['k'],
                        source_block_elements=block, outcome="original_native_source_unadmitted",
                        execution_measured=(kind != "expert"),
                        measurement_basis=("prepared_execution" if kind != "expert" else "original_native_source"),
                        reason="original source block is cut; legacy normalized prepared execution is separately legal, fused original-source construction is not"))
    plan['source_unadmitted'] = unsupported
    counts = {}
    for job in iter_requests(plan): counts[job['kind']] = counts.get(job['kind'],0)+1
    plan['logical_request_counts'] = counts; plan['total_logical_requests'] = sum(counts.values())
    batches = {}
    for job in iter_jobs(plan): batches[job['kind']] = batches.get(job['kind'], 0) + 1
    plan['job_counts'] = batches; plan['total_jobs'] = sum(batches.values())
    if expert_only:
        broad={key:value for key,value in plan.items() if key not in {'expert_role_formats','exclude_expert_role_formats'}}
        plan['broader_expert_job_count']=sum(1 for job in iter_jobs(broad) if job['kind']=='expert_rows')
        plan['broader_inventory_in_this_transaction']=plan['total_jobs']==plan['broader_expert_job_count']
        if not plan['total_jobs']:raise ValueError('expert selection contains no admissible native batch')
    plan['plan_digest'] = canonical_digest(plan)
    return plan


def iter_requests(plan):
    """Yield original-owner requests; unsupported source tuples stay in plan."""
    for build, runtime in plan['regimes']:
        for shape in plan['legacy_shapes']:
            for label in plan['formats']:
                spec = format_spec(label)
                for kind in ('legacy_decode','legacy_grouped'):
                    yield dict(kind=kind, build_isa=build, runtime_isa=runtime, source_format=label,
                        shape=shape['name'], n=shape['n'], k=shape['k'],
                        m_values=(1,) if kind=='legacy_decode' else tuple(plan['grouped_m']),
                        measurement_basis='prepared_execution',
                        original_source_aligned=(shape['k'] % (256 if spec.source_is_superblock else 32) == 0))
        for geometry in plan['expert_geometries']:
            for label in plan['formats']:
                spec = format_spec(label)
                role_format=(geometry['role'],label)
                if plan.get('expert_role_formats') and role_format not in map(tuple,plan['expert_role_formats']):continue
                if role_format in map(tuple,plan.get('exclude_expert_role_formats',())):continue
                if geometry['k'] % (256 if spec.source_is_superblock else 32): continue
                for operation, purpose, values in ((0,'decode',plan['grouped_m']),
                        (1,'mtp_verifier',plan['grouped_m']),(2,'prefill',plan['expert_prefill_m'])):
                    if plan.get('expert_only') and purpose not in plan['expert_scope']['purposes']:continue
                    for m in values:
                        if plan.get('expert_m') and m not in plan['expert_m']:continue
                        row_owners = ((m+3)//4)*((geometry['n']+63)//64)
                        count = max(2, math.ceil(plan['threads']/row_owners))
                        gate_up = geometry['role']=='gate_up'
                        experts = max(2, math.ceil(count/2)) if gate_up else count
                        for candidate in (('Pairwise','WideRows') if runtime=='AVX512' and m>=3 else ('Pairwise',)):
                            for holdout in (False,True):
                                yield dict(kind='expert_rows', build_isa=build,runtime_isa=runtime,
                                    source_format=label,shape=geometry['shape'],n=geometry['n'],k=geometry['k'],
                                    measurement_basis='original_native_source', original_source_aligned=True,
                                    policy_n=geometry['policy_n'],role=geometry['role'],operation=operation,
                                    purpose=purpose,m=m,candidate=candidate,holdout=holdout,
                                    expert_rows=[m]*experts+([1] if holdout else []))


def iter_jobs(plan):
    """Batch compatible requests through original retained fixture owners."""
    # Legacy owners already prepare/time all requested formats and grouped Ms.
    # M1 and grouped remain separate stages because installing M1 changes the
    # generation against which grouped evidence must be rebuilt/authenticated.
    experts = {}
    for request in iter_requests(plan):
        if request['kind'] != 'expert_rows':
            continue
        identity = tuple(request[name] for name in ('build_isa','runtime_isa','source_format',
            'shape','n','k','policy_n','role'))
        experts.setdefault(identity, []).append(request)
    for kind in ('legacy_decode', 'legacy_grouped'):
        for build, runtime in plan['regimes']:
            for shape in plan['legacy_shapes']:
                yield dict(kind=kind, build_isa=build, runtime_isa=runtime,
                    source_formats=plan['formats'], shape=shape['name'], n=shape['n'], k=shape['k'],
                    m_values=(1,) if kind == 'legacy_decode' else tuple(plan['grouped_m']),
                    measurement_basis='prepared_execution')
    # Shared expert/model geometry first supplies useful partial checkpoints;
    # the complete factorized inventory remains obligatory for final emission.
    for identity, requests in experts.items():
        first = requests[0]
        yield {**{name:first[name] for name in ('kind','build_isa','runtime_isa','source_format',
            'shape','n','k','policy_n','role')}, 'requests':requests,
            'measurement_basis':'original_native_source'}


def job_id(job) -> str:
    """Use a full declared request digest; names never masquerade as geometry."""
    return canonical_digest(job).split(':')[1]


def _binary(args, job):
    """Select the declared original build owner, independent of runtime ISA."""
    prefix='fused_' if job['kind']=='expert_rows' else ''
    return getattr(args, prefix+('avx2_bin' if job['build_isa']=='AVX2' else 'avx512_bin'))


def _command(args, plan, job, directory):
    """Retain original MPI session ownership, socket placement and30s deadline."""
    binary = _binary(args,job).resolve()
    helper = binary.parents[2]/'llaminar_mpi_session'
    session = directory/'session-base'; session.mkdir(exist_ok=True)
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(('LLAMINAR_CPU_FUSED_VERIFIER_','LLAMINAR_CPU_NATIVE_QUANTISED_VERIFIER_','LLAMINAR_CPU_NATIVE_QUANTISED_DECODE_')):
            del env[name]
    env.update(OMP_NUM_THREADS=str(plan['threads']),OMP_PLACES='cores',OMP_PROC_BIND='close',
        OMP_DYNAMIC='false',OMP_NESTED='false',HWLOC_COMPONENTS='-gl,-opencl',
        LLAMINAR_ISA_LEVEL=job['runtime_isa'].lower(),LLAMINAR_TP_COLLECT_TIMEOUT_MS='30000',
        LLAMINAR_TEST_MPI_SESSION_HELPER=str(helper))
    env.pop('LLAMINAR_NATIVE_QUANTISED_TOURNAMENT',None)
    env.pop('LLAMINAR_NATIVE_QUANTISED_TOURNAMENT_RECEIPT_PATH',None)
    env.pop('LLAMINAR_NATIVE_QUANTISED_PREFILL_ROW_POLICY',None)
    if plan.get('tournament')=='adaptive_v1' and not job.get('installed_auto_validation',False):
        env['LLAMINAR_NATIVE_QUANTISED_TOURNAMENT']='adaptive_v1'
        if job['kind']!='expert_rows':
            env['LLAMINAR_NATIVE_QUANTISED_TOURNAMENT_RECEIPT_PATH']=str((directory/'tournament.jsonl').resolve())
    if plan.get('prefill_row_policy')=='two-anchor' and not job.get('installed_auto_validation',False):
        env['LLAMINAR_NATIVE_QUANTISED_PREFILL_ROW_POLICY']='two_anchor_v1'
    if job['kind']=='expert_rows':
        env['LLAMINAR_PERF_STATS_JSON']=str((directory/'perfstats.json').resolve())
        env['LLAMINAR_PERF_STATS_FILTER']='kernel.cpu_native_quantised_fused_verifier_rows_projection_launch,kernel.cpu_native_quantised_execution_exact_policy_hits'
        requests=[]; maximum_experts=0; maximum_rows=0
        for request in job['requests']:
            identifier=job_id(request)
            evidence=directory/'evidence'/(identifier+'.json'); evidence.parent.mkdir(exist_ok=True)
            requests.append(dict(id=identifier,m=request['m'],purpose=request['purpose'],
                expert_rows=request['expert_rows'],target_descriptor=0,candidate=request['candidate'],
                evidence_path=str(evidence.resolve()),holdout=request['holdout']))
            maximum_experts=max(maximum_experts,len(request['expert_rows']))
            maximum_rows=max(maximum_rows,max(request['expert_rows']))
        native_plan=directory/'requests.json'
        _atomic(native_plan,dict(schema='cpu_execution_exact_batch_v2' if plan.get('tournament')=='adaptive_v1' and not job.get('installed_auto_validation',False) else 'cpu_execution_exact_batch_v1',requests=requests))
        fields=dict(M=maximum_rows,N_A=job['n'],K=job['k'],POLICY_N=job['policy_n'],
            PROJECTIONS=maximum_experts*(2 if job['role']=='gate_up' else 1),
            NUMERICAL_POLICY='GPUAlignedExpert',TEAM_LIFETIME='retained',OUTPUT_ROLE=job['role'],
            WARMUP=plan['warmups'],SAMPLES=plan['samples'],REPEATS=plan['repeats'],
            EXACT_REQUESTS_PATH=str(native_plan.resolve()),EXACT_SHAPE=job['shape'])
        env.update({'LLAMINAR_CPU_FUSED_VERIFIER_'+k:str(v) for k,v in fields.items()})
        filter_name=f"AllFormats/CPUFusedVerifier.ProjectionCounterBundleReplay/{job['source_format']}"
    else:
        env.pop('LLAMINAR_PERF_STATS_FILTER',None)
        grouped=job['kind']=='legacy_grouped'
        prefix='LLAMINAR_CPU_NATIVE_QUANTISED_VERIFIER_' if grouped else 'LLAMINAR_CPU_NATIVE_QUANTISED_DECODE_'
        fields=dict(FORMATS=','.join(job['source_formats']),M=','.join(map(str,job['m_values'])),
            SHAPE_NAME=job['shape'],N=job['n'],K=job['k'],THREADS=plan['threads'],
            WARMUP=plan['warmups'],ITERS=plan['samples'],MAX_CASES=len(job['m_values'])*len(job['source_formats']),
            STRONG_CSV=str(directory/'aggregate.csv'),TIMING_CSV=str(directory/'timing.csv'))
        env.update({prefix+k:str(v) for k,v in fields.items()})
        filter_name='*TrainerCsv_StrongVerifierRows_AllFormats' if grouped else '*TrainerCsv_StrongDecode_AllFormats'
    argv=[str(helper),'--session-base',str(session),'--','mpirun','-np','1',
        '--bind-to','socket','--map-by','socket','--mca','mpi_leave_pinned','1',
        '--mca','orte_allowed_exit_without_sync','1','--',str(binary),'--gtest_filter='+filter_name]
    return argv,env


def main(argv=None):
    """Plan or resume the full exact transaction; no implicit native job in plan mode."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    for name in ('avx2-bin','avx512-bin','fused-avx2-bin','fused-avx512-bin'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--threads',type=int,required=True)
    parser.add_argument('--warmups',type=int,default=5);parser.add_argument('--samples',type=int,default=30)
    parser.add_argument('--repeats',type=int,default=8);parser.add_argument('--batch-limit',type=int,default=0)
    parser.add_argument('--plan-only',action='store_true');parser.add_argument('--emit-only',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--install',action='store_true')
    parser.add_argument('--prefill-row-policy',choices=('exact-points','two-anchor'),default='exact-points')
    parser.add_argument('--model-scope',choices=('all-retained','modern'),default=None,
        help='New plans select Qwen3.5/3.6/3.8 and QFN; resume preserves the saved model scope')
    parser.add_argument('--reuse-completed-root',type=Path,help='Original paid cells; observations retain their original producer identity')
    parser.add_argument('--tournament',choices=('full','adaptive_v1'),default='full')
    parser.add_argument('--expert-model-family')
    parser.add_argument('--expert-tp-degree',type=int)
    from .shape_manifest import ExpertProjectionLayout
    parser.add_argument('--expert-layout',choices=tuple(layout.value for layout in ExpertProjectionLayout))
    parser.add_argument('--expert-isa',choices=tuple(':'.join(pair) for pair in REGIMES))
    parser.add_argument('--expert-purpose',choices=('decode','mtp_verifier','prefill'),action='append')
    parser.add_argument('--expert-m',type=int,action='append',default=[],metavar='M',
        help='Repeat to project expert-only collection onto actual M values admitted by every selected purpose; preserve on resume')
    parser.add_argument('--expert-role-format',action='append',default=[],metavar='ROLE:FORMAT',
        help='Explicit additive source-format slice; source/TP geometry stays catalog-owned')
    parser.add_argument('--exclude-expert-role-format',action='append',default=[],metavar='ROLE:FORMAT',
        help='Complement already qualified format slices without deleting installed entries')
    parser.add_argument('--stage',choices=('decode','rows','all','expert-only'),default='all')
    parser.add_argument('--cpu-list',help='existing profiler physical-core list; required for profile collection')
    parser.add_argument('--sudo-tool',action='store_true',help='run only isolated native counter owners with existing noninteractive sudo support')
    args=parser.parse_args(argv)
    args.output_dir=args.output_dir.resolve()
    if args.model_scope is None:
        saved=args.output_dir/PLAN_NAME
        args.model_scope=(json.loads(saved.read_text()).get('model_scope','all-retained')
            if args.resume and saved.is_file() else 'modern')
    if args.batch_limit < 0: parser.error('--batch-limit must be nonnegative')
    if args.install and (args.warmups < 5 or args.samples < 30):
        parser.error('exact installation requires at least5 warmups and30 raw samples')
    def role_formats(values):
        result=[]
        for value in values:
            fields=value.split(':')
            if len(fields)!=2 or fields[0] not in {'gate_up','down'}:
                parser.error('expert role/format must be gate_up:FORMAT or down:FORMAT')
            try:label=format_spec(fields[1]).label
            except ValueError as error:parser.error(str(error))
            result.append((fields[0],label))
        return tuple(result)
    expert_scope=None
    if args.stage=='expert-only':
        if any(getattr(args,name) is None for name in ('expert_model_family','expert_tp_degree','expert_layout','expert_isa')):
            parser.error('expert-only requires explicit model family, TP degree, layout and actual build/runtime ISA')
        expert_scope=dict(model_family=args.expert_model_family,tp_degree=args.expert_tp_degree,
            layout=args.expert_layout,isa=tuple(args.expert_isa.split(':')),
            purposes=tuple(args.expert_purpose or ('decode','mtp_verifier','prefill')))
    elif args.expert_m or any(getattr(args,name) is not None for name in ('expert_model_family','expert_tp_degree','expert_layout','expert_isa','expert_purpose')):
        parser.error('expert selection belongs to the expert-only transaction')
    try:
        plan=build_plan(args.threads,warmups=args.warmups,samples=args.samples,repeats=args.repeats, prefill_row_policy=args.prefill_row_policy,
            expert_only=args.stage=='expert-only',expert_scope=expert_scope,tournament=args.tournament,model_scope=args.model_scope,
            expert_role_formats=role_formats(args.expert_role_format),exclude_expert_role_formats=role_formats(args.exclude_expert_role_format),
            expert_m=tuple(args.expert_m))
        args.output_dir.mkdir(parents=True,exist_ok=True); target=args.output_dir/PLAN_NAME
        if target.exists():
            if json.loads(target.read_text()) != json.loads(json.dumps(plan)):
                raise ValueError('existing CPU exact plan identity changed; use a fresh output directory')
            if not args.resume and not args.plan_only: raise ValueError('existing exact transaction requires --resume')
        else: _atomic(target,plan)
        if args.plan_only:
            print(json.dumps({k:plan[k] for k in ('schema','plan_digest','job_counts','total_jobs')},indent=2));return 0
        required=_expert_program_names(args) if args.stage=='expert-only' else ('avx2_bin','avx512_bin','fused_avx2_bin','fused_avx512_bin')
        if any(getattr(args,name) is None for name in required):
            raise ValueError('collection/emission requires every selected original CPU native owner')
        args._remaining_batch_budget=args.batch_limit
        return run_transaction(args,plan)
    except (ValueError,OSError,KeyError) as error:
        parser.error(str(error))

    raise SystemExit(main())


def _producer_generation(args):
    """Authenticate original Release executable/core closures once per stage."""
    from .corpus_provenance import trainer_provenance, require_release_trainer
    result={}
    names=_expert_program_names(args) if getattr(args,'stage',None)=='expert-only' else ('avx2_bin','avx512_bin','fused_avx2_bin','fused_avx512_bin')
    for name in names:
        value=trainer_provenance(getattr(args,name),'cpu');require_release_trainer(value)
        result[name]=value
    return dict(programs=result,digest=canonical_digest(result))


def _context(args, plan, generation, paths, kind):
    """Reuse canonical strong-adapter provenance and native timing floors."""
    from .adapters.cpu_verifier import CPUVerifierAdapterContext
    from .adapters.cpu_decode import CPUDecodeAdapterContext
    from .adapters.evidence import raw_corpus_id
    from .profiles import MeasurementProfile
    contract=json.loads((KERNELS/'native_quantised_dispatch/manifests/cpu_native_quantised_serial_m1_arithmetic_v1.json').read_text())
    model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),None)
    if not model: raise ValueError('CPU collection has no original processor identity')
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    values=dict(profile=MeasurementProfile.PRODUCTION,run_id=plan['plan_digest']+'-'+kind,
        corpus_id=raw_corpus_id(paths),git_revision=revision,build_id=generation['digest'],
        compiler_id='cmake-release-closure:'+generation['digest'],
        architecture_class='cpu-'+model.replace('|','_'),device_name=model,
        driver_runtime='native-release-loader-closure:'+generation['digest'],
        raw_timing_sidecar_retained=True,minimum_promotion_warmups=plan['warmups'],
        minimum_promotion_samples=plan['samples'])
    if kind=='legacy_decode':
        return CPUDecodeAdapterContext(**values,frozen_serial_policy_hash=contract['contract_hash'])
    return CPUVerifierAdapterContext(**values,serial_m1_policy_hash=contract['contract_hash'])


def _admit_job(args, plan, generation, job, directory):
    """Validate original owner evidence before publishing a resumable completion."""
    from .adapters.cpu_decode import adapt_cpu_decode_csv
    from .adapters.cpu_verifier import adapt_cpu_verifier_csv
    from .adapters.cpu_execution_exact import adapt_cpu_execution_exact_native
    from .candidate_registry import cpu_native_quantised_verifier_registry
    from .candidate_observation import write_observation_csv
    from .corpus import ObservationCorpus
    if job['kind']=='expert_rows':
        rows=[]
        for request in job['requests']:
            path=directory/'evidence'/(job_id(request)+'.json')
            row=adapt_cpu_execution_exact_native(path,_context(args,plan,generation,[path],job['kind']),holdout=request['holdout'])
            witness=row.physical_launch_evidence;key=witness['key']
            if (row.source_format!=request['source_format'] or row.shape_name!=request['shape'] or
                key[0]!=request['operation'] or key[9]!=request['m'] or key[10]!=request['n'] or
                key[11]!=request['policy_n'] or key[12]!=request['k'] or key[8]!=plan['threads'] or
                witness['expert_rows']!=request['expert_rows'] or witness['target_descriptor']!=0 or
                witness['output_role']!=request['role'] or
                row.candidate_id!=cpu_native_quantised_verifier_registry().resolve(request['candidate']).candidate_id):
                raise ValueError('native expert request changed its declared source/geometry/vector')
            rows.append(row)
        corpus=ObservationCorpus(rows)
        sources=[directory/'evidence'/(job_id(request)+'.json') for request in job['requests']]
    else:
        sources=[directory/'aggregate.csv',directory/'timing.csv']
        context=_context(args,plan,generation,sources,job['kind'])
        adapter=adapt_cpu_decode_csv if job['kind']=='legacy_decode' else adapt_cpu_verifier_csv
        corpus=adapter([sources[0]],context,timing_sidecars=[sources[1]])
        expected={(label,m) for label in job['source_formats'] for m in job['m_values']}
        observed={(row.source_format,row.m) for row in corpus}
        if expected!=observed or any(row.aggregate_n!=job['n'] or row.k!=job['k'] or row.shape_name!=job['shape'] for row in corpus):
            raise ValueError('legacy native evidence changed or omitted a declared format/M/geometry')
    write_observation_csv(directory/'observations.csv',corpus)
    return [*sources,directory/'observations.csv']


def _collect_stage(args,plan,generation,stage):
    """Resume complete original-owner batches; failures never retry implicitly."""
    selected={'expert_rows'} if stage=='expert-only' else {'legacy_decode'} if stage=='decode' else {'legacy_grouped','expert_rows'}
    started=0;pending=0
    for job in iter_jobs(plan):
        if job['kind'] not in selected: continue
        directory=args.output_dir/'cells'/job['kind']/job_id(job);directory.mkdir(parents=True,exist_ok=True)
        # A paid local failure remains fatal, including emit-only/resume. An
        # older external success must never hide or replace this failed run.
        if (directory/'failure.json').exists():
            raise ValueError('an earlier local native batch failed; preserve it and use a fresh output directory')
        receipt=directory/'complete.json'
        from .cpu_completed_sweep import inspect_completed_job,retain_completed_reference
        contract=json.loads((KERNELS/'native_quantised_dispatch/manifests/cpu_native_quantised_serial_m1_arithmetic_v1.json').read_text())['contract_hash']
        if receipt.exists():
            inspected,reason=inspect_completed_job(directory,job,generation,plan,contract_hash=contract)
            if inspected is None:raise ValueError('completed CPU batch cannot be reused: '+reason+'; preserve it and use a fresh output directory')
            continue
        reuse_root=getattr(args,'reuse_completed_root',None)
        if reuse_root is not None:
            original=Path(reuse_root)/'cells'/job['kind']/job_id(job)
            if (original/'complete.json').exists():
                inspected,reason=inspect_completed_job(original,job,generation,plan,contract_hash=contract)
                if inspected is not None:
                    _atomic(receipt,retain_completed_reference(directory,inspected));continue
                _atomic(directory/'reuse-not-admitted.json',dict(original=str(original),reason=reason,pending=True))
        pending+=1
        if args.emit_only: continue
        budget=getattr(args,'_remaining_batch_budget',args.batch_limit)
        if args.batch_limit and budget<=0: continue
        command,env=_command(args,plan,job,directory)
        _atomic(directory/'request.json',dict(job=job,producer=generation,argv=command))
        with (directory/'process.log').open('w') as output:
            result=subprocess.run(command,env=env,stdout=output,stderr=subprocess.STDOUT,check=False)
        if result.returncode:
            _atomic(directory/'failure.json',dict(returncode=result.returncode,job_digest=canonical_digest(job)))
            raise ValueError(f'original CPU owner failed with exit{result.returncode}: {directory}')
        if list((directory/'session-base').iterdir()):
            raise ValueError('original CPU owner returned with an unretired session')
        artifacts=_admit_job(args,plan,generation,job,directory)
        _atomic(receipt,dict(job_digest=canonical_digest(job),producer_digest=generation['digest'],
            artifacts={str(path.resolve()):file_digest(path) for path in artifacts},native_exit=0,session_retired=True))
        started+=1;pending-=1
        if args.batch_limit:args._remaining_batch_budget=budget-1
    return pending


def _load_stage_corpus(args,plan,kind):
    """Read only authenticated common observations after the timing lane retires."""
    from .candidate_observation import read_observation_csv
    paths=[]
    for job in iter_jobs(plan):
        if job['kind']==kind:
            directory=args.output_dir/'cells'/kind/job_id(job)
            if not (directory/'complete.json').exists(): raise ValueError('cannot emit a partial CPU exact inventory')
            paths.append(directory/'observations.csv')
    return read_observation_csv(paths)


def _legacy_analyzer(kind):
    """Import existing pure adapter/emitter owners; never enter their old fit CLI."""
    name='analyze_cpu_native_quantised_decode_trainer' if kind=='decode' else 'analyze_cpu_native_quantised_verifier_trainer'
    path=KERNELS/'cpu'/(name+'.py');spec=importlib.util.spec_from_file_location('_exact_'+name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def _profile(args,corpus,directory,*,fused=False):
    """Run the existing isolated profiler transaction separately from timings."""
    from .candidate_observation import write_observation_csv
    from .profiler_evidence import build_profiler_request_manifest,write_profiler_request_manifest
    from .profiler_model import load_profiler_feature_catalog
    from . import profiler_collectors
    if not args.cpu_list: raise ValueError('exact promotion requires --cpu-list for the existing CPU profiler owner')
    directory.mkdir(parents=True,exist_ok=True)
    common=directory/'observations.csv';write_observation_csv(common,corpus)
    requests=directory/'requests.json';evidence=directory/'evidence.json'
    wanted=build_profiler_request_manifest(corpus)
    if requests.exists():
        from .profiler_evidence import read_profiler_request_manifest
        if read_profiler_request_manifest(requests).digest()!=wanted.digest():
            raise ValueError('existing CPU profiler request inventory changed')
    else:write_profiler_request_manifest(requests,wanted)
    profile_argv=['--requests',str(requests),'--output',str(evidence),
        '--raw-directory',str(directory/'raw'),'--backend','cpu','--cpu-list',args.cpu_list,'--resume']
    for flag,binary in (('--cpu-avx2-binary',args.fused_avx2_bin if fused else args.avx2_bin),
                        ('--cpu-avx512-binary',args.fused_avx512_bin if fused else args.avx512_bin)):
        if binary is not None:profile_argv += [flag,str(binary)]
    if args.sudo_tool:profile_argv.append('--sudo-tool')
    result=profiler_collectors.main(profile_argv)
    if result:raise ValueError('canonical isolated CPU profiling did not complete')
    load_profiler_feature_catalog(corpus,requests,evidence,source_corpus=corpus)
    return dict(requests_sha256=file_digest(requests),evidence_sha256=file_digest(evidence),timing_profile_separate=True)


def _emit_legacy(args,plan,kind):
    """Refresh existing literal exact selections with unchanged generic programs."""
    from .profiles import MeasurementProfile
    from .candidate_registry import candidate_registry_digest as candidate_digest
    from .cpu_exact_retention import refresh_decode_exact,refresh_verifier_exact
    corpus=_load_stage_corpus(args,plan,'legacy_'+('decode' if kind=='decode' else 'grouped'))
    owner=_legacy_analyzer(kind);manifest=load_shape_manifest()
    owner.validate_complete(corpus,manifest,require_full_inventory=False)
    if kind=='decode': entries=owner.select_entries(corpus)
    else:
        contract=json.loads((KERNELS/'native_quantised_dispatch/manifests/cpu_native_quantised_serial_m1_arithmetic_v1.json').read_text())
        entries=owner.select_entries(corpus,contract['contract_hash'])
    fresh=owner.generate_include(entries,[],corpus_digest=corpus.digest(),registry_digest=candidate_digest(),
        profile=MeasurementProfile.PRODUCTION,minimum_promotion_warmups=plan['warmups'],minimum_promotion_samples=plan['samples'])
    filename='CPUNativeQuantisedDecodePolicyGenerated.inc' if kind=='decode' else 'CPUNativeQuantisedVerifierRowsPolicyGenerated.inc'
    base=(args.output_dir/'retained'/filename).read_text()
    compose=refresh_decode_exact if kind=='decode' else refresh_verifier_exact
    result,receipt=compose(base,fresh)
    receipt.update(plan_digest=plan['plan_digest'],corpus_digest=corpus.digest(),
        profile=_profile(args,corpus,args.output_dir/'profiling'/kind))
    output=args.output_dir/'generated'/filename;output.parent.mkdir(exist_ok=True);output.write_text(result)
    receipt['output_sha256']=file_digest(output);_atomic(output.with_suffix('.receipt.json'),receipt)
    return output


def _install_file(source,target):
    """Atomically publish only authenticated generated output at its existing owner."""
    import shutil
    stage=target.with_name(target.name+'.inprogress');shutil.copyfile(source,stage);os.replace(stage,target)


def _rebuild_after_decode(args):
    """Honor M1→rebuilt producer→grouped order through original CMake owners."""
    names=_expert_program_names(args) if getattr(args,'stage',None)=='expert-only' else ('avx2_bin','avx512_bin','fused_avx2_bin','fused_avx512_bin')
    owners={}
    for name in names:
        build=getattr(args,name).resolve().parents[2]
        owners.setdefault(build,set()).add(name)
    for build,programs in owners.items():
        targets=['llaminar_mpi_session']
        if any(not name.startswith('fused_') for name in programs):targets.append('v2_perf_cpu_native_quantised_gemv')
        if any(name.startswith('fused_') for name in programs):targets.append('v2_perf_cpu_fused_verifier')
        command=['cmake','--build',str(build),'--parallel','--target',*targets]
        # Each logical owner belongs to exactly one build, so these disjoint
        # nonempty owner sets already give the logs unique, readable names.
        owner_names='+'.join(sorted(programs))
        log=args.output_dir/('decode-rebuild-'+owner_names+'.log')
        with log.open('w') as output:
            result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,check=False)
        if result.returncode:raise ValueError('CPU native rebuild after exact M1 publication failed')



def _post_install_auto_validation(args, plan, entries, generation):
    """Replay installed Auto on original batches; clocks are proof-only controls."""
    from dataclasses import replace
    from .adapters.cpu_execution_exact import adapt_cpu_execution_exact_native
    from .cpu_exact_overlay import CPUExecutionExactKey
    from .profiles import MeasurementProfile
    pending=0
    verification_plan={**plan,'warmups':1,'samples':1,'repeats':1}
    entry_digest=canonical_digest([(key.scalars(),value) for key,value in sorted(entries.items())])
    for original in iter_jobs(plan):
        if original['kind']!='expert_rows':continue
        source=args.output_dir/'cells'/'expert_rows'/job_id(original)
        requests=[];seen=set()
        for request in original['requests']:
            evidence=source/'evidence'/(job_id(request)+'.json')
            payload=json.loads(evidence.read_text());key=CPUExecutionExactKey.from_scalars(payload['exact_key'])
            identity=(key,request['holdout'])
            if identity in seen:continue
            seen.add(identity)
            if key not in entries:raise ValueError('installed verification target has no measured exact entry')
            requests.append({**request,'candidate':'Auto','expected_choice':entries[key]})
        job={**original,'requests':requests,'installed_auto_validation':True}
        directory=args.output_dir/'post-install-auto'/job_id(job);directory.mkdir(parents=True,exist_ok=True)
        receipt=directory/'complete.json'
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            if saved['producer_digest']!=generation['digest'] or saved['entries_digest']!=entry_digest:
                raise ValueError('installed Auto proof has stale binary or table identity')
            if any(file_digest(Path(path))!=digest for path,digest in saved['artifacts'].items()):
                raise ValueError('installed Auto proof evidence changed')
            continue
        pending+=1
        if args.emit_only:continue
        if (directory/'failure.json').exists():raise ValueError('installed Auto validation previously failed; review the retained failure')
        budget=getattr(args,'_remaining_batch_budget',args.batch_limit)
        if args.batch_limit and budget<=0:continue
        command,environment=_command(args,verification_plan,job,directory)
        _atomic(directory/'request.json',dict(job=job,producer=generation,argv=command))
        with (directory/'process.log').open('w') as output:
            result=subprocess.run(command,env=environment,stdout=output,stderr=subprocess.STDOUT,check=False)
        if result.returncode:
            _atomic(directory/'failure.json',dict(returncode=result.returncode))
            raise ValueError('installed Auto original native owner failed')
        artifacts=[]
        for request in requests:
            path=directory/'evidence'/(job_id(request)+'.json')
            context=replace(_context(args,verification_plan,generation,[path],'expert_rows'),profile=MeasurementProfile.QUICK)
            observation=adapt_cpu_execution_exact_native(path,context,holdout=request['holdout'],installed_choices=entries)
            witness=observation.physical_launch_evidence;key=CPUExecutionExactKey.from_scalars(witness['key'])
            if (key.operation!=request['operation'] or key.rows!=request['m'] or key.physical_n!=request['n'] or
                key.policy_n!=request['policy_n'] or key.k!=request['k'] or key.workers!=plan['threads'] or
                observation.source_format!=request['source_format'] or witness['expert_rows']!=request['expert_rows']):
                raise ValueError('installed Auto changed its declared target or mixed vector')
            artifacts.append(path)
        if list((directory/'session-base').iterdir()):raise ValueError('installed Auto native session did not retire')
        _atomic(receipt,dict(producer_digest=generation['digest'],entries_digest=entry_digest,
            artifacts={str(path.resolve()):file_digest(path) for path in artifacts},
            native_exit=0,session_retired=True,auto_dispatch_verified=True,timing_used_for_selection=False))
        pending-=1
        if args.batch_limit:args._remaining_batch_budget=budget-1
    return pending


def _finish_rows_install(args,plan,checkpoint,generated_owner):
    """Resume source publication/rebuild/Auto proof without relabelling old timings."""
    from .cpu_exact_overlay import read_execution_exact_include
    checkpoint_path=args.output_dir/'rows-stage.json'
    for name,record in checkpoint['programs'].items():
        source=Path(record['output'])
        if file_digest(source)!=record['sha256']:raise ValueError('staged exact source changed before installed validation')
        target=generated_owner/name
        if file_digest(target) not in {record['before_sha256'],record['sha256']}:
            raise ValueError('installed source was changed outside the exact transaction')
        if file_digest(target)!=record['sha256']:_install_file(source,target)
    if not checkpoint['rebuilt']:
        _rebuild_after_decode(args);checkpoint['rebuilt']=True
        checkpoint['producer']=_producer_generation(args);_atomic(checkpoint_path,checkpoint)
    generation=_producer_generation(args)
    if generation!=checkpoint['producer']:raise ValueError('installed exact producer changed before Auto proof')
    exact=Path(checkpoint['programs']['CPUNativeQuantisedExecutionExactGenerated.inc']['output'])
    entries=read_execution_exact_include(exact.read_text())
    pending=_post_install_auto_validation(args,plan,entries,generation)
    if pending:
        if args.emit_only:raise ValueError('installed exact validation is incomplete')
        _atomic(args.output_dir/'state.json',dict(complete=False,stage='post_install_auto',
            pending_batches=pending,plan_digest=plan['plan_digest']))
        print(f'Installed exact checkpoint: {pending} Auto validation batches remain; not qualified complete')
        return False
    if _producer_generation(args)!=generation:raise ValueError('installed binary changed during Auto validation')
    checkpoint['auto_verified']=True;_atomic(checkpoint_path,checkpoint)
    return True


def _publish_complete_state(args,plan,generated_owner):
    """Publish final scope and both staged/installed program identities."""
    produced=list((args.output_dir/'generated').glob('*.inc'))
    if args.install:produced += [generated_owner/path.name for path in produced]
    decode_stage=args.output_dir/'decode-stage.json'
    retained_decode=(json.loads(decode_stage.read_text()).get('retained_original_stage')
                     if decode_stage.is_file() else None)
    _atomic(args.output_dir/'state.json',dict(complete=True,installed=args.install,stage='complete',
        plan_digest=plan['plan_digest'],producer_digest=_producer_generation(args)['digest'],
        generated_files={str(path.resolve()):file_digest(path) for path in produced},
        installed_auto_dispatch_verified=args.install,
        retained_installed_m1_policy=retained_decode,
        scope=plan.get('scope','supported legacy decode/grouped and admitted expert-row exact projection'),
        selected_native_batches=plan['total_jobs'],broader_native_batches=plan.get('broader_expert_job_count'),
        broader_inventory_complete=plan.get('broader_inventory_in_this_transaction',False),
        ordinary_backend_native_prefill_installed=False,generic_fit=False))


def run_transaction(args,plan):
    """Collect/qualify/stage exact programs; a partial return is explicitly incomplete."""
    if args.stage=='expert-only':return _run_expert_only_transaction(args,plan)
    import shutil
    from .cpu_exact_overlay import (select_execution_exact_entries,generate_execution_exact_include,
        read_execution_exact_include,merge_exact_entries,CPUExecutionExactKey)
    from .corpus import ObservationCorpus
    generated_owner=ROOT/'src/v2/kernels/cpu/gemm'
    completed_state=args.output_dir/'state.json'
    if completed_state.exists() and json.loads(completed_state.read_text()).get('complete'):
        saved=json.loads(completed_state.read_text())
        if saved['plan_digest']!=plan['plan_digest'] or saved['producer_digest']!=_producer_generation(args)['digest']:
            raise ValueError('completed exact transaction identity changed')
        for path,digest in saved['generated_files'].items():
            if file_digest(Path(path))!=digest:raise ValueError('completed exact generated program changed')
        print('CPU exact transaction already complete; no collection or profiling repeated')
        return 0
    filenames=('CPUNativeQuantisedDecodePolicyGenerated.inc','CPUNativeQuantisedVerifierRowsPolicyGenerated.inc')
    rows_checkpoint=args.output_dir/'rows-stage.json'
    if rows_checkpoint.exists():
        if not args.install:raise ValueError('resuming installed exact validation requires --install')
        checkpoint=json.loads(rows_checkpoint.read_text())
        if checkpoint['plan_digest']!=plan['plan_digest']:raise ValueError('installed exact checkpoint plan changed')
        if _finish_rows_install(args,plan,checkpoint,generated_owner):
            _publish_complete_state(args,plan,generated_owner)
            print('CPU exact transaction complete with installed Auto dispatch verified')
        return 0
    retained=args.output_dir/'retained';retained.mkdir(exist_ok=True)
    manifest_path=retained/'manifest.json'
    if not manifest_path.exists():
        originals=[]
        for kind,filename in zip(('decode','verifier'),filenames):
            source=generated_owner/filename;target=retained/filename;shutil.copyfile(source,target)
            originals.append(dict(kind=kind,include=str(target.resolve()),sha256=file_digest(target)))
        exact=generated_owner/'CPUNativeQuantisedExecutionExactGenerated.inc'
        frozen=retained/exact.name;shutil.copyfile(exact,frozen)
        _atomic(manifest_path,dict(programs=originals,execution_exact=dict(include=str(frozen.resolve()),sha256=file_digest(frozen))))
    retained_manifest=json.loads(manifest_path.read_text());bases=retained_manifest['programs']
    prior_exact=retained_manifest['execution_exact']
    if file_digest(Path(prior_exact['include']))!=prior_exact['sha256']:
        raise ValueError('retained execution-exact artifact changed')
    authenticate_retained_programs(RetainedCPUProgram(item['kind'],Path(item['include']),item['sha256']) for item in bases)
    stages=('decode','rows') if args.stage=='all' else (args.stage,)
    decode_checkpoint=args.output_dir/'decode-stage.json'
    if args.stage=='all' and not decode_checkpoint.exists() and getattr(args,'reuse_completed_root',None) is not None:
        from .cpu_completed_sweep import retain_installed_decode_stage
        retain_installed_decode_stage(args.reuse_completed_root,args.output_dir,generated_owner,_producer_generation(args),plan)
    if args.stage=='all' and decode_checkpoint.exists():
        checkpoint=json.loads(decode_checkpoint.read_text())
        reference=checkpoint.get('retained_original_stage')
        if reference and file_digest(Path(reference['path']))!=reference['sha256']:
            raise ValueError('retained original M1 stage receipt changed')
        decode=args.output_dir/'generated'/filenames[0]
        if file_digest(decode)!=checkpoint['output_sha256']:
            raise ValueError('staged exact decode output changed after checkpoint')
        if not args.install or file_digest(generated_owner/filenames[0])!=checkpoint['output_sha256']:
            raise ValueError('resumed full transaction lost its installed exact decode generation')
        if not checkpoint['rebuilt']:
            _rebuild_after_decode(args);checkpoint['rebuilt']=True;_atomic(decode_checkpoint,checkpoint)
        stages=('rows',)
    for stage in stages:
        generation=_producer_generation(args);identity=args.output_dir/(stage+'-producer.json')
        if identity.exists() and json.loads(identity.read_text())!=generation:
            from .cpu_completed_sweep import relevant_producer_matches
            kinds={'legacy_decode'} if stage=='decode' else {'legacy_grouped','expert_rows'}
            if not relevant_producer_matches((job for job in iter_jobs(plan) if job['kind'] in kinds),json.loads(identity.read_text()),generation):
                raise ValueError('CPU used producer changed within a collected stage')
        if not identity.exists():_atomic(identity,generation)
        if stage=='rows':
            decode=args.output_dir/'generated'/filenames[0]
            if not decode.exists() or file_digest(decode)!=file_digest(generated_owner/filenames[0]):
                raise ValueError('grouped collection requires its completed installed exact M1 generation')
            previous=json.loads((args.output_dir/'decode-producer.json').read_text())
            base_hash=next(item['sha256'] for item in bases if item['kind']=='decode')
            from .cpu_completed_sweep import relevant_producer_matches
            if file_digest(decode)!=base_hash and relevant_producer_matches(
                    (job for job in iter_jobs(plan) if job['kind']=='legacy_decode'),previous,generation):
                raise ValueError('grouped producer was not rebuilt after changed M1 exact publication')
        pending=_collect_stage(args,plan,generation,stage)
        if pending:
            if args.emit_only:
                raise ValueError('exact emission requires every declared batch to be complete')
            _atomic(args.output_dir/'state.json',dict(complete=False,stage=stage,pending_batches=pending,plan_digest=plan['plan_digest']))
            print(f'CPU exact checkpoint: {pending} {stage} batches remain; full sweep incomplete')
            return 0
        if _producer_generation(args)!=generation:raise ValueError('CPU native closure changed during measurement')
        if stage=='decode':
            output=_emit_legacy(args,plan,'decode')
            if args.install:_install_file(output,generated_owner/filenames[0])
            checkpoint=dict(output_sha256=file_digest(output),installed=args.install,rebuilt=False,
                            producer_digest=generation['digest'])
            _atomic(decode_checkpoint,checkpoint)
            if args.stage=='all':
                if not args.install:raise ValueError('full transaction requires --install for M1→rebuild→grouped order')
                _rebuild_after_decode(args)
                checkpoint['rebuilt']=True;_atomic(decode_checkpoint,checkpoint)
            continue
        grouped=_emit_legacy(args,plan,'verifier')
        corpus=_load_stage_corpus(args,plan,'expert_rows')
        entries,receipt=select_execution_exact_entries(corpus)
        old_entries=read_execution_exact_include(Path(prior_exact['include']).read_text())
        merged=merge_exact_entries({key.scalars():choice for key,choice in old_entries.items()},
            {key.scalars():choice for key,choice in entries.items()},(key.scalars() for key in entries))
        entries={CPUExecutionExactKey.from_scalars(key):choice for key,choice in merged.items()}
        receipt['retained_execution_exact']=prior_exact
        training=ObservationCorpus(row for row in corpus if not row.physical_launch_evidence['holdout'])
        receipt.update(plan_digest=plan['plan_digest'],retained_programs=bases,
            profile=_profile(args,training,args.output_dir/'profiling/expert',fused=True))
        output=args.output_dir/'generated/CPUNativeQuantisedExecutionExactGenerated.inc'
        output.write_text(generate_execution_exact_include(entries));receipt['output_sha256']=file_digest(output)
        _atomic(output.with_suffix('.receipt.json'),receipt)
        if args.install:
            checkpoint=dict(plan_digest=plan['plan_digest'],rebuilt=False,auto_verified=False,
                programs={path.name:dict(output=str(path.resolve()),sha256=file_digest(path),
                    before_sha256=file_digest(generated_owner/path.name)) for path in (grouped,output)})
            _atomic(rows_checkpoint,checkpoint)
            if not _finish_rows_install(args,plan,checkpoint,generated_owner):return 0
    if args.stage=='decode':
        _atomic(args.output_dir/'state.json',dict(complete=False,installed=args.install,
            stage='awaiting_decode_rebuild',plan_digest=plan['plan_digest']))
        print('CPU exact M1 checkpoint requires rebuilt producers before grouped/expert collection')
    elif args.install:
        _publish_complete_state(args,plan,generated_owner)
        print('CPU exact transaction complete with installed Auto dispatch verified')
    else:
        _atomic(args.output_dir/'state.json',dict(complete=False,installed=False,
            stage='staged_not_installed',plan_digest=plan['plan_digest'],generic_fit=False))
        print('CPU exact candidates staged; installed Auto validation is still required')
    return 0


def expert_profile_observations(corpus, receipt):
    """Profile measured winners and actual Auto baselines, retaining raw losers.

    A final credible runner-up is valid timing evidence, but it does not create
    an extra profiling obligation. Historical full contests retain their prior
    behavior; the adaptive scope binds every selected key to its native baseline.
    """
    from .corpus import ObservationCorpus
    from .adaptive_tournament import is_promotion_grade
    selected={tuple(entry['key']):entry['candidate_id'] for entry in receipt['entries']}
    output=[]; observed={}; required={}
    for row in corpus:
        witness=row.physical_launch_evidence
        if witness['holdout']:continue
        key=tuple(witness['key']); tournament=witness.get('tournament')
        if tournament is None:
            output.append(row);continue
        if key not in selected:raise ValueError('profiler target lacks a selected exact key')
        wanted={selected[key],tournament['baseline_candidate_id']}
        if key in required and required[key]!=wanted:
            raise ValueError('profiler baseline changed within one exact contest')
        required[key]=wanted
        if row.candidate_id in wanted and is_promotion_grade(row):
            output.append(row);observed.setdefault(key,set()).add(row.candidate_id)
    if any(observed.get(key,set())!=wanted for key,wanted in required.items()):
        raise ValueError('selected winner or actual Auto baseline lacks full-grade profile evidence')
    return ObservationCorpus(output)


def _emit_expert_exact(args,plan,generation,prior_exact,retained_programs):
    """Stage only declared expert keys; retain every unrelated installed choice."""
    from .corpus import ObservationCorpus
    from .cpu_exact_overlay import (select_execution_exact_entries,read_execution_exact_include,
        generate_execution_exact_include,merge_exact_entries,CPUExecutionExactKey)
    corpus=_load_stage_corpus(args,plan,'expert_rows')
    entries,receipt=select_execution_exact_entries(corpus)
    retained_source=Path(prior_exact['include']).read_text()
    old=read_execution_exact_include(retained_source)
    merged=merge_exact_entries({key.scalars():choice for key,choice in old.items()},
        {key.scalars():choice for key,choice in entries.items()},(key.scalars() for key in entries))
    installed={CPUExecutionExactKey.from_scalars(key):choice for key,choice in merged.items()}
    training=expert_profile_observations(corpus,receipt)
    receipt.update(plan_digest=plan['plan_digest'],retained_execution_exact=prior_exact,
        retained_programs=retained_programs,
        profile=_profile(args,training,args.output_dir/'profiling/expert',fused=True))
    output=args.output_dir/'generated/CPUNativeQuantisedExecutionExactGenerated.inc'
    output.parent.mkdir(exist_ok=True)
    output.write_text(generate_execution_exact_include(installed))
    receipt['output_sha256']=file_digest(output);_atomic(output.with_suffix('.receipt.json'),receipt)
    return output


def _run_expert_only_transaction(args,plan):
    """Collect a central-catalog expert slice without any legacy recollection.

    Serial-M1 and grouped generic/previous exact authorities are retained inputs,
    not new qualification claims. Only the selected expert include is staged,
    rebuilt and proved through the original installed-Auto native owner.
    """
    import shutil
    owner=ROOT/'src/v2/kernels/cpu/gemm'
    state=args.output_dir/'state.json';checkpoint_path=args.output_dir/'rows-stage.json'
    if state.exists() and json.loads(state.read_text()).get('complete'):
        saved=json.loads(state.read_text())
        if (saved['plan_digest']!=plan['plan_digest'] or saved['producer_digest']!=_producer_generation(args)['digest'] or
                any(file_digest(Path(path))!=digest for path,digest in saved['generated_files'].items())):
            raise ValueError('completed expert slice changed its exact scope/producer/installed program')
        return 0
    if checkpoint_path.exists():
        checkpoint=json.loads(checkpoint_path.read_text())
        if not args.install or checkpoint['plan_digest']!=plan['plan_digest']:
            raise ValueError('expert installed checkpoint requires its original --install plan')
        if _finish_rows_install(args,plan,checkpoint,owner):_publish_complete_state(args,plan,owner)
        return 0
    retained=args.output_dir/'retained';retained.mkdir(exist_ok=True)
    receipt_path=retained/'expert-base.json'
    if not receipt_path.exists():
        programs=[]
        for kind,name in (('decode','CPUNativeQuantisedDecodePolicyGenerated.inc'),
                          ('verifier','CPUNativeQuantisedVerifierRowsPolicyGenerated.inc')):
            source=owner/name;copy=retained/name;shutil.copyfile(source,copy)
            programs.append(dict(kind=kind,include=str(copy.resolve()),sha256=file_digest(copy)))
        source=owner/'CPUNativeQuantisedExecutionExactGenerated.inc';copy=retained/source.name
        shutil.copyfile(source,copy)
        _atomic(receipt_path,dict(programs=programs,execution_exact=dict(include=str(copy.resolve()),sha256=file_digest(copy))))
    frozen=json.loads(receipt_path.read_text());prior=frozen['execution_exact']
    authenticate_retained_programs(RetainedCPUProgram(record['kind'],Path(record['include']),record['sha256'])
        for record in frozen['programs'])
    for record in [*frozen['programs'],prior]:
        if file_digest(Path(record['include']))!=record['sha256'] or file_digest(owner/Path(record['include']).name)!=record['sha256']:
            raise ValueError('expert slice lost its frozen installed serial/previous-exact authority')
    generation=_producer_generation(args);identity=args.output_dir/'expert-producer.json'
    if identity.exists() and json.loads(identity.read_text())!=generation:
        raise ValueError('expert slice native producer changed during resumed timing')
    if not identity.exists():_atomic(identity,generation)
    pending=_collect_stage(args,plan,generation,'expert-only')
    if pending:
        if args.emit_only:raise ValueError('expert slice cannot promote incomplete native batches')
        _atomic(state,dict(complete=False,stage='expert-only',pending_batches=pending,plan_digest=plan['plan_digest']))
        return 0
    if _producer_generation(args)!=generation:raise ValueError('expert native owner changed during measurement')
    output=_emit_expert_exact(args,plan,generation,prior,frozen['programs'])
    if not args.install:
        _atomic(state,dict(complete=False,stage='staged_not_installed',installed=False,plan_digest=plan['plan_digest']))
        return 0
    checkpoint=dict(plan_digest=plan['plan_digest'],rebuilt=False,auto_verified=False,
        programs={output.name:dict(output=str(output.resolve()),sha256=file_digest(output),before_sha256=file_digest(owner/output.name))})
    _atomic(checkpoint_path,checkpoint)
    if _finish_rows_install(args,plan,checkpoint,owner):_publish_complete_state(args,plan,owner)
    return 0


def _expert_program_names(args):
    """Name only the original prepared-bundle owner actually selected by this plan."""
    build=args.expert_isa.split(':')[0]
    return ('fused_avx2_bin' if build=='AVX2' else 'fused_avx512_bin',)





if __name__=='__main__':
    raise SystemExit(main())
