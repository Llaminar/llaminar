"""Reuse original completed CPU jobs without changing their measured producer.

The existing request, completion, native artifacts and common observations stay
in their original directory. A consumer only records a reference after matching
the actual job's executable/core closure and current semantic/timing contract.
"""
from pathlib import Path
import json
from .cpu_exact_overlay import canonical_digest, file_digest


def used_program_name(job):
    """Select the one native executable that actually executed this request."""
    return ('fused_' if job['kind']=='expert_rows' else '')+('avx2_bin' if job['build_isa']=='AVX2' else 'avx512_bin')


def relevant_producer_matches(jobs, original, current):
    """Ignore unused owners, while preserving exact used trainer/core identities."""
    if not original.get('programs') or not current.get('programs'):return False
    names={used_program_name(job) for job in jobs}
    return bool(names) and all(name in original.get('programs',{}) and
        original['programs'][name]==current.get('programs',{}).get(name) for name in names)


def inspect_completed_job(directory, job, current, plan, *, contract_hash):
    """Return original immutable receipts, or an explicit non-reuse diagnostic.

    A mismatch never rewrites a paid observation or substitutes a new producer.
    The caller may collect a new missing cell in its own fresh output directory.
    """
    from .candidate_observation import read_observation_csv
    from .profiles import require_promotion_timing
    directory=Path(directory)
    try:
        complete_path=directory/'complete.json'
        complete=json.loads(complete_path.read_text())
        if 'retained_original_receipt' in complete:
            original=complete['retained_original_receipt'];source=Path(original['path'])
            if file_digest(source)!=original['sha256']:return None,'original completion reference changed'
            complete_path=source;complete=json.loads(source.read_text());directory=source.parent
            if 'retained_original_receipt' in complete:return None,'nested reuse references are not original native evidence'
        request=json.loads((directory/'request.json').read_text());producer=request['producer']
        if (canonical_digest(request['job'])!=canonical_digest(job) or
                complete['job_digest']!=canonical_digest(job)):
            return None,'original job source/format/actual-M/arithmetic/ISA scope differs'
        if (producer.get('digest')!=canonical_digest(producer['programs']) or
                complete['producer_digest']!=producer['digest']):
            return None,'original producer receipt is incomplete or changed'
        if not relevant_producer_matches([job],producer,current):return None,'actual used trainer/core closure differs'
        if complete.get('native_exit')!=0 or complete.get('session_retired') is not True:
            return None,'original native session did not retire successfully'
        session=directory/'session-base'
        if session.exists() and any(session.iterdir()):return None,'original session directory remains live'
        artifacts=complete['artifacts']
        observations=directory/'observations.csv'
        if str(observations.resolve()) not in artifacts:
            return None,'original completion does not bind its common observations'
        if any(file_digest(Path(name))!=digest for name,digest in artifacts.items()):
            return None,'original paid artifact bytes changed'
        if job['kind']=='expert_rows':
            for request_row in job['requests']:
                raw_path=directory/'evidence'/(canonical_digest(request_row).split(':')[1]+'.json')
                if str(raw_path.resolve()) not in artifacts:return None,'original expert result is not receipt-bound'
                native=json.loads(raw_path.read_text())
                repeats=native.get('configured_full_repeat_count',native['timing']['repeat_count'])
                if repeats<plan['repeats']:return None,'original expert timing repeat floor is insufficient'
        corpus=read_observation_csv([observations])
        require_promotion_timing(corpus,label='retained original CPU job',minimum_warmups=plan['warmups'],minimum_samples=plan['samples'])
        expected_isa=(f"openmp:build={job['build_isa']}:requested={job['runtime_isa']}:"
                      f"effective={job['runtime_isa']}:threads={plan['threads']}")
        for row in corpus:
            if (row.build_id!=producer['digest']+'|cpu_isa='+job['build_isa'] or
                    row.compiler_id!='cmake-release-closure:'+producer['digest']):
                return None,'original observations are not bound to their recorded native producer'
            if row.threading_or_stream_mode!=expected_isa or row.serial_m1_policy_hash!=contract_hash:
                return None,'original thread/ISA/arithmetic contract differs'
            if row.supported and row.forced_route_ok and (not row.numerical_correctness or not row.bitwise_equal or not row.repeat_equal):
                return None,'original supported candidate lacks complete byte/repeat correctness'
        # Original job equality binds the finite matrix; raw artifacts and common
        # rows remain byte-bound by the native completion, not reconstructed.
        return dict(complete=complete,complete_path=complete_path,observations=observations),None
    except (OSError,ValueError,KeyError,TypeError) as error:
        return None,'original completed cell is inadmissible: '+str(error)


def retain_completed_reference(directory, inspected):
    """Link existing observations and retain their original completion identity."""
    directory=Path(directory);target=directory/'observations.csv';source=inspected['observations'].resolve()
    if target.exists() or target.is_symlink():
        if target.resolve()!=source:raise ValueError('reuse would overwrite existing observations')
    else:target.symlink_to(source)
    receipt=dict(inspected['complete']);original=inspected['complete_path'].resolve()
    receipt['retained_original_receipt']=dict(path=str(original),sha256=file_digest(original))
    return receipt


def retain_installed_decode_stage(original_root, output_root, owner, current, plan):
    """Retain an installed completed M1 stage, without reusing it as new clocks.

    Original M1 timing predates its required install/rebuild. Its observations
    therefore keep their original producer. The existing stage/output receipt
    proves the installed policy; the post-rebuild rows producer proves the
    current actual legacy executables/core contain that policy. Unused fused
    owners have no authority over this retained M1 stage.
    """
    from .cpu_exact_sweep import iter_jobs,_atomic
    import shutil
    original_root=Path(original_root);output_root=Path(output_root);owner=Path(owner)
    stage_path=original_root/'decode-stage.json'
    if not stage_path.is_file():return False
    checkpoint=json.loads(stage_path.read_text())
    source_plan=json.loads((original_root/'cpu-exact-plan.json').read_text())
    if source_plan['plan_digest']!=canonical_digest({key:value for key,value in source_plan.items() if key!='plan_digest'}):
        raise ValueError('retained M1 original plan identity changed')
    source=original_root/'generated/CPUNativeQuantisedDecodePolicyGenerated.inc'
    emission=json.loads(source.with_suffix('.receipt.json').read_text())
    source_jobs={canonical_digest(job) for job in iter_jobs({**source_plan,'expert_geometries':[]})
                 if job['kind']=='legacy_decode'}
    wanted=[job for job in iter_jobs({**plan,'expert_geometries':[]}) if job['kind']=='legacy_decode']
    if not wanted or any(canonical_digest(job) not in source_jobs for job in wanted):
        raise ValueError('retained installed M1 stage does not cover the complete declared decode scope')
    if (not checkpoint.get('installed') or not checkpoint.get('rebuilt') or
            emission.get('plan_digest')!=source_plan['plan_digest'] or
            emission.get('output_sha256')!=checkpoint['output_sha256'] or
            file_digest(source)!=checkpoint['output_sha256'] or
            file_digest(owner/source.name)!=checkpoint['output_sha256']):
        raise ValueError('retained M1 stage lost its completed emission or actual installed output binding')
    post=json.loads((original_root/'rows-producer.json').read_text())
    if (post.get('digest')!=canonical_digest(post['programs']) or
            not relevant_producer_matches(wanted,post,current)):
        raise ValueError('retained installed M1 stage does not match post-rebuild actual legacy trainer/core closures')
    before_path=original_root/'decode-producer.json';before=json.loads(before_path.read_text())
    if before.get('digest')!=checkpoint['producer_digest'] or before['digest']!=canonical_digest(before['programs']):
        raise ValueError('retained M1 timing lost its original pre-install producer identity')
    generated=output_root/'generated';generated.mkdir(exist_ok=True)
    target=generated/source.name
    if target.exists() and file_digest(target)!=file_digest(source):
        raise ValueError('retained M1 stage would overwrite different staged source')
    if not target.exists():shutil.copyfile(source,target)
    # These are existing stage files, retaining original producer fields. The
    # added reference labels installed authority rather than newly timed cells.
    _atomic(output_root/'decode-producer.json',before)
    retained=dict(checkpoint)
    retained['retained_original_stage']=dict(path=str(stage_path.resolve()),sha256=file_digest(stage_path),
        original_plan_digest=source_plan['plan_digest'],retained_policy_not_remeasured=True,
        covered_decode_jobs=len(wanted))
    _atomic(output_root/'decode-stage.json',retained)
    return True
