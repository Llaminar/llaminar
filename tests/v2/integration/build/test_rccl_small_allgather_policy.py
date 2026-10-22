#!/usr/bin/env python3
"""Builder-owned actual native policy proof; installed source/DSO verification.

No GPU initialization or mirrored policy participates. The original enqueue
negative keeps current private ABI headers and selected DSO deliberately: the
new communicator field makes mixing old layout with new helpers invalid.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[4]
PATCH='scripts/docker/patches/rccl-small-allgather-ll-policy.patch'
PROBE='tests/v2/integration/build/rccl_small_allgather_probe.cpp'
INPUTS=(PATCH,PROBE,'tests/v2/integration/build/test_rccl_small_allgather_policy.py',
        'tests/v2/integration/build/test_rccl_selector_taxonomy.py')
NATIVE=('src/enqueue.cc','src/graph/tuning.cc','src/rccl_wrap.cc','src/misc/archinfo.cc','src/SmallAllGatherPolicy.cpp',
        'src/init.cc','src/graph/connect.cc','src/misc/param.cc','src/register/register.cc','src/transport/nvls.cc','src/collectives.cc')
HEADERS=('src/include/comm.h','src/include/llaminar_small_allgather_scope.h','src/graph/topo.h',
         'src/include/transport.h','src/include/archinfo.h','src/include/enqueue.h','src/include/nccl_common.h',
         'src/SmallAllGatherPolicy.h')
SELECTION=('small_1','small_10240_fp32','boundary_16383','boundary_16384','large_16385','large_1048576',
           'other_collective','explicit_selector_excluded','overflow','unavailable_ll','nan_ll','infinite_ll','negative_ll',
           'zero_policy','nondivisible_policy',
           'native_gfx950_postpass','native_gfx942_pipeline')
SCOPE=('scope_complete_clique','scope_null_output_unowned','scope_null_output_completed','scope_nested_or_incomplete','scope_explicit_selector','scope_partial_count','scope_missing_output_owner','scope_duplicate_output_owner',
       'scope_nonblocking','scope_failed_initialization','scope_missing_original_owner','scope_foreign_original_owner',
       'scope_foreign_process','scope_foreign_host','scope_foreign_communicator','scope_missing_reciprocal_owner',
       'scope_mixed_architecture','scope_duplicate_topology_rank','scope_unconnected_endpoint','scope_foreign_transport',
       'scope_copy_fifo','scope_proxy_progress','scope_external_tuner','scope_multinode','scope_rank2','scope_rank3','scope_rank8','scope_runtime_connection','scope_unmeasured_channels')
OLD_FAILURES={'small_1','small_10240_fp32','boundary_16383','boundary_16384','overflow','unavailable_ll','nan_ll','infinite_ll','negative_ll'}
VARIANTS=('native','original_enqueue_body_current_private_ABI')


def require(condition,message):
    """Reject incomplete evidence without recovery or inferred successes."""
    if not condition:raise RuntimeError(message)


def digest(path):
    """Hash source/compiler-runtime proof inputs, never model/cache payloads."""
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def identities():
    """Bind every installed policy/probe/compiler-driver input."""
    return {name:digest(ROOT/name) for name in INPUTS}


def verify(report,library):
    """Authenticate complete native and meaningful original-consumer host outcomes."""
    require(report.get('schema')==1 and report.get('sources')==identities() and
            report.get('library_sha256')==digest(library),'Native policy source/runtime identity differs')
    require(report.get('negative_scope')==VARIANTS[1] and
            report.get('native_init_runtime_qualified') is False,'Negative/private ABI or host-proof scope differs')
    require(isinstance(report.get('rccl_receipt'),str) and
            report['rccl_receipt'].endswith('-smallag-'+report['sources'][PATCH]),'Missing native policy installer receipt')
    require(type(report.get('compile_database_sha256')) is str and
            re.fullmatch('[0-9a-f]{64}',report['compile_database_sha256']),'Missing original compile database identity')
    compiler=report.get('compiler',{})
    require(isinstance(compiler,dict) and Path(compiler.get('path','')).is_absolute() and
            type(compiler.get('stat')) is list and len(compiler['stat'])==5 and
            all(type(value) is int for value in compiler['stat']) and
            isinstance(report.get('compiler_version'),str) and bool(report['compiler_version']) and
            isinstance(report.get('architecture'),str) and report['architecture'].startswith('gfx') and
            Path(report.get('sdk','')).is_absolute(),'Missing compiler/SDK/architecture closure')
    pins=report.get('native_sources',{})
    require(type(pins) is dict and len(pins)==len(NATIVE)+len(HEADERS) and
            all(sum(name.endswith('/'+suffix) for name in pins)==1 for suffix in (*NATIVE,*HEADERS)) and
            all(type(value) is str and re.fullmatch('[0-9a-f]{64}',value) for value in pins.values()),
            'Missing native consumer/ABI/publication source pins')
    flags=report.get('native_compile_inputs',[])
    require(type(flags) is list and len(flags)==len(NATIVE) and all(type(row) is dict for row in flags) and
            {row.get('source') for row in flags}=={name for name in pins if name.endswith(tuple('/'+x for x in NATIVE))} and
            all(type(row.get('flags')) is list and row['flags'] and all(type(x) is str for x in row['flags']) for row in flags),
            'Missing native per-translation-unit compiler inputs')
    for row in flags:native_relocation_mode(row['flags'])
    outcomes=report.get('observations',[])
    require(type(outcomes) is list and all(type(row) is dict for row in outcomes) and
            [(row.get('variant'),row.get('mode')) for row in outcomes]==
            [(variant,mode) for variant in VARIANTS for mode in ('scope','selection')],
            'Missing/duplicate native policy modes')
    actual={}
    for row in outcomes:
        measured=row.get('measured',{})
        require(type(measured) is dict,'Malformed native measurement')
        cases=measured.get('cases',[])
        expected=SCOPE if row['mode']=='scope' else SELECTION
        require(type(measured) is dict and type(cases) is list and all(type(case) is dict for case in cases) and
                measured.get('mode')==row['mode'] and [case.get('name') for case in cases]==list(expected),
                'Vacuous or incomplete native case matrix')
        failed=set()
        for case in cases:
            require(type(case.get('checks')) is int and case['checks']>0 and
                    type(case.get('failures')) is int and 0<=case['failures']<=case['checks'] and
                    type(case.get('observed')) is list and len(case['observed'])==6 and
                    all(type(value) is int for value in case['observed']), 'Invalid native assertion/selection evidence')
            if case['failures']:failed.add(case['name'])
        negative=row['variant']==VARIANTS[1] and row['mode']=='selection'
        require(type(row.get('returncode')) is int and row['returncode']==int(negative) and
                failed==(OLD_FAILURES if negative else set()) and row.get('stderr')=='',
                'Native/original oracle failed, crashed or did not detect the old consumer')
        actual[(row['variant'],row['mode'])]={case['name']:case for case in cases}
    for mode,names in [('scope',SCOPE),('selection',tuple(name for name in SELECTION if name not in OLD_FAILURES))]:
        for name in names:
            require(actual[(VARIANTS[0],mode)][name]==actual[(VARIANTS[1],mode)][name],
                    'Excluded/postpass behavior changed across exact original enqueue: '+name)



def native_relocation_mode(tokens):
    """Retain the exact native relocation mode needed by included device headers.

    Host-only HIP parsing still sees RCCL's extern shared device declarations.
    Require one original enabled mode and reject missing, duplicate or opposing
    modes rather than substituting source declarations or compiler defaults.
    """
    modes=[token for token in tokens if token.startswith(('-fgpu-rdc','-fno-gpu-rdc'))]
    require(modes==['-fgpu-rdc'],'Native host proof requires exactly one original -fgpu-rdc mode')
    return modes


def native_inputs(build):
    """Reuse the existing actual compile-database extractor, not a second flags registry."""
    path=ROOT/'tests/v2/integration/build/test_rccl_selector_taxonomy.py'
    spec=importlib.util.spec_from_file_location('rccl_original_compile_inputs',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    database=json.loads((build/'compile_commands.json').read_text())
    result=[]
    for name in NATIVE:
        source,flags=module.inputs(build,name)
        rows=[row for row in database if
              (Path(row['directory'])/row['file']).resolve()==source.resolve()]
        require(len(rows)==1,'Native relocation mode requires one exact original compile row: '+name)
        row=rows[0]
        tokens=row['arguments'] if 'arguments' in row else shlex.split(row['command'])
        require(type(tokens) is list and all(type(token) is str for token in tokens),
                'Malformed native compiler arguments: '+name)
        result.append((source,[*flags,*native_relocation_mode(tokens)]))
    return result



def retire_proof(report):
    """Remove stale final/interrupted evidence before any rebuild can fail.

    Only the two exact builder-owned proof names are retired. Filesystem errors
    fail closed; directories and unrelated files are never recursively removed.
    A failed rebuild therefore cannot retain a previously successful receipt.
    """
    report.unlink(missing_ok=True)
    report.with_suffix(report.suffix+'.tmp').unlink(missing_ok=True)


def publish_verified_report(report, destination):
    """Atomically publish a fully validated report in its destination directory.

    The sole builder calls this only after verify succeeds. Interrupted writes
    remain temporary and cannot be accepted as a final proof; the next rebuild
    retires that exact temporary name before starting any work.
    """
    destination.parent.mkdir(parents=True,exist_ok=True)
    pending=destination.with_suffix(destination.suffix+'.tmp')
    with pending.open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    pending.replace(destination)


def compile_report(args):
    """Execute only actual host selectors and original body; retain exact dependency ABI."""
    retire_proof(args.report)
    source_flags=native_inputs(args.rccl_build)
    database_sha=digest(args.rccl_build/'compile_commands.json')
    receipt=(args.rccl_build/'.llaminar-rccl-commit').read_text().strip()
    require(receipt.endswith('-smallag-'+digest(ROOT/PATCH)),'Dependency receipt lacks exact native policy')
    require(digest(args.rccl_build/'librccl.so')==digest(args.library),'Selected DSO is not the supplied native build')
    generated_root=source_flags[0][0].parents[1]
    # enqueue source is hipify/src/enqueue.cc; original scope/header source and
    # every compiled helper use that same completed generated tree.
    pinned=[source for source,_ in source_flags]+[generated_root/name for name in HEADERS]
    require(len(set(pinned))==len(pinned),'Duplicate native proof input paths')
    # Clang selects its C++ driver from the invoked filename. Authenticate the
    # resolved executable, but preserve CMake's clang++ entrypoint at invocation.
    compiler_driver=Path(args.compiler).absolute()
    compiler=compiler_driver.resolve(strict=True);stat=compiler.stat()
    compiler_stat=[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]
    report=dict(schema=1,sources=identities(),library_sha256=digest(args.library),rccl_receipt=receipt,
                negative_scope=VARIANTS[1],native_init_runtime_qualified=False,
                compile_database_sha256=database_sha,
                compiler=dict(path=str(compiler),stat=compiler_stat),
                compiler_version=subprocess.check_output([str(compiler_driver),'--version'],text=True),
                sdk=str(args.sdk.resolve(strict=True)),architecture=args.architecture,
                native_sources={str(path):digest(path) for path in pinned},
                native_compile_inputs=[dict(source=str(source),flags=flags) for source,flags in source_flags],observations=[])
    with tempfile.TemporaryDirectory(prefix='llaminar-rccl-small-allgather-') as directory:
        tmp=Path(directory);old=tmp/'original';(old/'src').mkdir(parents=True)
        source=source_flags[0][0];body=source.read_bytes();preamble=b'#include "hip/hip_runtime.h"\n'
        prefix=preamble if body.startswith(preamble) else b''
        original=old/'src/enqueue.cc';original.write_bytes(body[len(prefix):])
        subprocess.run(['git','apply','--reverse','--include=src/enqueue.cc',str(ROOT/PATCH)],cwd=old,check=True)
        require(original.read_bytes()!=body[len(prefix):],'Negative control did not change the actual enqueue implementation')
        original.write_bytes(prefix+original.read_bytes())
        symbols=tmp/'probe.map';symbols.write_text('{ local: *; };\n')
        common=[str(compiler_driver),'-x','hip','--offload-host-only','--offload-arch='+args.architecture,
                '--hip-path='+str(args.sdk),'--rocm-path='+str(args.sdk),'-std=c++17','-O3',
                '-ffunction-sections','-fdata-sections','-fvisibility=hidden']
        # Helpers are identical and compiled once. Both enqueue variants retain
        # their exact current source/header ABI and the same selected DSO.
        def compile_helper(item):
            """Compile one independent original helper with its own native flags."""
            index,(source,flags)=item
            target=tmp/('helper-'+str(index)+'.o')
            subprocess.run([*common,*flags,'-c',str(source),'-o',str(target)],cwd=args.rccl_build,check=True)
            return str(target)
        with ThreadPoolExecutor(max_workers=min(len(source_flags)-1,os.cpu_count() or 1)) as executor:
            objects=list(executor.map(compile_helper,enumerate(source_flags[1:])))
        env={key:value for key,value in os.environ.items() if not key.startswith(('NCCL_','RCCL_','AMD_LOG_')) and
             key not in ('LD_PRELOAD','LD_LIBRARY_PATH')}
        for variant,enqueue in ((VARIANTS[0],source_flags[0][0]),(VARIANTS[1],original)):
            binary=tmp/variant
            subprocess.run([*common,*source_flags[0][1],'-DLLAMINAR_RCCL_ENQUEUE_SOURCE="'+str(enqueue)+'"',
                            str(ROOT/PROBE),'-x','none',*objects,str(args.library),'-Wl,--gc-sections',
                            '-Wl,--version-script='+str(symbols),'-Wl,-rpath,'+str(args.library.resolve().parent),
                            '-Wl,-rpath,'+str(args.sdk/'lib'),'-pthread','-o',str(binary)],cwd=args.rccl_build,check=True)
            for mode in ('scope','selection'):
                outcome=subprocess.run([str(binary),mode],env=env,capture_output=True,text=True,timeout=30)
                row=dict(variant=variant,mode=mode,returncode=outcome.returncode,stdout=outcome.stdout,
                         stderr=outcome.stderr,measured=json.loads(outcome.stdout))
                report['observations'].append(row);print(json.dumps(row),flush=True)
    require(all(digest(path)==value for path,value in report['native_sources'].items()),'Native source changed during proof')
    stat=compiler.stat()
    require([stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]==compiler_stat,'Compiler changed during proof')
    require(digest(args.rccl_build/'compile_commands.json')==database_sha and
            (args.rccl_build/'.llaminar-rccl-commit').read_text().strip()==receipt,'Original compile/receipt inputs changed during proof')
    verify(report,args.library)
    publish_verified_report(report,args.report)


def main():
    """Keep compilation in builder and installed verification source/compiler/device free."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('compile','verify'))
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--rccl-build',type=Path);parser.add_argument('--compiler');parser.add_argument('--architecture')
    parser.add_argument('--sdk',type=Path)
    args=parser.parse_args()
    if args.mode=='compile':
        require(all((args.rccl_build,args.compiler,args.architecture,args.sdk)),'Compile requires actual completed native build/compiler/SDK')
        compile_report(args)
    else:verify(json.loads(args.report.read_text()),args.library)


if __name__=='__main__':main()
