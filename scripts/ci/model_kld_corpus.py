"""Portable frozen GGUF references and complete full-vocabulary KL scoring.

Only small manifest metadata is digested. Model and logit payloads use explicit
extents and run-scoped stat identities; numerical comparison reads each row once.
The optional corpus checkout is materialized by the master workflow, never Unit
or production preflight. No reference acquisition/regeneration occurs here.
"""
from pathlib import Path, PurePosixPath
import concurrent.futures
import hashlib
import json
import math
import os
from model_kld_canary import definition, evaluate_rows

ROOT=Path(__file__).resolve().parents[2]
CATALOG=ROOT/'scripts/ci/approved_model_kld_corpus.json'


def require(value,message):
    """Reject missing or incompatible evidence without substituting a baseline."""
    if not value:raise ValueError(message)


def identity(path):
    """Pin an immutable input's physical metadata without reading its payload."""
    p=Path(path).resolve(strict=True);s=p.stat()
    return (str(p),s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def metadata_digest(value):
    """Authenticate the small reviewed catalog document, never model/cache bytes."""
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def relative_file(root,name):
    """Reject traversal, absolute spellings, aliases and symlink escapes."""
    require(type(name) is str and name and '\\' not in name,'Invalid corpus path')
    path=PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and path.as_posix()==name,'Invalid corpus path')
    result=(root/name).resolve(strict=True)
    require(result.is_relative_to(root.resolve(strict=True)) and result.is_file(),'Corpus payload escapes its root')
    return result


def load_corpus(root,*,catalog=None):
    """Admit only the source-pinned generation, exact histories and full payloads."""
    root=Path(root).resolve(strict=True)
    pin=json.loads((CATALOG if catalog is None else Path(catalog)).read_text())
    require(set(pin)=={'schema','path','manifest_digest'} and type(pin['schema']) is int and pin['schema']==1,'Invalid KL corpus approval')
    require(type(pin['path']) is str and not PurePosixPath(pin['path']).is_absolute() and '..' not in PurePosixPath(pin['path']).parts and PurePosixPath(pin['path']).as_posix()==pin['path'],'Invalid approved corpus path')
    manifest=json.loads((root/'manifest.json').read_text())
    require(metadata_digest(manifest)==pin['manifest_digest'],'Unapproved KL reference manifest')
    require(manifest.get('schema')=='llaminar.model-kld-corpus.v1' and manifest.get('temperature')==1.
            and manifest.get('vocabulary')==248320 and set(manifest.get('models',{}))=={'dense','moe'},'Incomplete KL reference inventory')
    inputs={str(root/'manifest.json'):identity(root/'manifest.json')};models={}
    for model,entry in manifest['models'].items():
        policy=definition(model,'cuda')
        require(entry['model_file']==policy.model_file and type(entry['model_bytes']) is int and entry['model_bytes']>0,'Wrong GGUF identity')
        require(entry['oracle']['complete'] is True and entry['oracle']['passed'] is True and entry['oracle']['cached_prefill'] is False,'Incomplete independent FP32 oracle')
        sources=entry.get('alpha_beta_sources')
        require(type(sources) is dict and len(sources)=={'dense':96,'moe':60}[model]
                and all(type(v) is dict and v.get('source_format')=='FP32' and
                        all(type(v.get(k)) is int and v[k]>0 for k in ['rows','columns']) for v in sources.values()),'Invalid original alpha/beta inventory')
        workload_path=relative_file(root,entry['workload']);reference_path=relative_file(root,entry['reference_report'])
        workload=json.loads(workload_path.read_text());reference=json.loads(reference_path.read_text())
        require(metadata_digest(workload)==entry['workload_digest'] and metadata_digest(reference)==entry['reference_report_digest'],'Frozen history/reference metadata changed')
        cases=workload.get('cases');require(type(cases) is list and len(cases)==3,'Missing short/long/reset histories')
        require([len(c['prompt_ids']) for c in cases]==[512,policy.prompt_tokens,512] and
                [len(c['continuation_ids']) for c in cases]==[5,129,5],'Changed canary history geometry')
        require(cases[0]['prompt_ids']==cases[2]['prompt_ids'] and cases[0]['continuation_ids']==cases[2]['continuation_ids'],'Missing identical terminal reset history')
        require(len({c['name'] for c in cases})==3 and all(type(c['name']) is str and c['name'] and '/' not in c['name'] and '\\' not in c['name'] for c in cases),'Invalid history names')
        require(all(type(v) is int and 0<=v<policy.vocabulary for c in cases for k in ['prompt_ids','continuation_ids'] for v in c[k]),'Invalid frozen token IDs')
        require(reference.get('complete') is True and reference.get('passed') is True and len(reference.get('cases',[]))==1,'Incomplete FP32 reference')
        case=reference['cases'][0];validate_case(case,cases[1],policy.vocabulary)
        payload=relative_file(reference_path.parent,case['logits_file'])
        require(payload.stat().st_size==129*policy.vocabulary*4,'Missing, truncated or unmaterialized LFS reference')
        for path in [workload_path,reference_path,payload]:inputs[str(path)]=identity(path)
        models[model]=dict(entry=entry,workload=workload,workload_path=workload_path,reference=reference,reference_payload=payload)
    return dict(manifest=manifest,manifest_digest=pin['manifest_digest'],inputs=inputs,models=models)


def unchanged(inputs):
    """Reject replacement/mutation throughout a model or scoring lifetime."""
    require(all(identity(path)==expected for path,expected in inputs.items()),'Canary input changed during execution')


def validate_case(case,wanted,vocabulary):
    """Authenticate row positions and full vocabulary against a frozen history."""
    count=len(wanted['continuation_ids']);first=len(wanted['prompt_ids'])
    require(case.get('name')==wanted['name'] and case.get('prompt_tokens')==first and case.get('continuation_ids')==wanted['continuation_ids'],'Changed teacher history')
    rows=case.get('rows');require(type(rows) is list and len(rows)==count,'Incomplete observed rows')
    require(all(type(row.get('consumed_tokens')) is int and row['consumed_tokens']==first+i and
                type(row.get('vocabulary')) is int and row['vocabulary']==vocabulary and
                type(row.get('argmax')) is int and 0<=row['argmax']<vocabulary for i,row in enumerate(rows)),'Incorrect consumed position/vocabulary metadata')


def physical_workers():
    """Use physical cores in this affinity, excluding duplicate SMT siblings."""
    cores=set()
    for cpu in os.sched_getaffinity(0):
        root=Path('/sys/devices/system/cpu')/f'cpu{cpu}'/'topology'
        cores.add(((root/'physical_package_id').read_text().strip(),(root/'core_id').read_text().strip()))
    require(cores,'No physical worker cores');return len(cores)


def row_metrics(reference,candidate,*,vocabulary=None):
    """Keep the historical row-scoring API through the shared FP64 authority.

    Corpus callers already authenticate the original vocabulary in validate_case
    and the exact memmap extent before selecting a row. This compatibility entry
    retains that row-sized contract; new direct consumers must pass their own
    independently authenticated vocabulary to full_vocabulary_metrics instead.
    Legacy metric keys remain present; added diagnostics never change gates.
    """
    import numpy as np
    from full_vocabulary_metrics import full_vocabulary_metrics
    return full_vocabulary_metrics(reference,candidate,vocabulary=int(np.asarray(reference).size) if vocabulary is None else vocabulary)


def score_observation(corpus,report_path,*,model,backend):
    """Verify fresh histories/reset and score prefill and decode independently."""
    import numpy as np
    policy=definition(model,backend);item=corpus['models'][model];path=Path(report_path).resolve(strict=True)
    document=json.loads(path.read_text());require(document.get('complete') is True and document.get('passed') is True,'Native observation incomplete')
    cases=document.get('cases');require(type(cases) is list and len(cases)==3,'Native reset history missing')
    pins={str(path):identity(path)};arrays=[]
    for case,wanted in zip(cases,item['workload']['cases']):
        validate_case(case,wanted,policy.vocabulary)
        # Native reports use absolute output paths. Require them to stay inside
        # their owned artifact directory; corpus reports are strictly relative.
        data_path=Path(case['logits_file']).resolve(strict=True)
        require(data_path.is_relative_to(path.parent),'Native logit payload escapes its owner')
        shape=(len(case['rows']),policy.vocabulary)
        require(data_path.stat().st_size==math.prod(shape)*4,'Truncated full-vocabulary output')
        pins[str(data_path)]=identity(data_path)
        values=np.memmap(data_path,dtype=np.float32,mode='r',shape=shape)
        require(np.isfinite(values).all(),'Nonfinite native logits')
        require(all(int(np.argmax(value))==meta['argmax'] for value,meta in zip(values,case['rows'])),'Stale native logit metadata')
        require(all(not np.array_equal(a,b) for a,b in zip(values,values[1:])),'Repeated stale decode logits')
        arrays.append(values)
    require(np.array_equal(arrays[0].view(np.uint32),arrays[2].view(np.uint32)),'Reset replay changed logit bytes')
    reference=np.memmap(item['reference_payload'],dtype=np.float32,mode='r',shape=(129,policy.vocabulary))
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(129,physical_workers())) as workers:
        metrics=list(workers.map(lambda index:row_metrics(reference[index],arrays[1][index],vocabulary=policy.vocabulary),range(129)))
    rows=[dict(consumed_tokens=policy.prompt_tokens+i,**metric) for i,metric in enumerate(metrics)]
    unchanged(corpus['inputs']);unchanged(pins)
    return dict(evaluate_rows(rows,model=model,backend=backend),rows=rows,reset_byte_equal=True,
                corpus_manifest_digest=corpus['manifest_digest'],full_vocabulary=True)
