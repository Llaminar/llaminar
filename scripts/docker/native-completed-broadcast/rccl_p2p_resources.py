#!/usr/bin/env python3
"""@file rccl_p2p_resources.py
@brief Extend the canonical dependency proof with complete final owned-entry resource evidence.

The original final-allocation producer must authenticate the selected DSO,
receipt and all output partitions first. This helper then audits every exact
owned entry and its complete call-free ISA closure; upstream ordinary kernels
remain outside this distinct entry proof. It never loads a runtime.
"""
from pathlib import Path
import importlib.util, sys, json, hashlib, re, argparse

def require(condition, detail):
    """Reject incomplete native evidence even under Python optimization."""
    if not condition:
        raise RuntimeError(str(detail))
def owned_record_files(allocation_root, library, architectures, producer):
    """Reuse original output-derived paths and bind every partition to its proof.

    The parent has authenticated the final-allocation report against this DSO
    and receipt. This narrower audit must consume the same configuration and
    all eight object partitions per target; neither legacy names nor unrelated
    matching files can substitute for a missing current output.
    """
    spec = json.loads((allocation_root / "configuration.json").read_text())
    require(spec.get("schema") == 1 and spec.get("architectures") == architectures and
            producer.get("resource_evidence_complete") is True and
            producer.get("architectures") == architectures,
            "Owned-entry audit changed its original final-allocation configuration")
    require(Path(spec["library"]).resolve(strict=True) == library.resolve(strict=True) and
            Path(spec["record_root"]).resolve(strict=True) == allocation_root.resolve(strict=True),
            "Owned-entry records belong to a different original output")
    artifact_path = Path(__file__).resolve().parents[1] / "llvm-lto-linker/artifact.py"
    entry = importlib.util.spec_from_file_location("owned_rccl_record_authority", artifact_path)
    artifact = importlib.util.module_from_spec(entry)
    sys.modules[entry.name] = artifact
    entry.loader.exec_module(artifact)
    observations = producer.get("observations", [])
    require(len(observations) == len(architectures) and
            {row.get("architecture") for row in observations} == set(architectures),
            "Owned-entry audit lost an original architecture observation")
    result = {}
    for arch in architectures:
        files = artifact.object_records(spec, arch)
        observation = next(row for row in observations if row["architecture"] == arch)
        records = [dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        bytes=path.stat().st_size) for path in files]
        require(observation.get("partitions") == artifact.PARTITIONS and
                observation.get("record_files") == records,
                "Owned-entry object records differ from authenticated final allocation: " + arch)
        result[arch] = files
    return result


def main():
    """Audit actual owned metadata/ISA after complete original allocation admission."""
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('--library', type=Path, required=True)
    args.add_argument('--allocation-root', type=Path, required=True)
    args.add_argument('--spill-guard', type=Path, required=True)
    args.add_argument('--llvm-bin', type=Path, required=True)
    args.add_argument('--output', type=Path, required=True)
    args.add_argument('--architectures', required=True)
    args.add_argument('--final-report', type=Path, required=True)
    args = args.parse_args()
    architectures = args.architectures.replace(',', ';').split(';')
    if not architectures or len(set(architectures)) != len(architectures) or (not all((re.fullmatch('gfx[0-9a-f]+', a) for a in architectures))):
        raise RuntimeError('Invalid original native architecture inventory')
    producer = json.loads(args.final_report.read_text())
    if producer.get('resource_evidence_complete') is not True or producer.get('architectures') != architectures or producer.get('library_sha256') != hashlib.sha256(args.library.read_bytes()).hexdigest():
        raise RuntimeError('Missing complete original final-allocation authority')
    r = args.output
    r.mkdir(exist_ok=True, parents=True)
    dso = args.library
    p = args.spill_guard
    sp = importlib.util.spec_from_file_location('final_canonical_guard', p)
    m = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = m
    sp.loader.exec_module(m)
    llvm = args.llvm_bin
    allkernels = m.audit_rocm_object(dso, llvm)
    kernels = [k for k in allkernels if ('ncclDevKernel_LocalP2p_' in k.name or 'ncclDevKernel_P2pProgressObservation' in k.name)]
    require(len(kernels) == 7 * len(architectures), len(kernels))
    alloc = []
    record_files = owned_record_files(args.allocation_root, args.library, architectures, producer)
    for arch in architectures:
        files = record_files[arch]
        functions, _ = m.allocation_evidence(''.join((f.read_text() for f in files)))
        owned = {n: v for n, v in functions.items() if ('ncclDevKernel_LocalP2p_' in n or 'ncclDevKernel_P2pProgressObservation' in n)}
        require(len(owned) == 7, (arch, owned.keys()))
        require(all((v.instances >= 1 and (not v.slots) for v in owned.values())), 'Incomplete actual owned-entry closure')
        alloc.append(dict(arch=arch, records=[dict(path=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files], entries={n: dict(instances=v.instances, slots=[vars(s) for s in v.slots]) for n, v in owned.items()}))
    require(all((k.private_bytes == 0 and (not k.dynamic_stack) and (k.vgpr == 0) for k in kernels)), 'Incomplete actual owned-entry closure')
    bundle = r / 'device.bundle'
    m.run_tool([str(llvm / 'llvm-objcopy'), '--dump-section', f'.hip_fatbin={bundle}', str(dso), str(r / 'host.o')])
    targets = m.run_tool([str(llvm / 'clang-offload-bundler'), '--list', '--type=o', f'--input={bundle}']).splitlines()
    isas = []
    for i, target in enumerate(targets):
        if not target.startswith(('hip-', 'hipv4-')):
            continue
        arch = next((x for x in architectures if target.endswith(x)), None)
        require(arch, target)
        code = r / (arch + '.co')
        m.run_tool([str(llvm / 'clang-offload-bundler'), '--unbundle', '--type=o', f'--targets={target}', f'--input={bundle}', f'--output={code}'])
        text = m.run_tool([str(llvm / 'llvm-objdump'), '--disassemble', '--demangle', str(code)])
        (r / (arch + '-isa.log')).write_text(text)
        chunks = re.split('(?=^[0-9a-f]+ <)', text, flags=re.M)
        entries = [x for x in chunks if ('ncclDevKernel_LocalP2p_' in x.split('\n', 1)[0] or 'ncclDevKernel_P2pProgressObservation' in x.split('\n', 1)[0])]
        require(len(entries) == 7, (arch, len(entries)))
        rows = []
        for entry in entries:
            calls = re.findall('\\b(?:s_swappc_b64|s_call_b64|s_setpc_b64)\\b', entry)
            require(not calls, entry.split('\n', 1)[0])
            rows.append(dict(symbol=entry.split('\n', 1)[0], out_of_line_calls=0))
        (r / (arch + '-owned-isa.log')).write_text(''.join(entries))
        isas.append(dict(target=target, entries=rows, code_sha256=hashlib.sha256(code.read_bytes()).hexdigest()))
    require(len(isas) == len(architectures), 'Incomplete actual owned-entry closure')
    summary = dict(architectures=architectures, final_resource_evidence_sha256=hashlib.sha256(args.final_report.read_bytes()).hexdigest(), dso=str(dso), dso_sha256=hashlib.sha256(dso.read_bytes()).hexdigest(), kernel_records=[vars(k) for k in kernels], final_allocation=alloc, isa=isas, scope='All14 actual owned-entry target records and full call-free code closures; NOT wholeRCCL/noNativeRuntime/noPerformanceQualification')
    (r / 'audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('PASS final canonical RDC/LTO:allownedentries, complete8partitions/ISApertarget,0private/0dynamicstack/0VGPRspills/0frameSlots/0calls')
    print(summary['dso_sha256'])


if __name__ == "__main__":
    main()
