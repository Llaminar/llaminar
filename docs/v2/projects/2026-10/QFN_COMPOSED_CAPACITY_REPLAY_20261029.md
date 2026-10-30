# QFN composed pipeline capacity replay

This investigation records metadata and admission evidence at `6f4c742e7`.
It is not an inference, numerical-accuracy or performance certificate. Current
qualification belongs in [the source-admission dashboard](QFN_SOURCE_ADMISSION_MATRIX.md).

## Method

The replay reads the real main and predictor GGUF metadata used by the canonical
source-admission cases. The source-case producer, topology resolver, fixed-memory
planner, routed-expert admission draft and PhysicalMemoryAuthority produce the
complete bill of materials. No table or weight payload is hashed or copied.

The diagnostic supplies synthetic 64 GiB GPU capacity to expose the complete
mandatory expert population. Its totals are then compared with the actual usable
capacity recorded by the native admission run. Synthetic capacity is never used
to admit a native run. The canonical minimum M64 prefill bucket, 15 retained MTP
draft slots, 16 verifier rows, source formats, expert placement and RAM ngram
policy remain unchanged. The disabled-MTP control retains its own normal policy.

The ROCm replay interposes and rejects CUDA/HIP context, allocation, copy, launch,
synchronization and device-query entrypoints. Its four completed cases made no
interposed runtime calls. All metadata evidence is in
`/tmp/qfn-rocm-pp-capacity-r13/`, including `main.cpp`, `build.py`, `deny_gpu.c`,
the complete `audit.log`, and the compact `result.json`.

## CUDA TP2 followed by CPU TP2

An earlier pipeline stage was incorrectly charged terminal-head and predictor
workspace when it retained main-model verifier rows. The installed fix projects
the same typed global-component ownership used by weight planning into workspace
planning. It preserves main-model rollback, actual terminal predictor storage,
and tied-head aliases. CPU, CUDA and ROCm consume the same selection contract.

For the original 16-GPU-layer/32-CPU-layer case, the fix removes **1,121,648,640
bytes per GPU**. Full retained MTP now requires 25,142,846,104 bytes per GPU,
against 25,005,457,408 usable bytes recorded in the native run. The remaining
137,388,696-byte deficit is not permission to discount another owner or weaken
the original positive case.

The separately named 15/33-layer family needs 24,005,714,888 bytes per GPU under
the same full retained MTP policy. Its complete **16-case native family passed**:
RAM, mmap, zero-capacity host-adaptive and positive-capacity host-adaptive ngrams,
each with Off, fixed1, fixed3 and Dynamic MTP. Every case proves retirement;
runtime identity and driver checks remained clean. Evidence is retained under
`parity-results/qwen-flash-next/source-admission-cuda-cpu15-full15-r12/`.

The original 16/32-layer positive declarations remain enabled. The 15/33 family
is an explicitly authored topology, never an automatic substitution. Full Unit
passed 790/790 and the 13 affected preflight registrations passed before this
slice was checkpointed. The original workspace implementation fails all three
new ownership regressions.

The later complete r13 prerequisites pass on this checkpoint: 790 Unit and
2,258 preflight passes, 29 unavailable-hardware skips, zero failures and a clean
driver window. The 3,077-registration run took 5,088.69 seconds; its canonical
receipt is `/tmp/qfn-admission-r13-prerequisites/prerequisites.json`. This does
not qualify the separate full source-admission or inference matrices.

## ROCm PP(TP2, TP2)

Each entry below is the complete per-GPU physical BOM, in bytes. Both GPUs in a
TP domain have the same total. The terminal domain also owns the predictor and
output path. The native run recorded 34,091,302,912 usable bytes per GPU.

| Explicit layer split | MTP policy | Earlier domain | Terminal domain |
|---|---|---:|---:|
| Original 24/24 | Off | 30,837,664,092 | 30,189,712,628 |
| Original 24/24 | Full15 retained | 31,691,153,776 | 34,812,493,532 |
| Diagnostic 25/23 | Off | 31,913,849,268 | 29,110,887,572 |
| Diagnostic 25/23 | Full15 retained | 32,799,192,520 | 33,700,348,540 |

The original terminal domain exceeds recorded usable memory by 721,190,620
bytes, about 688 MiB. Moving one main layer to the earlier domain leaves
390,954,372 bytes, about 373 MiB, on the tighter terminal domain. This is a
fitting declaration candidate, not native admission evidence. Its available
margin must also cover any newly implemented, explicitly owned native transport
storage before qualification.

The original 24/24 declarations remain positive obligations. The native
homogeneous PP(TP,TP) transport implementation is a separate unfinished
requirement; changing the split does not bypass that requirement. No 25/23
source-admission family or new runtime capability was installed by this replay.
