# NativeQuantised exact dispatch pipeline

Generic fitting is retired. Existing installed generic runtime decisions are
retained for unseen shapes; the refresh workflow only collects, selects and
composes measured exact entries.

## Authority map

| Concern | Authority |
|---|---|
| Exact refresh entrypoint | `scripts/refresh_native_quantised_dispatch_tables.sh` |
| Shared physical inventory | `native_quantised_dispatch/shape_manifest.py` and its JSON catalogs |
| Expert operation/source intervals | `expert_projection_geometries()` in the same module |
| Prefill row anchors | `prefill_matrix.py`, `PrefillGraphBuckets.def` |
| Source formats and candidate IDs | `format_registry.py`, `candidate_registry.py` |
| Native measurement | Existing CPU/CUDA/ROCm performance fixtures |
| Evidence admission | Common schema and backend adapters |
| CPU execution-exact identity | `CPUNativeQuantisedExecutionExactPolicy.h`, `cpu_exact_overlay.py` |
| CPU retained legacy programs | `cpu_exact_retention.py` |
| Dense/MoE GPU prefill | `production_dense_prefill_sweep.py`, `production_moe_prefill_sweep.py` |
| Exact winner generation | Existing common exact oracle and backend emitters |
| Profiler attribution | `profiler_evidence.py`, `profiler_collectors.py` |

Historical learner/scorer readers are not a production tuning entrypoint.
Do not invoke them to certify exact rows or create new generic rules.

## Evidence flow

```text
shared geometry + supported formats + actual operation contract
                         |
                         v
frozen production native owner -> raw samples + correctness + route witness
                         |
                         v
common evidence adapter -> exact candidate comparison
                         |
                         v
mixed-operation checks + separate physical-candidate profiling
                         |
                         v
complete declared exact delta + unchanged installed base
                         |
                         v
generated include -> actual Auto hit/miss + native byte gates -> model benchmark
```

Whole-operation timing stays whole. A fused bundle observation retains its full
descriptor vector and arithmetic/ownership class. A measured ordinary GEMM
cannot be relabeled as a fused expert execution. Phase is explicit; M alone
cannot identify MTP versus ordinary prefill.

Native source formats may require larger blocks than a prepared execution
layout. Legacy normalized prepared fixtures can prove supported small-K
arithmetic without claiming that an unsliced GGUF tensor has that source
shape. Source-faithful packing and migration need their own native-block
coverage. Explicit plan metadata must distinguish both claims.

## Retention and publication

A refresh freezes the original include, declares its key scope, requires all
applicable candidate evidence, and changes only those exact keys. Unrelated
rows and the generic program are retained. Reports separate old retained
provenance from fresh measurements; they do not reuse an old generic
certificate as certification of the changed include.

CPU execution-exact keys are full-width structured values. Build/runtime ISA,
source and prepared formats, numerical policy, phase, worker count, M/N/K,
policy N, K partition, layout and execution context must survive the
native-to-adapter-to-emitter round trip. Candidate selection cannot change the
ownership class under which its bundle was measured.

GPU emitters retain exact physical family identities. A streaming Auto family
is not automatically a forceable cooperative candidate. CUDA global and
CTA-local K partition families remain distinct. ROCm Q8 blockwise INT8 and
low-bit NativeQuantised candidates retain their respective launch controls and
resource witnesses. Unknown cross-family tuples fail validation.

## Failure and resume boundaries

| Condition | Required behavior |
|---|---|
| Partial run | Resume complete authenticated cells; keep total scope incomplete. |
| Changed native binary/core, policy or measurement contract | Start compatible new evidence explicitly; never silently reuse stale timings. |
| Missing candidate, byte mismatch or incompatible descriptor | Reject the affected observation and block its installation. |
| Source format cannot represent a native slice | Report source-unadmitted; do not pad, requantize or silently skip. |
| Exact miss at runtime | Use the existing complete runtime dispatcher; do not fail a legal unseen shape. |
| Missing GPU architecture in exact tables | Select the nearest compatible measured architecture within the backend; retain actual hardware limits and identity. Missing records alone never disable inference. |
| Emitter would modify generic rules | Reject composition; no learner or generic supplement is permitted. |
| New source/M obligation | Extend the shared plan and measure the missing exact cells. |
| Profiler failure | Keep canonical timing but do not fabricate resource evidence. |
| Failed installation/Auto verification | Keep the original production include; investigate using the retained candidate. |

CPU M1 and grouped evidence remain dependency ordered when a serial-policy
change affects grouped execution. Process batching may amortize preparation,
but must preserve separate request, phase, candidate and timing identities.
GPU capture/eager modes are independent measured modes where exposed.

GPU collection partitions the native immutable inventory by actual architecture.
Every group measures the full shared plan; splitting one plan across different
architectures is incomplete evidence. Context supplied by a caller can assert
observed facts, but cannot assign them. Masked children authenticate physical
UUID/PCI against discovery, since their local ordinal may have changed. An
architecture-scoped refresh retains other scopes and composes all M1 deltas
before their owning rebuild and grouped measurements.

Large timing/profiler corpora belong only in the optional `Llaminar/corpora`
repository. Ordinary builds and tests must not initialize it. Local plans,
partials and profiler output remain ignored; only small unit fixtures and
installed policy source belong in the engine repository.
