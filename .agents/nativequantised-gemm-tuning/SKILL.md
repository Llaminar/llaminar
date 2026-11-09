---
name: nativequantised-gemm-tuning
description: Tune NativeQuantised kernels and refresh exact dispatch overlays across CPU, CUDA and ROCm. Use for measured decode, MTP and prefill choices, TP shard inventories, candidate profiling, numerical equivalence, generated include installation and matched model benchmarks. Generic policy fitting is retired.
---

# NativeQuantised Exact-Overlay Tuning

Use the shared production inventory, native measurement owners, adapters,
exact winner selection and generated emitters. **Exact overlays are the only
tuning workflow.** Do not train generic trees, run generic cross-validation,
open generic seals, rebuild policy-search scorers or download a corpus to
retrain unseen-shape decisions. Existing runtime generic rules remain the
complete authority for unmeasured positive shapes and worker counts.

GPU exact entries retain their measured native architecture. Runtime prefers
an exact architecture match, then the nearest compatible same-backend tuning
architecture, subject to the actual card's kernel and resource limits. Keep
actual hardware identity separate from the chosen tuning identity. Missing
measurements must not disable inference or masquerade as native qualification.

Read the applicable [CPU](../cpu-tuning/SKILL.md),
[CUDA](../cuda-tuning/SKILL.md) or [ROCm](../rocm-tuning/SKILL.md) skill for
backend profiling. Use [kernel replay](../llaminar-testing/references/kernel-replay.md)
and the [testing workflow](../llaminar-testing/SKILL.md) for qualification.
The former `train_native_quantised_dispatch.sh` generic transaction is retired.
`scripts/refresh_native_quantised_dispatch_tables.sh --help` owns the supported
exact collection and installation interfaces.

## Plan the complete physical surface

1. Resolve shapes through `native_quantised_dispatch.shape_manifest`; its shared
   JSON/catalog sources and CMake bindings own CPU, CUDA and ROCm geometry.
   Add missing model and TP-shard obligations there, never in private backend
   lists. Preserve original source geometry and actual participant intervals.
   Project the requested model, topology and operation scope before collecting.
   Deduplicate identical physical execution keys shared by model aliases and
   retain existing qualified keys. Collect missing cells, not the entire model
   inventory on each refresh. Excluding a family from collection must not erase
   its installed dispatch or a shared shape required by a selected family.
2. Use the format and candidate registries. Include every supported source
   format, prepared encoding, runtime ISA, thread width and execution mode.
   Native source-block alignment and normalized prepared-execution support
   are different claims. Record source-unadmitted combinations explicitly;
   do not fabricate a source tensor, requantize it or silently omit a cell.
3. Cover serial decode M1, grouped MTP M2..16 and M31, and the applicable
   ordinary-prefill anchors. Physical GPU buckets come from
   `PrefillGraphBuckets.def`; the shared CPU expert row projection includes
   those buckets without changing dense CPU model-tier evidence budgets.
   Unmeasured routed row counts retain the existing generic policy.
4. Record execution, not just dimensions: arithmetic contract, original and
   prepared codebooks, phase, physical and policy N, K partition, output
   layout, actual worker count, compiled/runtime ISA and entry context.
   An ordinary single projection cannot certify a fused expert operation.
5. Declare the finite refresh scope before collection. Full means every
   declared applicable cell completed. A resumed batch, filtered plan or
   existing-table hit is not a full-sweep certificate.
6. Discover GPU targets through the native metadata authority and partition
   selected physical devices by architecture. Every architecture receives the
   complete format/shape/M inventory; only equivalent lanes share disjoint work.
   Retain UUID/PCI receipts through timing and profiling, normalize one visibility
   mask authority, and never infer architecture from a marketing name or ordinal.
   Table updates must preserve other architecture scopes.

The CPU plan can be inspected without native execution:

```bash
PYTHONPATH=tests/v2/performance/kernels python3 -m \
  native_quantised_dispatch.cpu_exact_sweep \
  --output-dir <work-dir> --threads <physical-cores-per-socket> --plan-only
```

Use the command's help for the Release trainer paths and resumable batch
controls. Keep missing runtime installation surfaces explicit; collection
alone must never be described as installed dispatch.

Fresh model sweeps select Qwen3.5/3.6/3.8 and QFN geometry, retaining shared
physical shapes under their original names. `--model-scope all-retained`
explicitly selects the older full inventory. `--resume` preserves the saved
scope. For CPU, `--reuse-completed-root PATH` admits compatible original cells
and an already installed M1 stage before collecting missing work. Retained M1
policy and reused raw measurements are distinct authorities; preserve both
original producers. CUDA/ROCm decode and grouped MTP can use the same flag to
retain completed shapes from an original collection or superset with matching
producer, hardware, arithmetic, formats, phase and sufficient timing floors.
Only missing shapes launch. Retain the original artifacts: resume revalidates
their references. Pre-install M1 clocks do not qualify a rebuilt producer;
historical routed-MoE receipts lack hardware identity for cross-root reuse and
must resume through their original corpus.

## Measure the actual production operation

Build with Ninja, ccache and unrestricted build parallelism. Canonical timings
use Release. Freeze the executable, loaded core/library closure, prepared
policy and collection settings before timing; a partial relink is not a new
coherent measurement owner. Do not overlap builds, profilers or competing
inference with a canonical timing lane.

CPU regimes are AVX2 build/runtime, AVX512 build with AVX2 runtime, and AVX512
build/runtime. Derive physical socket/core placement from the host. Keep one
native MPI rank per actual socket when measuring distributed execution and
reuse the existing MPI/session owner. A thread-count override is a separately
named measured regime, not permission to relabel its evidence.

Batch compatible requests to amortize source packing, process startup and
persistent worker-team setup. Preserve each request's actual M, phase,
candidate, descriptor vector, output extent, correctness and raw sample
identity. Use atomic per-cell completion and resume only authenticated complete
cells. Batching must not make one request's profiling interval include another.

Interleave or reverse candidate order across repeated samples and fresh
lifetimes. Retain all candidates and raw timings, including failures and
unsupported paths. Compare against the original installed policy and then
verify the emitted policy through actual Auto dispatch. Record whether Auto
means a frozen base or a deliberately excluded exact overlay; do not mix those
meanings in one result.

Where the native owner supports adaptive collection, use short interleaved
screening before full timing. Every candidate still passes its complete
correctness contract. Retain the actual installed baseline and credible
contenders at the original full timing floor; screening-only observations
cannot publish a winner. Profile the selected winner and actual baseline in
isolated runs, without profiling every loser. Two-candidate contests may keep
both candidates, so report measured collection cost rather than assuming a
universal tournament speedup. Older full-grade observations retain their
original producer identity and remain reusable when the physical execution
contract is compatible; a collector strategy change alone is not a reason to
recollect them. Existing exact-point evidence never becomes interval evidence
without the required checks and runtime support.

For CPU fused expert rows:

- Measure the whole fused bundle; never divide its latency into invented
  independent leaf timings.
- Use distinct expert input rows, with gate/up sharing only their own expert's
  input. Retain each original output placement and numerical contract.
- Keep one canonical training vector per complete exact key and separate mixed
  held-out vectors. Change only matching keys in a held-out bundle; unrelated
  descriptors retain their original policy.
- A measured row policy may apply only in its proved, policy-invariant
  execution class. Underfilled, differently partitioned or differently laid
  out work retains its existing implemented dispatch.
- M1 peers may occur in mixed bundles without becoming grouped row-policy
  candidates. Ordinary decode, MTP verifier and prefill keep typed identities.

Retain M1-to-grouped dependency order. If a new M1 overlay can affect the
inherited partition, arithmetic or schedule, install/stage it, rebuild the
measurement owner and collect grouped evidence against that exact generation.

## Keep arithmetic and resources qualified

Grouped MTP must be byte-identical to independent serial rows. Approximate KL
or aggregate agreement is not a substitute. Exercise production capture,
fixed/dynamic MTP where affected, reset, partial work, mixed source formats,
odd/tail rows, repeated execution and actual collective extents.

Profile each selected physical candidate separately from canonical timing:

- CPU: trainer-owned `perf_event_open` groups on the actual persistent worker
  TIDs, exact one-operation interval, no multiplexing. Inspect generated ISA,
  instruction count, IPC, cache traffic and physical-core scaling. Distinguish
  prologue/scalar bookkeeping from hot-loop vector spills.
- CUDA: exact stream and profiler range, registers, local memory/spills,
  occupancy, memory throughput and tensor/ALU utilization.
- ROCm: exact request-local range, VGPR/SGPR/LDS/private storage and ISA,
  occupancy, memory behavior and appropriate compatible counter passes.

Keep request batching within the actual profiler's qualified limits. Preserve
original raw reports and exact physical launch attribution. Profiler durations
never replace canonical latency samples. A faster candidate with changed
reduction order, missing resource evidence or a compute-bound hot-loop spill
cannot be installed.

## Compose and install exact observations

Freeze the installed base includes. Emit only measured exact entries, replace
only the declared refreshed keys and preserve unrelated exact entries and the
existing generic runtime program. A changed ABI or missing supported runtime
surface requires implementation and regression testing; an emitter must not
invent generic supplements or invoke the retired learner to hide the gap.

Exact keys must retain every runtime discriminator without lossy M packing,
model-name special cases or ambiguous aliases. The common exact oracle selects
from validated candidate samples. Held-out operation checks and a repeated
production Auto run must confirm that the selected candidate remains economical.
New evidence does not recertify retained timing or the old generic policy.

The existing GPU prefill transactions remain separate operation surfaces:

```bash
PYTHONPATH=tests/v2/performance/kernels python3 -m \
  native_quantised_dispatch.production_dense_prefill_sweep run \
  --backend <cuda|rocm> --output-dir <work-dir> \
  --binary <release-trainer> --devices <ordinals>

PYTHONPATH=tests/v2/performance/kernels python3 -m \
  native_quantised_dispatch.production_moe_prefill_sweep run \
  --backend <cuda|rocm> --output-dir <work-dir> \
  --binary <release-moe-trainer> --devices <ordinals>
```

Use each module's `status` and `combine` before its existing
`dense_production_overlay` or `moe_production_overlay` emitter. Retain immutable
base and retention receipts for additive refreshes. Do not erase unrelated
installed rows. Dense timings do not certify the routed MoE bundle.

## Regressions and model evidence

Register every discovered defect explicitly in ProductionTestPreflight and
keep metadata/parser/controller checks device-free. Run focused inventory,
phase, source-format, exact-key, retention and native dispatch tests during
iteration. After installation, verify actual exact hits, exact misses retaining
valid generic dispatch, all-format byte equivalence and relevant backend/ISA
paths. A handwritten table or injected test lookup alone cannot certify Auto.

Run the matched Release model benchmark with the same prompt/token IDs,
context, topology, expert placement, ngram policy, MTP and prefix settings.
Report prefill/decode changes separately from component speedups. For numerical
inference failures, use the canonical same-GGUF PyTorch parity framework to
locate the first diverging stage. Never relax a numerical gate to promote a
pure dispatch change.

## Evidence hygiene

Keep local timings and profiler results ignored. Publish large completed
corpora only in `Llaminar/corpora`, with the source repository pinning its
optional gitlink. Do not initialize/download corpora for ordinary builds or
Unit/preflight. Preserve historical artifacts as historical evidence; never
relabel them as a newly measured exact surface. No whole-model or cache-payload
hash passes are part of this workflow.

See [pipeline reference](references/pipeline.md) for ownership and failure
boundaries. Project dashboards record current measurements; this skill records
procedures rather than a tuning diary.
