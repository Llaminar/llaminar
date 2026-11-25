# P0 pooled paged-KV migration inventory

Baseline: implementation `a0b69f633`, adopting exact revision-4 design and
P0–P10 plan from `770bade4d1c6fc2f8566aef9de59f2ba7b812659`.

**Revision-5 reconcile (2026-11-25):** reviewer rewrite
`tiered-paged-design-rewrite.L7ICwq` delivered owner-approved design
`debc78dac0728194455e25cd461e8e1784af313aa29d10d39206f7847c1ff911` and plan
`925dc4d8e37ec57d4893b7cbcc70b094396834021ac88922808cd2d89ee18b35`
(diff `f1614994120883f0f7833641c71a7cacc5ee891423f7ff7db63ca2e416706c2f`).
Revision 5 is the current target: maximal PMA-admitted shared GPU KV residency
with configurable caps/tiers, UMA single accounting, complete S-held
suspension, a bounded all-resource completion witness and preserve/restore
first. PKV-01 is now owner-selected (§§1, 5), not pending. No numeric defaults
or installed CLI spellings are invented here; revision-4 language below is
retained as historical evidence, not current policy.

This is a source-linked working inventory, not P0 exit acceptance or P1
implementation evidence. Full refactoring is authorized by the replacement
plan. Historical fixed-partition tests remain oracles until replacement
qualification; they do not constrain the new physical interface.

## Preserved concurrent work

The three uncommitted files at adoption are DeviceGraphOrchestrator.cpp/.h and
Test__MTPGraphConstruction.cpp. Their complete diff and original bytes are in
`CB-checkpoints/paged-design-adoption.emZ6os`. They implement a private CPU
predictor transaction over mapped physical KV sequences; focused tests pass
but it is not enabled in ServingSession. Do not discard this work, advertise
it as paged access, or finish it as the new critical path.

| Work | Disposition | Replacement binding |
|---|---|---|
| `b4fe53a96` CPU sidecar row-map and native cache-identity tests | Refactor/oracle | Row maps bind generation-qualified published page views, not cache sequence partitions |
| `e0fd55147` main/draft hidden banks and exact PMA BOM | Keep request-state concept; refactor identity | Request-record generations own hidden publications independently of page frames |
| `213fd003f` terminal-row publication/reset | Keep numerical/ordering oracle | True request terminal rows publish under request-generation leases |
| Uncommitted compact resident predictor transaction | Preserve, suspend | Typed page-view input and speculative page reservations before P4 integration |
| CPU dense/hybrid shared-row tests | Keep behavioral/numerical oracle | New implementation must prove actual native paged access in addition to A/B/C behavior |
| Strict full-logit traces/live-ID/SSE tests | Keep invariants | Add page/table/commitment/access initial-state records; old traces do not qualify paging |

## Source census and scope

Source locators are preserved in `CB-checkpoints/paged-p0-census.2HESQB`:
328 cache/raw-view consumers, 250 prefix/reset/truncate locators, 64 estimator
locators, 249 registration locators and 96 backend owner paths. These are
search candidates, not correctness proofs or a claim that every occurrence is
a defect. The census is bound to the head and dirty status in that directory.
Template helpers, indirect kernel consumers and topology protocols still need
semantic closure before P0 exit. No absence claim is inferred from these scans.

All source paths below are relative to the repository root.

## Ownership transformation

| Owner / exact starting seam | Existing assumption | Action / target contract |
|---|---|---|
| `src/v2/planning/PhysicalMemoryAuthority.h`: PhysicalMemoryAuthority, allocation/reservation leases | Certified bytes and allocation lifetimes, not logical pages | Extend canonical authority with pool creation and real page suballocation. Backing and metadata each retain exactly one physical claim. No independent free-byte balance |
| `src/v2/planning/KVCacheMemoryEstimator.cpp`: estimate and compressed layout helpers | batch × context histories, per-entry anchors and format padding | Refactor into storage-class page geometry, pool counts independent of S, table/reference/retirement metadata and completion commitments. Preserve native codec sizing |
| `src/v2/planning/RankMemoryPlanInputs.*`, `MoEOverlayMemoryPlanInputs.*` | KV demand follows participant batch/context | Refactor typed pool demand per actual participant/shard and role; all-class prepare/commit/refusal; no remote ownership aliases |
| `src/v2/planning/ActivationMemoryEstimator.cpp`, `memory/GraphArenaStorageLayout.*` | Scratch/hidden work envelopes include ordinary and MTP batch capacities | Keep exact physical BOM; distinguish request records, work rows, table banks and actual simultaneous storage. Hidden/recurrent state is not page payload |
| `src/v2/kernels/IKVCache.h`: get_k/get_v/get_kv, append, logical block import/export, truncate/reset | Sequence index yields contiguous/ring backing and mutable cache metadata | Replace native serving seam with typed versioned page views, valid extents and read/write leases. Do not implement get_kv by gathering full histories |
| `src/v2/kernels/cpu/CPURingKVCache.cpp`: initializeLayerCache, append_kv_impl, logicalRead/get_kv, truncateSequence | Allocate K/V max_seq_len for each layer/sequence (lines 235–249 at baseline); ring heads/size; export flattening seq_idx × max_kv_len | Retire private full-context backing at P2 cutover. Native append and attention traverse arbitrary frames in logical order. Keep codecs and serial arithmetic controls |
| `src/v2/kernels/cpu/CPURingKVCacheAnchored.inl`, `CPUKVCache.h` | Anchors and helper views borrow sequence-owned payload | Refactor page-native format metadata and compatible anchor ownership; enumerate mutable versus shared metadata before COW |
| `src/v2/kernels/cpu/CPUHybridRingKVCache.h`, `kernels/IHybridKVCache.h` | Attention sequence and recurrent owner share slot coordinates | Keep request-owned mutable GDN/conv state; page attention histories separately; coherent immutable prefix checkpoints |
| `src/v2/kernels/cpu/attention/CPUFlashAttentionKernelT.*`, `CPUFlashAttentionAQ8.inl`, `CPUTQAttentionOperands.h`, `CPUQwenSparseAttention.cpp` | Contiguous/ring K/V tensors, ring geometry, per-request views | Native paged operand traversal. Retain exact logical reduction order, ISA/thread contracts and supported codecs; bounded tile staging only |
| `src/v2/kernels/cuda/kvcache/CUDARingKVCacheBase.*`, `CUDARingKVCache.cu`, `CUDARingKVCacheTQ.*`, `CUDAHybridRingKVCache.h` | Retained sequence banks and device count arrays | Replace with PMA-owned device arena/table authority. Runtime page assignments are device data, never capture keys or host mirrors |
| `src/v2/kernels/rocm/kvcache/ROCmRingKVCacheBase.*`, `ROCmRingKVCache*`, `ROCmHybridRingKVCache*` | Same fixed backing and native kernels | Equivalent ROCm page semantics, queued-access retirement and retained HIP transactions; no CPU/CUDA qualification substitution |
| `src/v2/kernels/cuda/attention/`, `src/v2/kernels/rocm/attention/` | Native ring/cache descriptors and launch families | Native paged kernels with exact live work, stable table bindings, tensor-core prefill and shipping-target spill evidence |
| `src/v2/kernels/attention/AttentionDeviceParams.h`, `AttentionExecutionPolicy.h`, `kvcache/KVCacheWorkspaceContract.h` | Immutable kernel policies plus cache pointers/workspace | Keep policy distinctions; replace cache operands with typed paged layouts, separately price lookup/tile work, bind all embedded identities |
| `src/v2/execution/compute_stages/stages/KVCacheAppendStage.*`, `AttentionComputeStage.*` | request_kv_slots_host maps rows to physical cache sequences; diagnostic K/V views | Replace row→request binding→published page-view contract. Diagnostics observe actual pages/valid extents without private full-history mirrors |
| `src/v2/models/qwen/QwenGraphBase.*`, `models/qwen35/Qwen35Graph.cpp`, `models/qwen4exp/Qwen4ExpPredictorGraph.cpp` | Host row maps and first_sequence_index threaded through builders | Declarative borrowed typed views; retain scoped construction and separate predictor source/role. Model files do not allocate pages |
| `src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp`, `graph/DeviceGraphExecutor.cpp`, `graph/IGraphBuilder.h` | Graph identity/output ownership and row selection | Stable table/view-bank addresses plus storage generations; publish complete immutable versions; retire exact borrowers |
| `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`: initializeInferenceStateFromArena, forwardServingRows, resetServingRow, populateCPUMTPShiftedCacheFromPrefill | Batch-sized private caches; row-indexed positions; first_seq_idx sidecar cache | Keep shared execution oracle; replace cache creation, table binding, prefix hooks and speculative ownership. Physical cache slots cannot remain the production address authority |
| `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestratorServingGraphFamily.cpp`, `ServingWindowDecode*` | Retained serving ABI remains an unimplemented GPU seam | Refactor complete transaction catalog for page-table ABI and all policy scalars; no host occupancy/count selection |
| `src/v2/execution/local_execution/orchestrators/RankOrchestrator.*`, `execution/runner/OrchestrationRunner.*`, MPI/global execution owners | Fixed batch/cache sequence wiring and participant commands | Preserve runner sole lifecycle authority. Topology-wide logical epoch; participant-local prepare/commit grants and retirement. No independent rank grant reordering |
| `src/v2/execution/serving/ServingSession.*`, `ServingBatch.h` | Resident row slot owns one cache sequence; retained logits and sampler | Keep lifecycle/sampler/output behavior; request generations own paged tables/commitments. Compact work rows are not physical frames |
| `src/v2/execution/mtp/`, terminal hidden and speculative checkpoint owners | Main/shifted cache slots, global mailbox/learner state, truncation for rollback | P4: generation-qualified request state and atomic main/predictor tentative page visibility. Preserve unpublished borrowers through cancellation/rejection |
| `src/v2/execution/prefix_cache/`, DeviceGraphOrchestrator restore/harvest/export paths | Materialized block archives copied into each request sequence | P3: resident pages actually shared, partial-tail COW, archive imports into exclusive frames, coherent checkpoints, effective-use recency |
| `src/v2/transfer/TransferEngine.*` public APIs | Movement/coherence events and allocation lifetimes | Bind page source-ready and exclusive destination-prior-access leases; exact completion generation. Prior readers and writers, not FIFO or disk delay, establish safety |
| `src/v2/execution/config/ServingCapacity.h`, `ServingPoolMemoryDemand.h`, config CLI/parser/document/startup owners | S/R/Q logical records but KV priced from batch/context; nominal response cap | Keep exact-or-reject logical limits. Add independent pool geometry/policy through canonical config and planning; P5 complete host envelope still open |
| `src/v2/app/modes/ServingBatchDriver.h`, `ServingResponseLeasePool.h`, `RequestGenerationLane.h`, `SSEPublishChannel.h`, frontends | Existing detached response/grant protocol with open aggregate bounds | Keep lifecycle contracts, close R1/R2/F5 through P5; no frontend page ledger or GPU control |
| `tests/v2/CMakeLists.txt`, cache/attention/prefix/MTP/serving/planning suites | Many tests assert fixed backing, ring wrap and sequence aliases | Classify each registration as preserved behavior/numerical oracle or replaced physical contract; add explicit production paging replay, geometry and access-lifetime entries |

## P1 prototype boundary and pressure decision history

Independent design review at exact `770bade4d` historically blocked treating
P1 as an unconditional contract (PKV-01/02/03). The retained review is
`CB-checkpoints/paged-kv-design-review-770bade4d.bv0UfJ/REVIEW.md`, SHA256
`95d3072625cb849e8cb51642332bce88bb59f25d28323f33943fb093eef473b2`.
Prior to revision 5 the implementation had not adopted a different pressure
policy and the operator answered **“ask the reviewer”**; that decision request
and the revision-4 resident-only proposal remain the historical record below.

**Current direction (owner-selected in revision 5):** bounded optimistic
retain/restore across GPU/RAM/disk with a concrete all-resource/topology
completion witness (design §§5.1–5.3), finite operation grants, cache-aware
bounded fairness and no-copy refresh. Old summed-maximum wording is not the
policy. Revision-4 proposal (historical): A request's declared maximum history
(including default remaining-context output) and simultaneous
COW/speculative overlap acquire fungible commitments across all required
storage classes before activation. They do not acquire fixed frame ranges. One
long request may consume more than an equal share of a pool.
Pending-retirement frames remain unavailable.

PMA will retain the actual pool and bounded metadata allocations. Its CPU
page-arena implementation owns frame/table generations, logical references,
commitments and access leases; those are ownership state inside charged
backing, not duplicated physical capacity accounting. GPU adapters must move
that canonical suballocation state to device-resident storage. The host keeps
immutable geometry and physical-allocation claims only. A CPU model is not a
GPU implementation or qualification.

The transaction seam must prepare all classes without publication, then commit
all or withdraw all. Published views retain table versions and actual access
frontiers. Closing a view prevents new readers; logical reference zero cannot
free a page until all submitted access completes. Restore/COW acquires exclusive
overwrite only after old readers/writers retire, then exact completion enables
new consumers. Late completions authenticate pool/frame/table/request
generations. No ad hoc dirty transition or raw byte-budget mutation is allowed.

## First unpublished authority prototype (not P1 exit)

`planning/PhysicalPagePool.*` and `PhysicalMemoryAuthority::createCPUPagePool`
now materialize PMA-charged CPU payload, frame descriptors, bounded request
records and frame-handle tables through `CPUStorageRoot`. Logical commitments
are fungible and convert to distinct frame incarnations without physical
allocation. An unpublished grant owns rollback through RAII. There is **no
native pointer, published page view, sharing, COW, access lease, transfer,
serving integration or GPU mutable authority exposed** in this prototype.
It cannot implement or certify the remaining P1 contract by itself. Follow-up
below supersedes the original no-payload-API description only for synchronous
private CPU byte append/publication; there is still no native KV/table/access
or serving interface.

Three focused production contract/lifecycle tests pass through
`V2_Integration_PMAPagePoolUnpublishedContract`: one request exceeds equal share,
exact-fit growth/refusal, independent peer ownership, frame-generation reuse,
exact physical materialization/release, invalid/overflowing geometry, and
one-byte-short construction rollback. Same-PMA multi-class preparation now
rolls back earlier unpublished grants if a later class refuses, preserves every
original peer owner and never recycles issued pool incarnations. Latest evidence:
`CB-checkpoints/paged-p1-multiclass.6dWosx`. The initial two-test packet
`CB-checkpoints/paged-p1-private-grants-r2.hXnQRq` remains preserved.

The reviewer's bounded-pressure recommendation was recorded as **not owner
adoption** at that time; revision 5 now records owner adoption of the bounded
optimistic retain/restore policy with completion witness. Policy integration
remains an implementation obligation (P0 decision record + P5 proof), not a
completed one. The working [authority transition specification](paged_kv_authority_transitions_20261125.md)
spells out conservation debits/returns, the pinned nine-token shared-tail case,
and separate write/operation-readable/committed objects. Those additional
semantics are not yet implemented or independently accepted.

The complete PMA Unit registration was also run: **FAIL**, 139/140 tests in
`paged-p1-private-grants.Im75zz`, with
`PreparedNativeLookupFixture.OriginalSourceFreezeAdmitAndPayloadResetReplay`
throwing `Explicit test placement requires an online NUMA node`. The new
unpublished pool case passed there. A focused pass does not clear that gate.
This prototype is being checkpointed under implementation-first cadence, not
accepted as P1 or advertised as production capability.

Latest follow-up adds `AppendPlan` and `CommittedPage`: exclusive writes affect
only bytes after the previous valid extent; a committed byte publication pins
its exact frame/content version and exposes only completed bytes. Concurrent
pin/writer overlap refuses before mutation. Aborted writes do not advance the
visible extent or recycle their issued version. Cancellation during a write
revokes publication but retains the frame until the plan retires. These are
single-owner synchronous CPU byte contracts, not operation-readable KV views,
stream/event access leases or shared-prefix adoption.

Five contract/lifecycle tests and two replay/validation tests pass through the
two focused preflight registrations at `paged-p1-append-replay.s6gpWE`. The
additional recorded byte sequence was captured from the actual authority at
`paged-p1-append-recording.1bKRNr`; the original ownership trace is unchanged.
Complete P1, independent acceptance, full gates, policy and native codecs remain
open. Latest checkpoint evidence is `paged-p1-foundation-checkpoint.ZmLTUE`:
147 PMA Unit tests plus explicit aggregate-CPU, six-contract and two-replay
preflight registrations pass. Registered-reader recording is retained at
`paged-p1-reader-recording.U0jSsn`; all three small traces are embedded into the
binary rather than loading builder-local fixture paths.

## Native storage-family closure in progress

`kernels/kvcache/KVStorageLayout.*` now owns the canonical typed cache family,
codec parser and immutable native token/basis payload layout. Planning's
`KVCacheMemoryEstimator::logicalPayload` consumes that factory rather than
owning a second codec definition. Existing case/alias semantics and archive
bytes remain unchanged. `pagePayload` excludes the separate request-wide basis;
`archivePayload` includes its serialized copy. No page allocator may use an empty
frame as permission to select a new AQ8 basis.

The factory covers the existing FP32/FP16/BF16/Q8_1/Q16_1/TQ4/TQ8 payload rules
for declared CPU/CUDA/ROCm devices and actual attention/hybrid families. It
retains GPU hybrid linear Q8, anchored CPU/attention-only Q8/TQ, CPU-only Q16
head blocks and existing hybrid-TQ rejection. Layout validation is device-free;
it does not qualify native paged execution or expand any supported combination.

Source inspection establishes additional distinct owners still to transform:

- `CPURingKVCacheAnchored.inl:49–85`: each layer/head's first serial key is the
  request basis; pages/COW/import retain that original basis. Page-local mean
  or first-token initialization would change encoded keys.
- `QwenSparseAttention.h:112–198`: QFN's persistent pooled indexer keys and raw
  four-position partial tail are separate from native K/V codec banks and
  per-query selected-index/count work. Dense paging alone cannot qualify or
  eliminate those full-history owners. Its current addressing enum lacks a
  native page-view case; a full-history gather is not the replacement.
- `MTPShiftedCacheOrigin.h:25–65`: successor-pair families offset by depth+1;
  QFN's visible-zero bootstrap has offset zero at depth0 and rejects other
  depths. Page ordinal/absolute-position/archive identity must retain the
  model-authenticated convention, not apply a universal main-minus-one rule.

Complete KV-cache estimator Unit, the new explicit native-page geometry
preflight and the complete PMA Unit registration pass at
`CB-checkpoints/paged-native-layout.inQG9J`. This verifies payload arithmetic
and existing planning tests, not native append/attention or a final page layout
ABI. QFN auxiliary paging, provenance/trust, numerical boundary compatibility,
quantized COW and all-target native evidence remain open.

## First direct FP32 operand witness (in progress)

`kernels/cpu/attention/CPUFP32PagedKVView.h` now provides a private CPU operand
view over actual retained `CommittedPage` owners. Each token stores native local
K then V; logical page order is independent of frame order. The final page may
have a partial live extent; padding/invalid head/token coordinates are rejected.
Moved-from page wrappers revoke their raw pointer/extent. No full-history gather
or payload allocation occurs in the view.

A production-type contract fixture assigns frames in physical order, publishes
logical pages in order 2/0/1, then reads five native FP32 tokens across a partial
tail after original request/pool handles retire. It checks actual source address
identity, bytes, padding refusal, moved-from access and malformed partial-page
ordering. Complete PMA Unit (148 tests) and the explicit
`V2_Integration_CPUFP32PagedKVViewContract` pass at
`CB-checkpoints/paged-fp32-operands.OLqOKN`.

This is **operand geometry/ownership**, not attention execution, a stable
versioned table bank, producer-ordered query visibility, native append, or P2
acceptance. The borrowed table must remain immutable through the operation;
that ownership still needs the bounded table/version authority before serving.
The existing optimized `CPUFlashAttentionKernelT::compute_flash_fp32` now has
an explicit paged entry point and native key/value row accessors. Both canonical
summary and single-summary branches retain their original dot, softmax, value
accumulation and reduction arithmetic. Admission validates each real interleaved
page bank and actual query/output extents before a worker team; it never asserts
that all history occupies one contiguous bank. No scalar replacement or full
history gather was introduced.

Native tests compare actual paged output bytes with the contiguous production
control over page-edge lengths, partial tails, GQA, grouped queries and lengths
257/513 crossing summary boundaries. All 56 CPU attention Unit tests and the
explicit `V2_Integration_CPUFlashAttentionPagedFP32` witness pass at
`CB-checkpoints/paged-fp32-attention.BpcBIc` (local Integration/aarch64). Malformed
query extents and query/output aliasing refuse before writes. Tests populate
native pages as fixtures; they do not implement production paged append.

Recorded-input native kernel replay now retains actual Q/K/V input bits,
initial pool/frame order and observed output bits in `attention_fp32_v1.txt`.
Capture first verified byte equality with the contiguous kernel; normal replay
does not regenerate expected output. Complete CPU attention Unit (58 tests),
byte-equality sweep and explicit replay preflight pass at
`CB-checkpoints/paged-fp32-kernel-replay.jcOmZH`; native capture is retained at
`paged-fp32-kernel-recording.tOAwPi`. The fixed replay panel is not a cross-ISA
or all-format certificate.

CPU profiling was attempted separately from correctness: `perf stat` could not
start (exit 127, profiler executable absent in `cb-run`). The shared VM is not a
dedicated economy host. Profiling/economy are unqualified, not waived. Work
is a private implementation checkpoint; ISA/thread qualification, native append,
stable table/version ownership, quantized paged operands and serving integration
remain open. The address source is now compile-time specialized so existing
contiguous calls do not acquire a runtime page-dispatch branch in their inner
loops. Final local evidence `paged-fp32-kernel-final.HQGWHb` passes CPU attention
Unit58, PMA Unit148, operand geometry, native paged byte comparison and recorded
kernel replay (five registrations). No throughput or latency win is asserted. These results do
not qualify P1/P2 exits or the default inference path.

## Bounded CPU native table banks (in progress)

`CPUFP32PagedKVTable.*` now materializes exact PMA-owned native page-object
cells and bounded bank metadata through `CPUStorageRoot`. Attachments authenticate the original PMA instance before
consuming a page; foreign-instance refusal preserves the supplied page owner.
Unpublished updates
cannot expose views; commit validates complete page geometry before publishing
an immutable snapshot. Snapshot versions are monotonic across all banks,
including aborted prepares. Banks and their actual page referents remain owned
until the last bounded snapshot borrower retires; no update allocates metadata
or copies a retained KV history.

The table authority can close an exact version to new borrowers before joining
its frontier. Existing snapshots remain readable after closure; a released
publisher does not authorize metadata reuse while another original borrower
survives. Native attention has an explicit snapshot-retaining entry point so
its synchronous operation retains a stable bank and every page publication.
This is CPU actor ownership, not GPU access/event or topology publication.

Native lifecycle tests exercise old/new bank outputs (4, 9 and 7) through the
actual optimized attention, two-bank exhaustion, reader limits, closure after
registration, aborted-version nonreuse, final PMA release, and one-byte-short
metadata rollback for bank counts 1/3/5. CPU attention Unit60 and explicit table
lifecycle preflight pass at `CB-checkpoints/paged-fp32-table-closure.YPd5as`.
These are lifecycle/geometry tests, not recorded table-controller replay.

Independent acceptance and full qualification are pending. Empty-history lifecycle tables, operation-specific causal
views, sharing/COW/reference liabilities, request-generation binding, atomic
page/table multi-class prepare/commit, native append, transfer frontiers remain open. Recorded table-controller replay now covers the
bounded private subset below. No P1/P2 exit or serving capacity is
qualified by this bounded CPU subset.

## Private-frame return before native append (in progress)

`PhysicalPagePool::Grant::discardFrame` now retires an original private frame
and removes only its allocation-ownership record, not a request's logical KV
table. Unused growth is restored only after the actual final pin retires. This
converts occupied capacity back into the original request's unconverted promise
without a second physical allocation or a duplicate debit. Pending retirement
cannot be used as immediately assignable growth.

Each frame records its original request generation. If that request retires
and its record is reused before the last old pin completes, the old completion
returns the frame to free capacity rather than crediting the replacement's
promise. Frame/pool generations still authenticate physical reuse independently.
This is required for bounded private-tail replacement; shared immutable prefix
adoption/COW liability policy is not implemented by this operation.

Complete PMA Unit149, CPU attention Unit60, seven page contracts, existing
private replay and table lifecycle pass at
`CB-checkpoints/paged-private-frame-return.5eNoVO`. The focused new test checks
pending-pin refusal, occupied-to-growth conversion, private frame reuse and no
credit to a reused request record. Actual native append remains open; these
changes are an implementation checkpoint, not accepted serving support.

### Recorded table/controller inputs

`private_tables_v2.txt` retains native FP32 input bytes, pool/table geometry,
empty request/frame/bank incarnations, borrower pressure and observed decisions
from the actual PMA/table implementations. It covers two simultaneous versions,
closed-version late refusal after publisher retirement, actual original page
reads, aborted version3 followed by publication4, real frame retirement and
reuse, and no credit to a replacement after the original request record retires.
Capture evidence is `paged-table-controller-recording-r2.2EQ6hF`; it extends the
retained first capture `paged-table-controller-recording.orSXd3` with the
request-record reuse panel, not a changed expected result for an old input.
Ordinary replay uses embedded observations and never writes new expectations.
The grammar rejects incompatible initial bank state, missing terminal state and
altered version, borrower or growth-credit observations. This is recorded
controller/native-address replay, not a model inference trace or Release economy
qualification.

## Independent findings carried into P0/P1

Findings map to the revision-5 design and plan sections in the resolved
canonical documents; implementation obligations remain attached to the
milestones below. Explicit PKV-01–09 section mappings are owned by the
canonical plan’s ownership-map design-reconciliation table (`plan_continuous_batching_implementation_20261125.md` §4);
the table below states obligations, not a duplicate mapping.

| Finding | Required resolution |
|---|---|
| PKV-01 | Owner-selected in revision 5: bounded optimistic retain/restore with all-resource completion witness (design §§1, 5; plan P0 decision record; P5 proof incl. tiers off/full) |
| PKV-02 | Per-participant/class conservation: unique occupied (including retiring) + unconverted private growth + unoccupied concurrent transients ≤ pool; explicit seal/share/COW/tentative/commit/cancel/real-retirement debits and returns; issuance exhaustion before admission |
| PKV-03 | Distinct exclusive write plan, producer-ordered operation-readable view with per-query visibility, and atomic committed publication; registered-but-not-enqueued readers close atomically; same-frame reject/reappend waits for real borrowers |
| PKV-04 | Early native CUDA/HIP live-shape/receipt/catalog feasibility witnesses before freezing ABI; late-known work inside complete transactions must launch economically |
| PKV-05 | Atomic direct admission versus Q-credit versus refusal; never-fit versus temporary pressure; bounded bypass/drain/fairness and external response incarnation |
| PKV-06 | Request-origin AQ8 basis, asymmetric V codecs, QFN compressed history/raw tail/selection tables, model-owned shifted origins, trustworthy compatible archives and request-neutral state sections |
| PKV-07 | Effective adoption receipt ordering at retention owner; no-copy hits update valid backing incarnations before eviction; stale/reordered receipts cannot touch replacements |
| PKV-08 | Common usable prefix/checkpoint boundary, participant set, prepare/vote/commit identity and pre/post-publication failure protocol, including expert-only participants and PP overlap |
| PKV-09 | Intake/parse capacity before copying and response lease through the last HTTP borrower, including final blocked write after producer exit; actual simultaneous transient/framing/tracing/non-streaming bytes |

R2 correction: the pinned implementation infers matched blocks and RAM tier;
it is not simply a constant-zero report. Exact lookup/selected boundary/tier
observations still require proof and cannot be inferred from restored tokens.

## Remaining P0 closure work

Revision-5 plan P0 (“Reconcile inventory and freeze explicit contracts”) owns
this closure; the list below is implementation-owned and keeps the same
obligations:

1. Resolve every census candidate to one of the owners above, including indirect
   template/device kernels, QFN sparse attention, native collective buffers and
   checkpoint/transfer closures; investigate gaps rather than whitelist them.
2. Freeze native page-format factories for each supported cache role/codec and
   alignment, including request-anchor semantics and TurboQuant metadata.
3. Specify bounded table banks and access/retirement capacity from actual
   concurrency, plus the shared numerical/window policy version at cutover.
4. Reconcile canonical config inputs (GPU cap, RAM off/auto/cap, disk
   capacity/directory, UMA domain identity) and per-participant pool BOM,
   including metadata and complete response state. No numeric default or
   installed CLI spelling is invented here; revision-5 config contract §4.3 is
   authoritative.
5. Implement P1 through the actual PMA factory and add production contract and
   recorded-input replay tests before claiming page support. P0/P1 remain open.
