# QFN consumer-minimal KV coverage, one through eight participants

Updated: 2026-10-22. This source-derived design extends the [TP4 ownership audit](QFN_TP4_COMPUTE_OWNERSHIP_AUDIT.md). The implementation propagates the same original-source declaration through automatic planning, admission, weights and graphs. Focused planning, captured projection, full Unit, frozen model accuracy, prefix restoration and matched Release throughput pass; complete current-source production preflight remains pending. The geometry and captured-scaling foundation is checkpointed at 11873a1a8, and original cache/snapshot consumers at 0552f5a34. The rejected correction 88 experiment does not change this plan.

**Conclusion:** retain the existing `TensorParallelConfig` assignment as the sole geometry authority and extend it to *consumer-proven overlapping KV intervals*. Q24/KV2/TP4 needs `[0,1),[0,1),[1,2),[1,2)` for Q intervals `[0,6),[6,12),[12,18),[18,24)`. Original source slicing and native sparse attention already support these origins. There is no existing general overlap-policy switch that makes this complete: cache origin reconstruction, partial-replica snapshots, and original-N arithmetic binding are genuine closure gaps. The minimum coherent candidate therefore crosses those existing owners; it is not a two-line splitter change.

The supported design range is **every participant count from one through
eight**, including three, five, six and seven. The count is an input to the
existing query splitter, never a dispatch table for KV placement. For QFN's
Q24/KV2 geometry and original GDN key16/value48 consumers, the canonical
aligned splitter gives the following expected original KV intervals (exclusive ends):

| Participants | Query counts, in assignment order | KV intervals, in the same order | Physical head copies across participants |
|---|---|---|---:|
| 1 | 24 | `[0,2)` | 2 |
| 2 | 12, 12 | `[0,1)`, `[1,2)` | 2 |
| 3 | 9, 9, 6 | `[0,1)`, `[0,2)`, `[1,2)` | 4 |
| 4 | 6, 6, 6, 6 | `[0,1)` twice, `[1,2)` twice | 4 |
| 5 | 6, 6, 6, 3, 3 | `[0,1)` twice, `[1,2)` three times | 5 |
| 6 | 6, 6, 3, 3, 3, 3 | `[0,1)` twice, `[1,2)` four times | 6 |
| 7 | 6, 3, 3, 3, 3, 3, 3 | `[0,1)` three times, `[1,2)` four times | 7 |
| 8 | 3, 3, 3, 3, 3, 3, 3, 3 | `[0,1)` four times, `[1,2)` four times | 8 |

These are test expectations, not production policy constants. With only two
KV heads, eight consumers cannot each own a distinct whole head. A head remains
present on each rank whose queries consume it; avoiding that necessary sharing
would require a different communication/attention algorithm. The intended win
is removing heads a rank never consumes. No new KV collective is introduced.
Unequal shares and reordered device IDs must derive the same minimal coverage
from their actual query origins. A query interval crossing a GQA group boundary
legitimately retains both neighboring heads.

Device-free geometry, snapshot and BOM tests cover all eight counts, including
MQA, GQA and MHA. Backend kernel tests can exercise every resulting local slice
on the available devices; those are distinct from a real eight-device topology
run. Record the actual native participant count with each integration result.
Existing invalid configurations (nonintegral GQA or participants with no Q,
FFN or vocabulary assignment) must fail precisely; do not add empty compute
owners or silently select full KV replication to admit them.

Head-layout support does not establish support for every complete topology.
The current `GateUpOwnedDownColumns` expert policy requires equal hidden-width
slices. QFN's H2560 cannot be divided equally at degrees 3, 6 or 7, so automatic
planning must retain that independent routed-policy rejection. The head/GDN
assignment itself is valid at those counts. Uneven expert down-projection
ownership and its collective contract are separate work; do not silently select
another expert policy or pad the hidden dimension to publish those candidates.

**Recurrent-head alignment is part of the same assignment.**
The model's GDN path has 16 key heads and 48 value heads. Its current
`GDNHeadAssignment` requires integral original head boundaries, so an ordinary
Q interval must start/end at multiples of three. An unconstrained equal split
would cut those heads at degrees 3, 5, 6 and 7. The new immutable
`QueryHeadPartitionDeclaration` derives the minimum granularity from every
retained consumer (`Q / gcd(Q, key_heads)`, combined with other required
head-boundary constraints), and passes it through the existing canonical
proportional splitter. There is no second query assignment in the graph,
planner or KV resolver. TP1/2/4/8 stay unchanged. Regressions cover both GDN
boundary validity and minimal KV coverage. Passing the same declaration through
every production construction owner remains required before promotion; local
geometry tests alone do not establish complete model admission at all counts.

## 1. One typed declaration, one assignment

Add one closed KV coverage policy to the existing declarative schema/assignment path: current full-bank behavior versus coverage of an original global Q interval. Put the small enum in an independent config header (avoiding a GraphSchema/TensorParallelConfig include cycle), expose it through `WeightShardingConfig` in `execution/local_execution/graph/GraphSchema.h`, and declare native sparse support in `models/qwen4exp/Qwen4ExpSchema.h`. Keep existing `ColumnParallel + KVHeads` weight patterns; no new sharding engine or ownership ledger.

Extend `TensorParallelConfig::{equalSplit,proportionalSplit,fromLocalTPContext}` with that typed declaration, defaulting existing callers to their current contract. Factories must retain the authenticated original global Q/KV bounds and policy inside this immutable configuration; a raw-assignment constructor cannot infer original KV total from a union maximum and thereby hide a missing head. Existing raw-constructor behavior remains unchanged unless that typed source declaration is supplied. A shared pure resolver computes `r=globalQ/globalKV`, `first=qStart/r`, `end=(qEnd-1)/r+1`, with checked bounds/divisibility and a nonempty query interval. Keep Q/FFN/vocabulary partitioning unchanged. Crossing a GQA boundary legitimately requires two heads; do not hardcode rank pairs.

`computeTotals()` and `validationError()` (`TensorParallelConfig.cpp:96–127,172–205`) currently special-case identical full replicas. For the explicit coverage policy, require every authored KV interval to equal its Q-derived cover and validate the ordered unique union is exactly `[0,globalKV)`. Count original heads once; independently price every real copy. Reject holes, foreign origins, integer overflow and unjustified overlaps. A generic interval-overlap allowance would be insufficient. Retain old full-replica/disjoint semantics for existing policy values.

Resolve the same immutable schema/consumer declaration at **every real construction owner**, passing it to the canonical splitter:

- `RankOrchestrator::initializeDeviceRunners` (`:1313`): WeightManagerConfig source assignments.
- `InferenceRunnerFactory` local/global assignment helpers (`:2510,:2581,:2731`): GraphConfig.tp_config and graph dimensions.
- `RankMemoryPlanInputs::bindRankLocalTPAssignment` (`:310`) and `MoEOverlayLocalCapacityPlanner` counterpart (`:771`): exact DevicePlanConfig binding before PMA admission.
- `TPPPValidator::validateRankLocalTPAssignment` (`:194,:208`): admitted topology and named-domain checks.

`AutomaticOrchestrationCandidates::preservesGDNHeadOwnership` (`:295`) only consumes unchanged Q intervals; it needs no different GDN formula. Ensure any policy propagation there uses the same resolver if it becomes a complete-assignment validator. No model-shape constants or per-device-index reconstruction.

## 2. Reuse the existing bytes, cache and consumer authorities

| Existing owner | Concrete change / invariant |
|---|---|
| `planning/WeightShardGeometry.cpp:111` | Reuse `logicalShardInterval(KVHeads, assignment)` and normal original-source slicing. Gate Q/K/V source identity, scale blocks, head width and prepared lifetime exactly as today. No dummy heads or repacking aliases. |
| `planning/MemoryPlanner.h:307–328`; `MemoryPlanner.cpp:889–925,1684`; `ActivationMemoryEstimator.cpp:278`; `WorkspaceMemoryEstimator.cpp:489` | Reuse `DevicePlanConfig.tensor_parallel_assignment` for resident weights, local KV count, projection/activation/workspace widths and source origin. Existing BOM contributions flow through PMA. Do not subtract a separately computed “KV saving.” |
| `DeviceGraphOrchestrator::initializeKVCaches`, main `:16532–16548`, shifted `:8107` | Replace participant*localCount reconstruction with the checked assignment interval. Validate assignment participant/device/global geometry against GraphConfig before constructing caches. Full-model sidecars remain origin 0/full-count; primary-shard sidecars use the same authoritative interval. No MTP capacity is introduced when absent. |
| `QwenGraphBase::addKVCacheAndAttention:4923–4926`; `Qwen4ExpSparseAttentionBuilder.cpp:174` | Existing native consumer binds Q origin and actual cache KV origin/count. Keep its coverage assertion. CPU `CPUQwenSparseAttention.cpp:406` and CUDA/HIP `QwenSparseGPU.cuh:608` already map global Q group minus original KV origin. No attention kernel algorithm change is needed for the selected sparse main consumer. |
| `prefix_cache/PrefixPayloadLayout.cpp:185,198,241` | Already compares/stores original main/shifted KV origins and local count from `logicalBlockLayout`. Preserve this authority: different original heads must not share a prefix identity. Codec logical block sizes and snapshots must shrink from actual local geometry. |

**Scope must follow actual retained consumers.** Generic `AttentionHeadMapping.h:21–47` has no visible-KV-bank origin; `QwenGraphBase.cpp:5058` assumes either full replicas or locally aligned groups. QFN's dense predictor explicitly returns to that generic consumer (`Qwen4ExpGraph.cpp:475`), while its native sparse main path is origin-aware. Phase-split replicated decode deliberately needs a full main cache (`initializeKVCaches:16428`) and its existing handoff cannot be reinterpreted as duplicate-subset concatenation. Preserve these existing typed full-bank demands; do not silently change them by a schema-wide count rule. The first private experiment can explicitly cover ordinary TensorParallel/GateUpOwnedDownColumns with no retained MTP, leaving other declarations intact. General promotion must either prove a consumer-specific resolved policy through every retained graph family or finish the missing generic origin contract. No fallback, hidden gather or weakened validation.

## 3. Two mandatory interface closures

**Snapshots:** `RankOrchestrator::applyRuntimeSnapshotShardingOverrides:954–987` forces ten K/V roles to REPLICATED whenever KV<TP. `getTPSnapshot:18590` also derives COLUMN_PARALLEL origin by concatenation. Both are wrong for subset replicas. Reuse the existing `PACKED_COLUMN_PARALLEL`/`SnapshotColumnGroup` mechanism in `execution/debug/TPSnapshot.h:353–375`, with a small explicit *replicated participant subset* group mode. Declare one group per original KV head: widths `[256,256,0,0]` and `[0,0,256,256]`. Every group must have an actual owner; nonowners consume zero local columns; emit one original group and validate every actual replica, preserving current diagnostic tolerances. Existing REPLICATED remains “all participants” and PARTITIONED remains disjoint. Derive groups from the assignment and real stage dimensions, not rank pairing or architecture string logic. Apply to projection/norm/RoPE/append/cache/effective K/V families; reject missing owners, shape/row mismatches and divergent replicas. Selecting one representative and ignoring the other is not a valid diagnostic repair.

**Arithmetic:** QFN uses `Qwen35Graph::buildFAAttentionGraph:1939` -> `FusedQKVGEMMStage::execute:328` -> shared-quantization `multiply_fused_tensor`. Reducing K/V physical N512 to256 can otherwise select another N-dependent numerical policy. Existing `OutputPartitionEquivalenceScope` in `tensors/TensorKernels.h:548–602` is directionally restricted: CPU NativeVNNI (`CPUNativeVNNIGemmKernel.h:238`), CUDA (`CUDAQuantisedGemmKernel.cpp:1698`) and ROCm (`ROCmQuantisedGemmKernel.cpp:1354`) explicitly reject policyN>physicalN. Therefore it cannot currently pin old N512 for an N256 slice. Fused stage Params/TensorProjectionDesc also carry no original-N arithmetic identity.

Use a narrow **typed original-output-slice overload** of the existing equivalence contract, authenticated from the frozen original source/view (originalN, first original output, physicalN, K). Keep the old two-int replicated-output validation unchanged. Carry this optional immutable fact per K/V projection through FusedQKVGEMMStage and TensorProjectionDesc; Q stays unchanged. Each backend's existing fused per-projection launch must bind the fact locally to that projection and restore it before the next leaf—an outer Q/K/V thread-local N override would corrupt Q's policy. Preserve original prepared/source family, M-dependent ordinary versus verifier choice, fixed partitions/tree and shared input quantization. This is arithmetic-identity plumbing, not a new dispatch policy. If existing backend fusion cannot honor it economically, stop and report that blocker; do not disable fusion or select another arithmetic. No claim of unchanged arithmetic follows merely from ROCm M1's fixed KB16.

Source review found a separate floating-prefill boundary: CPU ordinary floating
bundles use oneDNN, and ordinary CUDA/ROCm FP32 bundles use BLAS for applicable
shapes. Their existing floating equivalence scope can select fixed-column
verifier arithmetic for the entire bundle. Reusing it would change unselected Q
as well as K/V, so it cannot prove this optimization. The first arithmetic
implementation covers original native quantized sources, including the Q8_0
K/V projections in the current GGUF. An unimplemented floating original-slice
request must fail before enqueue; existing full-bank floating execution stays
as declared. Exact ordinary floating-source slicing remains an explicit open
implementation/qualification task, with no automatic arithmetic substitution.

## 4. Focused gates and decision boundary

Extend existing targets, with explicit new ProductionTestPreflight filters:

- `Test__TensorParallelConfig.cpp`, `_LocalTP.cpp`, `Test__TPPPValidator.cpp`: every degree 1–8 for Q24/KV2, integral multi-group GQA, MHA and one-bank MQA; independently enumerate each query's original KV head and prove exact minimal coverage, union totals and every physical copy. Include reordered/asymmetric/straddling intervals, invalid holes/overlaps/origins, participants exceeding the available Q heads, and unchanged default and dense/GDN declarations.
- `Test__MemoryPlanner.cpp`: weight/projection/workspace/KV/prefix/snapshot BOM matches exact admitted assignment; each physical replica charged; no new collective bank or MTP storage.
- `Test__RankOrchestrator.cpp` packed snapshot tests: at every degree 1–8, subset replicas reconstruct original head order, verify all copies, reject missing/mismatched/foreign groups; old replica/GDN modes unchanged. Include a head whose first owner is not participant zero and a participant consuming two heads.
- `Test__FusedQKVGEMMStage.cpp` plus existing CPU/CUDA/ROCm partition/fused suites: K/V physical256 with authenticated policy512, Q unchanged, scope restored between projections, ordinary/grouped paths and rejection before enqueue for foreign source/width. Preserve the old invalid N+1 two-int tests.
- Existing CPU Qwen sparse primitive and CUDA/ROCm `QwenSparseAttention` targets: compare each local original-head projection/cache/attention bytes to full-source slices, head1/nonzero origins, M1/grouped/chunked, cache codecs, prefix/reset and 20 large-to-small captures. Observe actual N256 and exact live append/storage extents, not only final tokens. Existing preflight families are at CMake:28759–28833.

Only after those gates: canonical 185-row panel/exact accepted-source logits, then the existing matched Release 512/256 ABBA with unchanged arithmetic/runtime/NUMA/ngram and explicit vocabulary sharding. Source layout estimates remain those in the prior audit; no latency or end-to-end gain is established. The three closure gaps above are the real minimum scope, not a reason to broaden PMA/admission architecture.

## Implemented foundation and validation

The explicit coverage declaration, original-consumer query granularity and
subset-replica snapshot assembler are implemented. Every degree 1–8 is covered,
with MQA/GQA/MHA geometry, uneven/reordered membership, exact minimum head
coverage, invalid source intervals and GDN boundary checks. This foundation
initially left production selection unchanged; the propagation follow-up below
now selects the qualified source family in the working candidate.

Focused geometry/ownership and CPU compact-cache tests passed in
`kv-coverage94-focused-r1`. CUDA and ROCm compact-cache tests passed in
`kv-coverage94-native-r3`: ten codec/block configurations, every local slice at
degrees 1–8, and twenty retained captured histories per slice. They prove exact
full-bank output bytes, compact native storage/strides, nonzero head/ring
origins, active read counts and inactive output guards. The initial GPU test
incorrectly required inactive diagnostic counters to retain their old bytes;
the existing GPU reducer deliberately zeros those counters. The corrected test
asserts that established contract and changes no kernel. These are same-device
consumer tests, not eight-physical-device model runs.

The cache/snapshot consumer follow-up now takes original KV origins and counts
from the retained assignment for both main and shifted caches. Explicit
full-model demands remain separate. Rank snapshots derive original-head groups
from each actual producer, verify all replicas, and reject foreign devices,
missing producers, mismatched widths and stale source layers. Absolute PP layer
ownership and historical prefill/condition scopes are checked before selecting
per-layer geometry; a complete-value publication still supersedes partials.
Prefix identity continues to use the actual cache geometry.

The focused consumer and neighboring TP/rank gate passes **19/19** in
`kv-cache95-focused-r1`; complete current-source Unit passes **771/771** in
`kv-cache95-unit-r1` (116.07 seconds). Both have clean driver and retirement
evidence. These device-free tests exercise real cache-geometry inputs and Rank
snapshot assembly, not native model admission. The initial build found two
GTest trace macros on one line; splitting them fixes the test compilation,
without changing runtime code. The two new consumer regressions are explicitly
registered in `ProductionTestPreflight`; complete current-source preflight is
still pending.

Automatic candidate generation, the validator, PMA inputs, weight slicing and
graph construction now consume the shared original declaration and canonical
assignment. The three explicit preflight aliases pass in
`query-kv95-propagation-focused-r1`, covering source/profile agreement,
physical-device identity, actual automatic candidates and admission inputs.
Head-layout checks cover logical degrees 1–8; the independent expert-column
restriction remains explicit. Complete Unit passes **773/773**, zero skips, in
`query-kv95-unit-r1` (117.67 seconds including the wrapper). The existing cost
model prices the actual assignments; no independent KV saving is subtracted.

Original K/V arithmetic facts now retain the exact prepared kernel and source
owners, so store replacement cannot invalidate or rebind an already prepared
stage. The native fixture exposed missing GPU `get_n/get_k` overrides; both
backends now expose the checked immutable prepared dimensions. The all-format
captured slice gate passes on CUDA and ROCm in `original-slice95-native-r5`:
21 source formats, ordinary/grouped M1/3/17, twenty changed/poisoned histories,
exact full-versus-slice bytes, unchanged Q, guard extents and physical-width
workspace declarations. Seven neighboring GEMM regressions pass in
`original-slice95-native-neighbors-r1`. Initial fixture failures also corrected
its capture dependency declaration and per-bundle workspace merge; they did
not require a runtime allocator or synchronization change.

Ordinary M512 and passive geometry evidence now pass together in
`query-kv95-evidence-bulk-focused-r1`: **9/9**, zero skips, clean driver and
retirement. Both GPU backends retain the original small-row histories and add
three M512 histories across all 21 source formats. The 84 captured bulk graphs
prove three native prefill producers each, exact full-versus-slice bytes and
physical-width workspace. All 252 observed producers report zero local-memory
bytes. Passive PerfStats metadata joins original prepared source intervals,
selected admission inputs and constructed cache geometry; it deliberately does
not claim per-instance leaf replay or replace native workspace tests.

The first Release model attempt, `query-kv95-release-accuracy-r1`, failed during
parallel participant setup, before inference. The driver window was clean. It
exposed a real TP assignment lifetime race: a factory borrowed the shared
WeightManager's raw configuration pointer while another participant replaced
its owner. Both factory readers now retain immutable shared snapshots, and all
WeightManager TP readers use atomically published owners. WeightSlicer also
retains a const owner. This fixes equal-assignment parallel setup without
serializing builders; it does not promise arbitrary topology reconfiguration
during nested weight materialization.

The explicit `WeightManagerTensorParallelSnapshotOwnership` preflight alias
and its existing Unit owner pass **2/2** in `query-kv95-ownership-focused-r1`.
The regression forces replacement through both APIs, checks final-owner
retirement, and validates 8,448 observations across four readers. Complete
current-source Unit passes **774/774**, zero skips, in
`query-kv95-ownership-unit-r1` (117.26 seconds including the wrapper).
The corrected Release build passes in `query-kv95-release-build-r2`. Its fresh
canonical required-group panel passes in `query-kv95-release-accuracy-r2`, with
all **26 files / 185 retained logit rows byte-identical** to accepted Four90,
including reset replay. The same observation validates selected admission,
all 24 original K/V source roles and compact cache geometry on all four GPUs.
Parent replay is observed; this cold metadata does not claim per-leaf replay
counts or native workspace measurements.

The exact registered TP4 parity cell also passes in `query-kv95-prefix-r1`
(91.07 seconds including staging/retirement), preserving all eight canonical
CSV artifacts, full and partial prefix restoration, a clean driver window and
normal native retirement.

### Matched Release throughput (L95)

The accepted ABBA cohort is `query-kv95-model-a1`,
`query-kv95-model-b1-repeat1`, `query-kv95-model-b2`, and
`query-kv95-model-a2`; `query-kv95-model-comparison.json` retains every sample
and validates all twelve natural 256-token sequences as identical. The six
samples per role have medians **786.251 / 36.746 tok/s** for the historical
Four90 control and **791.519 / 36.759 tok/s** for QueryInterval KV95
(prefill/decode): **+0.670% prefill / +0.034% decode**. Decode ranges overlap;
this is effectively unchanged decode, not a demonstrated decode speedup.

Both roles use explicit Four-worker owner execution and vocabulary sharding,
the same HIP76/RCCL80 runtime, four MI50s, 28 physical NUMA1 host workers,
RAM ngrams, context4096/M512, prompt512/output256, seed42 greedy/top-k1/top-p1,
MTP off with zero reserved capacity, one warmup and three measured repetitions
per arm. The frozen six arithmetic selectors are unchanged. This compares
different source/layout revisions, not different compile modes of one source.
Exact loaded-library equality and the complete cold dependency inventory are
checked separately; no model or cache payload checksums are introduced.

The first `query-kv95-model-b1` remains rejected: its native process completed
normally and its tokens matched, but the driver window contained an AMD
`amdgpu_amdkfd_restore_userptr_worker` workqueue CPU-hog warning during teardown.
The original failure and `attempt-disposition.json` are retained. Its exact
repeat passed without changing source, runtime, settings or driver policy;
the rejected timings are excluded and explicitly disclosed here. All four
accepted arms have clean driver windows and normal retirement. These are
descriptive paired results, not a statistical confidence claim. Complete
current-source preflight remains pending. The subsequent
[Automatic Four slice](QFN_ROUTE_OWNED_MOE_TUNING.md#automatic-four-worker-policy-qualification)
now passes its own model accuracy and matched-throughput qualification.
