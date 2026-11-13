# Early current-invocation lookup: concrete capture seam audit

Source-only against `/tmp/qfn-pooled-ngram-perf-r1/source`; no implementation,
build or native execution. This narrows [the all-tier prefetch plan](QFN_NGRAM_PREFETCH_PLAN.md) into
an implementation boundary. This is a detailed design note, not a dashboard.

## Scheduling answer

Removing the trunk edge alone is insufficient. `ComputeGraph::getExecutionOrder`
(`execution/local_execution/graph/ComputeGraph.cpp:255`) uses Kahn ordering with
initial ready roots discovered through an unordered_map. Independent roots have
no guaranteed relative execution order. `DeviceGraphExecutor_GraphCapture.cpp`
assigns every stage the same exact `gpu_stream` before capture (around 1774),
and replay preparation binds the segment capture stream again (around 2949).
A stage-bound stream set in a model builder will therefore be overwritten.
The ordinary executor's PARALLEL mode also executes sequentially; that is not
an alternative production mechanism for retained immutable-storage graphs.

**RAM/mmap can overlap without an extra GPU stream:** put hash and Publish
explicitly before the embedding/trunk root, and defer Consume until PLE. Once
Publish's system-release kernel runs, the existing host source worker can read
while the same GPU stream executes embedding/layers. This is real CPU-source
I/O versus GPU-compute overlap despite a single GPU stream. Keep the exact
existing trunk node order, reductions and collective edges after the new prefix.

**Resident VRAM gather cannot overlap trunk on this executor by merely removing
an edge:** all launches remain serialized on its one capture stream. Moving
complete gather before embedding is an exact-ordering experiment, not a proven
GPU overlap optimization. Actual gather/trunk concurrency needs executor-owned
persistent capture-side stream fork/join support, including preparation,
recording, replay identity, cross-stream arena/publication events and retained
parent membership. That is a larger separate slice. Do not advertise a resident
VRAM prefetch gain from the first RAM/mmap slice or introduce a nominal mode
that still runs the old placement. Preserve resident's current first-class path
until that complete implementation is justified and measured.

## Existing authorities and publication boundaries

`ForwardExecutionEngine::execute` establishes a `ScopedForwardLiveStateRead`
lease. The actual prelude calls `prepareLiveStateForForwardGraphExecution`
and `prepareDeviceTokenInputsForForwardGraphExecution` around lines 3598/3612
(and the corresponding native composition path around 6154/6168). The scoped
owner retires the mailbox read on the actual producer stream after execution,
including failed preludes. Materialization uses
`prepareGraphBuildStateForMaterialization` (~3136), which authenticates immutable
addresses/readiness, not a live request value. Early stages must stay inside
this same admitted execution envelope; graph construction must not read IDs,
counts or tokens from device to determine readiness.

`DeviceGraphOrchestratorPLE.cpp::fixedHeadPLEQuery` already authenticates the
actual source roles, native addresses and native row descriptors for fixed-head
cohorts. It is useful as the narrow source-authentication precedent, but its
cohort capability cannot be reused as an ordinary local-source authorization.
Factor shared producer-authentication facts into a generic native query token
only if required, retaining cohort checks in fixed-head construction.

| Invocation | Native producer inputs and valid early frontier |
|---|---|
| Ordinary prefill | Common request tokens `request_token_ids_dev_` plus `request_sequence_lengths_dev_`, or chunk tokens `prefill_chunk_token_ids_dev_` plus `prefill_chunk_real_rows_dev_`. Existing `device_prefill_chunk` authenticates actual request/progress and bucket. Initial ingress and each chunk producer must execute before the entire early prefix. Bucket capacity is not a live count. |
| Ordinary decode | `device_resident_logical_sequence_state_storage_.next_condition_tokens_device` (or authenticated scalar request ingress), with exactly one row; no sequence-length pointer accepted as a query count. `DeviceGraphOrchestratorGeneration::materializeOrdinaryGenerationForward` binds that address and the parent stream; actual frontier publication/sampling is in the retained generation transaction before the captured forward. The materialized address alone is not a sampled-token witness. |
| Grouped MTP verifier | `mtp_verifier_input_tokens_dev_` and `mtp_verifier_request_lengths_dev_`, produced by the current device token-plan. `appendVerifierPLEPrior` snapshots accepted token/convolution state and transaction live count into the dedicated verifier prior bank. Hash may run only after both token-plan and SnapshotPrior. SnapshotPrior currently depends on the incoming PLE trunk; moving it to input readiness requires changing this ownership edge deliberately. |
| MTP condition | `logical.next_condition_tokens_device`, exactly one query row. `target_cached_tokens_device` is sequence length/position, not the number of query rows. The fixed-head authority already distinguishes this case. |

Ordinary PLE uses `history.acceptedTokens`, while verifier uses `prior.tokens()`
from `Qwen4ExpAuxiliaryState::appendVerifierPLEPrior`.
`appendAcceptedPLECommit` computes next token/convolution banks and a PLEPromotion
node after PLE, publishing the accepted banks. In the complete sequential
captured transaction, previous publication must precede the next invocation's
input prefix. Early ordinary hash may borrow the accepted token bank only
through the existing invocation read lifetime; it must retire before any
promotion overwrites it. A split/pipelined parent cannot assume that ordering
from a pointer. MTP prior banks must remain immutable through both early hash
and retained verifier consumption, and release through the existing move-only
`PLEVerifierPrior` capability; do not manufacture another raw prior descriptor.

Important limit: source review identified existing input authority and parent
ordering APIs, not a reusable generic graph-node producer capability for every
role. Before enabling each role, prove its actual parent encloses *all* graph
roots with token/count publication. Start ordinary local single-participant;
MTP and PP require their own full parent proof. A graph prefix detached from
that envelope is invalid even when an embedding dependency used to hide it.

## Minimal typed API and wiring

1. Add a generic model hook in `models/qwen/QwenGraphBase.h/.cpp` for invocation
   input-prefix construction, called after authenticating ForwardInput and
   before adding embedding/external-hidden trunk roots. Default models return
   an empty/no-op **typed result**, not an execution fallback. The QFN override
   returns the actual terminal prefix dependency and a move-only lookup frame
   capability owned by this graph build. This is graph-local state, never a
   mutable member saved between graph builds or a host mirror of request state.

   The same hook must be called in the standard forward, hybrid path and PP
   construction paths (`QwenGraphBase.cpp` current layer loops ~2242,2587,2936).
   Scope first executable implementation to the ordinary local family only;
   unsupported requested early topologies must fail precisely rather than
   emitting a placeholder. Preserve existing supported execution paths until
   replacement is complete, and retire old combined ordinary wiring when the
   supported replacement is installed.

2. Give QFN's prefix builder the original token binding, exact `Qwen4ExpStageRows`,
   accepted/prior token-history capability, original arena IDs and retained
   source boundary. Append NgramHash -> source Publish. Return a typed frame
   bound to graph/device/row identity/table/source/version and its producer
   terminal. Attach the existing embedding node (or actual PP trunk root) to
   that terminal. Do not rely on root insertion order. An external producer
   readiness token should project the existing engine/parent ownership; adding
   a string naming a fictitious token-ready graph node is not sufficient.

3. Refactor `Qwen4ExpPLEBuilder::append` so lexical input is either a consumed
   typed early frame or a complete resident lookup produced by its real native
   path. Separate its `trunkReady` dependencies from the early Published frame. On
   the current single capture stream, **Consume must depend on trunkReady as
   well as Publish**, so its GPU wait cannot be scheduled ahead of trunk and
   destroy the intended overlap. Query norm depends on trunk; key/value depend
   on Consume; gate joins both. A future real side-stream implementation may
   make ready output independent of trunk, but this first slice cannot. Keep
   deterministic projection/reduction order and existing convolution/commit
   chain. The move-only frame is consumed exactly once; reject another graph,
   wrong rows/device, stale workspace generation, duplicate consume or a frame
   missing the retained storage boundary.

4. Split ordinary `ImmutableLookupStage`'s existing GPUStorage work into scoped
   Publish and Consume operations, or two stage types sharing one sealed frame
   constructor. Avoid a bool soup. Today execute at ~207 does publish then
   consume; these kernel APIs already exist. Publish reads IDs and live count,
   writes protocol device state and mapped request. Consume reads matching
   protocol/completion, writes native scratch and embedding, then preserves
   existing acknowledgement/sequence retirement. A typed builder supplies the
   Publish and trunk-ready dependencies automatically; callers cannot invoke
   Consume independently or place its wait before the overlap interval.

   `bufferContract()` (~230), arena root preparation and capture variant
   signature must be operation-specific: Publish must not claim to produce
   embedding/native scratch early; Consume must not overwrite/claim original
   IDs. Device state is InOut across both. Include operation, frame identity,
   boundary capture identity, all scalar geometry and workspace generations in
   capture identity. Stage preparation must preserve TransferEngine and exact
   stream/event publication. No hot-path allocation, channel rebinding or second
   outstanding sequence. Retain source boundary on the entire resulting graph.

5. Preserve interval lifetimes. The original arena ID bank is produced once,
   read by Publish and unchanged until that read retires; response remains
   pinned until Consume/ack and native scratch until decode. Moving lookup
   output earlier extends its liveness across all trunk work. Audit BufferArena
   contracts and aliases for `QFN_NGRAM_ROW_IDS`, protocol state/native scratch,
   and `QFN_PLE_EMBEDDING` before moving any resident gather. For RAM Publish,
   the earliest actual embedding write stays at Consume, so do not extend an
   output lifetime unnecessarily. Stage identity must name every retained bank.

No generic scheduler rewrite or side stream is needed for first RAM/mmap source
overlap. Resident VRAM concurrency is a separate, complete infrastructure task;
changing all stage scheduling to solve this narrow I/O opportunity is unnecessary.

## Required tests and current blockers

Device-free focused replay should prove graph edges explicitly order hash ->
Publish -> trunk root, Consume after matching Publish AND trunk terminal
(and before key/value),
query norm after trunk, unchanged trunk/reduction order, operation-specific
buffer contracts and capture identities, exact-once frame consumption, foreign
frame rejection, no duplicate generation/sequence, failure/cancel/reset and
large-to-small repeated replay. Register focused controller/graph replay in
ProductionTestPreflight. Test adversarially reordered insertion of independent
nodes; correct scheduling must not depend on unordered_map iteration.

CUDA/ROCm integration must capture the actual complete production parent with
real worker/channel, exact output/inactive-tail byte checks and terminal
retirement; test ordinary decode, prefill partial chunks and reset replay. Use
an existing worker read barrier/delay fixture for causal evidence: worker read
starts after Publish, independent trunk GPU work can retire while source is
blocked, and Consume waits without allowing key/value before completion. Keep
that fault-control fixture out of performance timing. Then run unprofiled
Release whole-request timings and a separate native timeline proving overlap,
CPU duty and actual source bytes. Broad model parity/prefix/HTTP gates remain
required before advertising production certification.

Blockers to implementation are concrete: (a) no reusable ordinary graph-local
input-ready capability yet, though existing producer ownership APIs exist;
(b) QwenGraphBase hook must cover all actual roots without detached PP nodes;
(c) split stage contracts/signatures must match arena lifetimes; (d) MTP prior
snapshot currently hangs off trunk and must move with its capability before
MTP enablement; (e) native executor currently has no per-stage side-stream
support, so real resident VRAM overlap cannot be claimed. The smallest complete
slice resolves (a)-(c) for local ordinary RAM/mmap and leaves (d)-(e) explicit.
