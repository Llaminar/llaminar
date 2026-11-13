# Q7 fixed VRAM n-gram ownership in a native pipeline

Q7 is the original four-shard Qwen3.8-Flash-Next-UD-Q4_K_XL model with FP32
activations and FP16 KV. Its two four-GPU apportioned EP stages use their own
native RCCL contexts. Both stages are on the same original NUMA owner, selected
by the existing observed-ISA pipeline binding from two-rank discovery. The
eight selected gfx906 devices own the one immutable n-gram source in VRAM. The
physical identities come from the original UUID, NUMA and PCI inventory.

QFN has exactly one main PLE decoder layer. The model parser authenticates
zero-based layer 1 and the graph calls PLE only there
(`Qwen4ExpConfig.cpp`, `Qwen4ExpGraph::addModelLayerInput`). In Q7's equal
24+24 layer split, the first EP4 owns the PLE computation and its history.
The second EP4 owns four storage ranges and continues executing its own model
layers. It has no PLE embedding consumer. Discover the PLE stage from the
actual layer interval, independent of stage count, width or position.

## Why lookup at the PLE stage

The table gather depends on current token IDs and accepted PLE token history.
Those authorities are already present at the original layer in ordinary,
prefill and MTP verifier construction. The lookup can run earlier on the same
stage once those dependencies are available. This could overlap the trunk in a
later tuning pass. Moving lookup to an embedding stage that does not own PLE
would require moving or publishing the PLE history and carrying resulting rows
to another stage. In Q7 the embedding and PLE layer already share a stage.
Sending PLE rows with its stage output would communicate bytes the tail never
uses. Keep accepted token and convolution history, grouped verifier prior, and
rollback publication with the PLE stage. If another legal layer partition puts
layer 1 after the head, route its current token/position metadata through the
existing PP mechanism to the actual PLE owner. Check prefill separately because
`PipelineForwardGraphEdges::append` already reverses ordinary/MTP metadata and
its prefill path has a different admitted token bank.

The existing fixed-head runtime conflates storage owners and compute consumers
because full-main TP uses the same set for both. Its fixed-head group validates
one LocalTP roster and performs a native allgather. Expanding that collective to
all eight would send useful PLE rows to storage-only GPUs and incorrectly
associate a full-model compute group with two distinct stages.

The adaptive pooled path already proves an original-rank native storage union,
addressed completion, prepared worker/arena permits, model-source retirement
and participant-local definition reuse. Use those ownership and wire primitives.
Full-VRAM immutable ranges do not require the adaptive cache directory, C0 or
positive slot accounting, Pin, victim selection, Free, host source service or
request-time movement. The adaptive archive's planned 22/33 operation families
are therefore not Q7 prerequisites. Extract a common low-level retained
addressed endpoint/child import facility only where the exact contracts match.
The recent shared-child proof covers one paired counted native edge on original
CUDA2/ROCm2 and two source parents. It does not certify a bidirectional
multi-edge service, partial teardown, distinct capture streams, or Q7 itself.

## One fixed placement and one useful data path

1. Extend the existing `FixedHeadSharded` policy with a typed rank-local
   pipeline execution provision. Derive the PLE consumer roster from the stage
   that covers the model-owned decoder layer and derive storage owners from the
   original homogeneous GPU pipeline roster. Validate each original rank,
   backend, UUID, NUMA, stage and local index. Preserve the existing full-main
   fixed-head arm. Continue rejecting unsupported cross-rank/mixed-vendor fixed
   placement unless it gains an explicit implementation. Compute geometry from
   the source's 16 heads and the actual cohorts. Q7 has two whole heads per
   storage GPU. Unequal positive widths and additional stages are valid within
   this geometry. An explicit whole-head policy cannot give a nonempty head to
   more owners than the source has heads. Any finer row partition would require
   a separate exact per-head row routing proof.
2. Build one immutable placement through
   `ImmutableRowTableExecutionProvision`,
   `ResolvedImmutableRowTablePlacement` and the existing final table admission
   after graph rows and expert quotas are selected. Keep storage membership
   distinct from `consumers()`. `RankMemoryPlanInputs`,
   `MemoryPlanner::fixedHeadParticipants`, the immutable placement input/wire
   and `Qwen4ExpBufferLayout` currently assume full-main TP. Add an explicit
   pipeline alternative rather than feeding an eight-way TP coordinate into
   four-way dense plans. Charge disjoint GGUF native row ranges once under
   `ImmutableTableStorage`, exact upload staging, exact reader/service
   request/result operands and native completion/graph resources under their
   existing PMA owners. Assert that all admitted ranges cover the original
   source exactly once. Size persistent per-participant banks by actual
   incident edges, maximum retained prefill/verifier rows and admitted
   concurrency. A table admission failure remains outside the expert/row
   search. No raw memory observations allocate storage independently.
3. Split owner preparation from consumer binding in `ModelContext`. The
   current batch upload loops over consumers and uses
   `prepareImmutableRowTableForConsumer`, so it cannot prepare a storage-only
   participant. Reuse the same native range uploader and original source/PMA
   transaction for all allocators, with the existing prepared worker and
   explicit drained loader stream for each actual device. Attach the resulting
   immutable source to the real PLE compute stage and service owners. Do not
   assign a fake reader or a copied source to the tail.
4. On the PLE stage hash once per actual reader as in fixed-head TP, retaining
   the existing exact query/prior identity. Assign each source head range to one
   original reader assembler by typed immutable geometry. On Q7, each reader
   packs its two local heads and obtains two remote heads from one tail owner.
   Use actual owner/reader locality for stable assignment, with original
   physical membership and head ordering carried separately. Interleave
   actual reader and paired storage-owner entries in source-range order so
   each Q7 reader assembles four adjacent original heads. The storage range
   order stays independent of pipeline/native communicator order. Send only the
   selected row IDs for the destination owner's heads. The owner validates the
   request and reads its admitted native table. Return only the corresponding
   packed native rows. Keep all data, live counts, status and receipts on
   device. The return must complete on the reader's original stream before its
   local gather consumes it.
5. Reuse the fixed-head packing/decoding and exact native transport pieces:
   `Qwen4ExpFixedHeadLookupStage`, `NativeByteEdge`,
   `NativeCollectiveBytes::prefix`, native completion frames, and
   `OriginalNativeGraphCaptureGroup`. A request and a response are distinct
   directional edge operations. Give both actual endpoints the same original
   invocation/edge identity and prove error/abort and receipt semantics. If
   the owner needs a separate retirement acknowledgement, its exact need must
   follow from buffer reuse/lifetime; the existing completion event and
   serialized parent may already close it. Use the existing
   `PipelineDeviceGeneration::collective_`, which already retains the full
   original native PP membership and exposes its exact coordinate through
   `PipelineForwardGraphEdges::CapturedDomain`. Q7 needs no additional
   storage communicator. The two stage compute contexts retain their original
   four members.
6. Collect reader-local native fragments only inside the actual PLE-stage
   collective. Q7 has equal four-head fragments per reader after pairing.
   Reuse native compact allgather and ordered decode on its original EP4
   context. For uneven geometry use existing device-counted allgather with
   exact live fragment extents and decode by explicit original head offsets.
   This sends no capacity padding, unused head slots or output to storage-only
   participants. For M live rows in Q7, each remote edge carries exactly
   M × 2 × sizeof(int64_t) of row IDs (the existing INT32 arena tensor
   represents 16 int64 IDs as 32 words per row) and M × 2 × native_row_bytes of returned
   data, plus required protocol metadata. Each reader owns M × 4 ×
   native_row_bytes for the stage gather. Empty invocations retain required
   metadata and carry no row payload. Receive checks authenticate exact row
   counts, ordering, heads and current invocation before decode.
7. Integrate the service inside each storage device's complete local parent.
   A downstream storage owner must run the addressed service before waiting
   for forward activation that depends on the lookup. Reverse row metadata for
   ordinary/MTP must already have reached the PLE stage before that service can
   wait for a query. For Q7, insert after `pipeline_mtp_rows` and its local
   dependency, before `pipeline_activation_receive`, in the relevant complete
   forward parent. Prefill has its actual admitted rows producer. An upstream
   storage owner requires the analogous edge before any operation that would
   wait on the PLE response. Verify the full dependency graph for other layer
   partitions. Attach only to transactions that run the PLE layer, including
   grouped verifier and FORCE/condition where applicable. The existing tail
   controller, HIP immutable ticket, and pipeline control/publication phases
   own branch selection. No hot host launch of a storage side graph.
8. Record each fixed request/response endpoint in its complete containing
   model parent through the existing full-pipeline capture wave. Use the same
   immutable role and range projection as admission. Direct recording is the
   initial implementation; the proved child-import facility is available only
   if measured preparation cost justifies it. Do not add the adaptive archive.
   If later justified, record a definition once per distinct
   immutable operand family on its original worker/stream and import the
   participant-local definition into each complete parent that uses it.
   Parent identity includes graph-local hash/count operands and exact root
   slices, placement/source/worker/arena generations, stage role, arithmetic
   and launch mode. Keep per-variant identity where these differ; never share
   graph pointers by matching capacity alone. Prove actual simultaneous
   endpoints and cross-stream imports before sharing beyond the existing
   bounded child proof. On failure, signal the original communicator, drain
   submitted work, and retire recorded definitions off-worker before their
   roots/permits/context. Request reset retains graph topology and resets the
   existing device-owned history/receipt state. A live borrower blocks source
   and archive retirement.

The participant-local service and exact forward-role map are detailed in
[the fixed resident service plan](QFN_FIXED_RESIDENT_PIPELINE_SERVICE_2026-11-10.md).
Implementation, functional qualification and performance qualification remain
separate: this approved design is not native or model evidence.

## Qualification using the existing owners

- Keep the full Unit and `ProductionTestPreflight` prerequisite ahead of
  performance work. Register focused model-free planner/codec/lifetime replay
  and native composition explicitly. Use synthetic immutable metadata for
  arbitrary stage widths/counts and reject duplicate physical owners, foreign
  ranks, wrong ranges, wrong stage, and premature retirement. Use real original
  RCCL/NCCL peers where present. Run partial/empty/skewed live counts,
  large-to-small reuse, stale invocation, partial recording/cancellation,
  resets, prefix/force/ordinary/fixed/dynamic branches. Assert actual addressed
  request/response/gather bytes and one materialization per source range, with
  no payload at irrelevant endpoints. Compare reconstructed lookup embeddings
  byte-for-byte against the existing fixed-head native oracle and include
  independent producer/consumer expected bytes and untouched guards. Build
  all shipping GPU architectures and account for each new kernel and native
  executable/definition resource. Retain evidence for cold cost and native
  communication efficiency separately from Release timing.
- Extend `Qwen4ExpObservedPipelineCases.h` in the existing
  `v2_integration_parity_qwen4exp_production_matrix`. Preserve RAM and mmap
  candidate IDs. Add an explicit eight-owner VRAM FixedHeadSharded cell product,
  retaining Ordinal/Random × Static/Dynamic and the existing source-only
  Off/fixed1/fixed3/Dynamic retained3 policies. Source admission stays a
  candidate until actual positive preparation and complete captured source
  ownership/retirement receipts pass. Authenticate eight nonoverlapping
  placement/upload owners, four real query consumers and both stage rosters in
  existing source evidence. The canonical source runner already handles
  original active/inactive retirement and the exact real main/predictor files.
- Add observed Q7 ordinary `NativeGeneration`, main `ProductionParity`,
  and enabled `MTPExactReplay` registrations by sharing the existing QFN
  generation/parity bodies and observed binding wrapper. They currently bind
  only the source-only candidate. Bind physical addresses before any runtime
  config is constructed, and retain the exact original owner and bound policy
  in receipts. Use the original full model/reference identity, real Off
  controls, complete prefix/reset/force trajectories and true Dynamic movement
  evidence. MTP exact replay must use the actual paired Q8_0 predictor and
  serial/grouped native output contract. Treat broader numerical MTP depth
  capability as a separate evidence obligation under the normal matrix.
- Export an exact Q7 paired numerical-eligible declaration through the
  canonical parity/corpus inventory, with the same lossless configuration
  projection and bound UUID/NUMA/selection receipt. The current observed
  source-only GTest method has no paired benchmark record. Preserve the same
  original discovery selection in Release and prove the active owner and
  stage/PLE roster in runtime receipts. Reuse
  `scripts/ci/run_model_parity_benchmarks.py` with its same-build Release binary
  and parser, exact cell, paired corpus and identical green numerical panel.
  The production workload is 512 prompt tokens and 256 outputs, one warmup,
  three measured iterations, greedy seed 42. Report accepted token identities,
  prefill and decode separately, per-stage expert/PLE evidence, actual movement,
  owner VRAM bytes, native communication volumes, clean driver window and
  retirement. Existing TP4/RAM numbers are controls, never Q7 throughput.

The nearest production owners for implementation are
`RankMemoryPlanInputs`, `MoEPipelineMemoryAdmission`,
`ImmutableRowTableExecutionProvision`, `ResolvedImmutableRowTablePlacement`,
`ModelContext`, `Qwen4ExpReplicatedPLECohort`, `Qwen4ExpPLEBuilder`,
`PipelineForwardGraphEdges` and the current original native byte/capture
owners. Keep planner, lifetime and arithmetic responsibilities in those
subsystems. No new transport or second model/allocator authority is needed.
