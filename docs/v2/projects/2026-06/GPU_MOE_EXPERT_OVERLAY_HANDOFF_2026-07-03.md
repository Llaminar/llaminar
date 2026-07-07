# GPU MoE Expert Overlay Handoff - 2026-07-03

This is a handoff for the next agent on branch
`feature/gpu-moe-expert-rebalancing`.

The immediate workstream is no longer the broad MoE rebalancing sprint in the
abstract. We are in a focused correctness/debugging phase for Qwen 3.6 MoE
ExpertOverlay, phase-split CUDA2/ROCm2, prefix cache, MTP, and graph-captured
parity/E2E. The current failure is a real production-path prefix restore state
coherence issue, not a test harness detour.

## Continuation Update - 2026-07-06

Focused long-context ROCm2TP LLEP coverage has been added and is passing.

Fixes from the continuation:

- LLEP prefill planner now propagates `max_weight_transfers` into CUDA and
  ROCm device-side config reconstruction. The missing field let compact
  transfer payload capacity be ignored, producing payload/plan overflow and
  stale transfer command materialization.
- Fresh prefix-cache misses no longer inherit prefix-cache block-size
  segmentation for LLEP prefill. Stable block windows are now used only for an
  actual restored prefix hit, or for an explicit configured
  `prefill_window_tokens`. The miss path stays equivalent to uncached one-shot
  prefill.
- Qwen3.6 MoE overlay graph lowering now passes apportioned local expert
  ranges for normal and LLEP transfer-slot views so presliced tensors do not
  walk the full 256-expert domain on each participant.

Regression coverage added:

- Synthetic CUDA and ROCm MoE kernel transfer-capacity checks in the existing
  `RuntimePrefillLeastLoadedTransferCommandsPreserveTransferOrder` tests.
- `V2_Integration_Qwen36MoEExpertOverlay_ROCm2TPLLEP_LongContextStateContinuity`
- `V2_Integration_Qwen36MoEExpertOverlay_ROCm2TPLLEP_NeedleRecallRegression`

Validation run after the fixes:

```bash
cmake --build build_v2_integration --target \
  v2_integration_prefix_cache_state_probe llaminar2 --parallel

ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Qwen36MoEExpertOverlay_ROCm2TPLLEP_NeedleRecallRegression$' \
  --output-on-failure --parallel

ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Qwen36MoEExpertOverlay_ROCm2TPLLEP_LongContextStateContinuity$' \
  --output-on-failure --parallel

LLAMINAR_E2E_LONG_CONTEXT=1 \
LLAMINAR_E2E_LONG_CONTEXT_TIER=lite \
LLAMINAR_E2E_CONTEXT_LENGTH=2048 \
LLAMINAR_E2E_LONG_MAX_TOKENS=128 \
LLAMINAR_E2E_LONG_REQUEST_TIMEOUT=420 \
tests/v2/e2e/server/test_server_e2e.sh \
  --binary build_v2_integration/llaminar2 \
  --suite '<Qwen3.6 MoE>|tp|64|<ROCm2 phase-split ExpertOverlay prefix-cache LLEP flags>|qwen36-moe-llep-prefix-ram-rocm2tp|no-prefill-graph-buckets,non-thinking-only,prefix-cache-rebalance-clear-probe,moe-rebalance-movement-probe'
```

Observed E2E result for all same-backend two-GPU phase-split prefix-cache
long-context slices run so far:

- ROCm2 LLEP: `ALL PASSED: 21/21 tests passed`.
- ROCm2 Dynamic: `ALL PASSED: 21/21 tests passed`.
- CUDA2 LLEP: `ALL PASSED: 21/21 tests passed`.
- CUDA2 Dynamic: `ALL PASSED: 21/21 tests passed`.

All covered beginning/middle/end long needle recall, multi-needle strict JSON
recall, structured long generation, near-boundary context, oversized context
rejection, movement probe, clean shutdown, VRAM release, and clean server log.

## Current Branch And Tree

- Branch: `feature/gpu-moe-expert-rebalancing`
- HEAD: `f59b2be6 WiP: Qwen 3.6 Dynamic/LLEP parity tests passing. Beginning perf tuning round.`
- Remote tracking: `origin/feature/gpu-moe-expert-rebalancing`
- Dirty tree: large and intentional. Do not reset it.
- Last check for orphaned jobs after stopping the in-flight parity diagnostic:
  no matching `llaminar2`, E2E, or Qwen parity process remained.

Important untracked/dirty items seen at handoff:

- Untracked: `docs/v2/projects/2026-06/JIT_MODEL_LOADING_PROJECT_PLAN.md`
- Untracked: `tests/v2/integration/parity/results/f59b2be6/`
- This file is newly added as the handoff.
- Many production and test files are modified. Treat the whole tree as the
  work product of this sprint, not as unrelated user dirt.

## User Steers That Are Invariants

These are not suggestions. Preserve them unless the user explicitly changes
direction.

- Use the existing production graph. Do not create shadow graphing or raw-stage
  parity-only execution.
- GPU parity tests must run full graph capture in prefill and decode. Snapshot
  collection must be reusable graph-captured machinery, not fallback raw stage
  execution.
- Parity tests should be serialized because they contend for GPUs and snapshot
  output.
- Reproduce failures in tests first where practical, then fix the production
  graph/kernel/plumbing bug. Do not loosen thresholds.
- Write regression tests for bugs found along the way.
- Do not use silent fallbacks. Unsupported behavior must fail fast and loudly.
- No CPU fallback for sampled logits. No Q8 fallback. Q8 either succeeds or
  hard fails.
- Never use CUDA/HIP default/null streams. Every GPU operation must use an
  explicit stream.
- CUDA and ROCm must stay aligned structurally and semantically.
- Do not hardcode two-card behavior. ExpertOverlay/LocalTP logic should support
  variable domains of size `n >= 2`.
- Same-backend homogeneous domains are in scope first: CUDA2 and ROCm2. Mixed
  CUDA+ROCm domains are out of scope for this sprint.
- Use grouped collectives where appropriate. Do not prefer explicit peer-copy
  branches when NCCL/RCCL can own transport and topology.
- RCCL graph capture is a feature target. Do not cripple ROCm by disabling
  collective capture or adding "drain" workarounds.
- Transfers and compute should overlap. Transfer stream ownership belongs with
  the device context, not a long-lived cache object.
- Rebalance transfer machinery should be async, wave/bucket oriented, and avoid
  moving empty slots.
- Avoid runtime VRAM allocation in hot paths. Prefer up-front pools/staging
  buffers and explicit slot ownership.
- Prefix cache blocks should be portable blobs. RAM reuse should work across
  request shapes. Disk/cross-run portability should not be broken by embedding
  runtime-local device pointers or table cookies.
- Do not reject prefix blocks solely because they were computed in a different
  bucket size. A prefix is a prefix; next request length should not invalidate it
  unless the payload schema/semantic state is genuinely incompatible.
- Avoid unnecessary backwards compatibility bloat. Remove dead/obsolete code
  instead of preserving aliases and legacy paths.
- Raw mallocs are acceptable in tests, not production hot paths.
- Keep docs updated with current design choices and benchmark status.
- Do not encode benchmark-specific hacks into policy. Real inference behavior
  matters.
- Expert movement language should be generic and not tied to "hot cache". Hot
  cache is only one consumer of expert movement capability.
- Dynamic policy should be one shared strategy across CPU, CUDA, and ROCm.
  Backend implementation details may differ, but the policy semantics should
  not fork into "ClassicDynamic" versus "Dynamic".
- LLEP and hot expert cache are separate concepts. LLEP should not depend on
  hot-cache economics.
- Main target policy direction:
  - Prefill: dense TP plus apportioned routed experts.
  - Decode: dense replicated plus apportioned routed experts, with optional hot
    expert replica/cache layer.
  - Shared experts follow dense policy, not routed-expert movement policy.
- Canonical terminology:
  - `ReplicatedExperts`: all experts on every participant.
  - `ApportionedExperts`: whole experts split across participants.
  - `ShardedExperts`: every participant owns a shard of every expert, with
    reduction where required.
  - `HybridTP_AE`: tensor parallel dense plus apportioned experts.
  - `PhaseSplitHybridTP_AE`: TP dense in prefill, replicated dense in decode,
    apportioned routed experts.
  - `HybridTP_RE`: TP dense plus replicated experts.
- `clear_cache()` naming is overloaded and architecturally suspect. Prefer
  explicit inference-state boundary semantics as work continues.
- The next strategic direction after correctness is a cleaner inference state
  data layer and eventually JIT model loading. The current prefix/MTP/runtime
  lifetime problems are adjacent to that architecture.

## Main Historical Task Tree

This is the sprint path so far, in compact but complete form.

1. Started a new branch for GPU MoE expert rebalancing.
2. Verified the GPU MoE packed expert format direction:
   CUDA and ROCm active GPU MoE path use a backend-neutral NativeVNNI-style
   packed layout, so same-backend GPU arrivals should be descriptor/blob copies,
   not CPU serialize/deserialize/repack.
3. Generalized expert movement beyond CPU NodeLocal domains:
   same-backend direct movement, async staging, transfer slots, device-side
   runtime tables, CUDA/ROCm parity.
4. Clarified parallel policy semantics:
   `ExpertParallel` renamed to `ApportionedExperts`;
   `HybridTP_EP` renamed to `HybridTP_AE`;
   `PhaseSplitHybridTP_EP` renamed to `PhaseSplitHybridTP_AE`;
   added/kept clear semantics for replicated/apportioned/sharded experts.
5. Shifted strategy:
   decode roofline is limited by splitting routed experts across cards, so
   maxing prefill and cleaning phase-split dense policy became important.
6. Added device-side MoE rebalance controller pieces:
   shared policy header, CUDA/ROCm kernels, runtime tables, async transfer
   streams, transfer slots, maintenance graphs, stats.
7. Explored sideband/collective transport:
   found histogram payload can be non-trivial, so avoid making normal decode
   collectives expensive. Keep grouped/fused sidebands a hard requirement when
   used, but do not overload allreduce payloads blindly.
8. Tuned static/dynamic speed and policy:
   CUDA static baseline previously reached around 121 tok/s decode in this
   branch lineage; dynamic around 119.5 tok/s before later correctness patches.
   ROCm dynamic showed longer-run health and some wins; CUDA economics remained
   more fragile.
9. Ingested LLEP paper direction:
   LLEP should be modular and separate from hot-cache. Dynamic remains the
   CPU-like default policy; LLEP is another algorithm option.
10. Added Qwen 3.6 MoE ExpertOverlay parity/E2E test matrix:
    Dynamic and LLEP, CUDA2 and ROCm2, prefill/decode, SnapshotInfrastructure,
    then prefix cache and MTP dimensions.
11. Found correctness failures in phase-split prefix/MTP/E2E. User explicitly
    said to keep phase-split AE in E2E and fix the real bug.
12. Started refactoring parity framework so GPU parity never needs raw stage
    execution and can take snapshots through graph-captured production paths.
13. Current focus: CUDA2 phase-split Dynamic prefix restore with MTP. Stale MoE
    runtime import was fixed; now there is a remaining prefix restore/GDN/KV
    suffix mismatch.

## Current Immediate Goal

Get Qwen 3.6 MoE ExpertOverlay CUDA2/ROCm2 E2E tests passing for Dynamic and
LLEP in phase-split AE, including:

- Long-context checks.
- Prefix cache restore.
- MTP interaction.
- Forced expert movement/migration during prefill/decode so movement paths are
  actually exercised.

Before E2E is the current parity gate:

- Reproduce numerical/state mismatch in PyTorch parity tests.
- Fix production graph/kernel/state plumbing.
- Keep graph capture enabled.
- Keep phase-split AE.
- Do not loosen canonical thresholds.

## Most Recent Concrete Fixes

### 1. Stale MoE Runtime State Import Fixed

Problem:

- A CUDA2 phase-split prefix restore imported a single-participant MoE runtime
  table into a two-participant decode graph.
- The dynamic decode bank guard failed with table participant metadata like:
  `table_participants=1 expected=2`.

Changes landed in the dirty tree:

- `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp`
  - Added dynamic-bank validation diagnostics.
  - Uses effective resident mask semantics:
    `effective_resident_mask = raw_resident_mask & valid_mask`.
  - Allows ownerless runtime entries with `owner_participant == -1`.
  - Rejects malformed owners `< -1` and owners outside domain.
  - Requires each expert to have effective residency.
  - Requires ready descriptors for local-compute experts.
  - Added prefix runtime capture compatibility gate:
    `expectedPrefixRuntimeParticipantCount(...)`
    and `portableRuntimeStateMatchesPrefixRestoreDomain(...)`.
  - Capture now skips model runtime state if the live table's participant count
    is incompatible with the expected restore domain.
  - Added perf counters for skipped uninitialized/incompatible portable runtime
    state.
- `src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp`
  - Runtime bank validation now treats mutable descriptors or explicit owner
    metadata as runtime-owner metadata.
  - Allows ownerless non-local entries.
  - Uses effective resident masks.
  - Refuses to synthesize a decode runtime bank when mutable descriptors or
    explicit owner/resident metadata are requested.
- `tests/v2/unit/execution/moe/Test__MoEGraphNative_ForbiddenDependencyScan.cpp`
  - Added/updated source-scan tests:
    - `Qwen35MoEDecodeGraphAcceptsLiveDynamicRuntimeBank`
    - `MoEExpertStageAcceptsOwnerlessRuntimeResidency`
    - `PrefixRuntimeStateDoesNotSerializeMoEPlacementPointers`

Validation:

```bash
cmake --build build_v2_integration \
  --target llaminar2_core \
           v2_unit_moe_forbidden_dependency_scan \
           v2_integration_parity_qwen36moe_expert_overlay_prefix_mtp \
  --parallel

ctest --test-dir build_v2_integration \
  -R "^V2_Unit_MoEForbiddenDependencyScan$" \
  --output-on-failure --parallel

ctest --test-dir build_v2_integration \
  -R "^V2_Unit_MoERuntimeTable$" \
  --output-on-failure --parallel
```

These passed after the fix.

### 2. Portable MoE Runtime Prefix State

Already present in the dirty tree:

- `src/v2/execution/moe/MoERuntimeTable.h`
- `src/v2/execution/moe/MoERuntimeTable.cpp`

Added portable runtime structs/methods:

- `DeviceMoEPortableExpertRuntimeState`
- `DeviceMoEPortableLayerRuntimeState`
- `capturePortableRuntimeState(...)`
- `restorePortableRuntimeState(...)`

Important semantics:

- Capture serializes logical fields and histograms only.
- It does not serialize runtime-local descriptor pointers.
- `TransferSlot` flag is stripped in portable capture and re-derived from the
  live descriptor on restore.
- Restore resolves live descriptors for local-compute experts and fails hard if
  required payload is missing.

Prefix runtime blob in `Qwen35MoEGraph.cpp`:

- Magic: `LMOERUN1`
- Version: `1`
- Obsolete non-magic pointer-bearing blobs are rejected loudly.
- No legacy compatibility path.

### 3. Prefix Restore Reset Ordering Was Adjusted

In `DeviceGraphOrchestrator::populatePrefix(...)`:

- It waits for pending live prefix mutation and live graph producers.
- It computes whether there is model runtime state to restore.
- It calls `clearInferenceState()`.
- It skips `graph_builder_->resetState()` only when a model runtime state blob
  exists. This was intended to preserve MoE logical placement while restoring
  portable runtime state.

Potential caution:

- `Qwen35Graph::resetState()` is currently a no-op because GDN state lives in
  the hybrid KV cache.
- `state_.clear()` does call `kv_cache->clear()`, and CUDA/ROCm hybrid KV cache
  `clear()` resets GDN host mirrors and kernel state.
- So the current failure is probably not simply "forgot to clear GDN state".

## Current Failing Test

Focused reproducer:

```bash
env \
  LLAMINAR_LOG_LEVEL=ERROR \
  LLAMINAR_PREFIX_PROBE_HASH_KV_PAYLOADS=1 \
  LLAMINAR_PREFIX_PROBE_HASH_KV_SEGMENTS=1 \
  LLAMINAR_PREFIX_PROBE_KV_SEGMENT_SPLIT=4 \
  LLAMINAR_PREFIX_PROBE_HASH_GDN_DEVICE_STATE=1 \
  ./build_v2_integration/tests/v2/v2_integration_parity_qwen36moe_expert_overlay_prefix_mtp \
  --gtest_filter='Qwen36MoEExpertOverlayPrefixMTPParity.PrefixRestorePartialHit_CUDA2TPDynamicPhaseSplit'
```

Before the stale-runtime fix, this failed earlier in the dynamic MoE bank guard.

After the stale-runtime fix, it reaches the intended parity assertion and fails:

- Location:
  `tests/v2/integration/parity/qwen36/Qwen36MoEParityTestBase.h:1409`
- Assertion:
  partial prefix restore must reproduce full-prefill state before decode.
- Failure:
  `prefill_state_match` is false.
- Representative failure detail from last completed run:
  - Main KV payload hash mismatch at cache 0, layer 3, sequence 0.
  - Full prefill:
    - `k_hash=10027302578529184405`
    - `v_hash=9187562274977824938`
  - Restored prefix plus suffix prefill:
    - `k_hash=9787604000488774168`
    - `v_hash=466624280843908388`
  - Cached-block boundary metrics for layer 3 are exact:
    - `K_PROJECTION`, `V_PROJECTION`, `K_ROPE`, `KV_APPEND_SOURCE_K`,
      `KV_APPEND_SOURCE_V`, `KV_CACHE_K`, `KV_CACHE_V`
    - rows=256, cosine=1, rel_l2=0, max_abs=0.
  - Leading KV segment matches; trailing segment differs.
  - Full/restored position is 640 and sequence length is 640.
  - `moe_epoch=0`, `live_epoch=0`, `live_mutations=0` in that diagnostic, so
    this is no longer obviously a MoE runtime placement issue.
  - GDN device/host hashes:
    - Layer 0 matched.
    - Layers 1, 2, and 4 diverged.
    - This suggests the cached prefix boundary itself is correct, but suffix
      prefill starts from incorrect recurrent/conv state or uses stale dynamic
      graph/kernel state after the restore.

I started a more expensive diagnostic with:

```bash
env \
  LLAMINAR_LOG_LEVEL=ERROR \
  LLAMINAR_PREFIX_PROBE_HASH_KV_PAYLOADS=1 \
  LLAMINAR_PREFIX_PROBE_HASH_KV_SEGMENTS=1 \
  LLAMINAR_PREFIX_PROBE_KV_SEGMENT_SPLIT=4 \
  LLAMINAR_PREFIX_PROBE_HASH_GDN_DEVICE_STATE=1 \
  LLAMINAR_PARITY_PREFIX_SPLIT_DIAG=1 \
  ./build_v2_integration/tests/v2/v2_integration_parity_qwen36moe_expert_overlay_prefix_mtp \
  --gtest_filter='Qwen36MoEExpertOverlayPrefixMTPParity.PrefixRestorePartialHit_CUDA2TPDynamicPhaseSplit'
```

It was stopped for this handoff before it reached the diagnostic comparison. No
failure conclusion came from that run. Re-run it next if you need the split:

- `full_vs_split`
- `split_vs_restored`
- `seed_vs_split_seed`

Those diagnostics are the fastest way to know whether split prefill without
cache restore is already divergent, or whether only the prefix restore import
path corrupts state.

## Current Working Hypotheses

These are hypotheses, not established facts.

### Hypothesis A: Phase-Split GDN Local/Full Bank Export Is Wrong

Phase-split prefill does dense TP. Decode uses replicated dense. For GDN layers,
`GDNLiveStateAllGatherStage` gathers TP-local live state into a full replicated
state at the end of prefill:

- File: `src/v2/models/qwen35/Qwen35Graph.cpp`
- Stage added when:
  - `total_tokens > 1`
  - `gdnLiveStateAllGatherAvailable(...)`
- Stage:
  - `layerX_gdn_live_state_allgather`

CUDA/ROCm GDN kernels maintain an active state and a secondary state:

- `CUDAGatedDeltaNet.h`
- `CUDAShortConvolution.h`
- `ROCmGatedDeltaNet.h`
- `ROCmShortConvolution.h`

`allocateState(full_size)` keeps the previous active local bank as secondary,
then makes the full bank active. `exportStateForSize(local_size, ...)` can later
export the secondary local bank if present.

Portable prefix cache layout stores both:

- host/local GDN state bytes: `metadata.host_bytes`
- device/full GDN state bytes: `metadata.device_bytes`

In RAM/disk prefix blocks, both may be host-staged in one byte vector:

- `RamPrefixStorageBackend` allocates `layout.hybrid_state_bytes`.
- `exportHybridPrefixPayload(...)` treats missing device payload plus host
  payload as host-staged device state.
- Export order is host/local first, then device/full.
- Import order mirrors that.

For partial prefix restore with suffix prefill:

- `importHybridPrefixPayload(...)` sees `restore_for_suffix_prefill=true`.
- It sets:
  - `include_host_state = false` if host payload hydrates device bank.
  - `include_device_state = true` when full device bytes exist.
  - `import_host_state_into_device_state = true` when host bytes exist and
    device bytes exist.
- This should import:
  - full bank from device/full section,
  - then local bank from host/local section into active state for suffix prefill.

Potential failure mode:

- The secondary local bank may not actually represent the post-prefix local
  state when exported after all-gather, or selection/import order may leave the
  wrong active bank for the suffix prefill.
- If so, cached-block boundary KV is exact but suffix prefill diverges because
  GDN starts from a stale or wrong local recurrent/conv state.

Next check:

- Run the split diagnostic. If `full_vs_split=match` and
  `split_vs_restored` mismatches, focus on `importHybridPrefixPayload` and
  `CUDA/ROCmHybridRingKVCache::importHybridPrefixState`.
- Add debug counters/logging for hybrid prefix import:
  - host bytes,
  - device bytes,
  - whether import host into device state happened,
  - active state size before/after import,
  - secondary state size before/after import.
- Consider adding explicit kernel inspection APIs if needed, but avoid
  production bloat.

### Hypothesis B: Captured Prefill Replay Reuses Stale GDN Kernel State

Prefix restore calls:

- `clearInferenceState()`
- `handleLivePrefixReplayStateAfterMutation(PrefixRestore, ...)`

That path should:

- clear transient handoffs,
- discard cached forward graphs on prefix restore,
- reset kernel dynamic state unless replay state is explicitly preserved.

Files:

- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`
  - `clearInferenceState`
  - `handleLivePrefixReplayStateAfterMutation`
  - `recordLivePrefixSessionReset`
  - `resetKernelDynamicState`
- `src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp`

Potential failure mode:

- A prefill graph cache or backend kernel dynamic object survives prefix restore
  with stale pointers/metadata to GDN request banks or effective-length state.
- The comments suggest the intended semantics are to discard cached graphs on
  prefix restore. Verify this in the actual failing path with perf/debug
  counters, not by assumption.

Next check:

- Confirm `handleLivePrefixReplayStateAfterMutation(PrefixRestore, ...)` is
  reached after `populatePrefix` and before suffix prefill graph launch.
- Confirm `forward_engine_->discardAllCachedGraphs()` runs for the suffix path.
- Confirm `KernelFactory::resetAllDynamicState()` is called after prefix restore,
  not bypassed by preserved replay state.
- If graph cache survives intentionally, this is probably wrong for prefix
  restore until there is a dedicated equivalence proof.

### Hypothesis C: Prefix Cache Hybrid Payload Is Portable But Bucket Coupling Remains

User rejected a previous direction that made prefix runtime state opaque,
serialized, or bound to runtime table identity/bucket shape.

Current expectation:

- Prefix blocks are portable logical blobs.
- RAM/disk/cross-run should remain viable.
- A prefix produced under one request/bucket shape should be reusable for a
  later request with a different suffix length, subject only to actual payload
  schema compatibility.

Potential issue:

- `PrefixPayloadLayout::compatiblePayloadShape(...)` still checks block size and
  hybrid byte sizes. That is acceptable for the block payload shape, but not a
  reason to reject the semantic prefix because "the next request is a different
  bucket". Be careful before adding more strict fields here.

## Files To Inspect First

Prefix restore / state lifecycle:

- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`
  - `populatePrefix`
  - `restoreLivePrefixState`
  - `captureLivePrefixState`
  - `harvestPrefix`
  - `clearInferenceState`
  - `handleLivePrefixReplayStateAfterMutation`
  - `resetHybridPrefixPayloadState`
  - `importHybridPrefixPayload`
  - `exportHybridPrefixPayload`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h`
  - `InferenceState::clear`
  - live prefix mutation state comments
- `src/v2/execution/prefix_cache/PrefixPayloadLayout.cpp`
- `src/v2/execution/prefix_cache/RamPrefixStorageBackend.cpp`
- `src/v2/execution/prefix_cache/DeviceHotPrefixStorageBackend.cpp`
- `src/v2/execution/prefix_cache/DiskPrefixStorageBackend.cpp`

Hybrid/GDN state:

- `src/v2/kernels/IHybridKVCache.h`
- `src/v2/kernels/HybridKVCacheConfig.h`
- `src/v2/kernels/cuda/kvcache/CUDAHybridRingKVCache.h`
- `src/v2/kernels/rocm/kvcache/ROCmHybridRingKVCache.h`
- `src/v2/kernels/cuda/gdn/CUDAGatedDeltaNet.h`
- `src/v2/kernels/cuda/gdn/CUDAShortConvolution.h`
- `src/v2/kernels/rocm/gdn/ROCmGatedDeltaNet.h`
- `src/v2/kernels/rocm/gdn/ROCmShortConvolution.h`
- `src/v2/execution/compute_stages/stages/GDNLiveStateAllGatherStage.cpp`
- `src/v2/execution/compute_stages/stages/GDNRecurrenceStage.cpp`
- `src/v2/execution/compute_stages/stages/ShortConv1dStage.cpp`

Phase-split graph:

- `src/v2/models/qwen35/Qwen35Graph.cpp`
  - `gdnLiveStateAllGatherAvailable`
  - `buildGDNAttentionGraph`
- `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp`
  - runtime table validation
  - prefix runtime state capture/restore
  - phase-split ExpertOverlay lowering

Parity test:

- `tests/v2/integration/parity/qwen36/Qwen36MoEParityTestBase.h`
  - partial prefix restore flow around the `PrefixRestorePartialHit` helper.
  - `LLAMINAR_PARITY_PREFIX_SPLIT_DIAG`.
  - snapshot comparison options around the failing assertion.
- `tests/v2/integration/parity/qwen36/Test__Qwen36MoE_ExpertOverlay_PrefixMTP_Parity.cpp`

Regression/source scans:

- `tests/v2/unit/execution/moe/Test__MoEGraphNative_ForbiddenDependencyScan.cpp`
- `tests/v2/unit/execution/moe/Test__MoERuntimeTable.cpp`
- `tests/v2/integration/kernels/cuda/Test__CUDAHybridKVCacheReset.cpp`
- `tests/v2/integration/kernels/rocm/Test__ROCmHybridKVCacheReset.cpp`

## Commands To Resume

Build focused targets:

```bash
cmake --build build_v2_integration \
  --target llaminar2_core \
           v2_unit_moe_forbidden_dependency_scan \
           v2_integration_parity_qwen36moe_expert_overlay_prefix_mtp \
  --parallel
```

Run current unit gates:

```bash
ctest --test-dir build_v2_integration \
  -R "^V2_Unit_MoEForbiddenDependencyScan$|^V2_Unit_MoERuntimeTable$" \
  --output-on-failure --parallel
```

Run failing CUDA2 parity:

```bash
env \
  LLAMINAR_LOG_LEVEL=ERROR \
  LLAMINAR_PREFIX_PROBE_HASH_KV_PAYLOADS=1 \
  LLAMINAR_PREFIX_PROBE_HASH_KV_SEGMENTS=1 \
  LLAMINAR_PREFIX_PROBE_KV_SEGMENT_SPLIT=4 \
  LLAMINAR_PREFIX_PROBE_HASH_GDN_DEVICE_STATE=1 \
  ./build_v2_integration/tests/v2/v2_integration_parity_qwen36moe_expert_overlay_prefix_mtp \
  --gtest_filter='Qwen36MoEExpertOverlayPrefixMTPParity.PrefixRestorePartialHit_CUDA2TPDynamicPhaseSplit'
```

Run split diagnostic:

```bash
env \
  LLAMINAR_LOG_LEVEL=ERROR \
  LLAMINAR_PREFIX_PROBE_HASH_KV_PAYLOADS=1 \
  LLAMINAR_PREFIX_PROBE_HASH_KV_SEGMENTS=1 \
  LLAMINAR_PREFIX_PROBE_KV_SEGMENT_SPLIT=4 \
  LLAMINAR_PREFIX_PROBE_HASH_GDN_DEVICE_STATE=1 \
  LLAMINAR_PARITY_PREFIX_SPLIT_DIAG=1 \
  ./build_v2_integration/tests/v2/v2_integration_parity_qwen36moe_expert_overlay_prefix_mtp \
  --gtest_filter='Qwen36MoEExpertOverlayPrefixMTPParity.PrefixRestorePartialHit_CUDA2TPDynamicPhaseSplit'
```

Run ROCm equivalent after CUDA root cause is understood:

```bash
env \
  LLAMINAR_LOG_LEVEL=ERROR \
  LLAMINAR_PREFIX_PROBE_HASH_KV_PAYLOADS=1 \
  LLAMINAR_PREFIX_PROBE_HASH_KV_SEGMENTS=1 \
  LLAMINAR_PREFIX_PROBE_KV_SEGMENT_SPLIT=4 \
  LLAMINAR_PREFIX_PROBE_HASH_GDN_DEVICE_STATE=1 \
  ./build_v2_integration/tests/v2/v2_integration_parity_qwen36moe_expert_overlay_prefix_mtp \
  --gtest_filter='Qwen36MoEExpertOverlayPrefixMTPParity.PrefixRestorePartialHit_ROCm2TPDynamicPhaseSplit'
```

Do not limit parallelism for build or CTest.

## After The Current Parity Bug Is Fixed

1. Add/verify regression coverage for the exact bug:
   - Prefer an integration parity regression if the bug is numerical/stateful.
   - Add source-scan/unit checks only for structural guarantees that can be
     validated cheaply.
2. Re-run:
   - CUDA2 Dynamic phase-split prefix+MTP parity.
   - CUDA2 LLEP phase-split prefix+MTP parity.
   - ROCm2 Dynamic phase-split prefix+MTP parity.
   - ROCm2 LLEP phase-split prefix+MTP parity.
3. Circle back to E2E:
   - CUDA2 Dynamic and LLEP.
   - ROCm2 Dynamic and LLEP.
   - Prefix cache dimensions.
   - MTP dimensions.
   - Full long-context checks including needle/haystack style tests.
   - Use short rebalance intervals/thresholds so moves definitely happen during
     the E2E cycle.
4. Run broader prefix-cache E2E tests because prefix-cache restore semantics
   have been changed.
5. Run the full unit suite before the next WiP commit.
6. Then make a `--no-verify` WiP commit only after requested tests are green or
   explicitly documented as still failing.

## Architectural Notes To Preserve

### Runtime Tables

The current answer to "Do we need runtime tables at all?" is:

- Yes, for dynamic/domain-shaped graph-captured paths:
  - mutable expert residency,
  - local-compute masks,
  - transfer-slot descriptors,
  - route scratch,
  - decode histograms,
  - device-side rebalance controller state.
- No, not for static full-local immutable descriptor paths.
- Prefix cache must not serialize runtime-local table pointers. It may capture
  portable logical runtime state when that state is compatible with the restore
  domain.

### Prefix Cache State Semantics

Prefix cache should be a portable data layer:

- KV logical blocks.
- Hybrid/GDN recurrent and conv state.
- MTP shifted KV state when enabled.
- Terminal hidden/logits when required.
- Portable MoE logical placement/runtime state only when it is actually
  portable across the expected restore domain.

It must not depend on:

- Live runtime table identity.
- Device pointer identity.
- Old runner object lifetime.
- The next request's prefill bucket size.

### Clear/Reset Semantics

`clear_cache()` is overloaded and should not accumulate more meanings.
Current work should prefer explicit concepts:

- request/session live inference state reset,
- graph replay/capture reset,
- prefix restore mutation boundary,
- kernel dynamic state reset,
- MoE placement/runtime reset,
- prefix cache storage invalidation.

Do not paper over this with more hidden side effects.

## What Not To Do Next

- Do not weaken parity thresholds.
- Do not skip phase-split AE in E2E.
- Do not switch the E2E to a non-phase-split policy just to pass.
- Do not add CPU/raw-stage snapshot fallbacks.
- Do not make prefix cache RAM-only by embedding live pointers.
- Do not make disk/cross-run prefix restore fail by design unless the user
  explicitly accepts that tradeoff. The user already rejected it.
- Do not add bucket-size rejection as a correctness fix.
- Do not add CUDA-only special modes without ROCm alignment.
- Do not leave background E2E/parity processes running.
- Do not commit until tests requested by the user are run and failures are fixed
  or explicitly called out.

## Suggested Next 60 Minutes

1. Re-run the split diagnostic with `LLAMINAR_PARITY_PREFIX_SPLIT_DIAG=1`.
2. If `full_vs_split` mismatches:
   - The suffix prefill graph itself is not equivalent to full prefill.
   - Focus on phase-split GDN live-state allgather, prefill graph capture, and
     effective-length replay metadata.
3. If `full_vs_split` matches but `split_vs_restored` mismatches:
   - The prefix restore/import path is corrupting state.
   - Focus on `importHybridPrefixPayload` and CUDA/ROCm hybrid import ordering.
4. Add targeted logging/counters for hybrid state import/export sizes and active
   bank selection. Remove noisy debug-only instrumentation after root cause, or
   keep only concise perfstats/regression counters that are useful long term.
5. Patch CUDA and ROCm together if touching shared hybrid/GDN semantics.
6. Rebuild focused targets and rerun the failing parity.
7. Add a regression that would have caught the bug.
