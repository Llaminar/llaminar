# HIP Graph Capture for Prefill — Tiered Implementation Plan

## The Opportunity

| Metric | Current | With Graph Capture |
|--------|---------|-------------------|
| GPU time | 441ms | 441ms (unchanged) |
| Host overhead | 53ms (524 stages × 101μs) | ~0.5ms (1 graph launch) |
| Wall time | 494ms | ~442ms |
| Throughput | 1179 tok/s | ~1346 tok/s (GPU-limited) |

**14% throughput gain** by eliminating host dispatch overhead entirely.

---

## Key Insight: Prefill Topology IS Fixed for Fixed `seq_len`

For a given `seq_len`, every kernel in the 524-stage pipeline has:
- **Fixed grid dimensions** (determined by `seq_len` + model constants)
- **Fixed device buffer addresses** (BufferArena allocates once, addresses stable)
- **Fixed kernel arguments** (pointers to pre-allocated device buffers)

The MoE routing is data-dependent (different tokens → different expert assignments), but the KERNELS are identical — router, sort, scan, scatter, grouped-GEMM all launch with the same grids. Data flows through fixed-address buffers with different values each time. The K2/K4 grids use `seq_len` as upper bound with per-expert early-exit — topology is routing-invariant.

---

## Tiered Roadmap

This work should land in three explicit tiers:

| Tier | Scope | Rebalancing | Capture Shape |
|------|-------|-------------|---------------|
| Tier 1 | Single ROCm device, Qwen3.5 MoE, full local expert ownership | Not active; rebalance controller must be off/absent for graph capture | Monolithic per-device prefill graph keyed by fixed `seq_len` or bucket |
| Tier 1.5 | Generic ExpertParallel rebalance domains | Domain-scoped observe/dynamic rebalance over named device participants | No new graph shape; establishes placement epochs and domain semantics for Tier 2 |
| Tier 2 | Expert-parallel collective domains in the tiered overlay system | Supported between prefill graph executions | Bucket/chunk graphs with explicit placement epochs and collective boundaries |

Tier 1 should stay deliberately small. Its job is to prove the core GPU graph machinery, the ROCm grouped-prefill kernels, dynamic token updates, and replay callbacks without the additional complexity of expert migration or cross-domain collectives.

Tier 1.5 should cleanly reframe the existing MoE rebalance controller. The current implementation grew out of a two-socket CPU GlobalTP regime and still uses terms like `socket_id`, but the underlying abstraction is a domain of participants that own or serve experts. Tier 1.5 promotes that into an **ExpertParallel rebalance domain** with explicit participant identities, domain-local placement epochs, decode/prefill histogram sources, and clear rules for single-device observe-only behavior.

Tier 1 must still leave the right seams for Tier 2:
- Graph cache keys should have room for `domain_id`, `placement_epoch`, and a collective/topology signature even when Tier 1 fills them with single-device defaults.
- Graph capture preflight should fail loudly on active MoE rebalancing, partial expert ownership, TP/PP collectives, and sparse overlay stages, not silently continue on the normal path.
- Rebalance/mask/replica update APIs should notify graph caches so future tiers can invalidate or advance placement epochs cleanly.
- Bucketed prefill should be implemented as a reusable scheduling primitive, not only as a server padding trick, because Tier 2 will use buckets as rebalance intervals.

---

## Tier 1 Architecture: Monolithic Single-Device Prefill Graph Cache

```
┌─────────────────────────────────────────────────────────────┐
│  PrefillGraphCache                                           │
│                                                             │
│  key: seq_len/domain=single/placement_epoch=0 →              │
│       value: { hipGraphExec_t, pinned_params }               │
│                                                             │
│  Lifecycle:                                                  │
│    1. Warmup: execute normally (lazy allocs stabilize)       │
│    2. Capture: beginCapture → execute all 524 stages →       │
│               endCapture → instantiate                       │
│    3. Replay: update pinned params → launch graph            │
└─────────────────────────────────────────────────────────────┘
```

Unlike decode's segmented approach (which splits into capturable/manual segments), prefill uses **monolithic capture** — all 524 stages in a single graph. This is possible because:
1. We removed the only `hipStreamSynchronize` (the max-reduction sync)
2. All buffer allocations must happen in warmup (capacity checks pass without `hipMalloc`)
3. All operations are stream-ordered (kernel launches + `hipMemsetAsync`)

The target replay path is one graph launch after updating pinned params. If Tier 1 Phase 3 starts with the lower-risk prelaunch token upload, replay becomes one small ordered H2D plus the graph launch; that variant must be measured and logged explicitly.

## Current Code Surface

The concrete implementation touches these existing areas:

| Area | Current State | Required Direction |
|------|---------------|--------------------|
| `ForwardExecutionEngine.cpp` | Prefill explicitly skips GPU graph capture and uses `executeFastDecode()` for the fast path. | Add a prefill graph state machine before the normal prefill fast path. |
| `ForwardGraphTypes.h` | `ForwardGraphCache` already owns stable graph objects, streams, dynamic-param stages, token vectors, and replay failure counters. | Extend it with prefill capture state, or wrap those fields in a `PrefillGraphCache` keyed by `seq_len`. |
| `DeviceGraphExecutor_GraphCapture.cpp` | `executeWithGraphCapture()` captures and launches in one call; it does not provide persistent replay for prefill and does not call replay callbacks after monolithic launch. | Add a reusable capture/replay entry point or let `PrefillGraphCache` drive `IGPUGraphCapture` directly. Ensure replay callbacks run after a monolithic graph launch. |
| `HIPGraphCapture.{h,cpp}` | Provides `beginCapture()`, `endCapture()`, `instantiate()`, `launch()`, and update support. | Reuse this object per `(model graph, seq_len)` instead of recapturing per prefill call. |
| `GraphCaptureGuard.h` | Existing capture-active guard used by graph capture infrastructure. | Use it to assert that ROCm MoE/GDN allocation paths are not entered while capture is active. |
| `MoERoutingStage.cpp` | `isGraphCapturable()` is decode-only and release-only; prefill still routes through paths that may populate host routing data. | Add a fixed-topology prefill capturable path that is device-only and snapshot-free. |
| `MoEExpertComputeStage.cpp` | Fixed-topology grouped prefill execution exists, but graph-capturability helpers are stubs/false and allocation readiness is not part of the predicate. | Implement strict prefill readiness checks and expose them via `isGraphCapturable()`. |
| `ROCmMoEKernel.cpp` | Grouping and grouped-prefill scratch allocate lazily with `hipMalloc`/`hipFree`; async grouping avoids D2H/sync but still can allocate. | Warm these in normal execution; during capture, assert capacity instead of reallocating. |
| `SharedExpertFFNStage.cpp`, `SharedExpertGateStage.cpp` | Capturable only for `seq_len == 1`. | Allow release ROCm prefill after warmup/prepared refs are ready. |
| `GDNRecurrenceStage.{h,cpp}` | Header says prefill is not capturable, but ROCm prefill runs `chunk_forward()` as stream work. | Make capturability depend on ROCm chunk-forward scratch readiness and no capture-time allocation. |
| `ROCmEmbeddingKernelT.cpp` | Dynamic token IDs are pinned and can be preloaded before execution; current behavior may skip the H2D copy inside the captured graph. | Decide and implement the token update contract for prefill replay. See Tier 1 Phase 3. |
| `MoERebalanceController.{h,cpp}` | Decode-boundary controller; terminology and placement APIs are still socket-oriented from the initial CPU NUMA/GlobalTP implementation. | Tier 1 rejects active rebalancing. Tier 1.5 reframes this as a domain-scoped ExpertParallel rebalance controller with participant ids, placement epochs, and observe-only single-device behavior. Tier 2 consumes that contract for prefill chunk rebalance. |
| `DecodeExpertHistogram.{h,cpp}` | Tracks decode expert utilization, can merge runtime-table counts, but names and window semantics are decode-centric. | Tier 1.5 generalizes this into a domain histogram source usable by decode now and prefill chunks later. |
| `DeviceGraphOrchestrator` / `RankOrchestrator` mask APIs | Rebalance mutates MoE stage masks, registers engines, clears grouped descriptor ids, and sets replica state; LocalTP paths use local device index, while single-DGO paths use MPI local rank as a socket id. | Tier 1 graph caches must be invalidated on any mask/replica mutation. Tier 1.5 replaces socket/local-rank assumptions with domain participant identity. Tier 2 formalizes this as placement epoch advancement. |
| `MoEExpertParallelPlan` / overlay execution plan | Overlay has ExpertParallel domains and residency policies, but runtime rebalance is not yet expressed as a controller attached to those domains. | Tier 1.5 declares where a domain-scoped controller attaches in the overlay system and what domain kinds are eligible. Tier 2 adds graph-captured execution around those domains. |
| Sparse overlay stages | `MoEExpertDispatchStage`, `MoESparseDispatchStage`, `MoELocalExpertStage`, `MoESparseReturnReduceStage`, and `MoEExpertParallelReduceStage` return non-capturable. | Tier 1 rejects these graphs. Tier 2 captures device-local segments around explicit overlay collective boundaries. |
| `MoERuntimeTable` | Already has stable per-layer device pointers and double-buffered placement banks. | Tier 2 should use this as the graph-stable placement indirection for rebalance between graph executions. |

### Tier 1 Supported Scope

The first supported scope is deliberately narrow:
- single ROCm device
- release/non-snapshot build
- Qwen3.5 MoE with full local expert ownership
- grouped prefill enabled
- MoE rebalance controller off/absent (`MoERebalanceMode::OFF` for graph-capture acceptance)
- no expert masks, no replicas, no CPU participation
- no TP/PP collectives
- no sparse/overlay MoE stages

The graph-native sparse stages (`MoEExpertDispatchStage`, `MoESparseDispatchStage`, `MoESparseReturnReduceStage`, `MoEExpertParallelReduceStage`, `MoELocalExpertStage`) currently return non-capturable and are out of scope for Tier 1.

### Tier 1 Future-Proofing Requirements

Even though Tier 1 rejects rebalancing and collectives, its interfaces should not paint Tier 2 into a corner:
- `PrefillGraphCacheKey` should include explicit fields for `seq_len_or_bucket`, `device_id`, `domain_id`, `placement_epoch`, and `graph_topology_signature`. Tier 1 can set `domain_id="single"` and `placement_epoch=0`.
- `DeviceGraphOrchestrator::applyExpertMasks()`, `setExpertReplicaSet()`, and any future runtime-table placement flip should call a graph-cache invalidation/epoch hook. Tier 1 will use this to fail fast if someone mutates expert placement while a single-device prefill graph is cached.
- `isGraphCapturable()`/preflight errors should distinguish unsupported feature classes (`active_moe_rebalance`, `partial_expert_output`, `overlay_collective`, `tp_collective`) so Tier 2 can turn them on one by one with targeted tests.
- Bucket support should expose a chunked prefill runner API: `runPrefillChunk(tokens, offset, bucket_len)`. Tier 1 may use it only for server bucketing; Tier 2 will use the same primitive for rebalance intervals.

---

## Tier 1 Phase 1: Make Single-Device Prefill Stages Graph-Capturable

Stages currently returning `false` for prefill `isGraphCapturable()`:

| Stage | Count | Current Guard | Fix |
|-------|-------|---------------|-----|
| `MoEExpertComputeStage` | 40 | Decode-oriented capturability; fixed-topology prefill helpers are not wired into `isGraphCapturable()` | Add prefill path: `true` only when runtime table, descriptors, engines, and scratch are ready |
| `MoERoutingStage` | 40 | Delegates to decode path | Return `true` only for release ROCm device-only prefill routing with a runtime table and no host routing output |
| `SharedExpertFFNStage` | 40 | `seq_len == 1` | Return `true` after prepared refs and prefill scratch are ready |
| `SharedExpertGateStage` | 40 | `seq_len == 1` | Return `true` after inputs/weights/outputs are device-resident and stable |
| `GDNRecurrenceStage` | 30 | `seq_len == 1` | Return `true` after ROCm `chunk_forward` scratch/state readiness is proven |

**Total: 190 stages** to flip from non-capturable to capturable for prefill.

The remaining 334 stages (GEMM, RMSNorm, ResidualAdd, RoPE, Attention, KVCacheAppend, Embedding, etc.) mostly already return `true`, but the full-graph preflight must still reject any stage that proves unsafe at runtime and Tier 1 Phase 2/3 must cover dynamic state and capture-illegal side effects.

### Required Code Changes

#### `MoERoutingStage`

Files:
- `src/v2/execution/compute_stages/stages/MoERoutingStage.h`
- `src/v2/execution/compute_stages/stages/MoERoutingStage.cpp`
- possibly `src/v2/kernels/rocm/moe/ROCmMoEKernel.{h,cpp}` if the current routing API cannot be used without host routing output

Required work:
- Add a prefill helper such as `isFixedTopologyPrefillGraphCapturable()` and make `isGraphCapturable()` return true for either the existing safe decode path or this new prefill path.
- Keep the existing release-only/snapshot-off guard. In practice, this means returning false under `ENABLE_PIPELINE_SNAPSHOTS` and false outside `HAVE_ROCM`.
- Require `params_.seq_len > 1`, a ROCm device, `debugEnv().rocm.moe_grouped_prefill`, a valid `moe_runtime_table`, valid `routing_indices`/`routing_weights`, and a supported `top_k`.
- Ensure the prefill routing execution path is device-only during capture. The current prefill `execute()` path eventually calls `routeWithTensors()` and may populate host-side routing data for CPU expert dispatch/snapshots. That cannot happen inside graph capture. If the ROCm implementation still performs D2H/sync for this call, add a separate device-only route method or a capture-safe flag that writes only the routing tensors and runtime table.
- Ensure `DecodeExpertHistogram`, snapshot stash logic, and any cached host routing side effects are skipped for prefill graph capture.
- Preserve current false results for CPU, snapshots, missing runtime table, masks/replicas/overlay routing, and unsupported top-k.

Acceptance gate:
- Update `tests/v2/unit/stages/Test__MoERoutingStage.cpp`: the current `GraphCapturableRejectsCPUAndPrefill` expectation must become narrower. It should still reject CPU and unsafe prefill, but add a ROCm grouped-prefill/runtime-table case that is capturable in a non-snapshot build.
- Add or update a ROCm integration route test that captures the routing stage in a HIP graph and replays it with different token values. It must show no D2H transfer, no stream synchronize, and identical routing tensors versus normal execution.

#### `MoEExpertComputeStage`

Files:
- `src/v2/execution/compute_stages/stages/MoEExpertComputeStage.h`
- `src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp`
- `src/v2/kernels/rocm/moe/ROCmMoEKernel.{h,cpp}`
- `src/v2/kernels/rocm/moe/ROCmMoEGroupedPrefillKernels.hip`

Required work:
- Implement the existing false/stub helpers:
    - `initializeMoERuntimeTableForGroupedPrefill()`
    - `initializeFixedTopologyGroupedPrefill()`
    - `canUseRuntimePrefillGrouping()`
    - `isFixedTopologyPrefillGraphCapturable()`
- Keep `canUseFixedTopologyGroupedPrefill()` as the normal execution predicate, but make graph capture use a stricter predicate that proves no capture-time setup is needed.
- The prefill graph-capturable predicate must require:
    - release/non-snapshot build
    - ROCm device and `params_.seq_len > 1`
    - `debugEnv().rocm.moe_grouped_prefill`
    - full local expert ownership
    - no expert masks, no replicas, and no graph-native sparse overlay path
    - valid `moe_runtime_table`
    - all required prepared GEMM engines are ready for every local expert used by the fixed-topology descriptors
    - grouped gate/up and down descriptor tables have already been built
    - grouping buffers, write-head buffers, and grouped prefill scratch have sufficient capacity for this `seq_len`, `top_k`, `d_model`, and intermediate dimension
- Make `initializeFixedTopologyGroupedPrefill()` perform all descriptor-table and scratch warmup outside capture. It may reuse the existing normal prefill execution warmup, but the ready bit must be explicit so `isGraphCapturable()` is not just assuming warmup happened.
- During capture, the execution path must call only stream-ordered work: `prepareExpertGroupsAsync()` and `executeGroupedPrefillPipeline()`, with capacity assertions instead of allocation.
- Do not enable decode MoE expert graph capture as part of this phase. The decode path has separate replay-state risks; this phase should only make fixed-topology prefill capturable.

Acceptance gate:
- `tests/v2/integration/kernels/rocm/Test__ROCmMoEKernel.cpp` already contains `FixedTopologyRuntimeGroupedPrefillMatchesExistingPrefillPath`; make it pass in the intended non-snapshot ROCm configuration and keep it skipped/false under snapshots.
- Add a direct stage-level test that runs one warmup prefill, verifies `MoEExpertComputeStage::isGraphCapturable()` becomes true, captures the stage, replays it, and compares output to the non-captured grouped prefill path within existing MoE tolerance.
- Add a negative test where scratch capacity is intentionally undersized before capture; the stage must fail loudly before any `hipMalloc`/`hipFree` path is reached.

#### Shared Expert Stages

Files:
- `src/v2/execution/compute_stages/stages/SharedExpertFFNStage.cpp`
- `src/v2/execution/compute_stages/stages/SharedExpertGateStage.cpp`
- related ROCm shared expert kernels under `src/v2/kernels/rocm/`

Required work:
- Change `isGraphCapturable()` from decode-only (`seq_len == 1`) to release ROCm decode-or-prefill when the stage has valid tensors, prepared weights, and preallocated scratch.
- For `SharedExpertFFNStage`, audit the prefill path for lazy scratch allocation. Warm the maximum required scratch in normal execution, and make capture assert that `seq_len <= scratch_seq_len_` rather than reallocating.
- For `SharedExpertGateStage`, ensure all inputs/outputs are arena-backed and device-resident before capture. Any `ensureOnDevice()` that can allocate or upload from host during capture must be moved to warmup or guarded.
- Keep false under snapshots and non-ROCm backends.

Acceptance gate:
- Add focused unit/integration coverage for shared expert capturability at `seq_len > 1` in the release ROCm configuration.
- Capture and replay the shared expert FFN+gate pair with fixed inputs and compare against the normal path.

#### `GDNRecurrenceStage`

Files:
- `src/v2/execution/compute_stages/stages/GDNRecurrenceStage.h`
- `src/v2/execution/compute_stages/stages/GDNRecurrenceStage.cpp`
- `src/v2/kernels/rocm/gdn/ROCmGatedDeltaNetKernels.hip`

Required work:
- Replace the current header comment/logic that says prefill is not capturable. ROCm prefill uses `chunk_forward()`, which can be graph-capturable if its scratch and state buffers are stable.
- Make `isGraphCapturable()` return true for release ROCm `seq_len > 1` only when the ROCm GDN kernel has already allocated all chunk-forward scratch for the requested dimensions.
- Audit `ROCmGatedDeltaNetKernels.hip` for `hipMalloc`, `hipFree`, host copies, and stream synchronizes in the `chunk_forward()` path. Any lazy allocation must happen during warmup, and capture must assert capacity.
- Ensure recurrence state and output buffers are persistent device buffers. Host state inspection or debug dumps must stay disabled during capture.

Acceptance gate:
- Add a GDN prefill capture test that performs warmup, captures `chunk_forward()`, replays, and compares with normal prefill output.
- Add a negative readiness test that proves prefill GDN is not capturable before scratch warmup.

---

## Tier 1 Phase 2: Eliminate Capture-Blocking Operations

Operations that are **illegal during graph capture**:

| Operation | Where | Fix |
|-----------|-------|-----|
| `hipMalloc` / `hipFree` lazy realloc | `groupTokensByExpertDevice` (write heads), `prepareExpertGroupsAsync` (group buffers), `ensureGroupedPrefillScratchCapacity`, GDN chunk scratch | Warmup should trigger these, but capture must add explicit capacity assertions and readiness predicates before every realloc path. |
| Sync D2H `hipMemcpy` | Synchronous grouping path and any host routing/snapshot/debug path | Prefill graph capture must stay on async grouping/device-only routing paths. Add tests/tracing to prove no D2H is entered. |
| `hipStreamSynchronize` | Synchronous grouping, debug/snapshot paths, possible legacy helper paths | Keep capture on stream-ordered async paths and fail if a sync path would be used. |
| Host-side conditionals on GPU results | Host routing output, histograms, snapshots, host decisions based on expert counts | Ensure capture decisions are based only on host metadata/capacity, not newly produced GPU results. |

**New guard**: During capture, assert that no realloc path is taken:
```cpp
if (isGraphCaptureActive()) {
    LLAMINAR_ASSERT(total_slots <= group_slots_cap_, "Prefill graph capture: buffer undersize");
}
```

### Required Code Changes

Files:
- `src/v2/kernels/rocm/moe/ROCmMoEKernel.cpp`
- `src/v2/kernels/rocm/moe/ROCmMoEKernel.h`
- `src/v2/kernels/rocm/gdn/ROCmGatedDeltaNetKernels.hip`
- `src/v2/backends/rocm/ROCmDeviceContext.*` if stream/capture state needs to be exposed
- `src/v2/execution/local_execution/graph/GraphCaptureGuard.h`

Required work in `ROCmMoEKernel.cpp`:
- Include/use `GraphCaptureGuard.h` and `utils/Assertions.h` (or return a hard failure that propagates) in every lazy allocation path used by grouped prefill.
- In `groupTokensByExpertDevice()`:
    - `hipMemsetAsync()` of counts/write heads is graph-safe and should stay.
    - The `d_write_heads_` realloc path is not graph-safe. If capture is active and `num_experts > max_write_heads_experts_`, assert/fail before `hipFree`/`hipMalloc`.
- In `prepareExpertGroupsAsync()`:
    - The grouping-buffer realloc path is not graph-safe. If capture is active and `total_slots > group_slots_cap_` or `num_experts > group_experts_cap_`, assert/fail before allocation.
    - Keep the async path free of D2H copies and `hipStreamSynchronize()`.
    - Ensure it does not call the synchronous `prepareExpertGroups()` alternate path while capture is active.
- In `ensureGroupedPrefillScratchCapacity()`:
    - If capture is active and any prefill scratch buffer would be reallocated, assert/fail before `hipFree`/`hipMalloc`.
    - Expose a read-only readiness helper such as `hasGroupedPrefillScratchCapacity(total_slots, d_model, intermediate)` for stage predicates.
- In `executeGroupedPrefillPipeline()`:
    - Keep `hipMemsetAsync(d_output, ...)` because it is graph-safe and required per replay.
    - Ensure `hidden->ensureOnDevice()` and `output->ensureOnDevice()` cannot allocate during capture. Prefer explicit pointer/capacity assertions when capture is active.
    - Ensure descriptor validation is host-only metadata validation and does not trigger GPU/host transfers.

Required work in GDN/embedding-adjacent paths:
- Add the same capture-active allocation guards around ROCm GDN `chunk_forward()` scratch.
- Audit `ROCmEmbeddingKernelT.cpp` for capture-time workspace allocation. The workspace token buffer must be allocated in warmup via the existing workspace path before capture begins.

Acceptance gate:
- Add a ROCm test that begins HIP stream capture around the grouped prefill MoE kernel sequence (`prepareExpertGroupsAsync()` + `executeGroupedPrefillPipeline()`) after warmup. It must instantiate and replay successfully.
- Run the same test with a deliberately smaller capacity and verify it fails with the new capture-capacity assertion instead of entering `hipMalloc`/`hipFree`.
- Run with transfer tracing enabled (`LLAMINAR_TRACE_TRANSFERS=1`, `LLAMINAR_TRACE_TRANSFERS_ONLY_D2H=1`) and confirm no D2H transfers occur during capture/replay.
- Run `V2_Integration_HIPGraphCapture` and the ROCm MoE grouped prefill tests in a release/non-snapshot ROCm build.

---

## Tier 1 Phase 3: Dynamic Parameter Mechanism for Prefill

Between graph replays, only **input token IDs** change (for server use). For benchmark, even these are constant.

**Pinned host buffer approach** (same pattern as decode's RoPE/Attention):

```cpp
struct PrefillDynamicParams {
    int* pinned_input_ids;       // Pinned host → captured H2D to device input_ids buffer
    int  seq_len;                // Fixed for this graph instance
};
```

The desired end state is that capture records the Embedding stage's H2D copy of `input_ids` as a graph node. Before replay, we write new token IDs to the same pinned buffer, and the captured H2D node re-reads them. The current embedding code does not guarantee this because its preload path can copy token IDs before stage execution and then skip the H2D in `apply_tensor()`, so Tier 1 Phase 3 must make the contract explicit.

For exact-length prefill graphs, GDN recurrence state and KV cache tensors remain persistent device buffers that kernels read/write in-place. Host-side metadata is still dynamic: KV append stages need post-replay callbacks, and Phase 6 bucketed replay adds real-token metadata so padded rows do not advance KV/GDN state.

### Required Code Changes

Files:
- `src/v2/execution/compute_stages/stages/EmbeddingStage.{h,cpp}`
- `src/v2/kernels/rocm/ROCmEmbeddingKernelT.cpp`
- `src/v2/execution/local_execution/engine/ForwardGraphTypes.h`
- `src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp`
- `src/v2/execution/compute_stages/stages/KVCacheAppendStage.h`

Important current behavior:
- `EmbeddingStage::hasDynamicParams()` and `updateDynamicParams()` already exist.
- `ROCmEmbeddingKernelT::setDynamicTokenIds()` already owns pinned host token storage and can async-copy tokens to the workspace token buffer.
- However, the current preload behavior can skip the token H2D copy inside `apply_tensor()`. A prefill graph that relies on the H2D node being recorded must make this explicit; otherwise token updates happen as a tiny prelaunch H2D outside the graph.

Choose and implement one token update contract:

| Option | Code Change | Pros | Cons |
|--------|-------------|------|------|
| Captured H2D node | Add a prefill graph mode where `ROCmEmbeddingKernelT::apply_tensor()` always records an async H2D from the same pinned token buffer during capture. Replay updates only write the pinned buffer. | Preserves the "one graph launch" model. | Requires care so `setDynamicTokenIds()` does not preload/skip the copy for this mode. |
| Prelaunch token upload | Keep current preload behavior. Before graph replay, call `EmbeddingStage::updateDynamicParams()` on the graph stream and then launch the graph after the H2D is ordered. | Smallest code change; token upload is only `seq_len * sizeof(int)`. | Replay is no longer literally one HIP call, though host overhead is still tiny. |

For the original throughput target, use the captured-H2D-node contract. If implementation risk is high, start with prelaunch token upload and record it as a measured compromise.

Additional dynamic-state requirements:
- `ForwardGraphCache::token_ids` is already stable for cached graphs. Prefill graph replay must update this cache-owned vector, not point embedding at a temporary request vector.
- `position_ids` must be stable for the captured `seq_len`; if position offset changes in server mode, it becomes another dynamic parameter.
- KV-cache append stages need replay callbacks. `KVCacheAppendStage` already has `needsOnGraphReplayed()` and `onGraphReplayed()`, but the monolithic graph capture path does not currently call replay callbacks after launch. Add callback collection/execution for prefill monolithic replay so host KV heads/counts advance correctly.
- GDN recurrence state lives in persistent device buffers; reset/sequence-boundary semantics must run before capture/replay just as normal prefill does. Padded bucket replay additionally requires the Phase 6 real-token GDN contract before it can be enabled.

Acceptance gate:
- Extend `tests/v2/unit/kernels/Test__KernelDynamicStateLifecycle.cpp` or `tests/v2/unit/execution/compute_stages/stages/Test__EmbeddingStage_GraphCapture.cpp` with a ROCm prefill case: capture embedding for `seq_len > 1`, replay with different token IDs, and verify output changes exactly as normal embedding would.
- Add a graph replay test containing `EmbeddingStage` plus `KVCacheAppendStage`; after two replays, host KV-cache head/count metadata must match normal execution.
- If using prelaunch token upload, add a benchmark counter/log line so the remaining host overhead is visible and bounded.

---

## Tier 1 Phase 4: `PrefillGraphCache` Implementation

```cpp
class PrefillGraphCache {
    struct Entry {
        std::unique_ptr<HIPGraphCapture> capture;
        int seq_len;
        bool ready = false;
    };

    // Key: seq_len (or bucket)
    std::unordered_map<int, Entry> cache_;
    hipStream_t capture_stream_;

    enum class Phase { None, Warmup, Capture, Ready };
    Phase phase_ = Phase::None;

public:
    bool hasGraph(int seq_len) const;
    void beginWarmup(int seq_len);          // Mark phase for this seq_len
    void beginCapture(int seq_len);         // hipStreamBeginCapture
    bool endCaptureAndInstantiate();        // hipStreamEndCapture + instantiate
    bool launch(int seq_len);              // hipGraphLaunch
};
```

### Required Code Changes

Files:
- new `src/v2/execution/local_execution/engine/PrefillGraphCache.{h,cpp}` or extensions to `ForwardGraphTypes.h`
- `src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp`
- `src/v2/execution/local_execution/graph/DeviceGraphExecutor_GraphCapture.cpp` if shared capture/replay helpers are added
- `src/v2/backends/rocm/HIPGraphCapture.{h,cpp}` only if additional metadata/callback hooks are needed
- `src/v2/utils/DebugEnv.h`

Recommended minimal design:
- Extend each `ForwardGraphCache` entry with a `PrefillGraphState` because the existing cache key already includes decode/prefill shape and `seq_len`.
- Introduce an explicit `PrefillGraphCacheKey` even if it initially wraps the existing cache key:
    - `int bucket_seq_len`
    - `DeviceId device_id`
    - `std::string domain_id` (`"single"` in Tier 1)
    - `uint64_t placement_epoch` (`0` in Tier 1)
    - `uint64_t topology_signature` (hash of stage types, collective-free topology, model/layer range)
- State fields:
    - `enum class Phase { Disabled, Cold, Warmed, Capturing, Ready }`
    - `int seq_len`
    - `PrefillGraphCacheKey key`
    - `std::unique_ptr<IGPUGraphCapture> capture`
    - `std::shared_ptr<IGPUDeviceContext> graph_ctx`
    - `GPUStreamHandle stream`
    - `std::vector<IComputeStage*> dynamic_param_stages`
    - `std::vector<IComputeStage*> replay_callback_stages`
    - `size_t node_count` if available from graph metadata/logging
- Reuse `ForwardGraphCache::gpu_stream`, `gpu_ctx`, `gpu_graph`, `gpu_graph_update_failures`, `token_ids`, `position_ids`, and `dynamic_param_stages` where possible instead of adding parallel ownership.

Capture/replay flow:
1. **Build + warmup run**: the initial bucketed prefill cache miss builds the reusable forward graph, preflights it, binds every GPU stage to the dedicated explicit prefill capture stream, and executes normal prefill once. This warms BufferArena device buffers, ROCm/CUDA grouping buffers, descriptor tables, shared expert scratch, GDN scratch, embedding workspace, and KV-cache device state.
2. **Readiness preflight**: before arming warmup, verify every stage in the prefill graph is capturable or warmup-dependent-capturable, there are no collective nodes, the backend is a supported GPU backend, snapshots are off, `seq_len >= LLAMINAR_PREFILL_GRAPH_MIN_SEQ`, MoE rebalancing is off/absent, and the graph is in the supported single-device/full-local scope.
3. **Capture run**: begin capture on the chosen explicit stream, execute the same prefill stages on that stream, end capture, instantiate, and mark the entry ready. Captured kernels are then launched once immediately so the capture request produces logits and advances device state.
4. **Replay**: update dynamic params, launch the executable graph, run replay callbacks, and perform only the existing final coherence/sync required to read logits.
5. **Failure handling**: any preflight, capture, instantiate, update, replay, or callback failure is fatal for this execution. Emit `LOG_ERROR` with the exact phase/stage/reason and throw/propagate failure; do not silently run the normal prefill path.

Executor integration details:
- Do not use the current `executeWithGraphCapture()` as-is for cached prefill replay; it captures/instantiates/launches per call and does not provide the desired persistent replay cache.
- Either add lower-level executor methods (`captureStagesToGraph()`, `launchCapturedGraph()`) or keep the capture orchestration in `PrefillGraphCache` using `IGPUGraphCapture` directly.
- Preflight must reject any stage with `isGraphCapturable() == false`. This is the safety net that prevents silently capturing unsupported MoE sparse/collective paths. Under `LLAMINAR_GPU_GRAPHS=1`, an unexpected rejection in an otherwise eligible prefill graph should be a loud configuration/runtime error, not a continuation onto normal execution.
- Monolithic replay must call `onGraphReplayed()` for stages that need it. The existing segmented replay path has similar callback concepts; prefill needs them too.
- Add an invalidation hook callable from `DeviceGraphOrchestrator::applyExpertMasks()`, `setExpertReplicaSet()`, and future `MoERuntimeTable` placement flips. In Tier 1, any such call while a prefill graph entry is ready should invalidate and fail the next replay with a clear `active_moe_rebalance_or_placement_mutation` reason. Tier 2 will convert this hook into placement-epoch advancement.

Environment variables in `DebugEnv.h`:
```cpp
struct ExecutionConfig {
    bool gpu_graphs = false;                // existing LLAMINAR_GPU_GRAPHS master switch
    int prefill_graph_min_seq = 256;        // LLAMINAR_PREFILL_GRAPH_MIN_SEQ
    bool prefill_graph_trace = false;       // LLAMINAR_PREFILL_GRAPH_TRACE
    bool prefill_graph_buckets = true;      // LLAMINAR_PREFILL_GRAPH_BUCKETS=0 opts out, Tier 1 Phase 6
};
```

Do not add a separate prefill graph master flag. Prefill graph capture is enabled by default whenever the existing `LLAMINAR_GPU_GRAPHS` debugenv flag enables GPU graphs and the prefill graph meets the supported-scope gates. `LLAMINAR_PREFILL_GRAPH_MIN_SEQ`, `LLAMINAR_PREFILL_GRAPH_TRACE`, and `LLAMINAR_PREFILL_GRAPH_BUCKETS` are tuning/diagnostic controls, not feature gates.

Acceptance gate:
- Add unit tests for the cache state machine: cold -> warmed -> ready -> replay, min-seq gating, active rebalance rejection, non-ROCm/snapshot/collective rejection, and stage-preflight rejection. Rejections and replay failures must report a fatal error path, not continue onto normal execution.
- Add a cache-key test proving Tier 1 records `domain_id="single"` and `placement_epoch=0`, and that a mask/replica invalidation hook prevents replay of a stale graph.
- Add an integration graph test with a small synthetic ROCm compute graph containing a dynamic stage and a replay-callback stage. Verify capture happens once and the second execution is a replay, not a recapture.
- Existing `V2_Integration_HIPGraphCapture` and `V2_Integration_GPUGraphCaptureExecution` must continue to pass.

---

## Tier 1 Phase 5: Integration with `ForwardExecutionEngine`

Replace the current prefill fast path:

```cpp
if (!is_decode) {
    const int seq_len = input.seq_len;

    if (prefill_graph_cache_.hasGraph(seq_len)) {
        // Replay
        updatePrefillDynamicParams(input);
        success = prefill_graph_cache_.launch(seq_len);
    } else if (prefill_graph_cache_.isCapturing(seq_len)) {
        // Capture run (stages execute into capture stream)
        success = executor_.executeFastDecode(*forward_cache.graph, ctx, ...);
        prefill_graph_cache_.endCaptureAndInstantiate();
        // First launch is already done (capture run executed the kernels)
    } else {
        // Warmup run (triggers all lazy allocs)
        success = executor_.executeFastDecode(*forward_cache.graph, ctx, ...);
        prefill_graph_cache_.beginCapture(seq_len);
        // Next call with same seq_len will be capture phase
    }
}
```

**Benchmark flow** (4 runs: 1 warmup + 3 measured):
1. Run 1 (warmup): Normal execute → triggers allocs → arm capture
2. Warmup epilogue: Capture run (hidden cost, not measured)
3. Run 2-4 (measured): All graph replays → ~441ms each → **1349 tok/s**

### Required Code Changes

Files:
- `src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp`
- `src/v2/execution/local_execution/engine/ForwardGraphTypes.h`
- benchmark runner/mode files under `src/v2/app/modes/` or `src/v2/utils/` that own warmup/measurement loops
- `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp` only if graph construction needs to expose prefill runtime-table readiness more explicitly

Required work in `ForwardExecutionEngine`:
- In the prefill branch, use the existing `debugEnv().execution.gpu_graphs` / `LLAMINAR_GPU_GRAPHS` switch. Do not add or check a separate prefill graph master flag.
- Reject Tier 1 graph capture when the MoE rebalance controller is active (`OBSERVE` or `DYNAMIC`) or when any expert mask/replica placement mutation has occurred. Tier 1 acceptance should run with `LLAMINAR_MOE_REBALANCE=off` or equivalent config.
- Reject graph capture when snapshots are active. The snapshot path intentionally uses full execution and may perform host inspection/dumps.
- Reject if `forward_cache.collective_nodes` is non-empty. Monolithic prefill graph capture should not include MPI/NCCL/RCCL collectives in the first implementation.
- Before capture/replay, ensure the graph stream/context is stable and that `host.ensureDeviceWorkspaceAllocated()` has run for the relevant workspace buffers.
- On warmup completion, call the prefill graph preflight. If it passes, arm capture for the next same-shape run.
- On capture completion, record a clear log message with seq_len, stage count, node count if available, and whether the capture run output is being used.
- On replay, call dynamic param updates and replay callbacks, then mark output/coherence state exactly as the normal fast path does.
- On preflight/capture/replay failure after `LLAMINAR_GPU_GRAPHS` has selected this path, log the exact reason and fail the request/process. Do not execute normal prefill instead.

Benchmark integration:
- Keep the measured benchmark loop clean. After the benchmark warmup run, if prefill graph capture is enabled and the graph is only warmed, run one hidden same-shape capture pass before measured iterations begin.
- If the benchmark harness cannot add a hidden capture pass cleanly, the first measured iteration must be labeled as capture and excluded from throughput averaging when graph capture is enabled.
- Compare performance with the same runtime placement and MPI bootstrap settings on both baseline and graph-capture runs. Use `--no-mpi-bootstrap` only for profiling/debug investigations, not for canonical benchmark claims.

Acceptance gate:
- Correctness: run Qwen3.5 MoE single-device prefill/inference with `LLAMINAR_MOE_REBALANCE=off`, `LLAMINAR_GPU_GRAPHS=0` and `LLAMINAR_GPU_GRAPHS=1`; generated greedy tokens and top-1 logits for the first decode step must match within existing parity tolerance.
- Parity: run the relevant Qwen3.5 MoE single-device parity test in the ROCm release/non-snapshot configuration, or a dedicated release parity target if integration snapshots make graph capture intentionally unavailable.
- Observability: logs must show `warmup`, `capture`, and `replay` phase transitions for a repeated `seq_len`. Failure logs must name the exact phase/stage/reason and terminate; there should be no message implying normal prefill execution will continue.
- Performance: for `seq_len >= 256` on the target Qwen3.5 MoE model, measured prefill wall time should be within 3-5% of GPU time, host dispatch overhead should drop from tens of milliseconds to low single-digit milliseconds, and prefill throughput should improve by at least 10% versus the same build/configuration with prefill graph capture disabled.

---

## Tier 1 Phase 6: Bucketed Graphs and Chunking Primitive

For variable-length prompts in server mode:

```
Buckets: 64, 128, 256, 384, 512, 544, 576, 608, 640, 672, 704, 736, 768, 1024, 1280, 1536, 2048, 2560, 3072, 4096
```

- Pad input to next bucket with `pad_token_id` (attention causal mask prevents contamination)
- Cache one graph per bucket/device, capped by `LLAMINAR_PREFILL_GRAPH_MAX_BUCKETS` (default 10)
- Capture cost: ~550ms per bucket, amortized over hundreds of requests
- Memory cost: ~50MB per graph instance (kernel metadata, not tensor data)

### Phase 6 Accepted Status

Phase 6 is accepted for Tier 1 bucketed single-device prefill graph capture. The
runtime now distinguishes real-token length from bucket length throughout the
bucketed path, fails loudly for unsupported graph shapes, and keeps bucket
eviction/recapture observable through logs and telemetry.

Implemented code surface:
- Bucket selection, token padding, absolute position IDs, chunk planning, and `ForwardExecutionEngine::runPrefillChunk()` all flow through shared helpers and `ForwardInput` real/bucket metadata.
- Bucketed `ForwardGraphSignature` values are keyed by bucket length and device. `LLAMINAR_PREFILL_GRAPH_MAX_BUCKETS` caps reusable bucketed forward graphs at the `ForwardExecutionEngine` level across bucket lengths/devices, while `PrefillGraphCache` retains per-entry lifecycle accounting.
- `LMHeadStage`, `HiddenStateRowSelectStage`, `KVCacheAppendStage`, `GDNRecurrenceStage`, and `ShortConv1dStage` consume replay metadata so padded rows do not drive logits, KV host counts, or GDN/short-conv recurrent state.
- Bucketed graph preflight still rejects unsupported collective, sparse overlay, snapshot, CPU-participation, or placement-mutation paths instead of silently falling back to normal prefill.

Accepted coverage:
- CPU, CUDA, and ROCm attention padding parity cover hostile bucket-tail rows.
- CUDA and ROCm GDN padded real-length tests cover recurrence/short-conv state.
- CUDA and ROCm row-select/LM-head graph-capture tests cover last-real-token readout across replay.
- CUDA and ROCm `PrefillGraphCacheExecution` tests cover exact capture/replay, padded same-bucket reuse across real lengths, raw/server-style same-bucket reuse, KV real-token advancement, and cross-bucket eviction/explicit recapture under `LLAMINAR_PREFILL_GRAPH_MAX_BUCKETS=1`.

Operational notes:
- Evictions are logged at `INFO` with bucket length, device, and configured cap.
- `prefillGraphCacheSnapshot()` reports engine-level bucket evictions even after the evicted bucket's per-entry cache has been destroyed.
- Tier 2 work remains about distributed/collective bucket graphs and ExpertParallel rebalance domains, not Tier 1 Phase 6 closeout.

---

## Tier 1.5: ExpertParallel Domain-Scoped Rebalance Controller

Tier 1.5 is a terminology and ownership cleanup that should land before Tier 2. The goal is to stop thinking of rebalance as a special CPU socket feature and instead define it as a controller attached to one **ExpertParallel rebalance domain**.

The domain owns a set of routed experts, a set of participant devices, a placement map, and one histogram window. Rebalancing spreads hot expert load across the participants in that domain. The participants may be CPU NUMA ranks, local GPUs, ROCm/CUDA devices, or overlay expert-domain participants. The controller should not care; it should operate on stable domain participant ids and `DeviceId`/`GlobalDeviceAddress` metadata.

The current two-socket CPU GlobalTP behavior becomes one instance of the generic model:

```
domain_id: cpu_global_tp
participants:
    0 -> cpu:0 on rank 0
    1 -> cpu:1 on rank 1
placement:
    expert_id -> participant_id
```

Single-device MoE becomes another instance:

```
domain_id: single
participants:
    0 -> rocm:0
mode:
    observe only; never rebalance because there is no alternate participant
```

Tiered overlay domains then become natural consumers of the same contract:

```
domain_id: routed_gpu_tier
participants:
    0 -> rocm:0
    1 -> rocm:1
placement:
    layer/expert -> participant_id
collectives:
    sparse dispatch/return outside the controller
```

### Tier 1.5 Phase 1: Domain Model and Vocabulary

Files:
- `src/v2/execution/moe/MoERebalanceController.{h,cpp}`
- `src/v2/execution/moe/DecodeExpertHistogram.{h,cpp}`
- `src/v2/execution/moe/MoEExpertParallelPlan.h`
- `src/v2/config/ExecutionDomainDefinition.h`
- `src/v2/execution/config/RuntimeConfig.h`

Required work:
- Introduce value types that describe the rebalance domain explicitly:
    - `ExpertParallelDomainId` or string domain name
    - `ExpertParallelParticipantId` as a dense 0..N-1 id inside the domain
    - participant metadata containing `GlobalDeviceAddress`, local `DeviceId`, world rank, optional NUMA node, and backend/domain kind
    - placement map fields named `expert_to_participant` and, where needed, `layer_expert_to_participant`
- Rename public controller concepts from socket language to participant/domain language. Prefer additive migration first:
    - `computeExpertMasks(int participant_id)` replaces `computeExpertMasks(int socket_id)`
    - `owner_participant` replaces `owner_socket`
    - `num_participants` replaces `num_sockets`
    - `assignForToken(..., participant_id, ...)` replaces `my_socket_id`
    - log lines say `domain`, `participant`, and `device`, not `socket`, unless they are explicitly reporting CPU topology
- Keep compatibility wrappers temporarily if needed, but make new call sites use domain vocabulary.
- Rename `DecodeExpertHistogram` only if the churn is acceptable; otherwise introduce a `DomainExpertHistogram` wrapper/alias and keep the implementation. The important contract is that histogram counts are scoped to a domain, not globally shared across unrelated ExpertParallel domains.
- Add `MoERebalanceDomainConfig` inside runtime graph config, carrying domain id, participant list, initial placement, mode, window sizes, replica/cache policy, and whether prefill histogram sources are enabled.

Acceptance gate:
- Unit-test domain construction for CPU GlobalTP, LocalTP, single-device, and overlay domain participants. Each test must produce stable participant ids and matching `DeviceId`/rank metadata.
- Unit-test that the old two-CPU-rank placement produces the same masks through the new participant APIs as the current socket APIs.
- Static grep gate: new code added for rebalance should not introduce new public `socket_id` terminology except in compatibility wrappers or CPU topology adapters.

### Tier 1.5 Phase 2: Wiring and Ownership

Files:
- `src/v2/execution/factory/InferenceRunnerFactory.cpp`
- `src/v2/execution/runner/OrchestrationRunner.cpp`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.{h,cpp}`
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.{h,cpp}`
- `src/v2/execution/mpi_orchestration/ExecutionPlanBuilder.cpp`
- `src/v2/execution/mpi_orchestration/RankExecutionPlan.h`

Required work:
- Move controller construction from implicit `world_size`/local-TP heuristics into an explicit domain resolver. Given the execution plan and optional overlay plan, produce zero or more `MoERebalanceDomainConfig` objects.
- Attach exactly one controller per rebalance domain. Do not rely on `RankOrchestrator::moeRebalanceController()` returning the first DGO controller for multi-domain setups.
- For single DGO/single participant domains, construct an observe-capable controller when requested, but mark `can_rebalance=false` and do not attempt mask mutation or weight transfer.
- For LocalTP, participants are the local TP devices in plan order; mask application fans out through `RankOrchestrator` using participant id, not incidental vector index without a domain id.
- For CPU `-d cpu`, preserve current behavior by resolving one CPU GlobalTP domain whose participant ids match the MPI ranks in that TP domain. The implementation must not use `mpi_ctx_->local_rank()` as a stand-in for global/domain participant id unless the domain resolver proves they are identical.
- For multi-node GlobalTP, use `global_tp_rank_in_domain` or a domain participation table as the participant id. This is the audit/fix for the current local-rank ambiguity.
- Expose controller lookup by domain id: `moeRebalanceController(domain_id)` and an iteration API for all active rebalance domains.

Acceptance gate:
- Unit-test controller wiring for:
    - single ROCm device: one observe-only domain, no rebalance application
    - `-d cpu`: one CPU GlobalTP domain, participant ids match MPI rank/domain rank
    - LocalTP: one local domain with one participant per local device
    - synthetic multi-node GlobalTP: participant ids do not alias when local ranks repeat
- Integration test that existing `-d cpu` MoE rebalance behavior is unchanged in masks, transfers, and generated greedy tokens.
- Negative test: if a placement update references a participant outside the domain, fail before any mask application.

### Tier 1.5 Phase 3: Domain Placement, Masks, and Replica Semantics

Files:
- `src/v2/execution/moe/MoERebalanceController.{h,cpp}`
- `src/v2/execution/moe/ExpertWeightTransfer.*`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp`
- `src/v2/execution/compute_stages/stages/MoEExpertComputeStage.{h,cpp}`

Required work:
- Treat expert masks as domain-local placement views: `mask[layer][expert] == participant owns/serves this expert`.
- Define separate concepts for:
    - base owner participant
    - replica participant(s)
    - active compute participant for a token when replicas exist
    - prefill owner participant when prefill chooses not to use dynamic per-token replica dispatch
- Update `ExpertReplicaSet` to use participant/domain terminology and to store a domain id. Replica placement must not be applied to a different domain by accident.
- Make weight transfer domain-aware. CPU cross-rank transfers can keep MPI send/recv, while local-device transfers and future overlay transfers should go through domain-specific transfer implementations.
- Apply masks via domain participant id. `DeviceGraphOrchestrator::applyExpertMasks()` should know which domain/participant the masks came from so graph caches can invalidate by `(domain_id, placement_epoch)`.
- Make placement changes publish a domain-local placement epoch. Tier 1 graph capture invalidation can remain conservative; Tier 2 will use this epoch to coordinate chunk graph replay.

Acceptance gate:
- Unit-test mask computation for base ownership, per-layer LPT placement, replicas, and GPU hot-cache masks using participant ids.
- Unit-test that applying a mask for domain A cannot mutate stages bound to domain B.
- Existing replica tests must pass after vocabulary migration.
- Add an epoch test: every successful mask/replica/base-placement update increments only the affected domain epoch.

### Tier 1.5 Phase 4: Decode and Prefill Histogram Contract

Files:
- `src/v2/execution/moe/DecodeExpertHistogram.{h,cpp}` or new `DomainExpertHistogram.*`
- `src/v2/execution/moe/MoERuntimeTable.{h,cpp}`
- `src/v2/execution/compute_stages/stages/MoERoutingStage.cpp`
- `src/v2/execution/runner/OrchestrationRunner.cpp`
- ROCm MoE routing/grouping kernels for runtime-table histogram accumulation

Required work:
- Keep decode as the first production histogram source. `MoERoutingStage` already records `seq_len == 1` host counts and ROCm runtime tables can merge device counts.
- Add an explicit histogram source type: `DecodeToken`, `PrefillChunk`, `SyntheticTest`, or similar. The controller should know whether a window was filled by decode tokens, prefill chunks, or both.
- Formalize prefill support as a chunk-boundary source. Prefill should not trigger mid-graph or mid-stage rebalancing; counts are merged only after a chunk completes.
- Window accounting must count real tokens, not padded bucket tokens, and must remain per-domain.
- Single-device domains should collect histograms in `OBSERVE`/profiling mode but `shouldRebalance()` must return false because `participant_count <= 1`.
- Rebalance decisions should expose reason codes: `window_not_full`, `single_participant_observe_only`, `mode_off`, `dynamic_disabled_for_domain`, `ready`.

Acceptance gate:
- Unit-test that single-device `DYNAMIC` configuration degrades to observe-only behavior with a clear reason and no mask mutation.
- Decode regression: after `window_size` decode tokens, a multi-participant domain becomes ready for rebalance exactly as before.
- Prefill histogram unit test: two chunks with padding merge only real-token counts into the domain window.
- Negative test: attempting to apply a rebalance while a graph capture/replay is active or while a chunk is incomplete fails loudly.

### Tier 1.5 Phase 5: Overlay Integration Contract

Files:
- `src/v2/execution/moe/MoEExpertParallelPlan.h`
- `src/v2/execution/moe/MoEExpertOverlayExecutionPlan.*`
- `src/v2/execution/moe/MoEExpertParallelPlanner.cpp`
- `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp`
- `src/v2/config/OrchestrationConfigParser.cpp`

Required work:
- Define which overlay domains are eligible for runtime rebalance:
    - routed expert domains: eligible
    - continuation/base dense domains: not expert-placement rebalance domains
    - shared expert domain: only eligible if it owns routed/shared expert placement explicitly; otherwise observe-only or disabled
- Attach controllers to overlay routed tiers by domain id. A tiered overlay can have multiple rebalance domains, each with independent participants, placement, histogram, and epoch.
- Make `RoutedTierRebalanced` residency policy consume the same domain-scoped histogram/placement machinery instead of being a separate planner-only concept.
- Define collective boundaries as outside the controller. The controller chooses placement and publishes masks/epochs; sparse dispatch/return collectives execute the data movement implied by that placement.
- Add diagnostics that render, per domain: participant list, mode, can_rebalance, current epoch, placement policy, window status, and last rebalance result.

Acceptance gate:
- Dry-run overlay config renders each routed ExpertParallel domain with participants and rebalance mode.
- Planner test: `RoutedTierRebalanced` with a histogram produces the same placement through the new domain controller/planner bridge as the current planner-only path.
- Negative test: trying to attach a runtime rebalance controller to the continuation/base dense domain fails validation with a clear message.
- Integration smoke: overlay execution without graph capture can run with domain controllers in observe mode and produce unchanged outputs.

Tier 1.5 exit criteria:
- Public rebalance APIs and diagnostics use domain/participant/device language.
- Existing CPU GlobalTP `-d cpu` behavior is preserved as one domain-scoped instance.
- Single-device MoE is formalized as observe-only/no-rebalance.
- LocalTP and multi-node GlobalTP have unambiguous participant ids.
- Overlay routed expert domains have a defined controller attachment point, even if Tier 2 graph capture is not implemented yet.

---

## Tier 2: Expert-Parallel Overlay Prefill Graph Capture

Tier 2 extends the Tier 1 machinery and the Tier 1.5 domain-scoped rebalance controller to expert-parallel collective domains in the tiered overlay system. The core idea is that prefill is no longer one monolithic graph over the whole prompt. It becomes a sequence of fixed-size bucket/chunk graph executions:

```
chunk 0: prefill graph replay for tokens [0, n)
         sync/merge prefill expert histogram
         optional rebalance + placement epoch flip

chunk 1: prefill graph replay for tokens [n, 2n)
         sync/merge prefill expert histogram
         optional rebalance + placement epoch flip

...
```

Rebalancing is allowed **between** graph executions only. It is never allowed during capture or replay. This keeps graph replay deterministic while allowing placement to adapt over a long prompt at a defined interval.

Tier 2 does not relax the fail-fast rule: if `LLAMINAR_GPU_GRAPHS=1` selects a Tier 2 graph path and the graph cannot satisfy its preflight/capture/replay contract, the request/process fails with a phase/domain/stage reason.

### Tier 2 Phase 1: Prefill Chunk Scheduler and Histogram Bridge

Files:
- `src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp`
- `src/v2/execution/local_execution/engine/ForwardGraphTypes.h`
- `src/v2/execution/moe/DecodeExpertHistogram.{h,cpp}` or a renamed/generalized histogram owner
- `src/v2/execution/moe/MoERuntimeTable.{h,cpp}`
- ROCm MoE routing/grouping kernels that can accumulate prefill expert counts

Required work:
- Promote the Tier 1 bucket primitive into a scheduler that can run a long prefill as multiple chunks with stable bucket sizes.
- Add a chunk policy: fixed token interval, bucket list, and minimum/maximum rebalance interval. The policy should be explicit in logs and config, for example `prefill_rebalance_interval_tokens` once Tier 2 is enabled.
- Collect expert counts for prefill chunks without host-side per-token routing decisions during graph replay. Preferred path: routing/grouping kernels write per-layer expert counts into `DeviceMoELayerRuntime` scratch or a domain-local counter buffer.
- After each graph chunk completes, sync/merge those counts into the existing rebalance controller outside graph replay. `DecodeExpertHistogram::mergeLayerCounts()` already has the right shape, even if the name is decode-centric.
- Reset per-chunk runtime counters after successful merge so repeated syncs are idempotent.
- Ensure padded bucket tokens do not contribute to expert histograms or rebalance decisions.

Acceptance gate:
- Unit-test chunk histogram merge: two chunks with known routed experts must produce the same per-layer counts as a direct host-side reference.
- Integration test with graph replay enabled: run two prefill chunks, merge counts between chunks, and verify the controller window advances only by real tokens, not padded bucket tokens.
- Negative test: attempting to rebalance while a chunk graph capture/replay is active must fail loudly.

### Tier 2 Phase 2: Placement Epochs and Runtime-Table Bank Flips

Files:
- `src/v2/execution/moe/MoERuntimeTable.{h,cpp}`
- `src/v2/execution/compute_stages/stages/MoEExpertComputeStage.{h,cpp}`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp`
- `src/v2/execution/runner/OrchestrationRunner.cpp`

Required work:
- Introduce a formal MoE placement epoch that advances whenever expert masks, replicas, owner maps, or runtime-table banks change.
- Make graph caches observe that epoch. There are two valid implementation levels:
    - Conservative: include `placement_epoch` in `PrefillGraphCacheKey` and recapture on epoch change.
    - Target: keep the graph executable reusable across epoch flips by making kernels read placement through stable `DeviceMoELayerRuntime*` pointers and double-buffered banks. In that model, the cache key includes only the runtime-table schema/capacity signature, while replay sees the new active bank values.
- Use `MoERuntimeTable::prepareInactiveBank()` and `flipActiveBank()` outside capture/replay to publish new placement to device memory.
- Ensure every active local expert descriptor in a new bank has resident, prepared gate/up/down payloads before the flip.
- Remove hidden stage-local placement state from the captured path. If kernels read runtime table masks/descriptors, host-side `expert_mask` and grouped descriptor table ids cannot be the source of truth for captured replay.
- Make `applyExpertMasks()`, `setExpertReplicaSet()`, and overlay owner-map updates notify the graph cache/epoch system consistently.

Acceptance gate:
- Unit-test epoch behavior: after a mask or replica update, ready graph entries either invalidate or observe a higher placement epoch.
- ROCm runtime-table test: capture/replay a graph that reads a stable runtime table pointer, flip the active bank between replays, and verify output changes according to the new local compute mask without changing graph kernel arguments.
- Negative test: a placement epoch flip with missing prepared expert descriptors must fail before graph replay.

### Tier 2 Phase 3: Domain-Local Graph Caches for Overlay Participants

Files:
- `src/v2/execution/moe/MoEExpertOverlayExecutionPlan.*`
- `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp`
- `src/v2/execution/local_execution/engine/PrefillGraphCache.{h,cpp}` or `ForwardGraphTypes.h`
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp`
- `src/v2/execution/runner/OrchestrationRunner.cpp`

Required work:
- Create graph-cache entries per overlay domain/participant, not just per root device. Cache key fields should include domain name/id, participant id, bucket length, placement epoch/schema, and layer range.
- Distinguish continuation/base-domain graphs from expert-domain graphs. A tiered overlay prefill chunk may execute several captured segments separated by dispatch/return collectives.
- Use domain participant index for expert masks and placement, not `mpi_ctx_->local_rank()` unless the domain contract proves they are identical.
- Ensure domain-local caches share the same prefill chunk schedule and placement epoch so all participants replay compatible graphs for the same chunk.
- Add a cross-rank/domain preflight handshake: all participants must agree on bucket length, placement epoch, graph topology signature, and collective keys before any participant starts capture/replay.

Acceptance gate:
- Unit-test cache keys for two domains with the same bucket length but different participant ids; they must not alias.
- Integration/dry-run test for a tiered overlay plan: all participants report the same chunk id and placement epoch, and incompatible signatures fail before any graph launch.
- GlobalTP audit test: mask application uses domain/global participant index, not node-local rank, for multi-node domains.

### Tier 2 Phase 4: Capturable Segments Around Sparse Overlay Collectives

Files:
- `src/v2/execution/compute_stages/stages/MoEExpertDispatchStage.*`
- `src/v2/execution/compute_stages/stages/MoESparseDispatchStage.*`
- `src/v2/execution/compute_stages/stages/MoELocalExpertStage.*`
- `src/v2/execution/compute_stages/stages/MoESparseReturnReduceStage.*`
- `src/v2/execution/compute_stages/stages/MoEExpertParallelReduceStage.*`
- `src/v2/execution/moe/MoEOverlaySparseCollective.*`
- `src/v2/execution/local_execution/graph/DeviceGraphExecutor_GraphCapture.cpp`

Required work:
- Keep sparse overlay collectives outside monolithic HIP graph capture until the collective backend itself has a graph-safe contract.
- Split the overlay prefill chunk into explicit segments:
    - captured base/route segment
    - manual sparse dispatch collective
    - captured expert-domain local compute segment(s)
    - manual sparse return-reduce collective
    - captured continuation segment
- Preserve `MoEOverlayCollectiveResult::collective_complete` gating. Follow-on movement such as continuation TP import/broadcast must not run until the relevant collective is complete.
- Convert host-staged sparse row metadata to fixed-capacity device-resident descriptors where possible. Until then, those stages remain manual boundaries and `isGraphCapturable() == false` is correct.
- For each manual collective boundary, define the exact tensors/coherence states that must be ready before the next captured segment launches.

Acceptance gate:
- Segmented graph execution test with mock sparse collectives: verify captured segments replay, collectives execute exactly once per chunk, and continuation work is gated on `collective_complete`.
- Negative test: if a collective reports incomplete/failed, the next captured segment must not launch.
- Integration test comparing segmented overlay prefill output against the existing non-captured overlay path for a small model/config.

### Tier 2 Phase 5: Rebalance Between Prefill Graph Executions

Files:
- `src/v2/execution/runner/OrchestrationRunner.cpp`
- `src/v2/execution/moe/MoERebalanceController.{h,cpp}`
- `src/v2/execution/moe/ExpertWeightTransfer.*`
- `src/v2/loaders/ExpertGemmRegistry.h`
- `src/v2/loaders/PreparedWeightStore.h`
- domain transfer code for overlay expert payloads

Required work:
- Add a prefill-chunk maintenance hook analogous to `maybeApplyMoERebalance()`, but called only after a chunk graph completes and histogram counts have been merged.
- Rebalance is allowed only at chunk boundaries. The scheduler must enforce: no active capture, no active replay, all participating domains at the same chunk boundary, and all required collectives complete.
- Apply rebalanced masks/replicas/owner maps by transferring or preparing expert payloads first, then publishing the new placement epoch/runtime-table bank.
- Decide whether the current graph executable remains reusable after a rebalance:
    - If runtime-table indirection is complete and capacities/schema are unchanged, replay the existing graph under the new placement epoch.
    - If stage topology, capacities, or descriptor schema changed, invalidate and recapture the next chunk graph explicitly.
- Keep raw expert weight release safe: do not release payloads needed for future rebalance arrivals or graph recapture.

Acceptance gate:
- End-to-end forced-rebalance test: run chunk 0, force a rebalance, run chunk 1, and compare logits/state against a non-captured chunked reference with the same rebalance schedule.
- Verify logs show `chunk`, `histogram_merge`, `rebalance`, `placement_epoch_flip`, and `graph_replay/recapture` in order.
- Negative test: a rebalance attempt during graph replay or before all domains reach the boundary must fail loudly.

### Tier 2 Phase 6: Performance and Correctness Gates

Required gates:
- Correctness: tiered overlay prefill graph capture must match the non-captured overlay path for fixed and rebalanced schedules.
- Rebalance determinism: the same prompt, bucket policy, and rebalance interval must produce identical placement epochs and generated tokens across runs.
- Collective safety: sparse dispatch/return collectives must preserve `collective_complete` ordering and avoid stale continuation transfers.
- Performance: graph-captured chunks should reduce host dispatch overhead inside each domain-local compute segment. Rebalance and collective overhead should be reported separately from graph replay time.
- Observability: each chunk log must include bucket length, real token range, domain id, participant id, placement epoch, capture/replay phase, and any recapture reason.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Hidden `hipMalloc` during capture | Capture fails | Assert-guard all alloc paths + warmup guarantees capacity |
| CK library internal state per-stream | Wrong results | Use same capture stream as warmup (existing decode pattern) |
| `hipMemsetAsync` inside captured graph | Unnecessary work on replay | Acceptable — `hipMemsetAsync` is fast (~1μs), write_heads reset is needed each time |
| Capture overhead too high for short prompts | Net negative for seq_len<128 | Only cache for seq_len >= 256 (configurable threshold) |
| Graph capture failure (ROCm driver bug) | Request/process fails | Fail fast with phase/stage/reason so driver/runtime issues are visible and reproducible |
| Monolithic replay skips KV host metadata updates | Decode starts from wrong KV position | Collect and call replay callbacks (`onGraphReplayed()`) after graph launch |
| Token H2D not actually captured | Replay uses stale or preuploaded token IDs | Make embedding token update contract explicit in Tier 1 Phase 3 and test changing-token replay |
| Integration build snapshots hide capturability | False failures in standard test build | Keep graph-capture acceptance in release/non-snapshot ROCm tests; retain snapshot negative tests |
| Tier 1 graph reused after expert placement mutation | Wrong experts compute after rebalance/mask update | Reject active rebalancing in Tier 1 and wire mask/replica updates to graph-cache invalidation hooks |
| Rebalance controller keeps socket/local-rank assumptions | Multi-node GlobalTP or overlay domains apply masks to the wrong participant | Tier 1.5 introduces explicit domain participant ids and tests local-rank aliasing |
| Domain-scoped controller accidentally mutates another domain | Overlay routed tier or TP domain changes unrelated expert placement | Carry `domain_id` through masks, replicas, transfers, epochs, and diagnostics; fail on domain mismatch |
| Single-device dynamic mode tries to rebalance | Unnecessary placement mutation or graph invalidation with no alternate participant | Formalize single-participant domains as observe-only with explicit reason codes |
| Tier 2 rebalance races graph replay | Placement changes while kernels are reading runtime tables | Allow rebalance only at chunk boundaries after all graph launches and overlay collectives complete |
| Runtime-table epoch flip changes descriptor schema/capacity | Captured graph reads incompatible descriptor/scratch layout | Include topology/capacity signature in cache key and recapture explicitly when schema changes |
| Sparse overlay collective ordering breaks under segmentation | Continuation stage sees stale or incomplete rows | Preserve `collective_complete` gating and never launch the next captured segment until the manual collective completes |

---

## Environment Variable Control

```bash
LLAMINAR_GPU_GRAPHS=1                  # Existing master enable for GPU graph execution
LLAMINAR_PREFILL_GRAPH_MIN_SEQ=256     # Minimum seq_len to capture (default: 256)
LLAMINAR_PREFILL_GRAPH_BUCKETS=1       # Bucketed capture default; set 0 to opt out
LLAMINAR_PREFILL_GRAPH_TRACE=0         # Verbose phase and failure logging
```

Tier 2 may add tuning controls for chunk intervals and bucket lists, but those should not be separate master feature flags. `LLAMINAR_GPU_GRAPHS` remains the graph-execution selector; Tier 2 controls should only shape eligible graph execution, such as `LLAMINAR_PREFILL_GRAPH_REBALANCE_INTERVAL_TOKENS`.

These should be parsed through `debugEnv()` in `src/v2/utils/DebugEnv.h`, not read with ad-hoc `std::getenv()` in hot paths.

---

## Implementation Order and Exit Criteria

### Tier 1 Exit Criteria

1. **Tier 1 Phase 1: stage readiness**
    - Implement capturability predicates and readiness helpers for MoE routing, MoE expert compute, shared expert, and GDN.
    - Exit when supported ROCm release single-device prefill stages return true only after warmup/readiness, while unsupported CPU/snapshot/rebalance/sparse/collective paths remain false.

2. **Tier 1 Phase 2: capture-safe ROCm kernels**
    - Add capture-time capacity assertions and eliminate capture-time lazy allocation/D2H/sync in grouped MoE prefill and GDN prefill.
    - Exit when direct HIP capture around the grouped prefill kernel sequence instantiates/replays and undersized buffers fail before allocation.

3. **Tier 1 Phase 3: dynamic state**
    - Finalize embedding token update semantics and monolithic replay callbacks for KV cache.
    - Exit when replay with changed token IDs and KV append metadata matches normal execution.

4. **Tier 1 Phase 4: cache infrastructure**
    - Add persistent prefill graph cache/state to the forward graph cache or a dedicated cache object, with future-proof key fields for domain and placement epoch.
    - Exit when tests prove one capture followed by repeated replays, with fatal errors for unsupported graphs and capture/replay failures.

5. **Tier 1 Phase 5: runtime integration and benchmark**
    - Wire the cache into `ForwardExecutionEngine` and keep measured benchmark iterations on graph replay, not capture.
    - Exit when Qwen3.5 MoE single-device, rebalance-off correctness/parity gates pass and prefill throughput improves by at least 10% for eligible prompts.

6. **Tier 1 Phase 6: buckets and chunk primitive**
    - Add bucket selection, padding, last-real-token logits handling, graph eviction, and chunked prefill bookkeeping.
    - Exit when bucket reuse is correct for variable prompt lengths and ineligible bucket requests fail loudly instead of silently using normal execution.

### Tier 1.5 Exit Criteria

1. **Tier 1.5 Phase 1: domain model and vocabulary**
    - Exit when controller configs, masks, replicas, histograms, and diagnostics use domain/participant/device terminology, with socket language isolated to CPU topology adapters or compatibility wrappers.

2. **Tier 1.5 Phase 2: wiring and ownership**
    - Exit when controllers are attached per ExpertParallel rebalance domain, with explicit participant ids for single device, CPU GlobalTP, LocalTP, and multi-node GlobalTP.

3. **Tier 1.5 Phase 3: placement, masks, and replicas**
    - Exit when mask/replica application is domain-scoped, weight transfers are domain-aware, and placement epoch increments are local to the affected domain.

4. **Tier 1.5 Phase 4: decode and prefill histogram contract**
    - Exit when decode remains behavior-compatible, prefill chunk histograms have a defined merge path, and single-participant domains observe without rebalancing.

5. **Tier 1.5 Phase 5: overlay integration contract**
    - Exit when routed overlay ExpertParallel domains have a controller attachment point, continuation/base dense domains are rejected for expert rebalance, and diagnostics render per-domain state.

### Tier 2 Exit Criteria

1. **Tier 2 Phase 1: chunk scheduler and histogram bridge**
    - Exit when prefill chunk histograms merge into the rebalance controller outside graph replay and count only real tokens.

2. **Tier 2 Phase 2: placement epochs and bank flips**
    - Exit when graph caches either invalidate on placement epoch changes or safely replay through stable runtime-table bank indirection.

3. **Tier 2 Phase 3: domain-local caches**
    - Exit when overlay domain/participant graph keys are unique and all participants agree on bucket, epoch, and topology before launch.

4. **Tier 2 Phase 4: segmented overlay collectives**
    - Exit when captured compute segments and manual sparse collectives preserve `collective_complete` ordering.

5. **Tier 2 Phase 5: rebalance between graph executions**
    - Exit when a forced chunk-boundary rebalance updates placement and subsequent graph execution matches the non-captured chunked reference.

6. **Tier 2 Phase 6: performance/correctness gates**
    - Exit when fixed and rebalanced tiered-overlay schedules match correctness gates and report graph replay, collective, and rebalance time separately.

## Suggested Test Commands

Use release/non-snapshot ROCm builds for positive graph-capture acceptance, because integration snapshots intentionally make many MoE graph-capture predicates return false.

```bash
# Release ROCm build for capture acceptance
cmake -B build_v2_release -S src/v2 -G Ninja -DCMAKE_BUILD_TYPE=Release -DHAVE_ROCM=ON
cmake --build build_v2_release --parallel

# Existing HIP graph capture coverage
ctest --test-dir build_v2_release -R "V2_Integration_.*HIPGraphCapture|V2_Integration_.*GPUGraphCapture" --output-on-failure --parallel

# ROCm MoE grouped prefill focused coverage
ctest --test-dir build_v2_release -R "ROCmMoEKernel.*FixedTopologyRuntimeGroupedPrefill" --output-on-failure --parallel

# Tier 1.5 domain-scoped rebalance coverage, once added
ctest --test-dir build_v2_integration -R "MoERebalance.*Domain|ExpertParallel.*Rebalance|MoEOverlay.*Rebalance" --output-on-failure --parallel

# Tier 1 benchmark A/B, same placement and environment for both runs
LLAMINAR_MOE_REBALANCE=off LLAMINAR_GPU_GRAPHS=0 ./build_v2_release/llaminar2 benchmark -d rocm:0 -m /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
LLAMINAR_MOE_REBALANCE=off LLAMINAR_GPU_GRAPHS=1 ./build_v2_release/llaminar2 benchmark -d rocm:0 -m /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

Tier 1.5 should preserve the existing CPU GlobalTP `-d cpu` dynamic rebalance behavior while adding domain-specific tests for single-device observe-only, LocalTP participant ids, multi-node GlobalTP participant ids, and overlay domain validation.

Tier 2 should add dedicated test filters once the phase tests exist, for example overlay segmented-capture, placement epoch, and chunk-rebalance suites. Do not validate Tier 2 by weakening Tier 1 rejection tests; Tier 1 should continue to fail fast on active rebalancing and overlay collectives.
