# Just-In-Time Model Loading Project Plan

**Date**: 2026-07-02  
**Status**: Proposed  
**Scope**: Immutable model source caching, plan-driven weight materialization, residency leases, graph-safe runner lifetimes, serving integration, and parity runtime reduction.  
**Primary files**: `src/v2/loaders/`, `src/v2/loaders/gpu_pipeline/`, `src/v2/execution/factory/`, `src/v2/execution/local_execution/`, `src/v2/execution/runner/`, `src/v2/app/modes/ServerMode.*`, `tests/v2/integration/parity/`, `tests/v2/e2e/server/`.

---

## Summary

The recent parity-cache failure around `release_raw_expert_weights` exposed the larger architectural issue: `ModelContext` is currently both the durable model source and a mutable holder of materialized/stripped runtime weight state. That is fine for a single cold runner, but it is unsafe for reusable runners, graph-captured parity tests, HTTP serving, prefix cache flows, and future just-in-time loading.

The target design is a split lifetime model:

1. `ModelImage`: immutable GGUF source image, metadata, tensor index, mmap/file lifetime, and source tensor descriptors.
2. `WeightPlan` / `FrozenModelWeightSet`: immutable declaration of what a runner needs.
3. `WeightMaterializer`: creates sharded/sliced/device-specific tensors from a `ModelImage` and a `WeightPlan`.
4. `WeightResidencyManager`: owns resident CPU/GPU prepared allocations through explicit leases and byte budgets.
5. `PreparedWeightStore`: remains the model-owned prepared handle registry, but entries are backed by residency leases.
6. `RunnerSession`: owns request/session state, KV/recurrent state, snapshots, graph replay state, and active weight leases.
7. `GraphContract`: exact graph reuse key that includes model image identity, materialization plan identity, residency generation, pointer-stability identity, stream family, and graph capture settings.

This is adjacent to the parity-runner lifetime work because the final parity optimization is exactly a constrained JIT-loading use case: reuse the immutable source and safe resident artifacts between tests without carrying stale request state or stripped raw tensors across graph-captured executions.

---

## Architecture Verdict

The split is sane and matches the direction of the existing codebase.

ExpertOverlay and LocalTP runner semantics must remain generic over a
multi-participant domain. The current CUDA2 and ROCm2 parity/bench fixtures are
2-participant proving grounds, not a runner type and not an architectural limit.

| Existing piece | Current role | JIT interpretation |
|----------------|--------------|--------------------|
| `ModelContext` / `ModelLoader` | Owns GGUF metadata, source tensor loading, `WeightManager`, and loader interface wrappers. | Should become a lightweight handle over immutable `ModelImage` plus per-use materialization services. |
| `WeightPlan`, `WeightBinding`, `FrozenModelWeightSet` | Already describe plan-driven graph bindings. | Keep as the planning boundary. Add stable plan hashes and stronger source-image identity. |
| `PreparedWeightStore` | Owns prepared GEMM, embedding, fused, sliced, and MoE expert slab state. | Keep as prepared-handle registry. Back entries with explicit resident leases so eviction and graph reuse are safe. |
| `LoadOrchestrator` / `DeviceLoadPipeline` | Already implements GPU-side H2D plus repack and VRAM pool lifetime. | Use as the GPU materialization backend under `WeightResidencyManager`. |
| `PrefillGraphCache` / `ForwardGraphSignature` | Tracks graph capture shape and replay state. | Extend with a `GraphContract`; graphs are reusable only while all required leases and pointer-stability ids are still valid. |
| `WeightStreamer` | Resident versus layer-streaming execution policy. | Fold in later as one residency policy, not the top-level JIT architecture. |

The unsafe piece is also clear: `releaseAllHostWeightData()` and `releaseRawExpertWeights()` are destructive operations on materialized runtime state. They must never destroy the immutable source image needed to create a new runner.

---

## Goals

- Load and index a GGUF model source once per process or model registry entry, then materialize runner-specific weights just in time.
- Make source model state immutable and safe to share across runners, parity tests, server requests, prefix-cache flows, and graph-captured executions.
- Keep `WeightPlan` and `FrozenModelWeightSet` as the graph-facing contract; graph construction should consume frozen bindings, not ad hoc `getWeightForDevice()` lookups.
- Own all resident prepared weights through explicit CPU/GPU leases with byte accounting, refcounts, stream/event safety, and eviction rules.
- Preserve graph-capture correctness by pinning pointer-stable resident allocations for the lifetime of captured graphs.
- Fail fast on missing source data, evicted graph-bound weights, unsupported residency policies, or stale graph contracts. No silent CPU fallback or hidden repack path.
- Reduce Qwen 3.6 MoE expert-overlay parity runtime by reusing immutable source state and exact safe resident artifacts between serialized GPU parity tests.

---

## Non-Goals

- Do not change model math, sharding semantics, MoE router top-k choices, or parity thresholds.
- Do not implement arbitrary per-token layer streaming in the first milestone.
- Do not support graph-captured eviction of weights that are still bound by an active graph.
- Do not hide OOM by falling back to CPU or changing placement silently.
- Do not solve cross-vendor direct expert transfer as part of JIT loading.
- Do not keep multiple full Qwen 3.6 multi-participant expert-overlay runners resident unless the residency budget explicitly says that fits.

---

## Current Code Findings

### `ModelContext` Is Too Stateful

`ModelContext` owns `ModelLoader` by value, exposes a loader wrapper that references it, and owns a `WeightManager`. That conflates durable model identity with mutable materialization state. The parity bug happened because a context that had raw expert weights released was still a tempting source for a later runner.

Target: `ModelContext` becomes a small compatibility shell over `shared_ptr<const ModelImage>` and a materialization/residency handle. Source data remains available until the `ModelImage` is evicted.

### `WeightPlan` Is The Right Boundary

`WeightPlan`, `WeightRequirement`, `WeightBinding`, and `FrozenModelWeightSet` already encode model id, device, role, layer, expert, TP/PP metadata, host policy, and expected prepared kind. This should become the mandatory bridge from orchestration into graph construction.

Target: every runner asks for a plan, materializes it, prepares it, receives a `FrozenModelWeightSet`, then builds graphs from those bindings.

### `PreparedWeightStore` Is Mostly Correct But Needs Leases

`PreparedWeightStore` already owns model-lifetime prepared GEMM handles, embeddings, fused adapters, sliced kernels, and expert slabs. It also has `resetDynamicState()` and `releaseAllPreparedState()`, which is the right distinction between request state and resident weight state.

Target: store entries must hold or reference `PreparedWeightLease` objects. Releasing an entry decrements leases; graph-bound entries cannot be evicted until the graph contract is invalidated and no session lease remains.

### GPU Loading Already Has The Materialization Backend

`LoadOrchestrator` and `DeviceLoadPipeline` already provide preplanned VRAM pools, pinned rings, explicit streams/events, and backend-neutral GPU repack. That is exactly the mechanism JIT loading should call for GPU materialization.

Target: the residency manager owns load orchestrators or their pools as resident allocation arenas and exposes stable leases to prepared handles.

### Graph Caches Need Weight Lifetime In Their Contract

`PrefillGraphCacheKey` and `ForwardGraphSignature` track shapes, device, bucket, placement epoch, and replay state. Captured CUDA/HIP graphs also depend on concrete device pointers and stream/lifetime contracts.

Target: extend graph cache identity with a `GraphContract` that includes model image id, plan hash, prepared store generation, residency generation, pointer-stability id, stream family, and capture settings. A graph launch must hard fail if the contract is no longer valid.

---

## Target Types

### `ModelImage`

Immutable process-wide source image:

- canonical path, inode/device, file size, mtime, optional content hash,
- GGUF metadata and tensor directory,
- mmap/file descriptor lifetime,
- raw source tensor descriptors and shape/type metadata,
- source image id for plan/cache keys.

It does not own prepared weights, device memory, graph state, KV cache, snapshots, request state, or rebalance state.

### `ModelImageCache`

Process or server-model-registry cache keyed by file identity and mmap policy. It returns `shared_ptr<const ModelImage>`.

Eviction is allowed only when no `ModelContext`, materializer, or active runner references the image.

### `WeightMaterializer`

Plan-driven creator of runtime tensors:

- consumes `ModelImage`, `WeightPlan`, target devices, precision, and TP/PP/MoE placement metadata,
- produces `FrozenModelWeightSet`,
- never mutates the source image,
- reports exact materialization stats and bytes.

Initially this can wrap `WeightManager::materialize()` and existing `getWeightForDevice()` paths, then peel responsibilities out of `WeightManager`.

### `WeightResidencyManager`

Owns resident allocations and prepared artifacts:

- CPU raw/sliced tensors,
- GPU packed pools from `LoadOrchestrator`,
- MoE expert slot pools and transfer staging buffers,
- prepared store generations,
- per-device byte budgets,
- eviction queues and active lease counts.

It returns explicit leases. No production path should own ad hoc resident weight allocations outside this manager.

### `PreparedWeightLease`

A ref-counted handle with:

- `model_image_id`,
- plan/materialization key,
- device,
- byte size,
- prepared kind,
- resident pointer/pool identity,
- residency generation,
- graph-bound pin count,
- completion event/stream lineage when preparation is async.

### `RunnerSession`

Owns everything that is not model-resident:

- active weight leases,
- KV and recurrent state,
- graph replay/cache state,
- snapshots and parity dump state,
- sampling/logits state,
- request boundary reset behavior.

`clear_cache()` resets session state. It does not unload immutable source data or evict graph-bound resident weights unless explicitly asked to unload the model.

### `GraphContract`

Exact validity contract for graph reuse:

- `model_image_id`,
- `weight_plan_hash`,
- `frozen_weight_set_hash`,
- prepared store generation,
- residency generation for every graph-bound lease group,
- pointer-stability id or resident pool id,
- device/domain ids,
- stream family id,
- placement epoch,
- bucket/shape signature,
- snapshot policy and graph capture flags.

Graph launch validates the contract and fails hard on mismatch.

---

## Lifetime State Machine

```text
Indexed
  ModelImage has metadata and tensor directory.

SourceMapped
  Raw GGUF source is mmap-backed or otherwise readable.

Planned
  Orchestration produced a WeightPlan.

Materialized
  Runtime tensors/slices/clones exist for a FrozenModelWeightSet.

Prepared
  GEMM/embedding/expert payloads are packed and registered.

Leased
  RunnerSession holds active PreparedWeightLease objects.

GraphBound
  Captured graph contract pins pointer-stable resident allocations.

Released
  Session/request state cleared; resident weights may remain.

Evicted
  Resident allocations released; immutable ModelImage may still remain.

Unloaded
  ModelImage evicted after all references are gone.
```

Destructive release operations apply only to `Materialized`, `Prepared`, `Leased`, or `GraphBound` state. They must never mutate `Indexed` or `SourceMapped`.

---

## Success Criteria

### Correctness

- Reusing a `ModelImage` after `release_raw_expert_weights` can create a fresh Qwen 3.6 MoE expert-overlay runner without missing raw expert source data.
- `clear_cache()` resets request state, graph replay state, KV/recurrent state, and snapshots without unloading immutable model source.
- Graph replay fails fast if a graph-bound lease was evicted, moved, or prepared under a different plan.
- CUDA and ROCm parity tests keep canonical thresholds unchanged.
- Server E2E long-context and prefix-cache tests pass with model-source reuse enabled.

### Performance

- Baseline instrumentation reports time spent in source indexing, source tensor materialization, GPU load/repack, prepared-store registration, graph build, graph capture, and parity snapshot collection.
- Repeated creation of the same model path in one process hits `ModelImageCache` and performs zero repeated GGUF metadata parse/index work.
- For serialized Qwen 3.6 MoE expert-overlay parity tests, wall time drops by at least 30 percent versus the current safe cache path, or the phase ends with a measured breakdown proving the remaining bottleneck is unavoidable GPU repack/graph capture work.
- Exact same-plan runner reuse avoids repeated GPU repack while respecting VRAM budgets.

### Memory

- Qwen 3.6 multi-participant parity runs never keep multiple full runners resident unless the budget explicitly allows it.
- Every resident weight allocation is attributed to a lease, model image, plan hash, device, and prepared kind.
- Eviction frees VRAM/host prepared memory deterministically once leases expire.
- The residency manager can explain why an allocation was retained or evicted.

### Observability

- Add `JitModelLoadStats` counters:
  - model image cache hits/misses,
  - source index ms,
  - materialization ms/bytes,
  - GPU repack ms/bytes,
  - resident lease bytes by device and kind,
  - graph contract hits/misses/rejections,
  - eviction counts and reasons.
- Add log lines with `model_image_id`, plan hash, prepared store generation, and graph contract id at runner creation.

---

## Phased Plan

### Phase 0: Baseline And Lifetime Instrumentation

**Goal**: Measure current costs and make hidden lifetime transitions visible.

Work:

- Add scoped stats around `ModelContext::create()`, `ModelLoader::loadModel()`, `WeightManager::materialize()`, `prepareWeightsForDevice()`, GPU `LoadOrchestrator::load()`, graph build, graph capture, and parity snapshot collection.
- Record whether a context has released raw host/expert data.
- Add a debug assertion that a stripped materialized context is never reused as a source for a new runner.
- Capture current Qwen 3.6 expert-overlay parity runtime for the CUDA2 and ROCm2 fixtures.

Tests:

- Unit test context/source reuse guards.
- Run the current Qwen 3.6 expert-overlay parity cache key tests.

Exit criteria:

- We know where parity time is going.
- The current destructive lifetime states are explicit in logs/stats.

### Phase 1: Immutable `ModelImage` And Source Cache

**Goal**: Separate durable source state from materialized runtime state.

Work:

- Add `ModelImage` and `ModelImageCache`.
- Move GGUF metadata/tensor directory/mmap lifetime into `ModelImage`, initially by wrapping or extracting from `ModelLoader`.
- Add `ModelImageLoader` or equivalent adapter implementing `IModelLoader` against immutable image state.
- Give every image a stable `ModelImageId`.
- Make `ModelContext::create()` acquire a `ModelImage` and keep compatibility APIs.

Tests:

- `V2_Unit_ModelImageCache_ReusesSamePathIdentity`.
- `V2_Unit_ModelImageCache_InvalidatesChangedFileIdentity`.
- `V2_Unit_ModelContext_RawReleaseDoesNotStripModelImage`.

Exit criteria:

- Multiple `ModelContext` objects can share the same immutable source.
- Destructive materialized-state release cannot remove the source needed for a later runner.

### Phase 2: Plan-Driven Materialization Service

**Goal**: Move materialization behind a reusable service keyed by immutable image plus plan.

Work:

- Introduce `WeightMaterializer` as the owner of materialization from `ModelImage` to `FrozenModelWeightSet`.
- Add stable `WeightPlanHash` and `FrozenModelWeightSetHash`.
- Split `WeightManager::materialize()` into compatibility shell plus materializer implementation.
- Ensure graph construction receives frozen bindings only.
- Make materialization fail if the requested source tensor no longer exists in `ModelImage`; no fallback to a stripped materialized tensor.

Tests:

- `V2_Unit_WeightMaterializer_SamePlanStableHash`.
- `V2_Unit_WeightMaterializer_ReleaseRawThenFreshMaterializeSucceeds`.
- Existing `V2_Unit_WeightPlan` and `V2_Unit_PreparedWeightStore`.

Exit criteria:

- A runner can be created from `ModelImage + WeightPlan` without relying on previous `WeightManager` mutable cache state.

### Phase 3: Residency Manager And Prepared Leases

**Goal**: Make prepared weight residency explicit, budgeted, and evictable.

Work:

- Add `WeightResidencyManager` and `PreparedWeightLease`.
- Move GPU prepared pool lifetime under leases backed by `LoadOrchestrator` and `WeightVRAMPool`.
- Teach `PreparedWeightStore` entries and expert slabs to hold leases.
- Add per-device residency budgets and hard fail on over-budget plans.
- Add eviction only for entries with no active runner or graph-bound lease.

Tests:

- `V2_Unit_WeightResidencyManager_LeasePinsAllocation`.
- `V2_Unit_WeightResidencyManager_EvictsOnlyUnleasedEntries`.
- `V2_Unit_PreparedWeightStore_ReleasesLeaseOnEntryRelease`.
- CUDA and ROCm smoke preparation tests for resident lease accounting.

Exit criteria:

- Every prepared allocation can be traced to a lease.
- Eviction frees memory deterministically and cannot invalidate active graphs.

### Phase 4: Graph Contracts And Runner Sessions

**Goal**: Make graph capture/replay safe under JIT residency.

Work:

- Add `GraphContract` and include it in `ForwardGraphSignature` and `PrefillGraphCacheKey` or adjacent cache metadata.
- Add runner/session-level lease bundles.
- Split `RunnerSession` state from model-resident state in `DeviceGraphOrchestrator` and `RankOrchestrator`.
- Validate graph contracts before replay.
- Ensure `clear_cache()` resets request/session state but preserves source image and eligible resident leases.

Tests:

- `V2_Unit_ForwardGraphTypes_GraphContractChangesOnPlanHash`.
- `V2_Unit_PrefillGraphCache_RejectsEvictedResidencyGeneration`.
- `V2_Integration_MultiTurnSessionReset_GraphContractSurvivesSafeReset`.
- CUDA and ROCm graph-captured prefill/decode parity smoke.

Exit criteria:

- Graph replay is never allowed against stale resident pointers.
- Safe request resets preserve reusable lazy/resident state without carrying request-local graph state.

### Phase 5: Serving And Prefix Cache Integration

**Goal**: Use JIT source/residency boundaries in production serving paths.

Work:

- Add a model registry for `ServerMode` backed by `ModelImageCache`.
- Make HTTP `clear_cache` and per-request reset operate on `RunnerSession`, not immutable model source.
- Wire prefix cache and long-context flows through session reset semantics.
- Add administrative unload path that evicts resident leases and then releases `ModelImage` when no references remain.

Tests:

- Existing server E2E suite in long-context mode.
- Prefix-cache E2E tests, including repeated requests and clear-cache behavior.
- Add a server test that unloads/reloads the same model path and verifies image cache and residency stats.

Exit criteria:

- Serving can reuse source model state across requests safely.
- Prefix cache behavior is unchanged except for reduced reload/rebuild overhead where applicable.

### Phase 6: MoE Expert Residency And Rebalance Integration

**Goal**: Make MoE dynamic movement a consumer of generic residency, not a separate allocation universe.

Work:

- Represent routed expert slots, shared expert residency, transfer staging, and hot-cache/LLEP arrivals as lease-backed residency objects.
- Keep pointer-stable expert slots for graph-captured decode/prefill.
- Reuse the existing async transfer and slot machinery, but allocate through `WeightResidencyManager`.
- Ensure expert movement never uses the immutable source as a mutable store and never transfers empty slots.

Tests:

- Qwen 3.6 Dynamic and LLEP parity tests on CUDA and ROCm LocalTP domains, starting with the current CUDA2 and ROCm2 fixtures.
- E2E Dynamic and LLEP long-context tests on CUDA and ROCm LocalTP domains, starting with the current CUDA2 and ROCm2 fixtures.
- Unit tests for expert arrival/departure lease release.

Exit criteria:

- Dynamic/LLEP movement works with JIT residency and graph contracts.
- Expert movement remains explicit, budgeted, and fail-fast.

### Phase 7: Parity Runtime Reduction

**Goal**: Use the JIT architecture to make parity tests faster without weakening what they test.

Work:

- Teach `ParityTestBase` to use `ModelImageCache`.
- Keep GPU parity tests serialized because they contend for GPU memory and snapshot directories.
- Replace ad hoc Qwen 3.6 expert-overlay runner caches with JIT leases and exact graph contracts.
- Allow exact same-plan runner reuse only when the graph contract matches.
- Otherwise evict the runner/session, keep `ModelImage`, and re-materialize through the JIT path.
- Keep parity snapshots collected from graph-captured prefill/decode; do not reintroduce raw stage execution.

Tests:

- Full Qwen 3.6 expert-overlay parity suite for the current CUDA2 and ROCm2 fixtures, with no runner assumptions that would block future n>2 fixtures.
- Full precommit parity gate additions for Dynamic and LLEP.
- Full unit suite after enabling source-image reuse.

Exit criteria:

- Parity tests remain numerically strict.
- The Qwen 3.6 expert-overlay parity suite is materially faster.
- Cache correctness is covered by reusable parity framework tests, not Qwen-specific special cases.

---

## Code Touch Points

| Area | Files | Planned change |
|------|-------|----------------|
| Immutable source | `src/v2/loaders/ModelLoader.*`, new `ModelImage.*`, new `ModelImageCache.*` | Split metadata/tensor index/mmap lifetime from runtime materialization. |
| Context compatibility | `src/v2/loaders/ModelContext.*`, `src/v2/interfaces/IModelContext.h` | Keep existing APIs while routing source access through `ModelImage`. |
| Planning | `src/v2/loaders/WeightPlan.*`, `WeightIdentity.*`, `WeightLifecycleTrace.*` | Add image id, plan hash, frozen-set hash, stronger trace events. |
| Materialization | `src/v2/loaders/WeightManager.*`, new `WeightMaterializer.*` | Move plan-to-binding materialization behind a reusable service. |
| Prepared state | `src/v2/loaders/PreparedWeightStore.*`, `ExpertSlabTypes.*` | Attach prepared entries and expert slabs to residency leases. |
| GPU loading | `src/v2/loaders/gpu_pipeline/*` | Use load orchestration as a residency backend rather than ad hoc owner. |
| Runner creation | `src/v2/execution/factory/InferenceRunnerFactory.*`, `src/v2/execution/runner/OrchestrationRunnerFactory.cpp` | Build runners from image + plan + lease bundle. |
| Device/rank runners | `DeviceGraphOrchestrator.*`, `RankOrchestrator.*`, `OrchestrationRunner.*` | Split session state from model residency; validate graph contracts. |
| Graph cache | `PrefillGraphCache.*`, `ForwardGraphTypes.*`, `ForwardExecutionEngine.*` | Add graph contract identity and invalidation behavior. |
| MoE residency | `src/v2/execution/moe/*`, MoE compute stages, CUDA/ROCm MoE kernels | Route expert arrivals/departures through lease-backed slots. |
| Serving | `src/v2/app/modes/ServerMode.*`, `ChatCompletionHandler.*`, `InferenceRunnerAdapter.*` | Add model registry/session reset semantics. |
| Prefix cache | `src/v2/execution/prefix_cache/*` | Verify cache state reset does not unload model source or graph-bound resident weights. |
| Parity tests | `tests/v2/integration/parity/ParityTestBase.h`, Qwen parity suites | Reuse `ModelImage`, exact runner leases, and graph-captured snapshot hooks. |
| E2E | `tests/v2/e2e/server/test_server_e2e.sh`, `long_context_checks.py` | Add JIT source reuse and unload/reload coverage. |

---

## Immediate Sprint Backlog

1. Add Phase 0 timing and lifetime stats.
2. Add `ModelImage`/`ModelImageCache` skeleton and unit tests.
3. Adapt `ModelContext::create()` to acquire a `ModelImage` while preserving current public behavior.
4. Add regression coverage proving `release_raw_expert_weights` cannot poison future runner creation from the same source image.
5. Measure Qwen 3.6 expert-overlay parity runtime before and after source-image caching.
6. Decide whether the next bottleneck is materialization, GPU repack, graph capture, or snapshot collection, then start the corresponding phase.

---

## Risks And Open Questions

- `ModelLoader` currently owns details that may not cleanly move into an immutable image in one patch. The first pass should wrap before extracting.
- Graph-captured CUDA/HIP pointer stability is non-negotiable; residency eviction must invalidate graphs before freeing memory.
- `WeightManager` has many compatibility responsibilities. Migration should keep it as a facade until materialization and residency are proven.
- Existing `WeightStreamer` may overlap conceptually with JIT loading, but it should not become the owner of graph-bound lifetime.
- Large MoE models on 24 GB cards cannot keep every useful artifact resident. The architecture must make eviction cheap and safe, not pretend every cache entry fits.
- Parity tests need speed, but they are also correctness tripwires. Exact cache keys and hard failures are preferable to broad reuse.

---

## Definition Of Done

The JIT loading project is done when:

- model source identity is immutable and shared,
- materialized/prepared state is lease-backed and evictable,
- graph replay validates resident weight contracts,
- server and prefix-cache paths use session resets without source reload,
- Qwen 3.6 CUDA and ROCm LocalTP Dynamic and LLEP parity and E2E tests pass for current 2-participant fixtures and remain architecturally open to n>2 domains,
- the parity framework no longer needs Qwen-specific unsafe runner/source cache hacks,
- measured parity runtime improves materially or we have a precise bottleneck report showing where time remains.
