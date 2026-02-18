# KernelFactory Device-Scoped Kernel Cache Refactor Plan (All Kernel Types)

## Summary

This plan expands the cache refactor from GEMM-only to **all kernel families**.
Goal: ensure kernel instances are cached **once per device + kernel kind (+ precision variant)**, never per tensor/stage.

## Status update (2026-02-17)

### Completed in code

- Phase A scaffold is implemented in `KernelFactory`:
  - `KernelKind`
  - `DeviceKernelKey`
  - generic `device_kernel_registry_` and helper APIs
- Phase A adapter/front-door shim is now applied to all non-GEMM device-scoped `getOrCreate*` paths:
  - `getOrCreateRoPE`
  - `getOrCreateRMSNorm`
  - `getOrCreateSwiGLU`
  - `getOrCreateSoftmax`
  - `getOrCreateResidualAdd`
  - `getOrCreateAttention`
  - `getOrCreateEmbedding`
- Softmax parity was added to match the other non-GEMM families:
  - `createSoftmax(const TensorBase*, DeviceType)` dynamic dispatch
  - `softmax_cache_` device+dtype cache
- Temporary registry instrumentation is active in those `getOrCreate*` methods:
  - logs `create`, `backfill`, and `hit` with `(kind, device, variant, ptr)`
- Phase C GEMM engine/prepared-weights architecture is now present:
  - `getOrCreateGemmEngine(exemplar_tensor, target_device)`
  - `getOrCreateGemmEngine(prepared_handle)`
  - `getOrCreatePreparedGemmWeights(tensor, target_device, prep_kind)`
  - `clearPreparedGemmWeightsFor(tensor)`
  - `gemmEngineRegistrySize()` / `preparedGemmRegistrySize()`
  - internal registries: `prepared_gemm_registry_`, `device_gemm_engine_registry_`
- Phase C stage call-site migration slice is applied (behavior-preserving):
  - `GEMMStage`, `LMHeadStage`, and `FusedQKVGEMMStage` now call
    `getOrCreatePreparedGemmWeights(..., device_id)` before `getOrCreateGemm(..., device_id)`
  - current execution still uses existing GEMM kernel path (no engine-switch semantics yet)
- Additional Phase C migration coverage is applied:
  - `FusedGateUpGEMMStage` now warms prepared handles before fused Gate/Up kernel lookup
  - fused factory helper paths warm prepared handles before creating fused adapters:
    - `getOrCreateFusedQKVGemm(...)`
    - `createFusedGateUpGemm(..., DeviceId)`
  - preload/packing paths now warm prepared handles before GEMM kernel creation:
    - `WeightPreloader::packWeight(...)`
    - `WeightManager::packWeight(...)`
- Engine API migration slice is applied (behavior-preserving):
  - main execution/preload paths now call `getOrCreateGemmEngine(..., DeviceId)`
    instead of direct `getOrCreateGemm(..., DeviceId)`
  - covered files include `GEMMStage`, `LMHeadStage`, `FusedQKVGEMMStage`,
    `KernelFactory` fused Gate/Up helper, `WeightPreloader`, and `WeightManager`
- Prepared-handle binding step is applied:
  - migrated call sites pass prepared handles into GEMM engine lookup
  - runtime logs confirm handle-bound lookups (`prepared=...`) in LocalTP ROCm smoke
- Device-scoped GEMM engine layer is now present:
  - added `device_gemm_engine_registry_` keyed by `(device, kind=GEMM, variant)`
  - `getOrCreateGemmEngine(prepared)` now touches this device-level registry first
  - runtime LocalTP ROCm logs show expected pattern (`device GEMM engine create` once per device+variant, then `hit`)
- Device-scoped GEMM engine object step is now applied:
  - `device_gemm_engine_registry_` entries are real engine objects (`PhaseCGemmDeviceEngine`) instead of markers
  - active `getOrCreateGemmEngine(prepared)` resolution now routes through this device engine object
- Legacy GEMM engine map cleanup is now applied:
  - removed per-prepared `gemm_engine_registry_` storage/type plumbing
  - `clearCacheFor` / `clearPreparedGemmWeightsFor` cleanup now targets prepared handles + device GEMM engines only
  - `gemmEngineRegistrySize()` is preserved as a compatibility alias to the device-scoped GEMM engine registry
- Typed device-engine interface slice is now applied:
  - introduced `KernelFactory::IGemmEngine` as the typed contract for device-scoped GEMM engines
  - `device_gemm_engine_registry_` now stores `shared_ptr<IGemmEngine>` instead of erased `shared_ptr<void>`
  - `PhaseCGemmDeviceEngine` now implements the typed interface, removing void-cast plumbing from engine lookup path
- Prepared-handle-only GEMM engine API slice is now applied:
  - added `getOrCreateGemmEngine(const PreparedGemmHandle*)` as the preferred lookup path
  - migrated execution, fused-kernel helpers, and weight preloading call sites to the prepared-only overload
  - `IGemmEngine::resolveKernel` now resolves from the prepared handle directly (no exemplar argument in active path)
- Prepared payload kernel-binding slice is now applied:
  - `PreparedGemmHandle` now carries a typed prepared payload (`prepared_weights`)
  - introduced `PreparedGemmWeights` to encapsulate prepared GEMM execution resources
  - `getOrCreatePreparedGemmWeights(...)` binds the GEMM kernel once and stores it in the prepared handle
  - `PhaseCGemmDeviceEngine::resolveKernel` now returns `prepared->prepared_weights->kernel` on the fast path
- Prepared-only resolve hardening is now applied:
  - removed lazy fallback from `PhaseCGemmDeviceEngine::resolveKernel` to legacy `getOrCreateGemm(...)`
  - device-engine resolution now fails fast if prepared payload, kind, or kernel binding is missing/mismatched
  - active GEMM engine resolution path now requires prepared payload integrity end-to-end
- Compatibility API cleanup is now applied:
  - removed deprecated `getOrCreateGemmEngine(prepared, exemplar)` overload from `KernelFactory`
  - refreshed API comments to describe the strict prepared-handle-first engine lookup contract
- Phase D fused-cache key migration slice is now applied:
  - `fused_qkv_cache_` and `fused_gate_up_cache_` are now keyed by prepared-handle identity instead of raw weight tensor pointers
  - fused cache lookups now resolve prepared handles first, then index cache by `(prepared handles, device)`
  - cleanup paths (`clearCacheFor`, `clearPreparedGemmWeightsFor`) now evict fused entries that reference prepared handles for the target tensor before prepared-handle erasure
- Phase D explicit-device fused QKV API slice is now applied:
  - added explicit `DeviceId` overloads for `createFusedQKVGemm(...)` and `getOrCreateFusedQKVGemm(...)`
  - `DeviceType` fused QKV lookup now delegates to the explicit-device overload after resolving ordinal
  - `FusedQKVGEMMStage` mixed/Q8 paths now call the `DeviceId` overload directly (`params_.device_id`), removing implicit thread-local ordinal dependence at stage call sites
- Phase D fused-stage prewarm cleanup slice is now applied:
  - removed redundant `getOrCreatePreparedGemmWeights(...)` prewarm calls from `FusedQKVGEMMStage` before fused-kernel lookup
  - removed redundant `getOrCreatePreparedGemmWeights(...)` prewarm calls from `FusedGateUpGEMMStage` execute/workspace lookup paths
  - updated fused QKV `KernelFactory` API comments to mark `DeviceId` overloads as preferred and `DeviceType` overloads as compatibility paths
- Phase D fused Gate/Up miss-path optimization is now applied:
  - `getOrCreateFusedGateUpGemm(..., DeviceId)` now constructs the fused adapter from the already-resolved prepared handles on cache miss
  - avoids a second prepared-handle resolution pass that previously occurred via `createFusedGateUpGemm(...)`
- Phase D internal helper unification for fused Gate/Up is now applied:
  - extracted a shared prepared-handle-native helper (`createFusedGateUpAdapterFromPrepared`) in `KernelFactory.cpp`
  - both `createFusedGateUpGemm(..., DeviceId)` and cache-miss `getOrCreateFusedGateUpGemm(..., DeviceId)` now route through that helper
  - this removes duplicated adapter-construction logic and keeps fused Gate/Up construction semantics centralized
- Phase D internal helper unification for fused QKV is now applied:
  - extracted a shared prepared-handle-native helper (`createFusedQKVAdapterFromPrepared`) in `KernelFactory.cpp`
  - both `createFusedQKVGemm(..., DeviceId)` and cache-miss `getOrCreateFusedQKVGemm(..., DeviceId)` now route through that helper
  - `createFusedQKVGemm(..., DeviceId)` now resolves prepared handles explicitly and delegates construction to the shared helper
  - this removes duplicated fused QKV construction logic and keeps prepared-handle-first semantics centralized
- Legacy retirement slice 1 is now applied:
  - migrated `cpu/gemm_v4/FusedGEMM` constructor kernel lookup from legacy `KernelFactory::getOrCreateGemm(weight)` to Phase C prepared-handle path:
    - `getOrCreatePreparedGemmWeights(weight, DeviceId(CPU,0))`
    - `getOrCreateGemmEngine(prepared)`
  - updated `FusedGEMM` comments/error text to reference prepared-handle engine resolution semantics
- Legacy retirement slice 2 is now applied:
  - removed eager legacy GEMM binding from `getOrCreatePreparedGemmWeights(...)`
  - prepared handles now record preparation metadata and defer executable kernel binding
  - `PhaseCGemmDeviceEngine::resolveKernel(...)` now performs first-use lazy binding when `prepared_weights->kernel` is null
  - this removes direct `getOrCreateGemm(..., DeviceId)` dependency from prepared-handle construction and centralizes binding in the engine resolution path
- Legacy retirement slice 3 is now applied:
  - removed direct legacy `getOrCreateGemm(...)` binding from `PhaseCGemmDeviceEngine::resolveKernel(...)`
  - added explicit-device prepared-kernel binding helper in `KernelFactory.cpp` using `createGemm(...)` with CUDA/ROCm ordinal guards
  - `getOrCreatePreparedGemmWeights(...)` now binds prepared kernels through that explicit-device helper
  - `PreparedGemmWeights` now owns the bound kernel (`owned_kernel`) and exposes a stable raw pointer (`kernel`) for engine resolution
  - engine resolve path is now strict again: missing prepared kernel binding fails fast
- Legacy retirement slice 4 is now applied:
  - routed `getOrCreateGemm(const TensorBase*, DeviceId)` through prepared-handle + device-engine APIs (`getOrCreatePreparedGemmWeights` → `getOrCreateGemmEngine`)
  - converted `getOrCreateGemm(const TensorBase*, DeviceType)` into a compatibility bridge that preserves legacy ordinal resolution and delegates to the `DeviceId` overload
  - this removes active runtime reliance on `device_targeted_cache_` for legacy GEMM callers while preserving API compatibility
- Legacy retirement slice 5 is now applied:
  - removed dormant `device_targeted_cache_` surface from `KernelFactory` state (`DeviceTargetedCacheKey`, hash, static map)
  - removed dead cleanup paths for `device_targeted_cache_` from `clearCacheFor(...)` and `clearCache()`
  - updated `cacheStats()` to exclude retired `device_targeted_cache_` entries
  - no runtime behavior change intended: active legacy GEMM APIs were already bridged to prepared-engine flow in slice 4
- Legacy retirement slice 6 is now applied:
  - converted no-device `getOrCreateGemm(const TensorBase*)` into a compatibility bridge that infers `DeviceId` from tensor residency and delegates to prepared-engine flow
  - updated legacy GEMM API comments in `KernelFactory.h` to mark all `getOrCreateGemm(...)` overloads as compatibility surfaces
  - effective active GEMM resolution path is now uniformly prepared-handle + device-engine based across all public `getOrCreateGemm(...)` entrypoints
- Legacy retirement slice 7 is now applied:
  - removed legacy `kernel_cache_` storage from `KernelFactory` state and static definitions
  - removed dead `kernel_cache_` cleanup/references from `clearCacheFor(...)`, `clearCache()`, and `cacheStats()`
  - `clearCache()` now derives tracked tensors from active registries (`prepared_gemm_registry_`, `sliced_cache_`, fused prepared-handle caches) before clearing tensor-owned packed-weight caches
  - no runtime behavior change intended: active GEMM API entrypoints were already prepared-engine bridges in slice 6
- Legacy retirement slice 8 is now applied:
  - removed obsolete `getOrCreateGemmEngine(const TensorBase*, DeviceId)` compatibility overload
  - GEMM engine resolution API surface is now prepared-handle-only (`getOrCreateGemmEngine(const PreparedGemmHandle*)`)
  - this tightens Phase C contract boundaries and removes a redundant adapter path around prepared-handle lookup
- Legacy retirement slice 9 is now applied:
  - removed legacy `getOrCreateGemm(...)` compatibility overloads from `KernelFactory` API/implementation (`TensorBase*`, `DeviceType`, `DeviceId`)
  - migrated remaining `tests/v2` call sites to prepared-handle + device-engine lookup (`getOrCreatePreparedGemmWeights` → `getOrCreateGemmEngine`)
  - refreshed source comments/examples that still referenced retired legacy overloads

### Runtime validation performed

- Release build validated.
- Integration build validated.
- LocalTP ROCm smoke run validated with:
  - `--tp-scope local --tensor-parallelism-degree 2 --tp-devices "rocm:0,rocm:1"`
  - observed expected registry behavior on both devices (`create` once, then repeated `hit`) with fused QKV / GateUp stages executing successfully.
  - after explicit-device fused QKV migration, fused QKV stage paths continue to execute successfully on `ROCm:0` and `ROCm:1` without new runtime errors.
  - after prewarm cleanup, fused QKV/GateUp stage execution remains stable in LocalTP ROCm smoke with no new runtime errors.
  - after fused Gate/Up miss-path optimization, integration build and LocalTP ROCm smoke remain stable.
  - after helper unification, integration build and LocalTP ROCm smoke remain stable.
  - after fused QKV helper unification, integration build and LocalTP ROCm smoke remain stable with fused QKV stage execution unchanged.
  - after legacy retirement slice 1 (`FusedGEMM` migration), integration build and LocalTP ROCm smoke remain stable with no new runtime errors.
  - after legacy retirement slice 2 (lazy kernel binding in `resolveKernel`), integration build remains stable.
  - after legacy retirement slice 3 (explicit-device `createGemm(...)` prepared binding + prepared-owned kernel), integration build remains stable.
  - after legacy retirement slice 4 (legacy `getOrCreateGemm(DeviceType/DeviceId)` bridged to prepared-engine flow), integration and release builds remain stable.
  - after legacy retirement slice 7 (`kernel_cache_` retirement with registry-derived packed-cache cleanup), release build remains stable.
  - after legacy retirement slice 8 (removal of tensor+device GEMM engine compatibility overload), release build and CUDA smoke remain stable.
  - after legacy retirement slice 9 (removal of all legacy `getOrCreateGemm(...)` compatibility overloads + remaining test migration), release build and CUDA smoke remain stable.
  - release was rebuilt before smoke (`build_v2_release/llaminar2` newer than `KernelFactory.cpp/.h` edits), then smoke was rerun.
  - this workspace currently exposes `CPU` and `CUDA:0` (no ROCm devices), so equivalent LocalTP smoke was executed on `--tp-scope local --tensor-parallelism-degree 1 --tp-devices "cuda:0"`.
  - CUDA smoke logs continue to show expected Phase C prepared/engine hits and fused QKV/GateUp stage execution with no new `ERROR` lines in filtered output.

### Remaining from this plan

- Remove temporary instrumentation once confidence window closes.
- Migrate any remaining direct GEMM callers outside the main execution/preload paths to consume Phase C APIs (if added later).
- Plan the next semantic step for true device-scoped GEMM engine sharing
  (decoupling engine identity from tensor identity, with prepared-handle-only weight binding).
- Introduce an actual tensor-independent GEMM engine object (or interface split)
  so prepared handles carry all weight-specific state and engine identity can collapse to device(+variant).
- Lift `PhaseCGemmDeviceEngine` from transitional wrapper to a true tensor-independent GEMM execution interface,
  so prepared handles are the only tensor-affine component and engine calls no longer delegate back through legacy GEMM creation APIs.
- Continue with Phase D/E after GEMM path migration.

## Why this is needed

Current behavior mixes two patterns:

1. **KernelFactory per-tensor cache** for GEMM/fused GEMM
   - `kernel_cache_` keyed by `TensorBase*`
   - `device_targeted_cache_` keyed by `(TensorBase*, DeviceType, ordinal)`
   - `fused_qkv_cache_`, `fused_gate_up_cache_` keyed by weight tensors

2. **Per-stage cached kernel instance** for non-GEMM kernels
   - `RoPEStage`, `RMSNormStage`, `SwiGLUStage`, `ResidualAddStage`, `EmbeddingStage`, `AttentionComputeStage`
   - each keeps `cached_kernel_` and initializes from `KernelFactory::create*`

Result: many duplicated kernel instances on same device and inconsistent lifecycle semantics.

## Historical touchpoints (pre-refactor audit)

### KernelFactory caches (tensor-affine)
- `src/v2/kernels/KernelFactory.h`
  - `kernel_cache_`
  - `sliced_cache_`
  - `device_targeted_cache_`
  - `fused_qkv_cache_`
  - `fused_gate_up_cache_`

- `src/v2/kernels/KernelFactory.cpp`
  - `getOrCreateGemm(...)` overloads
  - `getOrCreateFusedQKVGemm(...)`
  - `getOrCreateFusedGateUpGemm(...)`
  - `clearCacheFor(...)`, `clearCache()`

### Stage-level non-GEMM caching (effectively tensor/stage-affine)
- `src/v2/execution/compute_stages/stages/RoPEStage.*`
- `src/v2/execution/compute_stages/stages/RMSNormStage.*`
- `src/v2/execution/compute_stages/stages/SwiGLUStage.*`
- `src/v2/execution/compute_stages/stages/ResidualAddStage.*`
- `src/v2/execution/compute_stages/stages/EmbeddingStage.*`
- `src/v2/execution/compute_stages/stages/AttentionComputeStage.*`

## Target architecture

## 1) Device-scoped kernel registry

Introduce a global registry in `KernelFactory` keyed by:
- `DeviceId` (type + ordinal)
- `KernelKind` (`GEMM`, `ROPE`, `RMSNORM`, `SWIGLU`, `SOFTMAX`, `RESIDUAL_ADD`, `ATTENTION`, `EMBEDDING`, `FUSED_QKV`, `FUSED_GATE_UP`)
- `KernelVariant` (activation precision / layout / algorithm flags if required)

Returns a shared kernel service object reused by all stages on that device.

## 2) Prepared-weights registry (separate from kernels)

Weight/tensor-specific state moves out of kernel objects into prepared handles:
- key: `(TensorIdentity, DeviceId, PreparationKind)`
- value: packed/uploaded weight handle

Kernels become device engines consuming prepared handles at call time.

## 3) Stateless or explicitly scoped dynamic state

Kernel APIs must avoid sticky stage/tensor state:
- stream set per call or via short-lived execution context
- workspace bind/unbind must be call-scoped (or provided in args)
- dynamic params (e.g., RoPE pos offset, attention kv_len) passed per call

## Remaining execution plan (post Phase C)

### Phase D — Complete fused-path engine unification

- Convert fused QKV / GateUp execution paths to run strictly through device-scoped GEMM engines plus prepared handles.
- Remove any remaining fused tensor-tuple kernel caches once parity and perf are confirmed.
- Ensure fused execution no longer relies on legacy tensor-keyed GEMM creation paths.

### Phase E — Retire legacy tensor-affine cache APIs

- Remove obsolete tensor-keyed cache surfaces after all call sites are migrated:
  - `kernel_cache_`
  - `device_targeted_cache_`
  - tensor-keyed fused caches
- Keep only prepared-weight teardown for tensor lifecycle (`clearPreparedGemmWeightsFor`).
- Preserve compatibility aliases only where needed for external callers; otherwise delete stale API surface.

### Phase F — Harden device-engine semantics

- Lift `PhaseCGemmDeviceEngine` from a resolver wrapper to a true tensor-independent execution service.
- Keep tensor-affine state exclusively in `PreparedGemmWeights` and enforce that contract in all engine entry points.
- Remove temporary registry instrumentation after one additional validation window.

## API changes expected

- New preferred API shape (example):
  - `KernelFactory::getOrCreateKernel(DeviceId, KernelKind, KernelVariant)`
  - `KernelFactory::getOrCreatePreparedWeights(TensorBase*, DeviceId, PrepKind)`

- Deprecated:
  - `getOrCreateGemm(tensor, ...)`
  - stage-local `cached_kernel_` ownership patterns

## Risk areas

1. **Workspace lifecycle coupling**
   - Several kernels currently rely on `bindWorkspace()/unbindWorkspace()` and mutable internal pointers.
   - Must move to call-scoped workspace binding before sharing instances.

2. **Dynamic runtime params**
   - Attention/RoPE kernels use mutable dynamic state (`setDynamicAttnParams`, `setDynamicPosOffset`).
   - Must pass these as call params (or context object) to avoid races.

3. **Thread safety**
   - Shared per-device kernels will be concurrently used by stage threads.
   - Any mutable members must be removed or protected.

4. **clearCacheFor semantics**
   - Today tensor teardown clears kernels + packed caches.
   - After refactor, it should clear only prepared weight handles for that tensor.

## Validation plan

- Unit tests:
  - one kernel instance per `(device, kind, variant)`
  - no duplicate kernels when multiple stages request same kernel family
  - prepared-weight cache keyed by tensor+device only

- Integration tests:
  - run `V2_Unit_*`, `V2_Integration_*`, parity tests after each phase
  - targeted LocalTP ROCm repro for race and device-affinity regressions

- Logging instrumentation (temporary):
  - emit kernel pointer + device key on registry fetch/create
  - emit prepared-weight handle pointer + tensor/device key

## Next implementation slice

Implement **Phase D fused-path unification**:
- tighten fused QKV/GateUp runtime to prepared-handle + device-engine-only resolution
- remove now-redundant fused tensor-keyed cache paths guarded by validation
- run Integration + LocalTP ROCm smoke to confirm no regression

---

Prepared on 2026-02-17 for tensor-parallel branch.
