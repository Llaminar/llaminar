# KernelFactory Device-Scoped Kernel Cache Refactor Plan (All Kernel Types)

## Summary

This plan expands the cache refactor from GEMM-only to **all kernel families**.
Goal: ensure kernel instances are cached **once per device + kernel kind (+ precision variant)**, never per tensor/stage.

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

## Current touchpoints (audit)

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

## Migration strategy

### Phase A — Introduce common registry primitives (no behavior change)

- Add:
  - `enum class KernelKind`
  - `struct DeviceKernelKey`
  - `DeviceKernelRegistry` (`unordered_map<DeviceKernelKey, shared_ptr<IKernelBase>>`)
- Keep existing APIs and caches intact.
- Provide adapter wrappers so old code can retrieve kernels through registry without semantic changes.

### Phase B — Non-GEMM families first (lowest risk)

Move stage-level cached kernels to registry-backed `getOrCreate*` methods:
- Add `getOrCreateRoPE(DeviceId, TensorType/precision variant)`
- Add `getOrCreateRMSNorm(...)`, `getOrCreateSwiGLU(...)`, `getOrCreateResidualAdd(...)`, `getOrCreateSoftmax(...)`, `getOrCreateAttention(...)`, `getOrCreateEmbedding(...)`
- Remove `cached_kernel_` ownership from stages (keep non-owning/raw/weak references only).

### Phase C — GEMM split into engine + prepared weights

- Introduce `IGemmEngine` (device-scoped)
- Introduce `PreparedGemmWeights` (tensor+device scoped)
- Replace `getOrCreateGemm(tensor, device)` with:
  - `engine = getOrCreateGemmEngine(device, variant)`
  - `prepared = getOrCreatePreparedGemmWeights(tensor, device)`
  - `engine->multiply(prepared, A, C, ...)`

### Phase D — Fused kernels

- Rework fused QKV/GateUp to use same device GEMM engine + multiple prepared handles.
- Remove fused caches keyed by tensor tuples.

### Phase E — Remove legacy caches and APIs

- Remove:
  - `kernel_cache_`
  - `device_targeted_cache_`
  - tensor-keyed fused caches
- Keep compatibility shims temporarily behind compile/runtime guard.

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

## First implementation slice (recommended)

Implement **Phase B for RoPE + RMSNorm only**:
- minimal API extension
- easiest correctness envelope
- proves device-scoped sharing model before touching GEMM.

---

Prepared on 2026-02-17 for tensor-parallel branch.
