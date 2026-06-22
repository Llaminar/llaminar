# Graph System Vendor Abstraction Cleanup

**Status**: Planned  
**Date**: 2026-03-05  
**Scope**: Eliminate vendor-specific `#ifdef` leaks from the graph execution layer  
**Priority**: Medium — correctness is fine, but hygiene and diagnostic parity are not  

---

## Summary

The graph capture/replay architecture is structurally excellent. `IGPUGraphCapture`, `IWorkerGPUContext::createGraphCapture()`, `DeviceGraphCaptureController`, and `GraphCaptureGuard` form a clean, vendor-neutral abstraction stack with zero code duplication for capture/replay logic.

However, **18 vendor-specific `#ifdef HAVE_ROCM` / `#ifdef HAVE_CUDA`** sites exist in the four core graph-layer files. These are concentrated in context acquisition boilerplate, factory registration boilerplate, and ROCm-only diagnostic tooling that has no CUDA equivalent.

---

## Architecture Assessment

### What's Good

| Component | File | Verdict |
|-----------|------|---------|
| `IGPUGraphCapture` | `src/v2/backends/IGPUGraphCapture.h` | Clean lifecycle contract, zero vendor knowledge |
| `CUDAGraphCapture` | `src/v2/backends/cuda/CUDAGraphCapture.cu` | Properly isolated in `backends/cuda/` |
| `HIPGraphCapture` | `src/v2/backends/rocm/HIPGraphCapture.cpp` | Properly isolated in `backends/rocm/`, symmetric with CUDA impl |
| `IWorkerGPUContext::createGraphCapture()` | `src/v2/backends/IWorkerGPUContext.h` | Factory method pattern — callers never see vendor types |
| `DeviceGraphCaptureController` | `src/v2/execution/local_execution/graph/DeviceGraphCaptureController.cpp` | 1500 lines, only 2 vendor-specific lines |
| `GraphCaptureGuard` | `src/v2/execution/local_execution/graph/GraphCaptureGuard.h` | Thread-local flag, fully vendor-neutral |

### What Needs Work

18 vendor-specific sites across 4 files, categorized below.

---

## Inventory of Vendor-Specific Sites

### File 1: `DeviceGraphExecutor.cpp` — 9 occurrences

| # | Lines | Guard | Category | Description |
|---|-------|-------|----------|-------------|
| 1 | L24–27 | `HAVE_ROCM` | Include guard | `#include "HipDeviceGuard.h"`, `#include "ROCmBackend.h"` |
| 2 | L52–58 | `HAVE_ROCM` | Context resolution | `resolveWorkerDefaultStream()` — `pool.getAMDContext(rocm_ordinal()).defaultStream()` |
| 3 | L61–67 | `HAVE_CUDA` | Context resolution | `resolveWorkerDefaultStream()` — `pool.getNvidiaContext(cuda_ordinal()).defaultStream()` |
| 4 | L688–720 | `HAVE_ROCM` | Diagnostics | GPU pointer device validation in `executeSequential()` — `hipPointerGetAttributes` |
| 5 | L795–808 | `HAVE_ROCM` | Sync workaround | `pre_stage_sync` / `post_stage_sync` lambdas (currently no-op but structurally vendor-specific) |
| 6 | L1564–1621 | `HAVE_ROCM` | Diagnostics | `logWatchedPointerProducer()` — entire function is ROCm-only |
| 7 | L1859–1963 | `HAVE_ROCM` | Diagnostics | GPU pointer validation in `executeNode()` — ~100 lines of `hipPointerGetAttributes` + `ROCmBackend::queryPointerOwner` |
| 8 | L1973–1997 | `HAVE_ROCM` | Sync workaround | `sync_each_stage` — `hipDeviceSynchronize()` + `ROCmBackend::dumpRecentPointerEvents` |
| 9 | L2048–2049 | `HAVE_ROCM` | Diagnostics | `logWatchedPointerProducer()` call site |

### File 2: `DeviceGraphCaptureController.cpp` — 3 occurrences

| # | Lines | Guard | Category | Description |
|---|-------|-------|----------|-------------|
| 10 | L8–9 | `HAVE_ROCM` | Include guard | `#include <hip/hip_runtime.h>` |
| 11 | L1171–1173 | `HAVE_ROCM` | Error clearing | `hipGetLastError()` before `beginCapture()` — clears sticky error |
| 12 | L1217–1219 | `HAVE_ROCM` | Error clearing | `hipGetLastError()` after failed capture — consumes sticky error |

### File 3: `DeviceGraphOrchestrator.cpp` — 4 occurrences

| # | Lines | Guard | Category | Description |
|---|-------|-------|----------|-------------|
| 13 | L720–723 | `HAVE_ROCM` | Factory registration | `ensureAMDFactoryRegistered()` |
| 14 | L724–727 | `HAVE_CUDA` | Factory registration | `ensureNvidiaFactoryRegistered()` |
| 15 | L730–738 | *(runtime)* | Context resolution | `is_rocm()` → `pool.getAMDContext()` / `is_cuda()` → `pool.getNvidiaContext()` |

### File 4: `RankOrchestrator.cpp` — 4 occurrences (2 `#ifdef` + 2 runtime branches)

| # | Lines | Guard | Category | Description |
|---|-------|-------|----------|-------------|
| 16 | L669–671 | `HAVE_ROCM` | Factory registration | `ensureAMDFactoryRegistered()` |
| 17 | L673–675 | `HAVE_CUDA` | Factory registration | `ensureNvidiaFactoryRegistered()` |
| 18 | L684–700 | *(runtime)* | Context resolution | `DeviceType::ROCm` → `pool.getAMDContext()` / `DeviceType::CUDA` → `pool.getNvidiaContext()` |

### Category Totals

| Category | Count | ROCm-only | CUDA-only | Paired |
|----------|-------|-----------|-----------|--------|
| Context resolution | 4 | 0 | 0 | 2 pairs |
| Factory registration | 4 | 0 | 0 | 2 pairs |
| Diagnostics/debugging | 4 | **4** | 0 | 0 |
| Error clearing | 2 | **2** | 0 | 0 |
| Sync workaround | 2 | **2** | 0 | 0 |
| Include guards | 2 | **2** | 0 | 0 |
| **Total** | **18** | **10** | **0** | **4 pairs** |

**Notable**: 10 of 18 sites are ROCm-only with no CUDA equivalent. This means CUDA multi-GPU TP will lack the same diagnostic and error-recovery coverage.

---

## Cleanup Tasks

### Task 1: `GPUDeviceContextPool::getContext(const DeviceId&)`

**Impact**: Eliminates sites #2, #3, #15, #18 (4 sites)

Add a `DeviceId`-based accessor to `GPUDeviceContextPool`:

```cpp
// In GPUDeviceContextPool.h
IWorkerGPUContext& getContext(const DeviceId& device);

// In GPUDeviceContextPool.cpp
IWorkerGPUContext& GPUDeviceContextPool::getContext(const DeviceId& device) {
    if (device.is_cuda()) return getNvidiaContext(device.cuda_ordinal());
    if (device.is_rocm()) return getAMDContext(device.rocm_ordinal());
    throw std::invalid_argument("Not a GPU device: " + device.to_string());
}
```

Then replace all 4 if/else chains with `pool.getContext(dev_id)`.

### Task 2: `ensureGPUFactoryRegistered(const DeviceId&)` or auto-register in `getContext()`

**Impact**: Eliminates sites #13, #14, #16, #17 (4 sites)

Option A — standalone helper:

```cpp
void ensureGPUFactoryRegistered(const DeviceId& device);
```

Option B — fold into `getContext()` so factory registration is automatic on first access. This is arguably cleaner since the factory should always be registered before a context is used.

### Task 3: `IWorkerGPUContext::clearLastError()`

**Impact**: Eliminates sites #10, #11, #12 (3 sites including include)

```cpp
// In IWorkerGPUContext.h (default no-op for CPU mock contexts)
virtual void clearLastError() {}

// AMDDeviceContext: hipGetLastError()
// NvidiaDeviceContext: cudaGetLastError()
```

This also fixes a latent issue: CUDA graph captures can leave sticky errors too, and we don't clear them today.

### Task 4: `IWorkerGPUContext::validatePointerDevice()` and `dumpPointerDiagnostics()`

**Impact**: Eliminates sites #1, #4, #6, #7, #8, #9 (6 sites including includes)

Move the GPU pointer validation and tracing behind a vendor-neutral virtual method:

```cpp
// In IWorkerGPUContext.h
struct PointerValidationResult {
    bool valid = true;
    int actual_device = -1;
    std::string owner_info;  // human-readable
};

virtual PointerValidationResult validatePointerDevice(void* gpu_ptr, int expected_ordinal) {
    return {true, expected_ordinal, ""};  // default: assume valid
}

virtual void dumpRecentPointerEvents(int count) {}
```

The ROCm implementation would call `hipPointerGetAttributes` + `ROCmBackend::queryPointerOwner`. A CUDA implementation would call `cudaPointerGetAttributes`. The executor code would become:

```cpp
// Before (ROCm-only, ~30 lines of #ifdef)
#ifdef HAVE_ROCM
    hipPointerAttribute_t attr{};
    hipError_t err = hipPointerGetAttributes(&attr, gpu_ptr);
    ...
#endif

// After (vendor-neutral, 3 lines)
auto result = gpu_ctx->validatePointerDevice(gpu_ptr, expected_ordinal);
if (!result.valid) {
    LOG_ERROR("[GPU_PTR_VIOLATION] Stage='" << name << "' " << result.owner_info);
}
```

This also enables CUDA diagnostic parity automatically.

---

## Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| `#ifdef HAVE_ROCM` in graph layer | 12 | 0 |
| `#ifdef HAVE_CUDA` in graph layer | 3 | 0 |
| Runtime vendor branches in graph layer | 3 | 0 |
| CUDA diagnostic parity | None | Full |
| Lines of vendor code in executor | ~200 | ~10 (virtual dispatch overhead) |

---

## Files Modified

| File | Changes |
|------|---------|
| `src/v2/backends/GPUDeviceContextPool.h` | Add `getContext(const DeviceId&)` |
| `src/v2/backends/GPUDeviceContextPool.cpp` | Implement `getContext(const DeviceId&)` with auto factory registration |
| `src/v2/backends/IWorkerGPUContext.h` | Add `clearLastError()`, `validatePointerDevice()`, `dumpRecentPointerEvents()` |
| `src/v2/backends/rocm/AMDDeviceContext.h/.cpp` | Implement `clearLastError()`, `validatePointerDevice()` |
| `src/v2/backends/cuda/NvidiaDeviceContext.h/.cu` | Implement `clearLastError()`, `validatePointerDevice()` |
| `src/v2/execution/local_execution/graph/DeviceGraphExecutor.cpp` | Remove all `#ifdef` blocks, use new APIs |
| `src/v2/execution/local_execution/graph/DeviceGraphCaptureController.cpp` | Remove `#ifdef HAVE_ROCM` + `hip_runtime.h`, use `clearLastError()` |
| `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp` | Use `getContext(DeviceId)` |
| `src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp` | Use `getContext(DeviceId)` |

---

## Scorecard (current state)

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Graph capture abstraction | **A** | `IGPUGraphCapture` + factory method — textbook |
| Capture controller | **A** | 1500 lines, only 2 vendor lines |
| Executor core logic | **B+** | Clean happy path; diagnostics leak vendor details |
| Context acquisition | **B-** | 4 identical if/else chains |
| Diagnostic parity | **C** | ROCm has extensive GPU pointer diagnostics; CUDA has none |
| Factory registration | **B** | Works but boilerplate |
