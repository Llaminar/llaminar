# GPU Decode Overhead Elimination Plan

## Problem Statement

On MI50 (gfx906) with ROCm 7.1.1, Qwen2.5-7B-Q4_0 decode achieves only **21 tok/s** despite the GPU kernels running at **237 tok/s** (4.2ms/token). The ~90% overhead comes from CPU-side dispatch:

| Metric | Value |
|--------|-------|
| Wall-clock per token | 47.7ms |
| Kernel GPU time | 4.2ms |
| CPU overhead | **43.5ms (91%)** |
| Stages per token | 339 |
| HIP launches per token | ~500-700 |
| Per-launch cost (MI50) | ~20-40µs |

### Overhead Breakdown

The Fast Decode Executor (Phase 1 of prior plan) eliminated **~2ms/token** of framework overhead. Remaining sources:

| Source | Estimated ms/token | Root Cause |
|--------|-------------------|------------|
| HIP kernel launch latency | 10-14ms | ~500+ `hipLaunchKernelGGL` calls × 20-40µs each on gfx906 |
| ROCm runtime overhead | 25-30ms | HSA queue management, driver command buffer, hardware dispatch |
| Uncached kernel allocation | 0.1-0.3ms | 120 `make_unique` + destructor per token |
| Redundant `hipSetDevice` | 0.3-1.7ms | 339+ no-op calls per token |
| KernelFactory mutex | ~0.01ms | 72+ lock/unlock cycles |
| `dynamic_cast` (RTTI) | ~0.01ms | 200+ casts per token |

---

## Strategy

Two phases targeting the two overhead categories:

1. **Phase 1: Quick Wins** — Eliminate unnecessary C++ overhead inside `stage->execute()` calls (~1-3ms savings)
2. **Phase 2: HIP Graphs** — Replace 500+ individual kernel launches with a single `hipGraphLaunch` (potential 3-9x decode speedup)

---

## Phase 1: Quick Wins (C++ Stage Overhead)

### Target: 1-3ms/token savings

### 1.1 Cache Ops Kernels in Stages

**Problem**: RMSNormStage, SwiGLUStage, and ResidualAddStage call `KernelFactory::create*()` on every `execute()`, allocating a new `unique_ptr<>` each time. With 24 layers:
- 48 RMSNorm creations/token (2 per layer)
- 24 SwiGLU creations/token
- 48 ResidualAdd creations/token
- **Total: 120 heap allocations + destructions per token**

**Solution**: Add `cached_kernel_` member to each stage (same pattern used by RoPEStage).

| Stage | File | Current | Fix |
|-------|------|---------|-----|
| RMSNormStage | `execution/compute_stages/stages/RMSNormStage.h/.cpp` | `createRMSNorm()` every call (line 82) | Add `cached_kernel_` lazy init |
| SwiGLUStage | `execution/compute_stages/stages/SwiGLUStage.h/.cpp` | `createSwiGLU()` every call (line 68) | Add `cached_kernel_` lazy init |
| ResidualAddStage | `execution/compute_stages/stages/ResidualAddStage.h/.cpp` | `createResidualAdd()` every call (lines 189, 234, 265) | Add `cached_kernel_` lazy init |

**Reference pattern** (from RoPEStage.cpp:107-113):
```cpp
if (!cached_kernel_) {
    auto dev_type = KernelFactory::getDeviceType(params_.device_id);
    cached_kernel_ = KernelFactory::createRoPE(Q_base, dev_type);
}
```

**Stages already correctly cached** (no changes needed):
- RoPEStage: `cached_kernel_` (lazy init)
- EmbeddingStage: `cached_kernel_` (via `getOrCreateKernel()`)
- AttentionComputeStage: `cached_kernel_` (lazy init)
- FusedAttentionWoStage: `kernel_` / `q16_kernel_` (constructor)
- GEMMStage: KernelFactory `getOrCreateGemm()` (global cache)
- FusedQKVGEMMStage: KernelFactory `getOrCreateFusedQKVGemm()` (global cache)
- FusedGateUpGEMMStage: `cached_kernel_` (lazy init)
- LMHeadStage: KernelFactory `getOrCreateGemm()` (global cache)

**Estimated savings**: 0.1-0.3ms/token (120 fewer heap alloc/dealloc per token)

### 1.2 Eliminate Redundant `hipSetDevice`

**Problem**: Every ROCm kernel wrapper calls `hipSetDevice(device_idx)` at entry. On single-GPU decode, all 339+ stages set the same device — pure overhead (~1-5µs each × 339 = 0.3-1.7ms).

**Files with `hipSetDevice` (46 calls across 10 files)**:
| File | Calls |
|------|-------|
| `kernels/rocm/ROCmGemvKernel.hip` | 2 |
| `kernels/rocm/HipBLASGemmKernel.cpp` | 4 |
| `kernels/rocm/ROCmQuantisedGemmKernel_CK.hip` | 2 |
| `kernels/rocm/attention/ROCmFlashAttentionKernels.hip` | 1 |
| `kernels/rocm/kvcache/ROCmRingKVCache.cpp` | 5 |
| `kernels/rocm/ops/ROCmSwiGLUKernels.hip` | 3 |
| `kernels/rocm/ops/ROCmRoPEKernels.hip` | 11 |
| `kernels/rocm/ops/ROCmRoPEKernelT.cpp` | 5 |
| `kernels/rocm/ops/ROCmEmbeddingKernelT.cpp` | 5 |
| `kernels/rocm/ops/ROCmRMSNormKernels.hip` | 3 |
| `kernels/rocm/ops/ROCmResidualAddKernels.hip` | 3 |

**Solution**: Set device once at the executor level before the fast decode loop, and use a thread-local guard to skip redundant calls inside kernels.

**Approach**: Add `hipSetDevice(device_idx)` once in `executeFastDecode()` before the stage loop. Add a `DeviceGuard` utility that tracks the current device per-thread and skips `hipSetDevice` when it's already set. Alternatively, remove `hipSetDevice` from all per-kernel calls and document that callers must set device before calling kernel functions.

**Estimated savings**: 0.3-1.7ms/token

### 1.3 Summary

| Fix | Effort | Savings | Risk |
|-----|--------|---------|------|
| Cache ops kernels | Low (3 stages) | 0.1-0.3ms | None — same pattern as RoPEStage |
| Eliminate hipSetDevice | Medium (10 files) | 0.3-1.7ms | Must ensure device is set at correct level |
| **Total Phase 1** | | **0.4-2.0ms** | |

---

## Phase 2: HIP Graphs

### Target: 3-9x decode speedup (47ms → 5-15ms/token)

### 2.1 Background

HIP Graphs (ROCm equivalent of CUDA Graphs) capture a sequence of GPU operations into a graph object, instantiate it, and replay it with a single API call. This eliminates per-kernel CPU dispatch overhead entirely.

**Key APIs**:
| Phase | API | Purpose |
|-------|-----|---------|
| Capture | `hipStreamBeginCapture(stream, mode)` | Start recording |
| | `hipStreamEndCapture(stream, &graph)` | Produce `hipGraph_t` |
| Instantiate | `hipGraphInstantiate(&exec, graph)` | Compile to executable |
| Launch | `hipGraphLaunch(exec, stream)` | Replay entire graph |
| Update | `hipGraphExecUpdate(exec, new_graph, &info)` | In-place update |
| Per-node | `hipGraphExecKernelNodeSetParams(exec, node, &p)` | Update single node |

### 2.2 Why Llaminar Decode is an Ideal Candidate

| Property | Status | Why It Matters |
|----------|--------|----------------|
| Fixed graph topology | ✅ | Same 339 stages, same order, every decode step |
| Stable buffer pointers | ✅ | `ForwardGraphCache` maintains stable input/output buffers |
| KV cache pointer stability | ✅ | Pre-allocated, write at incrementing offsets |
| Single HIP stream | ✅ | All work on one `hipStreamNonBlocking` stream |
| Minimal dynamic params | ✅ | Only `pos_offset` and `seq_len` change per step |
| No host-device sync in loop | ✅ | `syncLogitsAtBoundary()` only after all stages complete |
| No dynamic allocation | ✅ | Zero `hipMalloc`/`hipFree` on hot path |

### 2.3 What Changes Between Decode Steps

Only 3 stage types have dynamic parameters via `updateDynamicParams()`:

| Stage | Changed Param | Update Method |
|-------|---------------|---------------|
| RoPEStage | `pos_offset` | Passed as kernel argument |
| AttentionComputeStage | `pos_offset`, `seq_len` | Passed as kernel arguments |
| FusedAttentionWoStage | `pos_offset`, `seq_len` | Passed as kernel arguments |

Plus the orchestrator updates:
- `token_ids[0]` — new token ID (stable buffer, contents change)
- `position_ids[0]` — new position (stable buffer, contents change)

**Strategy for parameter updates**:
- **Option A**: Re-capture + `hipGraphExecUpdate()` each step. Since topology is unchanged, the in-place update will succeed. Simpler to implement.
- **Option B**: `hipGraphExecKernelNodeSetParams()` on affected nodes. Requires tracking stage→node mapping. More efficient.
- **Option C**: Store dynamic params in device memory, update via small `hipMemcpyAsync` before graph launch. Kernels read from device memory instead of kernel args. Most efficient — zero graph updates needed.

**Recommended**: Start with Option A (simplest), optimize to Option C if update overhead is measurable.

### 2.4 Integration Architecture

```
DeviceGraphOrchestrator::forward() [cache-hit decode path]
  │
  ├─ updateCachedGraphParams()          // Update pos_offset, seq_len, token_ids
  ├─ graph->reset()                      // Clear completion flags
  │
  └─ if hip_graph_cache_.valid:
  │     ├─ Re-capture (Option A) or update params (Option B/C)
  │     ├─ hipGraphExecUpdate() or hipGraphExecKernelNodeSetParams()
  │     └─ hipGraphLaunch(hip_graph_cache_.exec, stream)
  │
  └─ else (first decode):
        ├─ hipStreamBeginCapture(stream, Relaxed)
        ├─ executeFastDecode()            // All stages launch into captured stream
        ├─ hipStreamEndCapture(stream, &graph)
        ├─ hipGraphInstantiate(&exec, graph)
        ├─ hip_graph_cache_.valid = true
        └─ (first token already computed during capture)
```

### 2.5 New Structures

Add to `DeviceGraphOrchestrator.h` inside `ForwardGraphCache`:
```cpp
struct HIPGraphCache {
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    bool valid = false;
    int consecutive_updates = 0;
    bool disabled = false;           // Fallback if updates fail repeatedly

    void invalidate() {
        if (exec) hipGraphExecDestroy(exec);
        if (graph) hipGraphDestroy(graph);
        exec = nullptr; graph = nullptr;
        valid = false; consecutive_updates = 0;
    }
};
```

### 2.6 Environment Variable Gate

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMINAR_HIP_GRAPHS` | `false` | Enable HIP Graph capture/replay for decode |

Default OFF until validated. Can be promoted to default ON after parity tests pass and benchmarks confirm improvement.

### 2.7 Constraints and Edge Cases

| Constraint | How to Handle |
|------------|---------------|
| **Prefill (variable seq_len)** | Never use graphs for prefill — grid sizes change |
| **TP with RCCL** | RCCL supports stream capture since ROCm 5.x — test on MI50 |
| **`hipSetDevice` during capture** | `hipSetDevice` is a host-only API, not captured — set before capture |
| **`hipMemcpy` (sync) in embedding** | Convert to `hipMemcpyAsync` before capture, or move token_id upload outside capture |
| **Profiling mode** | Skip graph capture when `LLAMINAR_PROFILING=1` (same gate as fast decode) |
| **Snapshot callbacks** | Skip graph capture when snapshots are active |
| **Graph update failure** | Fallback to `executeFastDecode()` (sequential) after 4 consecutive failures (llama.cpp pattern) |
| **Memory overhead** | `hipGraphInstantiate` may allocate driver-internal memory — monitor with `hipMemGetInfo` |

### 2.8 Known Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| MI50 `hipGraphInstantiate` latency | HIGH | Measure empirically — if >50ms, amortization over many tokens still wins |
| `GGML_HIP_GRAPHS` marked "experimental, slow" in llama.cpp | MEDIUM | Different architecture — Llaminar's kernel structure may be more graph-friendly |
| rocBLAS internal streams breaking capture | MEDIUM | Test hipBLAS GEMM capture — if broken, use CK kernels only |
| Debug difficulty (errors at launch, not at kernel) | LOW | Gate behind env var, always test with graphs OFF first |
| Device memory for dynamic params adds complexity | LOW | Start with Option A (re-capture), only optimize if needed |

### 2.9 Expected Performance

| Scenario | Decode ms/token | tok/s | Speedup |
|----------|----------------|-------|---------|
| Current (fast decode executor) | 45.7ms | 21.9 | 1.0x |
| Phase 1 only (quick wins) | ~43-44ms | ~23 | ~1.05x |
| Phase 2 best case | ~5-7ms | 143-200 | 6-9x |
| Phase 2 realistic | ~10-15ms | 67-100 | 3-5x |
| Phase 2 worst case | ~20-25ms | 40-50 | 2x |

---

## Implementation Plan

### Phase 1 Tasks (Quick Wins)

| # | Task | Files | Effort | Dependencies |
|---|------|-------|--------|--------------|
| 1.1 | Add `cached_kernel_` to RMSNormStage | `RMSNormStage.h/cpp` | 30min | None |
| 1.2 | Add `cached_kernel_` to SwiGLUStage | `SwiGLUStage.h/cpp` | 30min | None |
| 1.3 | Add `cached_kernel_` to ResidualAddStage | `ResidualAddStage.h/cpp` | 45min | None |
| 1.4 | Add `DeviceGuard` or set-once pattern for `hipSetDevice` | `kernels/rocm/*.hip`, `executeFastDecode()` | 2hr | None |
| 1.5 | Benchmark Phase 1 | (run benchmark) | 15min | 1.1-1.4 |

### Phase 2 Tasks (HIP Graphs)

| # | Task | Files | Effort | Dependencies |
|---|------|-------|--------|--------------|
| 2.1 | Add `LLAMINAR_HIP_GRAPHS` to DebugEnv | `DebugEnv.h` | 15min | None |
| 2.2 | Add `HIPGraphCache` struct | `DeviceGraphOrchestrator.h` | 30min | None |
| 2.3 | Prototype: simple capture/replay in `executeFastDecode()` | `DeviceGraphExecutor.h/cpp` | 2hr | 2.1, 2.2 |
| 2.4 | Convert sync `hipMemcpy` in embedding to async | `ROCmEmbeddingKernelT.cpp` | 1hr | None |
| 2.5 | Wire graph capture in DeviceGraphOrchestrator | `DeviceGraphOrchestrator.cpp` | 2hr | 2.3 |
| 2.6 | Add graph update mechanism (Option A: re-capture + update) | `DeviceGraphOrchestrator.cpp` | 2hr | 2.5 |
| 2.7 | Benchmark Phase 2 (single GPU) | (run benchmark) | 30min | 2.6 |
| 2.8 | Parity test: greedy output graph ON vs OFF | (run tests) | 30min | 2.7 |
| 2.9 | TP support: verify RCCL capture | `DeviceGraphOrchestrator.cpp` | 2hr | 2.7 |
| 2.10 | Optimize: Option C device-memory params (if needed) | `RoPEStage`, `AttentionStage` kernels | 4hr | 2.7 |

### Validation Plan

| Test | Purpose | Command |
|------|---------|---------|
| Parity tests (6/7 suites) | No regression | `ctest -R "^V2_Integration_Parity_" -E GlobalTP` |
| Greedy output comparison | Token-exact output, graph ON vs OFF | `LLAMINAR_HIP_GRAPHS=0 ./llaminar2 -d rocm:0 -m model.gguf -p "..." -n 50 -t 0` vs `LLAMINAR_HIP_GRAPHS=1 ...` |
| Benchmark | Throughput improvement | `./llaminar2 --benchmark -d rocm:0 -m model.gguf -n 20` |
| Profiling | Kernel throughput unchanged | `LLAMINAR_PROFILING=1 ./llaminar2 --benchmark -d rocm:0 -m model.gguf -n 20` |

---

## Success Criteria

| Metric | Phase 1 | Phase 2 |
|--------|---------|---------|
| Decode throughput (MI50, Qwen2.5-7B-Q4_0) | ≥23 tok/s | ≥50 tok/s |
| Token-exact greedy output match | ✅ | ✅ |
| Parity tests pass (6/7) | ✅ | ✅ |
| No regression in prefill | ✅ | ✅ |

---

## Reference

### Key Source Files

| File | Purpose |
|------|---------|
| `execution/local_execution/graph/DeviceGraphExecutor.h/cpp` | `executeFastDecode()` — primary integration point |
| `execution/local_execution/orchestrators/DeviceGraphOrchestrator.h/cpp` | Forward graph cache, decode orchestration |
| `execution/compute_stages/stages/RMSNormStage.h/cpp` | Uncached kernel creation (line 82) |
| `execution/compute_stages/stages/SwiGLUStage.h/cpp` | Uncached kernel creation (line 68) |
| `execution/compute_stages/stages/ResidualAddStage.h/cpp` | Uncached kernel creation (lines 189, 234, 265) |
| `execution/compute_stages/stages/RoPEStage.h/cpp` | Cached kernel pattern (reference implementation) |
| `kernels/rocm/ops/ROCm*Kernels.hip` | `hipSetDevice` calls in kernel wrappers |
| `backends/rocm/AMDDeviceContext.cpp` | Single HIP stream creation (line 226) |
| `utils/DebugEnv.h` | Environment variable gating |

### llama.cpp Reference

| File | Purpose |
|------|---------|
| `ggml/src/ggml-cuda/common.cuh` (lines 1117-1175) | `ggml_cuda_graph` struct, node property tracking |
| `ggml/src/ggml-cuda/ggml-cuda.cu` (lines 3922-3983) | Graph capture/replay flow, 4-update disable heuristic |

### MI50 (gfx906) Specifics

- ROCm 7.1.1, HIP Graphs supported since ROCm 4.x
- `hipSetDevice` overhead: ~1-5µs (driver call, even when no-op)
- `hipLaunchKernelGGL` overhead: ~20-40µs (vs ~5-10µs on MI200+)
- Wave size: 64
- HBM2 bandwidth: ~900 GB/s
