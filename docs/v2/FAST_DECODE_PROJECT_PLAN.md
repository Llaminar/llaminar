# Fast Decode Executor — Project Plan

**Goal**: Reduce CPU-side stage execution overhead during decode from ~43ms/token to <3ms/token, increasing decode throughput from ~21 tok/s to ~125+ tok/s on MI50 (gfx906).

**Date**: February 2026  
**Branch**: `tensor-parallel`  
**Status**: Planning

---

## Problem Statement

During decode (seq_len=1), each forward pass processes 339 stages through `DeviceGraphExecutor::executeNode()`. Each call performs ~10 operations, most of which are **no-ops during cached decode**:

| Per-Stage Operation | Time (µs) | Needed During Cached Decode? |
|---|---|---|
| `getDumpInfo()` | ~5-10 | **No** — dump info never changes between decode steps |
| `extractInputBuffers()` | ~5-10 | **No** — buffers don't change |
| `extractWeightBuffers()` | ~5-10 | **No** — buffers don't change |
| `extractOutputBuffers()` (pre-execute) | ~5-10 | **No** — buffers don't change |
| `cohereInputs()` | ~10-20 | **No** — tensors are already on GPU, every call is a no-op (dynamic_cast + is_on_device check) |
| `cohereOutputs()` | ~10-20 | **No** — output GPU buffers already allocated |
| `StageDumper::shouldDump()` | ~2-5 | **No** — env vars don't change mid-inference |
| `printStageOutputs()` | ~1-2 | **No** — gated by env var, always false in production |
| Profiling timestamps (×10) | ~5-10 | **No** — profiling disabled in production |
| `execute()` | ~5-15 | **Yes** — the actual kernel launch |
| `markOutputsDirty()` (via extractOutputBuffers + loop) | ~10-20 | **Partially** — need to mark dirty, but can do it cheaper |
| Collective intercept check | ~2-5 | **Only for TP>1** |

**Total overhead**: 339 stages × ~80-130µs = ~27-44ms  
**Kernel time**: ~4.7ms  
**Result**: 90% of forward pass time is wasted on bookkeeping

### Current Numbers (Release Build, MI50, Qwen2.5-7B-Q4_0)

| Metric | Current | Target (Phase 1) | Target (Phase 2) |
|---|---|---|---|
| Decode wall-clock/token | 47.8ms | ~7-8ms | <5ms |
| Decode throughput | 21 tok/s | 125-143 tok/s | 200+ tok/s |
| Kernel time/token | 4.7ms | 4.7ms (unchanged) | 4.7ms (unchanged) |
| CPU overhead/token | 43ms | 2-3ms | <0.5ms |
| Overhead ratio | 90% | 30-40% | <10% |

---

## Architecture

### Current Execution Path (Cached Decode)

```
DeviceGraphOrchestrator::forward()
  → forward_cache_ HIT
  → updateCachedGraphParams()          // Updates pos_offset, seq_len in all stages
  → forward_cache_.graph->reset()      // Clears completion flags
  → executor_.execute(graph, ctx)      // <-- THIS IS THE BOTTLENECK
      → executeSequential()
          → for each of 339 stages:
              → executeNode()          // ~127µs of overhead per stage
                  → getDumpInfo()
                  → extractInputBuffers() + cohereInputs()
                  → extractWeightBuffers() + cohereInputs()
                  → extractOutputBuffers() + cohereOutputs()
                  → shouldDump()
                  → stage->execute()   // ~14µs avg actual kernel work
                  → extractOutputBuffers() + markOutputsDirty()
                  → printStageOutputs()
                  → profiling stats
  → syncLogitsAtBoundary()             // hipStreamSynchronize
```

### Proposed Phase 1 Path (Fast Decode)

```
DeviceGraphOrchestrator::forward()
  → forward_cache_ HIT
  → updateCachedGraphParams()
  → forward_cache_.graph->reset()
  → executor_.executeFastDecode(graph, ctx)   // NEW: minimal loop
      → for each of 339 stages:
          → stage->execute(ctx)               // Just the kernel launch
      → // No coherence, no dumps, no profiling, no validation
  → syncLogitsAtBoundary()
```

### Proposed Phase 2 Path (HIP Graph)

```
DeviceGraphOrchestrator::forward()
  → forward_cache_ HIT
  → updateCachedGraphParams()
  → hipGraphLaunch(cached_hip_graph, stream)  // Single API call replaces 339 launches
  → syncLogitsAtBoundary()
```

---

## Phase 1: Fast Decode Executor

**Priority**: HIGH  
**Effort**: 1-2 days  
**Risk**: Low  
**Expected improvement**: 21 → 125-143 tok/s

### 1.1 — Add `executeFastDecode()` to DeviceGraphExecutor

**File**: `src/v2/execution/local_execution/graph/DeviceGraphExecutor.h`

Add a new method to the `DeviceGraphExecutor` class:

```cpp
/**
 * @brief Execute a cached decode graph with minimal overhead
 *
 * This is a stripped-down execution path for cached decode graphs where:
 * - All tensors are already on the target GPU device
 * - Output GPU buffers are already allocated
 * - No stage dumping, validation, or profiling is needed
 * - Stage objects are reused (not rebuilt)
 *
 * Skips: getDumpInfo, extractBuffers, cohereInputs/Outputs,
 *        shouldDump, printStageOutputs, profiling, assertions
 *
 * @param graph The cached compute graph (stages already configured)
 * @param ctx Device context for execution
 * @return true on success
 */
bool executeFastDecode(ComputeGraph &graph, IDeviceContext *ctx);
```

**File**: `src/v2/execution/local_execution/graph/DeviceGraphExecutor.cpp`

Implementation — a tight loop that only calls `execute()` and handles collective intercept:

```cpp
bool DeviceGraphExecutor::executeFastDecode(ComputeGraph &graph, IDeviceContext *ctx)
{
    auto order = graph.getExecutionOrder();

    for (const auto &name : order)
    {
        auto *node = graph.getNode(name);

        // Collective intercept (TP>1 only)
        if (collective_ctx_)
        {
            auto stage_type = node->stage->type();
            if (stage_type == ComputeStageType::ALLREDUCE)
            {
                if (!executeCollectiveAllreduce(*node, ctx))
                    return false;
                graph.markCompleted(name);
                continue;
            }
            else if (stage_type == ComputeStageType::ALLGATHER)
            {
                if (executeCollectiveStridedAllgather(*node, ctx))
                {
                    graph.markCompleted(name);
                    continue;
                }
            }
        }

        // Direct execute — no coherence, no dumps, no profiling
        if (!node->stage->execute(ctx))
        {
            LOG_ERROR("[DeviceGraphExecutor] Fast decode stage failed: " << name);
            return false;
        }

        graph.markCompleted(name);
    }

    return true;
}
```

**Key insight**: We can skip `markOutputsDirty()` entirely in this path because:
1. During cached decode, all tensors remain on GPU between steps
2. No code between stages calls `data()` (which would trigger D2H sync)
3. Only `syncLogitsAtBoundary()` at the end does a stream sync
4. The logits tensor uses mapped memory with `markMappedSynced()` — no coherence state needed

However, if we later need to support debug tools during fast decode, we should add a lightweight dirty-marking path. Test this assumption by running parity tests.

### 1.2 — Pre-compute Collective Node Set

To avoid the `stage->type()` virtual call + enum comparison on every node (339 per pass), pre-compute which nodes are collective stages when the graph is cached.

**File**: `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h`

Add to `ForwardCache` struct:

```cpp
struct ForwardCache {
    // ... existing fields ...
    std::unordered_set<std::string> collective_nodes;  // Pre-computed collective stage names
};
```

**File**: `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`

When building the cache, populate `collective_nodes`:

```cpp
// After graph build, before cache.valid = true:
for (const auto &name : graph->getExecutionOrder()) {
    auto *node = graph->getNode(name);
    auto type = node->stage->type();
    if (type == ComputeStageType::ALLREDUCE ||
        type == ComputeStageType::ALLGATHER ||
        type == ComputeStageType::ALLGATHER_V)
    {
        cache.collective_nodes.insert(name);
    }
}
```

Then pass this set to `executeFastDecode()` or store it in the executor config:

```cpp
bool executeFastDecode(ComputeGraph &graph, IDeviceContext *ctx,
                       const std::unordered_set<std::string> *collective_nodes = nullptr);
```

### 1.3 — Wire Into DeviceGraphOrchestrator Cache-Hit Path

**File**: `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`

In the cached decode code path (~line 610), replace `executor_.execute()` with `executor_.executeFastDecode()`:

```cpp
// Current:
bool success = executor_.execute(*forward_cache_.graph, ctx);

// New:
bool success = executor_.executeFastDecode(
    *forward_cache_.graph, ctx,
    &forward_cache_.collective_nodes);
```

### 1.4 — Environment Variable Gate

Initially gate this behind `LLAMINAR_FAST_DECODE` for safe rollout.

**File**: `src/v2/utils/DebugEnv.h`

Add to `ExecutionConfig`:

```cpp
struct ExecutionConfig {
    // ... existing fields ...
    bool fast_decode = true;  // Default ON — disable with LLAMINAR_FAST_DECODE=0

    void reload() {
        // ... existing ...
        fast_decode = getEnvBool("LLAMINAR_FAST_DECODE", true);
    }
};
```

**File**: `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`

```cpp
bool use_fast_decode = debugEnv().execution.fast_decode && !config_.enable_profiling;
if (use_fast_decode) {
    success = executor_.executeFastDecode(*forward_cache_.graph, ctx,
                                          &forward_cache_.collective_nodes);
} else {
    success = executor_.execute(*forward_cache_.graph, ctx);
}
```

Note: Disable fast decode when profiling is enabled so per-stage timing still works.

### 1.5 — Verification

1. **Correctness**: Run parity tests with fast decode enabled (default):
   ```bash
   ctest --test-dir build_v2_integration -R "V2_Integration_Parity_Qwen2_SingleDevice" -V
   ```

2. **Greedy output match**: Compare output text with fast decode on vs off:
   ```bash
   # Fast decode ON (default)
   ./build_v2_release/llaminar2 -d rocm:0 -m model.gguf -p "What is the capital of France?" -n 20 -t 0

   # Fast decode OFF
   LLAMINAR_FAST_DECODE=0 ./build_v2_release/llaminar2 -d rocm:0 -m model.gguf -p "What is the capital of France?" -n 20 -t 0
   ```

3. **Performance benchmark**:
   ```bash
   ./build_v2_release/llaminar2 --benchmark -d rocm:0 -m model.gguf -p "Hello world" -n 50 -t 0
   ```

4. **Validate with assertions** (one-time check):
   ```bash
   LLAMINAR_VALIDATE_BUFFERS=1 LLAMINAR_FAST_DECODE=1 \
   ./build_v2_integration/llaminar2 -d rocm:0 -m model.gguf -p "Hello" -n 5 -t 0
   ```

---

## Phase 2: HIP Graph Capture (Future)

**Priority**: MEDIUM  
**Effort**: 2-3 weeks  
**Risk**: Medium (gfx906 maturity, RCCL capture compat)  
**Expected improvement**: Phase 1 result → 200+ tok/s

### 2.1 — Overview

Layer HIP Graph capture on top of Phase 1's fast decode path. On the first cached decode step, capture the kernel launch sequence as a HIP Graph. On subsequent steps, replay the graph with a single `hipGraphLaunch()` call.

### 2.2 — Key APIs

| HIP API | Purpose |
|---|---|
| `hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed)` | Start recording kernel launches |
| `hipStreamEndCapture(stream, &graph)` | Stop recording, get graph handle |
| `hipGraphInstantiate(&exec, graph, 0)` | Compile graph into executable form |
| `hipGraphLaunch(exec, stream)` | Replay all captured kernels |
| `hipGraphExecUpdate(exec, graph, &result)` | Update executable with new params |
| `hipGraphExecKernelNodeSetParams(exec, node, &params)` | Update single kernel's params |
| `hipGraphDestroy(graph)` / `hipGraphExecDestroy(exec)` | Cleanup |

### 2.3 — Parameter Update Strategy

Each decode step changes:
- `token_ids[0]` — the new token to process (affects embedding lookup)
- `position_ids[0]` — incremented by 1 (affects RoPE)
- `position_offset` — incremented by 1 (affects KV cache write position, attention masking)

Options:
1. **GPU-side indirection**: Store changing values in GPU memory, kernels read from pointers (no graph update needed). Requires modifying stages to accept pointers-to-pointers.
2. **`hipGraphExecKernelNodeSetParams`**: Update individual kernel node params. Requires enumerating nodes and mapping them to stages.
3. **`hipGraphExecUpdate`**: Re-capture graph and do in-place update if structure matches. Simplest but may have overhead.
4. **Recapture**: Destroy and re-capture if params change. Acceptable if recapture is rare (e.g., only on batch boundary).

**Recommended**: Start with option 3 (`hipGraphExecUpdate`). If overhead is too high, move to option 1 for the 3 changing parameters.

### 2.4 — Constraints

| Constraint | Impact | Mitigation |
|---|---|---|
| No `hipMalloc` during capture | All GPU buffers must be pre-allocated | Already done — `DeviceWorkspaceManager` allocates on first decode |
| No `hipMemcpy(H2D)` sync | Host→device copies drain pipeline | Already fixed (inv_freq), embedding uses `hipMemcpyAsync` |
| No `hipDeviceSynchronize` | Full device sync breaks capture | Audit all kernel paths (see §2.6) |
| RCCL may not capture | Multi-GPU collectives can fail capture | Piecewise capture between collectives (vLLM pattern) |
| MI50 (gfx906) maturity | llama.cpp disables graphs below Ampere | Test explicitly; add arch gate if needed |

### 2.5 — Implementation Outline

Create `HipGraphCapture` class in `src/v2/execution/local_execution/graph/`:

```cpp
class HipGraphCapture {
public:
    enum class State { IDLE, CAPTURING, READY, FAILED };

    // Capture the fast decode execution
    bool beginCapture(hipStream_t stream);
    bool endCapture(hipStream_t stream);

    // Replay captured graph
    bool launch(hipStream_t stream);

    // Update for new decode step (returns false if recapture needed)
    bool update();

    // Invalidate (e.g., on phase transition, KV cache resize)
    void invalidate();

    State state() const;

private:
    hipGraph_t graph_ = nullptr;
    hipGraphExec_t exec_ = nullptr;
    State state_ = State::IDLE;
    int consecutive_update_failures_ = 0;
    static constexpr int MAX_UPDATE_FAILURES = 4;  // llama.cpp's heuristic
};
```

### 2.6 — Sync Audit Required

Before capture, audit all ROCm kernel files for operations that break stream capture:

| File | Issue | Status |
|---|---|---|
| `ROCmRoPEKernels.hip` | `hipOps_rope_populate_inv_freq` — was sync hipMemcpy, now GPU kernel | ✅ Fixed |
| `ROCmRoPEKernelT.cpp` | Non-contiguous position_ids — was sync hipMemcpy, now hipMemcpyAsync | ✅ Fixed |
| `ROCmEmbeddingKernelT.cpp:243` | `hipMemcpy(d_token_ids, ...)` — sync H2D per forward pass | ❌ Needs fix |
| `ROCmQuantisedGemmKernel_CK.hip:1966` | `quantizeActivations()` sync H2D — prefill M>1 only | ⚠️ Decode M=1 uses GEMV, skips this |
| `ROCmQuantisedGemmKernel_CK.hip:1981` | `applyScaling()` sync D2H — prefill M>1 only | ⚠️ Decode M=1 uses GEMV, skips this |
| `TensorBase.cpp` | `ensureOnHost()` uses `hipEventSynchronize` | ⚠️ Not called during fast decode |
| `TensorBase.cpp` | `ensureOnDevice()` may use sync copy | ⚠️ Not called during fast decode |

### 2.7 — Verification

Same as Phase 1, plus:
- `rocprof --hip-trace` to verify single graph launch per forward pass
- Compare kernel timing with/without graph capture
- Test with `LLAMINAR_HIP_GRAPH=0` fallback

---

## Phase 3: Optimizations (Future Future)

**Priority**: LOW  
**Effort**: 2-3 weeks

### 3.1 — `getExecutionOrder()` Caching

Currently `getExecutionOrder()` is called every forward pass and returns a `std::vector<std::string>`. For a cached graph, this is always the same order. Cache it.

### 3.2 — Flat Stage Array

Replace the `std::vector<std::string>` order + `getNode(name)` (hash map lookup) with a flat `std::vector<ComputeNode*>` built once during cache setup.

### 3.3 — Embedding Token ID Upload

`ROCmEmbeddingKernelT.cpp:243` uses synchronous `hipMemcpy` to upload `token_ids` every forward pass. For decode (single token), this is a tiny copy but still a sync point. Fix:
- Use `hipMemcpyAsync` with pinned host memory for token_ids
- Or pass token_id as a kernel argument directly (it's just one int during decode)

### 3.4 — Fused Layer Execution

For single-device decode with no TP, fuse groups of stages into "super-stages":
- `attn_norm` + `qkv_proj` → single kernel launch
- `rope` + `kv_append` + `attention` + `wo_proj` → fused attention
- `ffn_norm` + `gate_up_proj` + `swiglu` + `down_proj` → fused FFN

This reduces stage count from ~339 to ~56 (28 layers × 2 fused blocks).

### 3.5 — Copy-Indirection for HIP Graphs

llama.cpp style: store mutable parameters (position_offset, KV write index) in GPU-side buffers. Kernels read from these buffers instead of scalar arguments. This eliminates the need for graph param updates entirely — just update the GPU buffer, and the captured graph reads the new values automatically.

---

## File Inventory

### Files to Modify (Phase 1)

| File | Change |
|---|---|
| `src/v2/execution/local_execution/graph/DeviceGraphExecutor.h` | Add `executeFastDecode()` declaration |
| `src/v2/execution/local_execution/graph/DeviceGraphExecutor.cpp` | Implement `executeFastDecode()` |
| `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h` | Add `collective_nodes` to `ForwardCache` |
| `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp` | Populate `collective_nodes` on cache build; call `executeFastDecode()` on cache hit |
| `src/v2/utils/DebugEnv.h` | Add `fast_decode` to `ExecutionConfig` |

### Files to Create (Phase 2)

| File | Purpose |
|---|---|
| `src/v2/execution/local_execution/graph/HipGraphCapture.h` | HIP Graph wrapper class |
| `src/v2/execution/local_execution/graph/HipGraphCapture.cpp` | Implementation |

### Test Files

| File | Purpose |
|---|---|
| `tests/v2/unit/Test__FastDecodeExecutor.cpp` | Unit test for executeFastDecode path |
| `tests/v2/integration/Test__FastDecodeE2E.cpp` | End-to-end decode output comparison |

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| Skipping `markOutputsDirty()` causes stale data | `syncLogitsAtBoundary()` does full stream sync before logits are read. Intermediate tensors are only consumed by GPU kernels (never host). Validate with parity tests. |
| Skipping coherence breaks TP/PP | Fast decode only activates on single-device cached decode path. Multi-device paths use normal executor. |
| Edge case: stage modifies host data | Audit all stage `execute()` implementations for host-side data access. None should — all GPU stages operate on device pointers. |
| Performance regression if fast decode has a bug | Gate behind `LLAMINAR_FAST_DECODE` env var (default ON). Easy to disable. |
| HIP Graph capture fails on gfx906 | Phase 2 has automatic fallback to Phase 1 fast decode. |

---

## Success Criteria

### Phase 1
- [ ] Decode throughput ≥ 100 tok/s on MI50 with Qwen2.5-7B-Q4_0
- [ ] All parity tests pass with fast decode enabled
- [ ] Greedy output matches between fast decode ON and OFF
- [ ] No memory leaks or GPU errors under ASAN
- [ ] `LLAMINAR_FAST_DECODE=0` correctly falls back to full executor

### Phase 2
- [ ] Decode throughput ≥ 180 tok/s on MI50
- [ ] HIP Graph capture succeeds on gfx906
- [ ] Automatic fallback when capture fails
- [ ] `LLAMINAR_HIP_GRAPH=0` correctly disables graph capture
