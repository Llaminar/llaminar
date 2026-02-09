# GPU Graph Decode Speedup — Research Plan

**Date**: 2026-02-09
**Context**: GPU graph capture/replay on MI50 (ROCm) and RTX 3090 (CUDA) is now **correct** after fixing `FusedGateUpGemmAdapter::setGPUStream()` propagation, but throughput is **neutral** because transition overhead between alternating capturable/manual segments offsets the kernel-launch savings from graph replay.

**Current numbers** (MI50 0.5B Q4_0): baseline 53.66 tok/s, GPU graphs 53.65 tok/s — zero improvement.

**Root cause of neutrality**: A 28-layer model produces ~57 capturable + ~28 manual = ~85 segments with ~56 cross-type transitions. Each transition requires a `synchronizeStream()` call (~5-10μs CPU stall), totalling ~280-560μs per decode step. This wipes out the ~200-300μs saved by graph replay avoiding per-kernel `hipLaunchKernelGGL` overhead.

---

## Table of Contents

1. [Strategy A: Make Attention Stages Graph-Capturable](#strategy-a-make-attention-stages-graph-capturable)
2. [Strategy B: Event-Based Sync Instead of Stream Synchronize](#strategy-b-event-based-sync-instead-of-stream-synchronize)
3. [Strategy C: Stage Coalescing / Segment Reduction](#strategy-c-stage-coalescing--segment-reduction)
4. [Combined Impact Analysis](#combined-impact-analysis)
5. [Recommended Implementation Order](#recommended-implementation-order)

---

## Strategy A: Make Attention Stages Graph-Capturable

### Status Quo

Seven stage types explicitly return `isGraphCapturable() = false`:

| Stage | Reason | File |
|-------|--------|------|
| `FusedAttentionWoStage` | `position_offset` changes each step | `stages/FusedAttentionWoStage.h:102` |
| `AttentionComputeStage` | KV length changes each step | `stages/AttentionComputeStage.h:82` |
| `AttentionWithKVCacheStage` | KV length + cache mutation | `stages/AttentionWithKVCacheStage.h:98` |
| `KVCacheAppendStage` | KV lengths change each step | `stages/KVCacheAppendStage.h:70` |
| `KVCacheGatherStage` | KV lengths change each step | `stages/KVCacheGatherStage.h:44` |
| `RoPEStage` | `pos_offset` changes each step | `stages/RoPEStage.h:69` |
| `EmbeddingStage` | `token_ids` change each step | `stages/EmbeddingStage.h:57` |

Base class default is `true` (in `IComputeStage.h:490`).

### Barrier Analysis

For each non-capturable stage, the specific technical blockers are:

#### AttentionComputeStage (primary GPU attention path)
1. **Host-side `kv_len` query** — calls `kv_cache->get_cached_tokens()` at runtime (`AttentionComputeStage.cpp:87`). This host read determines effective KV length, which changes every decode step.
2. **Host-side causal mask allocation** — `std::make_unique<FP32Tensor>(...)` at `AttentionComputeStage.cpp:160`. Allocates and fills a mask tensor **on the CPU every call**. Completely incompatible with stream capture.
3. **Dynamic `num_splits`** — Flash Decoding grid at `CUDAFlashAttentionKernelT.cpp:310` selects split count via host-side `if/else` on `kv_len`. Changes grid dimensions.

#### RoPEStage
1. **Scalar `pos_offset` parameter** — passed as a host scalar to the GPU kernel. Increments by `seq_len` each decode step. The blocker is trivial: the kernel reads the value as a launch parameter.

#### KVCacheAppendStage
1. **Cache state mutation** — `kv_cache->append()` modifies ring buffer head pointer on the host.
2. **Dynamic write position** — depends on current cache length.

#### EmbeddingStage
1. **Fundamentally different inputs** — `token_ids` change every step. Not fixable.

### Key Insight: Grid Dimensions Are Already Static

The CUDA Flash Decoding kernel uses grid `dim3(n_heads, num_splits, batch_size)` (`CUDAFlashAttentionKernels.cu:960`). For decode mode:
- `n_heads`: constant (model architecture)
- `num_splits`: host-selected but could be fixed at max (e.g., 8)
- `batch_size`: constant (1 for decode)

The variable `kv_len` is a **kernel argument**, NOT a grid dimension. This means the kernel launch shape is already graph-capturable — only the argument value changes.

### Proposed Solution: Graph Node Parameter Updates

CUDA and HIP both support updating kernel arguments on an instantiated graph without re-capturing:
- `cudaGraphExecKernelNodeSetParams()` — CUDA
- `hipGraphExecKernelNodeSetParams()` — HIP/ROCm

**Approach**: Capture the attention kernel with initial parameters, then update `kv_len` and `position_offset` between graph replays via parameter update API. No re-capture needed.

#### Sub-tasks Required

1. **Eliminate host-side mask allocation**: The attention kernel already supports `causal=true` mode with kernel-internal masking. Remove the host-side `FP32Tensor` mask allocation and pass `causal=true` flag instead.

2. **Fix `num_splits` for decode**: Set `num_splits = 8` (or the maximum) for all decode steps. The kernel handles shorter KV sequences correctly with the split mechanism — extra splits just return zeros.

3. **Move `kv_len`/`position_offset` to device-side params**: Either:
   - Use graph node parameter update API (preferred — zero re-capture cost)
   - OR maintain a device-side parameter buffer updated via `cudaMemcpyAsync` before graph launch

4. **Pre-resolve KV cache pointers**: `CUDARingKVCache` uses stable device pointers (`CUDARingKVCache.h:84`). Resolve once during graph setup, not per-execute.

5. **Create `GraphCapturableAttentionStage`**: New stage class wrapping the above patterns, returning `isGraphCapturable() = true` for decode mode only.

### Impact Estimate

Making attention capturable would merge the 3-stage manual block (`rope + kv_append + attention`) into the surrounding capturable segments. Per-layer pattern would change from:

```
Current:  [C,C] [M,M,M] [C,C,C,C,C,C,C]  (3 segments per layer)
After:    [C,C,C,C,C,C,C,C,C,C,C,C]       (1 segment per layer, merges across layers)
```

This would reduce from ~85 segments / ~56 transitions to ~3 segments / ~2 transitions (embedding, one massive capturable, final sync). **Saves ~270-540μs per decode step.**

### Complexity: HIGH

- Requires new graph-aware attention stage class
- Must integrate with graph node parameter update API (not yet used anywhere)
- Must handle KV cache append within graph or as a special pre-graph operation
- Must handle RoPE position offset within graph
- Must eliminate all host-side allocations from attention execute path
- Risk of correctness regressions in attention (numerically sensitive)

### Prefill: NOT a Candidate

Prefill `seq_len` varies, causing grid dimension changes. Graph capture for prefill is not feasible without major kernel redesign. Focus on decode only.

---

## Strategy B: Event-Based Sync Instead of Stream Synchronize

### Status Quo

Phase 3 replay uses `gpu_ctx->synchronizeStream()` at type transitions:

```cpp
// manual→graph transition (GraphExecutor.cpp:1063)
gpu_ctx->synchronizeStream(nullptr);  // blocks CPU until stream 0 completes

// graph→manual transition (GraphExecutor.cpp:1314)
gpu_ctx->synchronizeStream(capture_stream);  // blocks CPU until capture stream completes
```

These are CPU-blocking operations:
- CUDA: `cudaStreamSynchronize()` (~5μs, blocks CPU thread)
- ROCm: `hipStreamSynchronize()` (~5-10μs, blocks CPU thread)

### Key Finding: Event API Already Exists But Is Unused

The codebase has a complete event-based inter-stream dependency API that was **built for exactly this purpose** but never wired in:

**`IWorkerGPUContext::insertStreamDependency()`** (`IWorkerGPUContext.h:293-307`):
- Records an event on `dependency_stream`, makes `dependent_stream` wait for it
- GPU-side only — **no CPU blocking**
- Comment says: *"Used by segmented graph capture to order graph launches (on capture_stream) with manual stage dispatches (on legacy stream 0) without CPU overhead."*

**Implementations exist**:
- CUDA: `NvidiaDeviceContext.cu:552-588` — `cudaEventCreateWithFlags(cudaEventDisableTiming)` + `cudaEventRecord` + `cudaStreamWaitEvent`
- ROCm: `AMDDeviceContext.cpp:509-540` — `hipEventCreateWithFlags(hipEventDisableTiming)` + `hipEventRecord` + `hipStreamWaitEvent`

**Cached sync event exists**:
- `GraphSegmentCache::sync_event` (`GraphExecutor.h:450`) with `ensureSyncEvent()`/`destroySyncEvent()` methods (`GraphExecutor.cpp:73-95`)
- Created and destroyed but **never used for recording/waiting**

### Proposed Change

Replace `synchronizeStream()` with event record/wait at transitions:

```cpp
// Before replay loop:
segment_cache.ensureSyncEvent(gpu_ctx);

// manual→graph transition:
gpu_ctx->recordEvent(segment_cache.sync_event, nullptr);         // mark stream 0 progress
gpu_ctx->waitEvent(segment_cache.sync_event, capture_stream);    // GPU-side wait on capture_stream

// graph→manual transition:
gpu_ctx->recordEvent(segment_cache.sync_event, capture_stream);  // mark capture_stream progress
gpu_ctx->waitEvent(segment_cache.sync_event, nullptr);           // GPU-side wait on stream 0

// Final sync unchanged — CPU must wait before reading output:
gpu_ctx->synchronize();
```

Or equivalently using the higher-level API:
```cpp
// manual→graph:
gpu_ctx->insertStreamDependency(capture_stream, nullptr);

// graph→manual:
gpu_ctx->insertStreamDependency(nullptr, capture_stream);
```

### Cost Comparison

| Operation | Per-call cost | × 56 transitions | Total |
|-----------|--------------|-------------------|-------|
| `synchronizeStream` (current) | ~5-10μs | 56 | **280-560μs** |
| Event record + wait (cached event) | ~0.1μs | 56 | **~5.6μs** |
| `insertStreamDependency` (creates/destroys event) | ~0.3-0.5μs | 56 | **~17-28μs** |
| **Savings** | | | **~250-555μs** |

### Implementation Complexity: VERY LOW

- **~10 lines** changed in one file (`GraphExecutor.cpp`)
- All API infrastructure already exists and is tested
- Event reuse pattern (single cached event, alternating record/wait) is safe
- Add env var `LLAMINAR_GPU_GRAPH_EVENT_SYNC` for A/B testing

### Risks

1. **Event reuse correctness**: Single event, alternating record/wait. Safe because each `recordEvent` overwrites previous state, and each event is consumed by `waitEvent` before the next `recordEvent`.

2. **GPU work overlap**: With event-based sync, the CPU won't wait for GPU completion between segments. This means the CPU can race ahead issuing more graph launches. Should be fine since graph replay is asynchronous by design.

3. **`createEvent()` uses full timing**: The existing `createEvent()` at `NvidiaDeviceContext.cu:432` uses `cudaEventCreate()` with timing enabled. For minimal overhead, the `ensureSyncEvent` should use `cudaEventCreateWithFlags(cudaEventDisableTiming)`. The `insertStreamDependency` implementation already does this.

4. **Debug visibility**: With event-based sync, GPU execution may overlap across segment boundaries, making GPU debuggers (compute-sanitizer) harder to use. Keep `synchronizeStream` as fallback via env var.

---

## Strategy C: Stage Coalescing / Segment Reduction

### Current Segmentation

The segmentation algorithm (`GraphExecutor.cpp:822-848`) already performs cross-layer merging — consecutive capturable stages are grouped regardless of layer boundaries.

Per-layer pattern (GPU decomposed path):
```
[C] attn_norm, [C] qkv_proj, [M] rope, [M] kv_append, [M] attention, [C] wo_proj, [C] attn_residual, [C] ffn_norm, [C] gate_up_proj, [C] swiglu, [C] down_proj, [C] ffn_residual
```

Cross-layer merge: layer N-1's `ffn_residual` (C) merges with layer N's `attn_norm` (C) and `qkv_proj` (C).

**Actual pattern**: 3 segments per layer (capturable → manual → capturable), with adjacent capturables merging across boundaries:

| Segment | Content | Type | Stages |
|---------|---------|------|--------|
| 1 | embedding | Manual | 1 |
| 2 | layer0: attn_norm, qkv_proj | Capturable | 2 |
| 3 | layer0: rope, kv_append, attention | Manual | 3 |
| 4 | layer0: wo_proj...ffn_residual + layer1: attn_norm, qkv_proj | **Capturable (merged)** | 9 |
| 5 | layer1: rope, kv_append, attention | Manual | 3 |
| ... | ... | ... | ... |
| 84 | layer27: wo_proj...ffn_residual + final_norm, lm_head | Capturable (merged) | 9 |

Total: ~29 capturable + ~29 manual = ~58 segments, ~57 transitions.

### Key Opportunity: Make RoPE Capturable

RoPE's only blocker is the scalar `pos_offset` parameter. If RoPE becomes capturable, it merges with the pre-attention capturable run:

```
Before: [C: attn_norm, qkv_proj] [M: rope, kv_append, attention] [C: wo_proj...]
After:  [C: attn_norm, qkv_proj, rope] [M: kv_append, attention] [C: wo_proj...]
```

This still leaves 3 segments per layer. However, the first capturable segment now has 3 stages and still merges with the previous layer's tail capturable segment. The per-layer manual segment shrinks from 3 to 2 stages.

**Impact**: Reduces manual segments from 3 stages to 2 stages each, but **does NOT reduce segment count** (still 3 per layer). The number of transitions stays the same.

Wait — actually, if RoPE is capturable AND `kv_append + attention` remain manual, the pattern is still alternating C/M/C per layer. No segment count reduction.

### Actual High-Value Coalescing: Attention + KVCache Combined

If we could make the entire `[rope, kv_append, attention]` block capturable (Strategy A), that would collapse each layer to a single capturable segment, reducing from ~58 to ~3 segments total.

### Existing Stage Fusion

Already fused:
- `FusedQKVGEMMStage`: Q+K+V projections (shared input quantization)
- `FusedGateUpGEMMStage`: Gate+Up projections (shared input quantization)
- `FusedAttentionWoStage`: Attention+Wo (CPU-only, JIT)

These fuse stages **within the same capturable segment** — they reduce kernel launches (graph nodes) but not segment count.

### Kernel-Level Fusion Opportunities

| Fusion | Saves | Impact | Complexity |
|--------|-------|--------|------------|
| attn_norm + qkv_proj (RMSNorm + GEMM) | 1 kernel launch | ~2μs/layer | HIGH |
| ffn_norm + gate_up_proj (RMSNorm + GEMM) | 1 kernel launch | ~2μs/layer | HIGH |
| swiglu + down_proj (Activation + GEMM) | 1 kernel launch | ~2μs/layer | HIGH |
| wo_proj + attn_residual (GEMM + Add, β≠0) | 1 kernel launch | ~2μs/layer | MODERATE |

All of these are within capturable segments, so they reduce graph node count but not segment transitions. At 28 layers × 2μs saved = ~56μs — marginal.

### Conclusion for Strategy C

Stage coalescing has **limited standalone value** because:
1. Cross-layer merging already happens automatically
2. The non-capturable attention block creates hard segment boundaries
3. Kernel fusion reduces graph nodes (~0.5μs each) but not segment transitions (~5-10μs each)

The only coalescing that meaningfully reduces transitions is making entire stage blocks capturable (Strategy A).

---

## Combined Impact Analysis

| Strategy | Segment Reduction | Transition Savings | Time Savings | Complexity | Standalone Value |
|----------|-------------------|-------------------|-------------|-----------|-----------------|
| **B: Event-based sync** | Same segments | Same transitions but ~50× cheaper | ~250-555μs | VERY LOW | **HIGH** |
| **C: Make RoPE capturable** | 58 → 58 | 0 | 0 (same transitions) | LOW | LOW |
| **A: Make attention capturable** | 58 → ~3 | ~55 transitions eliminated | ~275-550μs | HIGH | **HIGH** |
| **C: Kernel fusion** | Same | 0 | ~56μs (node overhead) | HIGH | LOW |

### **Interaction effects**:

- **B alone**: Saves ~250-555μs immediately. No other changes needed.
- **A alone**: Saves ~275-550μs by eliminating transitions entirely. B becomes unnecessary if A succeeds.
- **B + A together**: B covers the residual 2-3 transitions that remain even after A.
- **C alone**: Negligible impact on segmentation.

### Decode budget context

For the 0.5B model at 53 tok/s → 18.9ms per token:
- ~280-560μs overhead = **1.5-3.0%** of decode time
- Eliminating this overhead → theoretical max improvement: **+0.8-1.6 tok/s** (to ~54.5-55.2 tok/s)

For the 7B model at 16 tok/s → 62.5ms per token:
- ~280-560μs overhead = **0.45-0.9%** of decode time
- Marginal improvement

**The transition overhead is a small fraction of total decode time.** The bulk of decode time is spent in actual kernel computation (GEMM, attention), not in launch/sync overhead.

---

## Recommended Implementation Order

### Phase 1: Event-Based Sync (Strategy B) — DO FIRST
- **Effort**: ~2 hours, ~10 lines in one file
- **Risk**: Very low — infrastructure already built and tested
- **Value**: Immediate ~250-555μs savings, confirms measurement methodology
- **Deliverable**: A/B benchmarkable via `LLAMINAR_GPU_GRAPH_EVENT_SYNC` env var

### Phase 2: Measure Actual Overhead — BEFORE Strategy A
After Phase 1, instrument the replay loop to measure:
1. Actual time spent in graph launches vs manual execution
2. Time spent in remaining sync operations
3. GPU utilization during decode (are we compute-bound or launch-bound?)

If GPU utilization is already >90%, graph optimizations have diminishing returns. The real bottleneck may be kernel compute efficiency, not launch overhead.

### Phase 3: Graph-Capturable Attention (Strategy A) — ONLY IF MEASURED VALUE
- **Effort**: ~2-4 weeks for full implementation
- **Risk**: High — attention is numerically sensitive, KV cache state management is complex
- **Value**: Eliminates nearly all transitions, but actual speedup depends on whether we're launch-bound or compute-bound
- **Prerequisites**: 
  1. Graph node parameter update API integration
  2. Eliminate host-side mask allocation (use kernel-internal causal masking)
  3. Fix `num_splits` for decode mode
  4. GPU-side KV cache position tracking
  5. New `GraphCapturableAttentionStage` class

### Phase 4: Kernel Fusion (Strategy C) — SEPARATE TRACK
Kernel fusion (norm+GEMM, SwiGLU+GEMM) is valuable for reducing **compute overhead** independent of graph capture. It should be pursued on its own merits for decode throughput improvement, not specifically for graph speedup.

---

## Summary

The GPU graph infrastructure is now **correct** but **throughput-neutral** because transition sync overhead equals graph replay savings. There are three approaches to break this deadlock:

1. **Make the sync cheaper** (Strategy B: events) — easy win, ~250μs savings, do first
2. **Eliminate the transitions** (Strategy A: capturable attention) — hard but high-value, measure first
3. **Reduce graph node count** (Strategy C: kernel fusion) — marginal for graphs, valuable independently

The honest assessment is that **graph capture may provide only 1-3% decode speedup** on these models because the decode bottleneck is kernel compute time, not launch overhead. The real value of GPU graphs may emerge with:
- Smaller models where launch overhead dominates (sub-0.5B)
- Multi-batch decode where graph replay amortizes over batch dimension
- Future hardware with higher kernel launch latency
