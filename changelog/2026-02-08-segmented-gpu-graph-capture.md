# Segmented GPU Graph Capture/Replay for Decode

**Date**: 2026-02-08  
**Type**: Performance optimization  
**Status**: Complete (CUDA verified, ROCm pending)

## Summary

Implements **segmented GPU graph capture** that partitions the decode compute graph
into capturable segments (GEMMs, norms, SwiGLU, residuals) and non-capturable
segments (attention, KV cache, RoPE, embedding), then replays the capturable
portions via GPU graph launch to eliminate per-kernel dispatch overhead.

## Problem

During decode, the DeviceGraphExecutor issues ~291 individual kernel launches per token
(for 0.5B, 24 layers) or ~339 per token (for 7B, 28 layers). On ROCm MI50, the
CPU-side dispatch overhead is ~43ms per token — dominating the actual compute time
and limiting decode throughput to ~15.88 tok/s vs a kernel-only theoretical
~238 tok/s.

Prior single-graph approach (captured all stages into one graph) achieved +35.2%
on the first decode token but failed on subsequent tokens because attention stages
have variable `kv_len` parameters that change each token, causing graph update
failures.

## Solution

### Segmented Architecture

Instead of one monolithic graph, partition stages into alternating segments:

```
[embedding(manual)] → [attn_norm+qkv_proj(graph)] → [rope+kv+attn(manual)] →
[wo_proj+residual+ffn(graph)] → [rope+kv+attn(manual)] → ... → [lm_head(graph)]
```

**25 capturable segments** (218 stages) + **25 manual segments** (73 stages) on
the 0.5B model.

### Non-Capturable Stages

Stages marked `isGraphCapturable() = false` because their parameters change each
decode token:
- `AttentionComputeStage` — kv_len grows
- `AttentionWithKVCacheStage` — kv_len grows
- `KVCacheAppendStage` — appends new entries
- `KVCacheGatherStage` — KV lengths change
- `RoPEStage` — position_offset increments
- `EmbeddingStage` — token_ids are different each token
- `FusedAttentionWoStage` — position_offset in updateDynamicParams

### 3-Phase Execution

1. **Phase 1 (Warmup)**: First decode token — build segment list, execute via
   `executeFastDecode()`. Ensures lazy kernel initialization (workspace mallocs)
   completes before capture.

2. **Phase 2 (Capture)**: Second decode token — create capture stream, iterate
   segments: capturable segments go through beginCapture→execute→endCapture→
   instantiate→launch; manual segments execute normally with explicit sync.

3. **Phase 3 (Replay)**: All subsequent tokens — capturable segments: direct
   `launch()` (no re-capture); manual segments: execute on legacy default stream.
   Stream sync via `synchronizeStream()` at type transitions only.

### Synchronization Strategy

Graph segments launch on `capture_stream` (non-blocking). Manual stages dispatch
to the legacy default stream (stream 0). These are separate streams requiring
explicit synchronization at every type transition:

```
[graph on capture_stream] → syncStream(capture_stream) → [manual on stream 0]
[manual on stream 0]      → syncStream(nullptr)        → [graph on capture_stream]
```

Per-stream `synchronizeStream()` (~5μs) is ~10× cheaper than device-wide
`synchronize()` (~50μs). Event-based GPU-side sync (`insertStreamDependency`)
was implemented but benchmarked slower due to per-call event create/destroy
overhead; the `synchronizeStream()` approach proved optimal.

### Capture Stream Creation

**Critical finding**: The capture stream must be created on the calling thread,
not on the device context's worker thread. On ROCm (MI50/gfx906, ROCm 7.1),
streams created on a different thread produce corrupted output when used for
graph capture on the main thread. Solution: `GraphSegmentCache::ensureCaptureStream()`
creates the stream via `IWorkerGPUContext::createStream()` directly.

## CUDA Results (RTX 3090, Qwen2.5-0.5B-Instruct-Q4_0)

| Metric | Baseline | GPU Graphs | Change |
|--------|----------|------------|--------|
| Decode (100 tok) | ~130 tok/s | ~130 tok/s | Neutral |
| Prefill | ~430 tok/s | ~520 tok/s | +20% |

On CUDA RTX 3090, kernel launch overhead is ~1μs/launch (negligible), so graph
replay savings (~0.3ms/token) are offset by `synchronizeStream()` overhead
(~0.25ms/token for 49 transitions). Net result: neutral decode, improved prefill.

**The primary target is ROCm MI50** where CPU overhead is ~43ms/token — segmented
graphs should recover the majority of this overhead.

## Code Changes

### New Interface Methods (`IWorkerGPUContext.h`)
- `synchronizeStream(void *stream)` — per-stream sync (cheaper than device sync)
- `insertStreamDependency(void *dependent, void *dependency)` — GPU-side event sync
- `createGraphCapture(void *stream)` — factory for graph captures on specific stream

### New Virtual Method (`IComputeStage.h`)
- `isGraphCapturable()` — returns `true` by default; overridden to `false` by
  7 stage types with dynamic parameters

### New Structures (`DeviceDeviceGraphExecutor.h`)
- `GraphSegment` — stage names + capturable flag + graph capture handle
- `GraphSegmentCache` — persistent segment cache with stream/event lifecycle,
  move semantics, RAII cleanup

### New Execution Path (`DeviceDeviceGraphExecutor.cpp`)
- `executeWithSegmentedGraphCapture()` — 3-phase capture/replay engine

### Modified Files
- `DeviceGraphOrchestrator.h/cpp` — `ForwardGraphCache` stores `GraphSegmentCache`,
  calls `executeWithSegmentedGraphCapture()` when `LLAMINAR_GPU_GRAPHS=1`
- `NvidiaDeviceContext.h/.cu` — implements `synchronizeStream`, `insertStreamDependency`,
  `createGraphCapture(void*)`
- `AMDDeviceContext.h/.cpp` — same implementations for ROCm
- `Test__KernelBaseDeviceContext.cpp` — MockGPUContext implements new pure virtuals

### Environment Variables
- `LLAMINAR_GPU_GRAPHS=1` — enable segmented GPU graph capture/replay (default: off)

## Next Steps

- [ ] Benchmark on ROCm MI50 with 7B model (primary target)
- [ ] Investigate running manual stages on capture_stream (eliminate all sync overhead)
- [ ] Consider removing old single-graph `executeWithGraphCapture()` code path
- [ ] Cache events in `insertStreamDependency()` to avoid per-call create/destroy
