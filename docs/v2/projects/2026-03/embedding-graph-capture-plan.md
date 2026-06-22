# Embedding Stage GPU Graph Capture Plan

**Date**: 2026-03-05  
**Status**: Planned  
**Complexity**: LOW  
**Priority**: Low (saves ~10-20μs per decode step — 1 manual segment + 2 type transitions)

## Problem

The `EmbeddingStage` is the only remaining stage at the start of the decode graph that forces a manual (non-capturable) segment. It returns `isGraphCapturable() = false` because `token_ids` change each decode step.

Current segment layout (first 3 segments):

| Segment | Content | Type |
|---------|---------|------|
| 1 | `embedding` | Manual |
| 2 | `layer0: attn_norm, qkv_proj` | Capturable |
| 3 | `layer0: rope, kv_append, attention` | Manual |

This creates 2 unnecessary type transitions (manual→capturable at the start) costing ~10-20μs per decode step.

## Root Cause

The CUDA/ROCm embedding kernel (`CUDAEmbeddingKernelT::apply_tensor()`) performs a host-to-device `cudaMemcpyAsync` of `token_ids` **inside** `apply_tensor()`:

```cpp
// CUDAOpsKernels.cpp ~line 1285
cudaMemcpyAsync(d_token_ids, token_ids, token_bytes, cudaMemcpyHostToDevice, stream);
```

The host source pointer (`token_ids`) points to `ForwardGraphCache::token_ids`, a plain `std::vector<int>` in **pageable memory**. For CUDA/HIP graph replay, H2D memcpy sources must be in **pinned (page-locked) host memory**.

## Solution: Device-Side Parameter Buffer Pattern

The codebase already has a proven pattern for this exact problem, used by three other stage types:

| Stage | Dynamic Param | Kernel Method | Pinned Buffer |
|-------|--------------|---------------|---------------|
| `AttentionComputeStage` | `kv_len`, `position_offset` | `setDynamicAttnParams()` | `h_attn_params_` |
| `RoPEStage` | `pos_offset` | `setDynamicPosOffset()` | `h_device_params_` |
| `KVCacheAppendStage` | ring buffer `head` | `setDynamicHead()` | `h_head_params_` |

The pattern is:
1. Allocate a **pinned host buffer** lazily (`cudaMallocHost`)
2. Write new values into the pinned buffer in `updateDynamicParams()` (called before graph replay)
3. Issue `cudaMemcpyAsync` from pinned host → device workspace buffer
4. The captured graph kernel reads from the stable device workspace pointer

## Changes Required

### Layer 1: Kernel — Add `setDynamicTokenIds()` to CUDA/ROCm Embedding Kernels

**Files**: `src/v2/kernels/cuda/ops/CUDAOpsKernels.cpp`, ROCm equivalent

Following the pattern from `CUDAFlashAttentionKernelT::setDynamicAttnParams()`:

```cpp
void CUDAEmbeddingKernelT::setDynamicTokenIds(const int* token_ids, int num_tokens) {
    // 1. Lazy-allocate pinned host buffer
    if (!h_token_ids_) {
        cudaMallocHost(&h_token_ids_, max_tokens_ * sizeof(int));
    }

    // 2. Copy into pinned buffer
    std::memcpy(h_token_ids_, token_ids, num_tokens * sizeof(int));

    // 3. H2D to the existing workspace buffer (runs before graph replay)
    int* d_buf = static_cast<int*>(workspace_->getBuffer(EmbeddingWorkspaceBuffers::TOKEN_IDS));
    if (d_buf) {
        cudaMemcpyAsync(d_buf, h_token_ids_, num_tokens * sizeof(int),
                        cudaMemcpyHostToDevice, static_cast<cudaStream_t>(gpu_stream_));
    }
}
```

Add member fields to `CUDAEmbeddingKernelT`:
```cpp
int* h_token_ids_ = nullptr;   // Pinned host buffer
int max_tokens_ = 0;           // Max decode batch size (typically 1)
```

Clean up in destructor:
```cpp
if (h_token_ids_) { cudaFreeHost(h_token_ids_); h_token_ids_ = nullptr; }
```

The `apply_tensor()` method also needs modification: when the kernel is being used inside a graph-captured path, it should skip the inline H2D memcpy of `token_ids` (since `setDynamicTokenIds` already handled it) and instead read directly from the device workspace buffer. One approach is a boolean flag `dynamic_params_active_` set by `setDynamicTokenIds()`.

### Layer 2: Interface — Add Virtual Method to `ITensorEmbedding`

**File**: `src/v2/tensors/TensorKernels.h` (or wherever `ITensorEmbedding` is defined)

```cpp
/// Pre-stage dynamic parameter update for graph-capturable embedding.
/// Copies token_ids to pinned host buffer → device workspace via H2D memcpy.
virtual void setDynamicTokenIds(const int* token_ids, int num_tokens) {
    (void)token_ids; (void)num_tokens;
}
```

CPU kernels don't need this (CPU embedding is not graph-captured), so the default no-op is correct.

### Layer 3: Stage — Wire Up `EmbeddingStage`

**File**: `src/v2/execution/compute_stages/stages/EmbeddingStage.h`

```cpp
// Change from false to true:
bool isGraphCapturable() const override { return true; }

// Add dynamic param support:
bool hasDynamicParams() const override { return true; }

void updateDynamicParams(int pos_offset, int seq_len) override {
    (void)pos_offset; // Embedding doesn't use position offset
    if (cached_kernel_ && params_.token_ids) {
        cached_kernel_->setGPUStream(gpuStream());
        cached_kernel_->setDynamicTokenIds(params_.token_ids, params_.num_tokens);
    }
}
```

No `onGraphReplayed()` callback is needed — embedding has no host-side state to advance after replay (unlike KVCache which must advance the ring buffer head).

### Layer 4: Orchestrator — No Changes Needed

The orchestrator already:
- Precomputes `dynamic_param_stages` by checking `hasDynamicParams()` (DeviceGraphOrchestrator.cpp ~line 668)
- Calls `updateDynamicParams()` on all registered stages before each decode step
- Updates `ForwardGraphCache::token_ids` contents via `std::memcpy` before dispatch

## Impact on Segmentation

After making embedding capturable, segment 1 **merges into segment 2**:

| Before | After |
|--------|-------|
| Seg 1: `embedding` (Manual) | Seg 1: `embedding + layer0_attn_norm + layer0_qkv_proj` (Capturable) |
| Seg 2: `layer0: attn_norm, qkv_proj` (Capturable) | Seg 2: `layer0: rope, kv_append, attention` (Manual) |
| Seg 3: `layer0: rope, kv_append, attention` (Manual) | ... |

Eliminates 1 manual segment and 2 type transitions. Saves ~10-20μs per decode step.

## Preconditions & Safety

### Embed Table Upload Must Complete Before Capture

The quantized embedding path (IINT8Unpackable → EmbedQ8 repack) does a one-time synchronous `cudaMemcpy` for table upload, gated by the `needs_upload` check in `apply_tensor()`. This **must not execute during graph capture**.

This is already safe: Phase 1 (Warmup) runs a full forward pass before Phase 2 (Capture), so the table is guaranteed to be uploaded and cached in `s_workspace_embed_cache_` before any capture occurs.

### Device Workspace Buffer Already Exists

`EmbeddingWorkspaceBuffers::TOKEN_IDS` is already allocated by `DeviceWorkspaceManager` with size `max_seq_len * sizeof(int)` and 256-byte alignment. No new workspace buffers are needed.

### Decode-Only

This change only affects decode mode (graph capture is decode-only). Prefill uses variable `seq_len` and is never graph-captured.

## Files to Touch

| File | Change |
|------|--------|
| `src/v2/execution/compute_stages/stages/EmbeddingStage.h` | `isGraphCapturable() → true`, add `hasDynamicParams()`, `updateDynamicParams()` |
| `src/v2/tensors/TensorKernels.h` | Add `setDynamicTokenIds()` virtual method to `ITensorEmbedding` |
| `src/v2/kernels/cuda/ops/CUDAOpsKernels.cpp` | Add `setDynamicTokenIds()` with pinned buffer; modify `apply_tensor()` to skip inline H2D when dynamic params active |
| `src/v2/kernels/cuda/ops/CUDAOpsKernels.h` | Add `h_token_ids_`, `max_tokens_`, `dynamic_params_active_` members |
| `src/v2/kernels/rocm/ops/ROCmOpsKernels.hip` | Same changes as CUDA |
| `src/v2/kernels/rocm/ops/ROCmOpsKernels.h` | Same members as CUDA |
| Tests | Unit test verifying embedding is graph-capturable and correct after parameter updates |

## Testing Strategy

1. **Unit test**: Create test that builds a graph with `EmbeddingStage`, verifies `isGraphCapturable() == true`, calls `updateDynamicParams()` with different token_ids, and validates output correctness
2. **Integration test**: Run full decode with `LLAMINAR_GPU_GRAPHS=1` and verify segment count decreased by 1
3. **Parity test**: Verify decode output matches non-graph-captured baseline (existing parity tests should cover this)
