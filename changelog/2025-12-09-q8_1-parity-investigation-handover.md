# Q8_1 Activation Precision Parity Investigation - Handover Document

**Date**: December 9, 2025  
**Branch**: `feature/typed-residuals`  
**Author**: GitHub Copilot

---

## Task Goal

Investigate and fix Q8_1 → FP32 → Q8_1 round trips in the Llaminar V2 inference pipeline that cause parity degradation between FP32 and Q8_1 activation precision modes.

**Original User Request**:
> "Can you examine the Q8_1 path and determine whether we are doing Q8_1 -> FP32 -> Q8_1 round trips unnecessarily?"

**Success Criteria**:
- Q8_1 activation precision should produce similar output to FP32 (Top-5 overlap ≥60%, reasonable KL divergence)
- Currently: **Top-5 = 0%, KL = 10.43** (complete divergence)

---

## Current Parity Test Results

```
Running: ./build_v2_release/tests/v2/v2_test_activation_precision_parity --gtest_filter=ActivationPrecisionParity.Q8_1VsFP32Parity

Results:
- FP32 top-1 prediction: token 576
- Q8_1 top-1 prediction: token 43912
- Top-1 Match: NO
- Top-5 Overlap: 0%
- KL Divergence: 10.4313

Sequences diverge at step 0:
- FP32: [52643]
- Q8_1: [21817]
```

---

## What Has Been Fixed

### 1. Logits Buffer Precision (FIXED)
- **Issue**: `logits_buffer_` was created using activation precision, causing Q8_1 logits
- **Fix**: `logits_buffer_` now always uses FP32 (`Qwen2Pipeline.cpp:1276-1278`)

### 2. MPI Byte-Level Operations (FIXED)
- **Issue**: MPI allreduce forced FP32 dequantization for Q8_1 tensors
- **Fix**: Added `allgather_bytes()` and `allgatherv_bytes()` for native Q8_1 communication

### 3. Attention Output Buffer (FIXED)
- **Issue**: `buffers.attn_output` was hardcoded FP32
- **Fix**: Now uses `createActivation()` to respect activation precision

### 4. Q8_1 Attention Kernel Dispatch (FIXED)
- **Issue**: `compute_attention_with_kv_cache` only handled FP32 kernel cast
- **Fix**: Added Q8_1 kernel dispatch in `PipelineBase.cpp:1585-1610`

### 5. Q8_1 Attention Output Handling (FIXED)
- **Issue**: Attention tried to call `mutable_data()` on Q8_1 output tensor (throws)
- **Fix**: Added temporary FP32 buffer for attention compute, then quantize to Q8_1 output (`PipelineBase.cpp:1555-1570`)

---

## What Has Been Verified Working

1. **RMSNorm Q8_1 Path**: Confirmed via debug logging - 49 "Taking Q8_1 path" messages (24 layers × 2 norms + 1 final = 49)
2. **KV Cache Q8_1**: Confirmed 3.375 MB allocation (vs 12 MB for FP32)
3. **EmbeddingOp Q8_1**: Uses `mutable_q8_1_blocks()` which clears dequant cache
4. **`mutable_q8_1_blocks()`**: Confirmed it calls `dequant_cache_.clear()` (Tensors.h:2307)

---

## Current Investigation Focus

### The Mystery: Stale `dequant_cache_`

Debug logging revealed that `Q8_1Tensor::data()` returns a stale `dequant_cache_` with:
- **Wrong size**: 1835008 elements (expected: 458752 for `[512, 896]` tensor)
- **Identical values**: Same `first=-0.0340338` across all 24 layers

**Key Observations**:
1. Size 1835008 = 512 × 3584 = 512 × (4 × 896) — suggests 4 tensors worth of data
2. The cache should be empty after `mutable_q8_1_blocks()` is called
3. Yet the early-exit condition at `Q8_1Tensor.cpp:245` fires: `is_mutable_ && !dequant_cache_.empty()`

### Hypothesis

Either:
1. **Wrong tensor being logged**: The tensor with cache size 1835008 is NOT `buffers.normalized`
2. **Cache repopulated**: Something calls `.data()` BEFORE `mutable_q8_1_blocks()` clears it
3. **Multiple tensor instances**: Different Q8_1Tensor instances are being used

### Data Flow to Trace

```
Embedding → current_hidden_ (Q8_1)
    ↓
save_residual → buffers.residual (Q8_1)
    ↓
RMSNorm(buffers.residual → buffers.normalized) [calls mutable_q8_1_blocks() - clears cache]
    ↓
QKV Projection reads buffers.normalized->data() [should dequantize fresh blocks]
```

**Question**: Why does `buffers.normalized->data()` find a non-empty cache?

---

## Key Code Locations

### Q8_1Tensor Cache Logic
- `src/v2/tensors/Q8_1Tensor.cpp:245-260`: `data()` early-exit for mutable tensors with cache
- `src/v2/tensors/Tensors.h:2302-2310`: `mutable_q8_1_blocks()` clears cache

### Pipeline Data Flow
- `src/v2/pipelines/qwen/Qwen2Pipeline.cpp:619`: RMSNorm writes to `buffers.normalized`
- `src/v2/pipelines/qwen/Qwen2Pipeline.cpp:691`: QKV projection reads `buffers.normalized->data()`

### RMSNorm Operation
- `src/v2/pipelines/ops/RMSNormOp.h:396-410`: Q8_1 path calls `mutable_q8_1_blocks()`

### Attention Compute (recently fixed)
- `src/v2/pipelines/PipelineBase.cpp:1541-1700`: `compute_attention_with_kv_cache`

---

## Debug Logging Added

1. **RMSNormOp.h:344-350**: Logs which code path is taken (FP32/Q8_1/etc.)
2. **Q8_1Tensor.cpp:256-259**: Logs cache stats when returning stale cache

**Note**: Attempted to add logging to `mutable_q8_1_blocks()` in Tensors.h but it failed due to `LOG_DEBUG` not being declared in header scope.

---

## Next Steps for Investigation

1. **Add tensor address tracking**: Log `this` pointer in both `mutable_q8_1_blocks()` and `data()` to verify same tensor
2. **Trace first `data()` call**: Find what code path first calls `data()` on the Q8_1 tensor (before RMSNorm)
3. **Check `save_residual()`**: Does it call `data()` on the input hidden state?
4. **Verify buffer shapes**: Confirm `buffers.normalized` has shape `[512, 896]` at runtime

---

## Test Commands

```bash
# Build
cmake --build build_v2_release --target v2_test_activation_precision_parity -j

# Run parity test
export OMP_NUM_THREADS=28 OMP_PLACES=sockets OMP_PROC_BIND=close
export LLAMINAR_LOG_LEVEL=DEBUG
timeout 120 mpirun -np 1 --bind-to socket --map-by socket \
  ./build_v2_release/tests/v2/v2_test_activation_precision_parity \
  --gtest_filter=ActivationPrecisionParity.Q8_1VsFP32Parity

# Check RMSNorm path taken
... 2>&1 | grep -E "Taking.*path"

# Check dequant_cache usage
... 2>&1 | grep -E "dequant_cache|Mutable tensor"
```

---

## Architecture Context

### Q8_1Block Structure (36 bytes)
```cpp
struct Q8_1Block {
    uint16_t d;      // FP16 scale
    int16_t sum_qs;  // Pre-computed sum of qs values
    int8_t qs[32];   // 32 quantized values
};
```

### Q8_1Tensor Cache Invalidation Design
- `mutable_q8_1_blocks()`: Returns mutable block pointer, clears `dequant_cache_`
- `data()`: For mutable tensors with non-empty cache, returns cache directly (early exit)
- `data()`: For mutable tensors with empty cache, dequantizes blocks and populates cache

**Design Intent**: After writing Q8_1 blocks, next `data()` call should dequantize fresh blocks.

---

## Files Modified in This Session

1. `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` - logits_buffer_ fix
2. `src/v2/pipelines/PipelineBase.cpp` - Q8_1 attention kernel dispatch + output handling
3. `src/v2/pipelines/ops/RMSNormOp.h` - Debug logging for path tracking
4. `src/v2/tensors/Q8_1Tensor.cpp` - Debug logging for cache stats
5. `src/v2/tensors/Tensors.h` - (attempted, reverted) mutable_q8_1_blocks logging

---

## Related Documentation

- `.github/copilot-instructions.md` - Project development guidelines
- `.github/instructions/llaminar-architecture-v2.instructions.md` - V2 architecture overview
- `docs/v2/projects/2025-10/SNAPSHOT_FRAMEWORK_DESIGN.md` - Snapshot system for debugging
