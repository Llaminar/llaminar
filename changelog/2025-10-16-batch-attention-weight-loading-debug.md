# Batch Attention Debugging - Weight Loading Issue

**Date**: 2025-10-16  
**Component**: MPIAttentionBatchOperator, BatchQwenPipeline  
**Status**: ROOT CAUSE IDENTIFIED

## Problem Summary

`MPIAttentionBatchOperator` was producing NaN values. Through systematic debugging with `TensorHealthCheck` validation, we discovered:

```
K projection output: min=-8.55471e+35, max=3.69997e+35  ❌ EXTREME!
Q projection output: min=-35.641, max=47.75             ✅ NORMAL
V projection output: min=-0.0948628, max=0.20147        ✅ NORMAL
```

##Root Cause

**The K weight tensor has incorrect dimensions:**
- **Actual**: `[64, 896]` (single KV head, MPI-sliced)
- **Expected**: `[128, 896]` (both KV heads, full tensor)

From logs:
```
[MpiSlicingHelper.cpp] Successfully loaded blk.0.attn_k.weight with shape [64, 896] for rank 1/2
```

When rank 1 tries to extract rows 64-127 from a 64-row matrix → **reading uninitialized memory** → extreme values (10^35).

## Architecture Mismatch

### How MPIAttentionOperator Works (Correct)
1. **PipelineBase** loads MPI-sliced weights during initialization
2. Each rank gets only its subset: K weight = `[n_kv_heads_local * head_dim, d_model]`
3. Operator uses these pre-sliced weights directly
4. **No runtime weight extraction needed**

### How MPIAttentionBatchOperator Works (Incorrect Implementation)
1. **PipelineBase** loads MPI-sliced weights (same as above)
2. Operator **expects FULL weights**: `[n_kv_heads * head_dim, d_model]`
3. Operator **extracts local slice at runtime**:
   ```cpp
   int local_rows = n_kv_heads_local_ * head_dim_;  // 64
   int row_offset = (head_offset_ * n_kv_heads_ / n_heads_) * head_dim_;  // rank 1: offset=64
   std::memcpy(wk_local->data(), wk->data() + offset_elements, ...);  // ❌ Reading past end!
   ```
4. **Result**: Rank 1 reads uninitialized memory

## Solution Options

### Option A: Change Weight Loading (Recommended)
`BatchQwenPipeline` should load **REPLICATED** (full) weights, not MPI-sliced weights.

**Rationale**:
- Batch operators handle their own distribution internally
- Simpler to reason about (operators are self-contained)
- Matches how `MPILinearBatchOperator` works (full weights + runtime slicing)

**Implementation**:
- Override `getWeightForLayer()` in `BatchQwenPipeline` to request `REPLICATED` tensors
- Or: Add a pipeline mode flag to `PipelineBase` to disable MPI slicing

### Option B: Change Operator Logic
Remove runtime weight extraction from `MPIAttentionBatchOperator` and use pre-sliced weights.

**Rationale**:
- Matches existing `MPIAttentionOperator` pattern
- No memory overhead from duplicating weights

**Implementation**:
- Remove weight extraction loops
- Use `wk`, `wv`, `wo` tensors directly
- Requires all ranks to have their subset pre-loaded

## Validation Infrastructure Added

### Shared Headers Created
1. **`src/operators/common/TensorHealthCheck.h`** - Detects NaN/Inf/uninitialized data
2. **`src/operators/attention/AttentionStageContracts.h`** - Already existed, now used by both operators

### Weight Validation Added to MPIAttentionBatchOperator
```cpp
// Validates weight dimensions match expected architecture
const int expected_wk_rows = n_kv_heads_ * head_dim_;  // 128 for 2 KV heads
const int wk_rows = static_cast<int>(wk->shape()[0]);
if (wk_rows != expected_wk_rows || wk_cols != D) {
    LOG_ERROR("wk dimension mismatch: got [" << wk_rows << "," << wk_cols
              << "], expected [" << expected_wk_rows << "," << D << "]");
    return false;
}
```

This would have caught the issue immediately with a clear error instead of cryptic NaN failures.

## Next Steps

1. **Immediate**: Implement Option A (change BatchQwenPipeline to load REPLICATED weights)
2. Update `MPIAttentionBatchOperator` to work with full weights
3. Add integration test to verify weight shapes before execution
4. Document weight loading contract in both pipeline types

## Files Modified

- ✅ `src/operators/common/TensorHealthCheck.h` - Created shared validation utility
- ✅ `src/operators/MPIAttentionOperator.cpp` - Now includes shared TensorHealthCheck
- ✅ `src/operators/MPIAttentionBatchOperator.cpp` - Added weight validation and health checks
- 🔄 `src/BatchQwenPipeline.cpp` - **NEEDS FIX**: Must load REPLICATED weights

## Lessons Learned

1. **Early validation saves hours**: Weight dimension checks would have caught this in <1 second
2. **Health checks are critical**: `TensorHealthCheck` immediately identified uninitialized memory
3. **Architecture assumptions must be explicit**: Document whether operators expect pre-sliced or full weights
4. **Reuse validation infrastructure**: Moving `TensorHealthCheck` to shared header enables consistency

## Performance Impact

None - this is a correctness fix. The validation adds ~1ms overhead but only runs when `LLAMINAR_ATTN_VALIDATE_OUTPUT=1`.
