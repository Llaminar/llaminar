# RoPE Implementation Deduplication and ROPE_APPLICATION Fix
**Date:** 2025-10-16  
**Author:** David Sanftenberg  
**Issue:** Batch operator had duplicated RoPE implementation causing divergence from sequential operator

## Overview
Fixed ROPE_APPLICATION divergence (max_abs=97.3449 → 0) by eliminating code duplication. The batch operator was maintaining its own RoPE implementation instead of reusing the proven `AttentionPrimitives::apply_rope()` function, leading to subtle differences in computation.

## Problem Statement

**Original Issue:**
- Batch operator: Custom RoPE loops in `MPIAttentionBatchOperator::applyRoPE()`
- Sequential operator: Uses `llaminar::attn::apply_rope()` from AttentionPrimitives
- Result: ROPE_APPLICATION divergence with max_abs=97.3449, rel_l2=0.18586
- Maintenance nightmare: Two implementations that could drift apart

**Code Smell:**
```cpp
// BAD: Duplicated implementation in batch operator
#pragma omp parallel for collapse(3)
for (int b = 0; b < batch_size; ++b) {
    for (int h = 0; h < n_heads_local_; ++h) {
        for (int t = 0; t < seq_len; ++t) {
            // 30+ lines of rotation math...
        }
    }
}
```

## Solution

### 1. Extended AttentionPrimitives for Batch Support

**Added to `src/operators/common/AttentionPrimitives.h`:**
```cpp
void apply_rope_batched(float *q, float *k, int batch_size, int seq_len, int head_dim,
                        int q_heads, int k_heads, int n_past, float freq_base);
```

**Implementation in `AttentionPrimitives.cpp`:**
- Computes inverse frequencies once (shared across batches)
- Processes each batch element independently
- Delegates to proven `apply_rope_to_tensor()` for actual computation
- Layout: `[batch_size, seq_len, num_heads * head_dim]`

Key insight: Each batch element has layout `[seq_len, num_heads, head_dim]` which matches the single-batch function's expected layout. We just process multiple slices.

### 2. Simplified Batch Operator

**Replaced 100+ lines with ~30 lines:**
```cpp
void MPIAttentionBatchOperator::applyRoPE(float *q, float *k, int batch_size, int seq_len) {
    const int n_past = 0;  // Prefill mode
    
    llaminar::attn::apply_rope_batched(
        q, k,
        batch_size, seq_len, head_dim_,
        n_heads_local_, n_kv_heads_local_,
        n_past, rope_freq_base_
    );
}
```

### 3. Cleanup

**Removed from `MPIAttentionBatchOperator`:**
- `std::vector<float> rope_freqs_` member variable
- Frequency precomputation in constructor
- 100+ lines of custom RoPE loops
- All RoPE-related debug logging (now in primitives)

## Technical Details

### RoPE Formula (Consistent Across Both Implementations)
```
x[2i]   = x0 * cos(θ) - x1 * sin(θ)
x[2i+1] = x0 * sin(θ) + x1 * sin(θ)

where θ = position * inv_freq[i]
      inv_freq[i] = 1.0 / freq_base^(2i / head_dim)
```

### Layout Expectations
- **Batch operator input**: `[B, T, n_heads * head_dim]`
- **Per-batch slice**: `[T, n_heads, head_dim]` ← matches single-batch layout
- **AttentionPrimitives expects**: `[seq_len, num_heads, head_dim]`

### n_past Handling
- Batch operator is prefill-only → n_past = 0
- Sequential operator tracks position → n_past varies
- Position in RoPE: `position = n_past + t`

## Results

### Before Fix
```
✓ Q_PROJECTION layer 0 (max_diff=0)
✓ K_PROJECTION layer 0 (max_diff=0)
✓ V_PROJECTION layer 0 (max_diff=0)
✗ ROPE_APPLICATION layer 0 (max_abs=97.3449, rel_l2=0.18586)
  Worst index: 4064
  Sequential: 47.7905
  Batch:      -49.5544
```

### After Fix
```
✓ Q_PROJECTION layer 0 (max_diff=0)
✓ K_PROJECTION layer 0 (max_diff=0)
✓ V_PROJECTION layer 0 (max_diff=0)
✓ ROPE_APPLICATION layer 0 (max_diff=0)  ← FIXED!
✗ ATTENTION_CONTEXT layer 0 (max_abs=0.162069, rel_l2=1.01258)  ← Next issue
```

## Files Changed

### Modified
- **src/operators/common/AttentionPrimitives.h** (+15 lines)
  - Added `apply_rope_batched()` declaration
  - Documentation for batch-aware RoPE

- **src/operators/common/AttentionPrimitives.cpp** (+60 lines)
  - Implemented `apply_rope_batched()` 
  - Delegates to existing `apply_rope_to_tensor()`

- **src/operators/MPIAttentionBatchOperator.h** (-3 lines)
  - Removed `rope_freqs_` member variable

- **src/operators/MPIAttentionBatchOperator.cpp** (-80 lines net)
  - Added `#include "common/AttentionPrimitives.h"`
  - Replaced custom RoPE with primitive call
  - Removed frequency precomputation
  - Removed 100+ lines of custom loops

## Benefits

1. **Code Maintainability**: Single source of truth for RoPE logic
2. **Correctness**: Both operators now use identical, well-tested implementation
3. **Reduced Surface Area**: ~100 fewer lines of complex math to maintain
4. **Future-Proof**: RoPE improvements automatically benefit both operators
5. **Testability**: Primitives are unit-testable without MPI context

## Design Principles Applied

✅ **DRY (Don't Repeat Yourself)**: Eliminated RoPE duplication  
✅ **Single Responsibility**: Primitives handle math, operators handle distribution  
✅ **Composability**: Batch operator composes single-batch primitive  
✅ **Testability**: Math logic isolated from MPI concerns  

## Next Steps

The divergence now occurs in **ATTENTION_CONTEXT** (the attention computation itself):
- max_abs: 0.162069
- rel_l2: 1.01258
- Likely cause: Attention score computation or softmax differences

This is a separate issue from RoPE and should be investigated next.

## Validation

```bash
# Test passes ROPE_APPLICATION stage
timeout 60 mpirun -np 2 ./build/test_batch_correctness \
  --gtest_filter="BatchCorrectnessTest.FindFirstDivergenceStage"

# Expected output:
# ✓ ROPE_APPLICATION layer 0 (max_diff=0)
```

## Lessons Learned

1. **Reuse over Reimplementation**: Always check for existing implementations before writing new code
2. **Primitives First**: Build composable, testable primitives that operators can orchestrate
3. **Layout Awareness**: Understanding tensor layouts enables reuse (batch is just N single-batch slices)
4. **Code Smells**: Duplicated math logic is a red flag for maintenance issues
5. **Trust Tests**: Divergence tests caught the subtle implementation differences

## Related Work

- **2025-10-16**: Q/K/V projection gather pattern fix (max_diff → 0)
- **2025-10-16**: Test framework improvements (proper failure on divergence)
- **Next**: ATTENTION_CONTEXT divergence investigation
