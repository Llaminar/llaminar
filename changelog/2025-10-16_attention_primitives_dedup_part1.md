# Attention Primitives Deduplication - Part 1
**Date:** 2025-10-16  
**Author:** David Sanftenberg  
**Status:** ✅ RoPE fixed, ⚠️ ATTENTION_CONTEXT divergence remains (GQA expansion needed)

## Overview
Replaced duplicated attention computation code in batch operator with proven primitives from `AttentionPrimitives` and `SoftmaxCore`. This is part 1 of a multi-phase refactoring to eliminate code duplication and ensure consistency between batch and sequential operators.

## What Was Fixed

### ✅ RoPE Application (Completed Earlier)
-  **Before**: Custom RoPE loops (~100 lines)
- **After**: Call to `apply_rope_batched()` (~30 lines)
- **Result**: ROPE_APPLICATION max_diff=0

### ✅ Attention Score Computation
- **Before**: Custom `cblas_sgemm` loops with manual GQA handling
- **After**: Call to `compute_qk_scores_batched()`
- **Result**: Uses proven Q @ K^T implementation with causal masking

### ✅ Softmax Application
- **Before**: Custom softmax with manual max-finding and normalization (~50 lines)
- **After**: Call to `llaminar::kernels::softmax_row_major()`
- **Result**: Uses proven numerically stable softmax

### ✅ Attention Output Computation
- **Before**: Custom implementation (not fully visible in changes)
- **After**: Call to `apply_scores_to_v_batched()`
- **Result**: Uses proven scores @ V implementation

## Changes Made

### New Functions Added to AttentionPrimitives.h
```cpp
void compute_qk_scores_batched(const float *q, const float *k, float *scores,
                               int batch_size, int seq_len, int head_dim, int heads, bool causal);

void apply_scores_to_v_batched(const float *scores, const float *v, float *out,
                               int batch_size, int seq_len, int head_dim, int heads);
```

Both functions process batches by looping over batch dimension and delegating to proven single-batch implementations.

### Modified Files
- **src/operators/common/AttentionPrimitives.{h,cpp}** (+90 lines)
  - Added `compute_qk_scores_batched()`
  - Added `apply_scores_to_v_batched()`
  
- **src/operators/MPIAttentionBatchOperator.cpp** (-120 lines net)
  - Added `#include "common/SoftmaxCore.h"`
  - Replaced `computeAttentionScores()` custom implementation
  - Replaced `applyCausalMaskAndSoftmax()` custom implementation  
  - Replaced `computeAttentionOutput()` custom implementation

## Current Test Status

```
✅ EMBEDDING (max_diff=0)
✅ ATTENTION_NORM layer 0 (max_diff=0)
✅ Q_PROJECTION layer 0 (max_diff=0)
✅ K_PROJECTION layer 0 (max_diff=0)
✅ V_PROJECTION layer 0 (max_diff=0)
✅ ROPE_APPLICATION layer 0 (max_diff=0)
❌ ATTENTION_CONTEXT layer 0 (max_abs=0.162069, rel_l2=1.01258)
```

## Root Cause of Remaining Divergence

**GQA (Grouped Query Attention) Expansion Missing**

The sequential operator performs GQA expansion BEFORE computing attention:
```cpp
// Sequential operator (MPIAttentionOperator.cpp:2072)
llaminar::attn::expand_kv_for_gqa(
    global_k_cache->data(), global_v_cache->data(),
    result.local_k_expanded->data(), result.local_v_expanded->data(),
    attn_seq_len, head_dim_, local_heads, n_head_kv_, head_offset, n_head_,
    world_size > 1, kv_offset);
```

The batch operator tries to handle GQA inline during attention computation with manual `kv_h` indexing, but the primitives expect all heads to already be expanded.

**GQA Concept:**
- Qwen-0.5B: 14 Q heads, 2 KV heads
- Group size: 14 / 2 = 7
- Q heads [0-6] use KV head 0
- Q heads [7-13] use KV head 1
- Each KV head must be replicated 7 times to match Q head count

## Diagnostic Clues

**Identical Statistics, Different Values:**
```
Sequential: min=-0.0948628 max=0.158855 mean=0.00118892
Batch:      min=-0.0948628 max=0.158855 mean=0.00118892
```

This indicates:
1. ✅ Data is present (not zeros/garbage)
2. ✅ Computations are happening
3. ❌ Data is permuted/arranged differently (likely due to GQA head mapping)

**High Relative L2 (1.01258) but Matching Ranges:**
- Suggests systematic offset rather than random error
- Consistent with incorrect head-to-KV-head mapping

## Next Steps

### Phase 2: Add GQA Expansion to Batch Operator

1. **Add GQA expansion before attention computation**
   ```cpp
   // Expand K, V from [B, T, n_kv_heads_local * head_dim] 
   //              to [B, T, n_heads_local * head_dim]
   auto k_expanded = expandKVForGQA(k_local, ...);
   auto v_expanded = expandKVForGQA(v_local, ...);
   ```

2. **Consider adding batched GQA expansion primitive**
   ```cpp
   void expand_kv_for_gqa_batched(const float *k, const float *v,
                                  float *k_expanded, float *v_expanded,
                                  int batch_size, int seq_len, int head_dim,
                                  int n_heads, int n_kv_heads, ...);
   ```

3. **Update batch operator flow:**
   ```
   Current:  Q/K/V proj → RoPE → Attention (with inline GQA)
   Target:   Q/K/V proj → RoPE → GQA expand → Attention
   ```

## Benefits Achieved So Far

1. **Eliminated ~200 lines of duplicated code**
2. **RoPE now perfectly consistent** (max_diff=0)
3. **Softmax uses proven numerically stable implementation**
4. **Attention scores use battle-tested primitives**
5. **Future improvements to primitives benefit both operators**

## Design Principles

✅ **Single Source of Truth**: RoPE, softmax, and attention math now unified  
✅ **Composability**: Batch operator composes single-batch primitives  
✅ **Testability**: Primitives can be unit-tested independently  
⏳ **Completeness**: GQA expansion needs same treatment  

## Lessons Learned

1. **Code Duplication → Divergence**: Custom implementations inevitably drift
2. **Primitives First**: Build composable, tested building blocks
3. **GQA is Complex**: Head replication must happen before primitive calls
4. **Stats are Diagnostic**: Matching min/max/mean but different values = permutation issue

## Files Changed

- `src/operators/common/AttentionPrimitives.h` (+30 lines)
- `src/operators/common/AttentionPrimitives.cpp` (+90 lines)
- `src/operators/MPIAttentionBatchOperator.cpp` (-120 lines, +40 lines)

## Validation

```bash
# Test shows progress
timeout 60 mpirun -np 2 ./build/test_batch_correctness \
  --gtest_filter="BatchCorrectnessTest.FindFirstDivergenceStage"

# Expected: ROPE passes, ATTENTION_CONTEXT fails with GQA-related divergence
```

## Related Work

- **2025-10-16**: RoPE deduplication (ROPE_APPLICATION → max_diff=0)
- **2025-10-16**: Test framework improvements (proper failure detection)
- **2025-10-16**: Q/K/V gather pattern fixes
- **Next**: GQA expansion in batch operator (Phase 2)
