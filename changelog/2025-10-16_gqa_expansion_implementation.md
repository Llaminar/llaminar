# GQA Expansion Implementation in Batch Operator
**Date:** 2025-10-16  
**Author:** David Sanftenberg  
**Status:** ⚠️ Implemented but divergence persists

## Overview
Added GQA (Grouped Query Attention) expansion to the batch attention operator, matching the pattern used in the sequential operator. The expansion properly replicates KV heads to match Q head count for models using GQA.

## Changes Made

### 1. Added `getKVHeadDistribution()` Methods

**File:** `src/operators/MPIAttentionBatchOperator.h`
```cpp
std::pair<int, int> getKVHeadDistribution() const;
std::pair<int, int> getKVHeadDistribution(int rank) const;
```

**File:** `src/operators/MPIAttentionBatchOperator.cpp`
- Implements KV head distribution calculation across MPI ranks
- Returns (local_kv_heads, kv_head_offset) for proper GQA mapping

### 2. Integrated GQA Expansion in execute()

Added GQA expansion step between RoPE application and attention score computation:

```cpp
// Step 2.5: Expand K and V for GQA (Grouped Query Attention)
if (n_kv_heads_ < n_heads_) {
    // Allocate expanded tensors [B, T, n_heads_local * head_dim]
    k_expanded_local = std::make_shared<SimpleTensor>(...);
    v_expanded_local = std::make_shared<SimpleTensor>(...);
    
    // Expand for each batch element
    for (int b = 0; b < B; ++b) {
        llaminar::attn::expand_kv_for_gqa(
            k_src, v_src, k_dst, v_dst,
            T, head_dim_, n_heads_local_, n_kv_heads_local_,
            head_offset_, n_heads_,
            false,  // gathered_rank_major
            kv_offset);
    }
}
```

### 3. Updated Attention Computation to Use Expanded Tensors

- `computeAttentionScores()` now uses `k_expanded_local` instead of `k_local`
- `computeAttentionOutput()` now uses `v_expanded_local` instead of `v_local`

### 4. Removed Obsolete Code

Deleted `computeAttentionOutputOLD()` method that contained custom GQA handling logic (which was incorrect).

## GQA Configuration (Qwen-0.5B)

- **Total Q heads:** 14
- **Total KV heads:** 2
- **Group size:** 7 (each KV head serves 7 Q heads)
- **Rank 0:** Q heads 0-6 → KV head 0
- **Rank 1:** Q heads 7-13 → KV head 1

## Test Results

### Before GQA Expansion
```
✅ ROPE_APPLICATION layer 0 (max_diff=0)
❌ ATTENTION_CONTEXT layer 0 (max_abs=0.162069, rel_l2=1.01258)
```

### After GQA Expansion  
```
✅ ROPE_APPLICATION layer 0 (max_diff=0)
❌ ATTENTION_CONTEXT layer 0 (max_abs=0.166133, rel_l2=1.07539)
```

**Observation:** Divergence persists with similar magnitude. Identical statistics (min/max/mean) but different values suggest a permutation/indexing issue unrelated to GQA expansion itself.

## Diagnostic Clues

**Identical Statistics:**
```
Sequential: min=-0.0948628 max=0.158855 mean=0.00118892
Batch:      min=-0.0948628 max=0.158855 mean=0.00118892
```

This indicates:
1. ✅ GQA expansion is running (verified with debug logs)
2. ✅ Data is present and computed (not zeros/garbage)
3. ✅ Value ranges match
4. ❌ **Values are permuted/indexed differently**

## Root Cause Hypothesis

The divergence is likely NOT due to missing GQA expansion (now implemented), but rather due to:

1. **Tensor layout mismatch** in snapshot capture
   - Sequential operator may gather/rearrange differently than batch operator
   - Attention context snapshot might use different head ordering

2. **Head-to-tensor-offset mapping**
   - Batch operator uses `[B, n_heads_local, T, head_dim]` layout
   - Sequential operator uses different intermediate layouts
   - Snapshot capture may not account for these differences

3. **MPI gather ordering**
   - Multi-rank gather might use different patterns
   - Rank-major vs token-major layout differences

## Next Steps for Investigation

1. **Compare snapshot capture logic:**
   - Check how ATTENTION_CONTEXT is assembled in both operators
   - Verify head ordering and tensor strides match

2. **Add detailed per-head comparison:**
   - Compare individual heads instead of flattened tensor
   - Identify which specific heads diverge

3. **Validate GQA expansion output:**
   - Add debug logging to verify expanded K/V tensors match expected layout
   - Compare first few values of expanded tensors between ranks

4. **Check attention primitive layouts:**
   - Verify `compute_qk_scores_batched()` and `apply_scores_to_v_batched()` 
   - Ensure they match the sequential operator's layout expectations

## Files Modified

- `src/operators/MPIAttentionBatchOperator.h` (+15 lines)
  - Added `getKVHeadDistribution()` method declarations
  
- `src/operators/MPIAttentionBatchOperator.cpp` (+60 lines, -55 lines deleted)
  - Added `getKVHeadDistribution()` implementations
  - Added GQA expansion logic in `execute()`
  - Updated attention computation to use expanded tensors
  - Removed obsolete `computeAttentionOutputOLD()` method

## Related Work

- **2025-10-16**: RoPE deduplication (ROPE_APPLICATION → max_diff=0) ✅
- **2025-10-16**: Attention primitives deduplication (Part 1) ✅  
- **2025-10-16**: GQA expansion implementation (this document) ⚠️

## Lessons Learned

1. **Compilation errors aren't always reported correctly** - Build task reported success despite compile failure
2. **Identical statistics = permutation issue** - Classic debugging clue
3. **GQA expansion is necessary but not sufficient** - Other layout issues may remain
4. **Debug with cerr for critical checkpoints** - Bypasses logging system for guaranteed output

## Open Questions

1. Why do statistics match perfectly but individual values differ?
2. Is the snapshot capture for ATTENTION_CONTEXT using consistent layouts?
3. Are the batched attention primitives correctly handling head dimensions?
4. Should we add per-head validation to the parity test?
