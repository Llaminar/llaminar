# Critical Fix: Incremental Decode Causal Masking Bug

**Date**: October 13, 2025  
**Author**: David Sanftenberg  
**Impact**: High - Fixes correctness of all incremental token generation  
**Test**: ParityFramework.TrueIncrementalDecodeVsPyTorch now PASSES (1170/1170 stages)

## Summary

Fixed a critical bug in attention softmax for incremental decode mode where causal masking was incorrectly applied, causing softmax to output one-hot vectors instead of proper probability distributions. This resulted in completely incorrect token generation.

## Problem Description

### Symptom
- **TrueIncrementalDecodeVsPyTorch** test was failing
- Token sequences diverged: PyTorch [6 → 25010 → 10] vs Llaminar [6 → 62162 → 11]
- First divergence at ATTENTION_NORM_layer1 (max_abs=6.30324, rel_l2=0.436228)

### Root Cause Investigation

Traced backward through the pipeline:
1. **EMBEDDING**: ✓ Perfect (max_abs=0.0, rel_l2=0.0)
2. **ATTENTION_NORM_layer0**: ✓ Perfect (max_abs=5.96e-08)
3. **ATTENTION_OUTPUT_layer0**: ✗ FAILED (max_abs=0.0255, rel_l2=0.470)
4. **ATTENTION_SOFTMAX_layer0**: ✗ **CRITICAL BUG FOUND**

```python
# Expected (PyTorch)
[0.188, 0.199, 0.347, 0.139, 0.079, 0.047]  # Smooth probability distribution

# Actual (Llaminar - BEFORE FIX)
[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # One-hot vector!
```

### Technical Analysis

In incremental decode mode:
- **Query**: Single new token at position `n_past` (e.g., position 5)
- **Keys**: Full KV cache with all previous tokens [0, 1, 2, 3, 4, 5]
- **Attention shape**: `[rows=1, cols=6]` (1 query × 6 cache positions)

The causal masking logic was:
```cpp
bool masked = causal && c > r;  // Mask position c if c > row_index r
```

For `r=0` (the only row), this masks positions `c ∈ {1, 2, 3, 4, 5}`, leaving only position 0 unmasked:
```
Row 0: [KEEP, MASK, MASK, MASK, MASK, MASK]
Softmax result: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  ← One-hot vector!
```

**The Bug**: Causal masking used **relative row index** instead of **absolute query position**.
- Row index `r=0` does NOT mean "query at position 0"
- In incremental decode, row 0 is the NEW token at absolute position `n_past + 0 = 5`
- This token should attend to ALL cache positions [0..5], not just position 0!

## Solution

Disable causal masking entirely for incremental decode (`seq_len=1`):

```cpp
// BEFORE (INCORRECT)
args.causal = true;  // Always use causal mask

// AFTER (CORRECT)
const bool use_causal_mask = (seq_len > 1); // Only for prefill/batch mode
args.causal = use_causal_mask;
```

### Rationale

**Incremental Decode** (`seq_len=1`):
- Query token is NEW, attending to PAST cache
- Cache only contains tokens with position < current position
- No future tokens to mask → causal masking not needed

**Prefill/Batch Mode** (`seq_len>1`):
- Multiple query tokens attending to each other
- Position `i` must not attend to future positions `j > i`
- Causal masking required

## Code Changes

**File**: `src/kernels/MPIAttentionKernel.cpp`  
**Location**: Lines 2131-2137

```cpp
// Apply softmax to each head (sequential loop - softmax_row_major parallelizes internally)
// CRITICAL FIX: In incremental decode (seq_len=1), disable causal masking because
// the query token is attending to the full cache which only contains PAST tokens.
// Causal masking with rows=1 would incorrectly mask based on relative position (row 0 -> mask all c > 0).
const bool use_causal_mask = (seq_len > 1); // Only use causal mask in prefill/batch mode

for (int h = 0; h < local_heads; ++h)
{
    llaminar::kernels::SoftmaxRowArgs args;
    args.scores = scores.data() + static_cast<size_t>(h) * seq_len * attn_seq_len;
    args.rows = seq_len;
    args.cols = attn_seq_len;
    args.causal = use_causal_mask;  // FIX: Disable for incremental decode
    args.scale = 1.0f;
    // ... rest of softmax application
}
```

## Verification

### Test Results

**BEFORE FIX**:
```
[TRUE_INCR] Token Sequence Comparison:
  PyTorch tokens:  [6 → 25010 → 10]
  Llaminar tokens: [6 → 62162 → 11]  ← WRONG!
  ✗ DIVERGENCE at position 1

[STAGE-LEVEL VALIDATION]
  Stages failed: 2
  ✗ ATTENTION_NORM_layer1.npy (max_abs=6.30324, rel_l2=0.436228)
```

**AFTER FIX**:
```
[TRUE_INCR] Token Sequence Comparison:
  PyTorch tokens:  [6 → 25010 → 10]
  Llaminar tokens: [6 → 25010 → 10]  ← CORRECT!
  ✓ Token sequences MATCH

[STAGE-LEVEL VALIDATION]
  Tokens passed:      3/3
  Stages compared:    585
  Stages passed:      1170
  Stages failed:      0
  ✓ ALL STAGES PASS
```

### Softmax Output Validation

**Layer 0, Head 0, Token 0** (after fix):
```python
# Llaminar now matches PyTorch exactly
PyTorch:  [0.188, 0.199, 0.347, 0.139, 0.079, 0.047]
Llaminar: [0.188, 0.199, 0.347, 0.139, 0.079, 0.047]  ✓
```

## Impact Assessment

### Affected Functionality
- ✅ **Incremental token generation**: Now produces correct outputs
- ✅ **Multi-turn conversation**: Token sequences match reference
- ✅ **KV cache utilization**: Proper attention over full history
- ⚠️ **Prefill mode**: Unaffected (still uses causal masking correctly)

### Performance Impact
- **None**: The fix only changes when causal masking is applied, not how it's computed
- Incremental decode may be slightly faster (no unnecessary masking operations)

## Related Work

This fix complements our earlier work on:
- **GQA (Grouped Query Attention)** expansion for multi-head attention
- **KV cache management** for incremental decode
- **RoPE (Rotary Position Embeddings)** for position-aware attention

All three systems now work correctly together for production-quality incremental token generation.

## Testing Strategy

### Regression Prevention
1. **TrueIncrementalDecodeVsPyTorch**: Primary validation (PASSES)
2. **OpenBLASPrefillVsPyTorch**: Ensure prefill still works (PASSES)
3. Future: Add unit test specifically for softmax causal masking logic

### Manual Verification
```bash
# Run incremental decode test
./build/test_parity_framework --gtest_filter="*TrueIncremental*"

# Expected: All 1170 stages pass, tokens match
```

## Lessons Learned

1. **Causal masking semantics vary by context**:
   - Prefill: row index = absolute query position
   - Incremental decode: row index = relative offset in batch

2. **One-hot softmax outputs are a red flag**:
   - Indicates masking bug (not enough valid positions)
   - Can manifest as extreme distribution collapse

3. **Test at multiple abstraction levels**:
   - Started with token-level divergence
   - Traced to layer-level stage failures
   - Identified component-level softmax bug
   - Fixed primitive-level masking logic

4. **Snapshot-driven debugging is powerful**:
   - Layer-by-layer comparison pinpointed exact failure
   - Intermediate stage validation (ATTENTION_SOFTMAX) was critical

## Future Improvements

Consider these enhancements:
1. **Absolute position-aware masking**: Pass `n_past` to softmax for more flexible masking
2. **Unified masking interface**: Single function handling both prefill and decode modes
3. **Softmax output validation**: Add contract to detect one-hot or invalid distributions
4. **Performance profiling**: Measure impact of causal masking on/off

## References

- Test file: `tests/test_parity_framework.cpp` (TrueIncrementalDecodeVsPyTorch)
- Softmax implementation: `src/kernels/common/softmax_core.cpp`
- Attention kernel: `src/kernels/MPIAttentionKernel.cpp`
- Related issue: GQA expansion (commit `cb974e7`)
