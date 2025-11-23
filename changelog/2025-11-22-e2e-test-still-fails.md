# E2E Test Results After Mask Fix

**Date**: 2025-11-22 23:43
**Test**: `BatchPaddingDivergenceTest.SequentialVsBatchedWithPadding`
**Status**: ❌ **STILL FAILING**

## Summary

Fixed the attention mask construction bug (`window_size=0` issue), but the E2E batch padding divergence test **still fails** with the same 97 billion % divergence. This indicates there are **multiple bugs** in the batch processing pipeline.

## Test Results

### Mask Fix Verification
- ✅ `Test__AttentionMaskDiagnostic::SimpleCausalMask` - **PASSES**
- ✅ `Test__AttentionMaskDiagnostic::SimplePaddingMask` - **PASSES**
- Mask construction is now correct

### E2E Test Results
```
Seq0 (no padding): 0% divergence ✅
Seq1 Position 4: 47 billion % divergence ❌
Seq1 Position 5: 97 billion % divergence ❌
```

**Same divergence as before the mask fix!**

## Analysis

### What We Fixed
1. Attention mask construction in `AttentionUtils.h`
2. Changed `window_size >= 0` to `window_size > 0`
3. Masks now correctly allow tokens to attend to valid positions

### What's Still Broken
The batch processing pipeline has **additional bugs** beyond mask construction:

1. **Possible culprits**:
   - Attention score computation with batch layout
   - Softmax application across batch boundaries
   - K/V cache addressing for batched sequences
   - Residual connections with padding
   - Buffer stride calculations

2. **Evidence**:
   - Seq0 works perfectly (0% divergence)
   - Seq1 completely broken (97B% divergence)
   - Padding positions [6-7] may be contaminating real positions [4-5]

## Logits Analysis

### Sequential Baseline (Correct)
```
Seq1 Position 0 (K/V[4]): [3.21354, -3.20449, 1.78312, ...]
```

### Batch Execution (Broken)
```
Position 4 (Seq1 token0): [2.3493, -0.370318, 0.351106, ...]  ← Wrong!
Position 5 (Seq1 token1): [-0.00926827, 0.279036, 1.25631, ...]  ← Wrong!
Position 6 (PAD): [1.02804, ...]  ← Should be zero!
Position 7 (PAD): [-1.64296, ...]  ← Should be zero!
```

**Key observation**: Padding positions have **non-zero values**! This shouldn't happen if masking was working correctly.

## Next Investigation Steps

### 1. Verify Mask Application in Attention Kernel
The mask construction is fixed, but is it being **applied** correctly in the attention computation?

Check in `CpuAttentionKernelT.cpp`:
- Is the mask pointer being passed to the kernel?
- Is `apply_attention_mask()` being called?
- Are scores being properly masked before softmax?

### 2. Check Padding Position Zeroing
Padded positions should produce zero output. Currently they don't:
- Position 6 sum: 18.78 (should be 0)
- Position 7 sum: 25.78 (should be 0)

This suggests:
- Padding mask not being applied, OR
- Padding outputs not being zeroed after computation

### 3. Inspect Batch Attention Flow
Trace through batch attention execution:
```cpp
GQAAttention::compute_batch() 
  → CpuAttentionKernelT::compute_batch()
    → Per-batch attention computation
      → Mask application
        → Softmax
          → Output accumulation
```

Check if any step mishandles the batch layout.

### 4. Add Attention Score Logging
Instrument attention kernel to log:
- Raw scores before masking
- Scores after masking (should have -inf for padding)
- Scores after softmax (padding should be ~0)
- Context vectors (padding should be zero)

### 5. Check Sequence Length Handling
Verify `actual_lengths` is being passed correctly:
- `GQAAttention::compute_batch(actual_lengths = {4, 2})`
- Are these lengths being used to construct the mask?
- Are they being used to zero padding outputs?

## Code Locations to Investigate

1. **`src/v2/kernels/cpu/CpuAttentionKernelT.cpp`**:
   - Line ~400-600: `compute_batch()` implementation
   - Check mask application before softmax
   - Check output zeroing for padding positions

2. **`src/v2/pipelines/attention/GQAAttention.cpp`**:
   - Line 242-327: `compute_batch()` orchestration
   - Verify mask is constructed (we know it is, tests pass)
   - Verify mask pointer is passed to kernel

3. **`src/v2/pipelines/qwen/Qwen2Pipeline.cpp`**:
   - Line 355+: Batch processing flow
   - Check if `actual_lengths` is being passed correctly

## Hypothesis

**Most Likely**: The mask is constructed correctly, but **not being applied** in the attention kernel for batched computation. The `compute_batch()` path may have a different code path that skips mask application or applies it incorrectly.

**Evidence**:
- Mask construction tests pass
- Padding positions have non-zero outputs
- Only affects batched path (sequential works fine)

## Test Commands

```bash
# Mask diagnostic (these now pass)
./build_v2/tests/v2/v2_test_attention_mask_diagnostic

# E2E test (still fails)
export LLAMINAR_DEBUG_BATCH=0 LLAMINAR_LOG_LEVEL=ERROR OMP_NUM_THREADS=28
timeout 300 mpirun -np 2 --oversubscribe \
  ./build_v2/tests/v2/v2_test_batch_padding_divergence \
  --gtest_filter="BatchPaddingDivergenceTest.SequentialVsBatchedWithPadding"
```

## Conclusion

We've made progress:
1. ✅ Found and fixed mask construction bug
2. ✅ Verified masks are now correct
3. ❌ E2E test still fails - **mask application** or **output zeroing** is broken

The next debugging session should focus on:
1. Tracing mask application in `compute_batch()` path
2. Verifying padding outputs are zeroed
3. Adding attention score logging to pinpoint exact failure point

**Status**: Mask construction fixed, but batch attention still has critical bugs.
