# Kernel Unit Tests Pass - Bug is Upstream

**Date**: 2025-11-22 23:50
**Status**: ✅ **KERNEL TESTS ALL PASS** - Bug is in pipeline orchestration, not kernel!

## Summary

Created comprehensive unit tests for `CpuAttentionKernelT::compute_batch()` to verify mask application and padding handling. **All 4 tests pass perfectly**, proving the kernel correctly:
1. Applies padding masks to attention scores
2. Zeros out padding position outputs  
3. Matches sequential and batch execution exactly
4. Prevents cross-sequence contamination

**Critical Finding**: The E2E test fails with the same kernel that passes all unit tests. This means **the bug is in how the pipeline calls the kernel**, not in the kernel itself.

## Test Results

### Test File
- **`Test__CpuAttentionKernelT_PaddingMask.cpp`** (677 lines)
- 4 comprehensive tests targeting the batch padding bug

### Test 1: MaskIsAppliedToScores ✅ PASS
```
[MASK VERIFICATION]
Mask for positions 4-7 (Seq1):
  Row 4: [-INF, -INF, -INF, -INF,     0,     0, -INF, -INF]
  Row 5: [-INF, -INF, -INF, -INF,     0,     0, -INF, -INF]
  Row 6: [-INF, -INF, -INF, -INF, -INF, -INF, -INF, -INF]  ← Padding row
  Row 7: [-INF, -INF, -INF, -INF, -INF, -INF, -INF, -INF]  ← Padding row

[ATTENTION SCORES - SEQ1]
Head 0:
  Row 2 (PAD): [0.00000, 0.00000, 0.00000, 0.00000]  ← All zeros!
  Row 3 (PAD): [0.00000, 0.00000, 0.00000, 0.00000]  ← All zeros!

[OUTPUT VERIFICATION]
Position 6 (PAD): sum(abs) = 0.00000  ✅
Position 7 (PAD): sum(abs) = 0.00000  ✅
```

**Result**: Mask correctly applied, padding outputs correctly zeroed.

### Test 2: SequentialVsBatchWithPadding ✅ PASS
```
[COMPARISON: SEQUENTIAL VS BATCH]
✅ Seq0 matches perfectly
✅ Seq1 matches perfectly  
✅ Padding output is zero
```

**Result**: Batch execution with padding produces identical results to sequential execution.

### Test 3: NoCrossSequenceContamination ✅ PASS
```
Seq0 output (expect ~1.0): [1.00000, 1.00000, ...]  ✅
Seq1 output (expect ~10.0): [10.00000, 10.00000, ...]  ✅
```

**Result**: No cross-sequence contamination - sequences remain independent.

### Test 4: AttentionScoreInspection ✅ PASS
```
Row 0: [0.28496, 0.71504, 0.00000, 0.00000] sum=1.00000  ← Attends only to [0,1]
Row 1: [0.22444, 0.77556, 0.00000, 0.00000] sum=1.00000  ← Attends only to [0,1]
Row 2: [0.00000, 0.00000, 0.00000, 0.00000] sum=0.00000  ← Padding row all-zero
Row 3: [0.00000, 0.00000, 0.00000, 0.00000] sum=0.00000  ← Padding row all-zero
```

**Result**: Attention scores correctly show zero weights for padding, sum to 1.0 for real tokens.

## Analysis: Why E2E Fails When Kernel Works

The kernel tests prove that `CpuAttentionKernelT::compute_batch()` works correctly. So why does the E2E test fail?

### Kernel Test Setup (PASSES)
```cpp
// Direct kernel call with proper mask
CpuAttentionKernelT<FP32Tensor> attention;
attention.compute_batch(
    Q, K, V, output,
    batch_size, seq_len, n_heads, n_kv_heads, head_dim,
    false, -1, 
    scores_workspace,
    nullptr, nullptr,
    mask_tensor,  // ← Mask provided here
    false, nullptr, -1
);
```

### E2E Test Setup (FAILS)
```cpp
// Pipeline execution
Qwen2Pipeline pipeline;
pipeline.forward(input_ids, batch_size);

// Internally calls GQAAttention::compute_batch()
//   → CpuAttentionKernelT::compute_batch()
```

**Hypothesis**: The mask is constructed correctly (Test__AttentionMaskDiagnostic proves this), but something in the **pipeline orchestration** is:

1. **Not passing the mask to the kernel** in batch mode
2. **Passing the wrong mask pointer**
3. **Using the wrong batch layout** (e.g., wrong offsets)
4. **Corrupting K/V cache** during batch processing

## Next Investigation: GQAAttention::compute_batch()

The chain is:
```
Qwen2Pipeline::forward()
  → Qwen2Pipeline::attention_block()
    → MpiAttentionOrchestrator::compute()
      → GQAAttention::compute_batch()
        → CpuAttentionKernelT::compute_batch()  ← Works perfectly!
```

Since the kernel works, the bug must be in **`GQAAttention::compute_batch()`** or how it's called from `MpiAttentionOrchestrator`.

### Key Questions

1. **Is the mask being passed to the kernel?**
   - Check if `GQAAttention::compute_batch()` forwards `workspace_mask` to kernel
   - Could be nullptr when it should have a value

2. **Is the batch layout correct?**
   - Check if Q/K/V buffer offsets match what the kernel expects
   - Kernel expects: `[batch_size, seq_len, n_heads/n_kv_heads, head_dim]`

3. **Is the K/V cache being written correctly?**
   - Batch mode might write padding entries to cache
   - Cache reads might include padding data

4. **Is the attention output being post-processed?**
   - Something after the kernel might be overwriting the correct output
   - Residual connection or output projection might not respect padding

## Code Locations to Investigate

### 1. GQAAttention::compute_batch()
**File**: `src/v2/pipelines/attention/GQAAttention.cpp`

Check:
- Line ~242-327: Does it pass `workspace_mask` to kernel?
- Are buffer offsets calculated correctly for batch layout?

### 2. MpiAttentionOrchestrator::compute()
**File**: `src/v2/pipelines/attention/MpiAttentionOrchestrator.cpp`

Check:
- How does it construct the mask for batch mode?
- Does it handle `actual_lengths` correctly?

### 3. Qwen2Pipeline::attention_block()
**File**: `src/v2/pipelines/qwen/Qwen2Pipeline.cpp`

Check:
- Line ~355+: Batch processing flow
- Are `actual_lengths` being passed to orchestrator?
- Is the mask being constructed with correct lengths?

## Test Commands

### Kernel Unit Tests (PASS)
```bash
./build_v2/tests/v2/v2_test_cpu_attention_kernel_t_padding_mask
# Result: 4/4 PASSED
```

### E2E Test (FAIL)
```bash
export LLAMINAR_DEBUG_BATCH=0 LLAMINAR_LOG_LEVEL=ERROR OMP_NUM_THREADS=28
timeout 300 mpirun -np 2 --oversubscribe \
  ./build_v2/tests/v2/v2_test_batch_padding_divergence \
  --gtest_filter="BatchPaddingDivergenceTest.SequentialVsBatchedWithPadding"
# Result: FAILED (97 billion % divergence)
```

## Recommended Next Steps

1. **Add logging to GQAAttention::compute_batch()**
   ```cpp
   LOG_INFO("compute_batch: mask=" << (workspace_mask ? "present" : "nullptr"));
   LOG_INFO("compute_batch: batch_size=" << batch_size << ", seq_len=" << seq_len);
   ```

2. **Verify mask is passed through the call chain**
   - Add assertions: `ASSERT(workspace_mask != nullptr)`
   - Log mask pointer addresses at each level

3. **Check buffer layouts**
   - Add diagnostic prints of Q/K/V shapes and strides
   - Verify batch offsets match kernel expectations

4. **Inspect K/V cache writes**
   - Check if padding positions are being written to cache
   - Verify cache reads skip padding entries

## Conclusion

**Progress**:
1. ✅ Mask construction is correct (previous session)
2. ✅ Kernel mask application is correct (this session)
3. ❌ E2E test still fails → **Bug is in pipeline orchestration**

**Next Focus**: Trace through `GQAAttention::compute_batch()` and `MpiAttentionOrchestrator::compute()` to find where the mask gets lost or the batch layout gets corrupted.

The kernel works perfectly in isolation. The bug is in how the **pipeline wires everything together** for batch processing.
