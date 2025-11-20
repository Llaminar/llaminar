# Batched Attention Debugging Session Summary
**Date**: 2025-01-XX  
**Issue**: E2E batch parity tests fail (ComprehensiveBatchParity, MultiSequenceBatch, BatchScaling)  
**Status**: Root cause identified but E2E tests still failing

## Summary

Successfully identified and fixed a **heap buffer overflow** in the batched attention mask allocation, but E2E tests still fail despite unit tests passing. The issue appears to be in pipeline-level integration rather than the attention kernel itself.

## Discoveries

### 1. Mask Buffer Sizing Bug (FOUND AND FIXED)

**Location**: `create_combined_batch_mask()` in `AttentionUtils.h:429`

**Root Cause**: The mask buffer must be `[total_len, total_len]` where `total_len = batch_size * seq_len`, but tests were allocating `[total_len, seq_len]`.

**Evidence**: AddressSanitizer caught heap-buffer-overflow:
```
WRITE of size 4 at 0x5100000dcfc0 thread T0
mask[i * total_len + j] = can_attend ? 0.0f : neg_inf;
```

**Fix**: Changed all test workspace_mask allocations from:
```cpp
// WRONG
config.workspace_mask = create_fp32_tensor(batch_size * seq_len, seq_len);

// CORRECT  
config.workspace_mask = create_fp32_tensor(batch_size * seq_len, batch_size * seq_len);
```

**Status**: ✅ PipelineBase.cpp already had correct allocation (line 587)  
**Status**: ✅ Unit tests now pass with corrected allocation

### 2. Unit Test Results

**Test File**: `tests/v2/unit/Test__GQAAttention_SecondSequence.cpp`

**All 6 Tests PASS**:
1. ✅ NonZeroOutput - Sequence 1 produces non-zero results
2. ✅ IsolatedComparison - Sequence 1 batched vs standalone matches
3. ✅ BufferOffsetValidation - Buffer offsets correct with sequential patterns
4. ✅ CausalMaskApplication - Causal mask applied correctly to sequence 1
5. ✅ CrossSequenceIsolation - Sequence 0 changes don't affect sequence 1
6. ✅ SymmetricProcessing - Swapping sequences produces swapped outputs

**Conclusion**: The `GQAAttention::compute_batch()` kernel works correctly in isolation with proper workspace buffer allocation.

### 3. E2E Test Results (Still Failing)

**Test**: `Qwen2E2ECorrectness.ComprehensiveBatchParity`

**Results**:
- Sequence 0: max_diff=19.1991, mean_diff=2.83433 (FAIL)
- Sequence 1: max_diff=21.349, mean_diff=2.70251 (FAIL)

**Key Observation**: Both sequences fail in E2E tests, but unit tests pass. This suggests the issue is NOT in the attention kernel itself, but in how the pipeline invokes batched attention or manages batched data flow.

## Technical Analysis

###Workspace Buffer Sizes (Correct in PipelineBase)

```cpp
// src/v2/pipelines/PipelineBase.cpp (lines 560-590)
const int total_len = batch_size * max_seq_len;

// Scores: [n_heads * total_len, total_len] ✅
attention_workspace_scores_ = FP32Tensor({n_heads * total_len, total_len});

// QKV: [max_threads * total_len * head_dim * 3] ✅  
attention_workspace_qkv_buffer_ = FP32Tensor({max_threads * total_len * head_dim_ * 3});

// Context: [max_threads * total_len * head_dim] ✅
attention_workspace_context_ = FP32Tensor({max_threads * total_len * head_dim_});

// Mask: [total_len * total_len] ✅
attention_workspace_mask_ = FP32Tensor({total_len * total_len});
```

**All buffers correctly sized for batched execution.**

### Data Flow Chain

```
Qwen2Pipeline::forward_batched()
  ↓
PipelineBase::attention_gqa_batch()  
  ↓
MpiAttentionOrchestrator::compute_batch()
  ↓  
GQAAttention::compute_batch()
  ↓
create_combined_batch_mask() → ASAN caught overflow here
  ↓
CpuAttentionKernelT::compute_batch()
```

## Next Steps

### 1. Verify Pipeline Batched Path Setup

Check if `Qwen2Pipeline::forward_batched()` correctly:
- Initializes workspace buffers with `initializeDeviceInfrastructure(max_seq_len, batch_size)`
- Passes correct `batch_size` and `actual_lengths` to attention calls
- Handles batched tensor layout: `[batch_size * seq_len, d_model]`

### 2. Check Batched Forward Implementation

Verify `Qwen2Pipeline::forward_batched()`:
- Correctly concatenates input sequences
- Properly distributes Q/K/V projections across batched inputs
- Correctly extracts sequence-specific outputs from batched results

### 3. Add Instrumentation

Add logging to trace:
- Input tensor shapes at each pipeline stage
- Actual buffer sizes allocated
- Batch parameters passed to attention calls
- Output tensor layouts

### 4. Compare Single vs Batched Execution

Run the same input through:
1. Two separate single-sequence forwards (ground truth)
2. One batched forward with batch_size=2 (test)
3. Compare outputs sequence-by-sequence

### 5. Investigate Projection Layers

Check if FFN/attention projection layers correctly handle batched tensors:
- GEMM operations on `[batch_size * seq_len, d_model]` inputs
- Weight broadcasting
- Output tensor reshaping

## Files Modified

1. `tests/v2/unit/Test__GQAAttention_SecondSequence.cpp` (NEW, 636 lines):
   - 6 diagnostic tests for sequence 1 handling
   - All tests PASS after mask size fix
   - Tests: NonZeroOutput, IsolatedComparison, BufferOffsetValidation, CausalMaskApplication, CrossSequenceIsolation, SymmetricProcessing

2. `tests/v2/CMakeLists.txt`:
   - Added v2_test_gqa_attention_second_sequence test target
   - Labels: V2, Unit, Attention, Batched, SecondSequence, BufferOffsets, GQA

## Key Insights

1. **Kernel is Correct**: `GQAAttention::compute_batch()` works correctly with proper buffer sizes
2. **Production Code is Correct**: `PipelineBase.cpp` allocates buffers correctly
3. **Test Bug**: Unit test originally had wrong mask size (caught by ASAN)
4. **E2E Still Fails**: Issue must be in pipeline-level batched execution, not attention kernel
5. **Hypothesis**: Problem likely in:
   - How batched inputs are prepared/concatenated
   - How batched outputs are extracted/split  
   - Projection layer handling of batched tensors
   - RMSNorm/RoPE application on batched data

## Reproduction

### Unit Tests (PASS)
```bash
cd /workspaces/llaminar/build_v2
ctest -R "V2_Unit_GQAAttention_SecondSequence" --output-on-failure
# Result: 100% tests passed (6/6)
```

### E2E Tests (FAIL)
```bash
timeout 180 mpirun -np 2 --bind-to socket --map-by socket \
  ./build_v2_e2e/tests/v2/v2_test_qwen2_e2e_correctness \
  --gtest_filter="Qwen2E2ECorrectness.ComprehensiveBatchParity"
# Result: Both sequences fail with max_diff > 19
```

## Conclusion

The attention kernel bug (buffer overflow) has been fixed and unit tests confirm correct behavior. However, E2E tests still fail, indicating the issue is at a higher level in the pipeline's batched execution path. The next debugging session should focus on:
1. Pipeline batched forward implementation
2. Input/output tensor layout handling  
3. Projection layer batched execution
4. Per-layer operation handling of batched data

**Recommendation**: Add detailed logging to `Qwen2Pipeline::forward_batched()` to trace tensor shapes and identify where divergence begins.
