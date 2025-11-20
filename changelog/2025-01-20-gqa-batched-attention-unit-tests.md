# GQA Batched Attention Unit Tests - Success Report

**Date:** 2025-01-20  
**Status:** ✅ ALL TESTS PASSING (6/6)

## Summary

Created comprehensive unit tests for `GQAAttention::compute_batch()` to isolate and debug the divergence issues seen in E2E tests. **Tests immediately revealed a critical bug in workspace mask allocation**.

## Test Coverage

### File: `tests/v2/unit/Test__GQAAttention_BatchedPath.cpp`

**6 comprehensive tests** covering all batched attention scenarios:

1. **BasicBatchedCompute**: Verifies `compute_batch()` executes without errors
   - Tests: batch_size=2, seq_len=4, n_heads=2, non-causal attention
   - Validates: No NaN/Inf in outputs, correct execution

2. **BatchSize1EquivalentToSingleSequence**: Critical equivalence test
   - Tests: batch_size=1 via `compute_batch()` vs `compute()` single-sequence path
   - Validates: Outputs are identical (tolerance: 1e-4)
   - Purpose: Ensures batched path doesn't add artifacts when batch_size=1

3. **BatchedEquivalentToSequentialExecution**: Multi-sequence validation
   - Tests: batch_size=3 batched execution vs 3 sequential single-sequence calls
   - Validates: Batched processing matches independent sequential processing
   - Purpose: Ensures batch independence - sequences don't interfere

4. **CausalVsNonCausalMasking**: Masking behavior validation
   - Tests: Same inputs with causal=true vs causal=false
   - Validates: Outputs differ meaningfully (causal blocks future, non-causal allows)
   - Purpose: Confirms masking logic correctly controls attention patterns

5. **PaddingMaskCorrectness**: Variable-length sequence handling
   - Tests: batch_size=2 with actual_lengths=[4,3], seq_len=6 (padded)
   - Validates: Real tokens have meaningful outputs, padding tokens near-zero
   - Purpose: Ensures padding mask prevents padding from influencing attention

6. **GQAHeadBroadcasting**: GQA-specific functionality
   - Tests: n_heads=4, n_kv_heads=2 (K/V head broadcasting)
   - Validates: Executes without errors, correct output shape
   - Purpose: Validates GQA head broadcasting mechanism

## Critical Bug Found

### Issue: Workspace Mask Buffer Overflow

**Symptom**: AddressSanitizer heap-buffer-overflow in `create_combined_batch_mask`

**Root Cause**: Workspace mask tensors allocated with incorrect dimensions:
```cpp
// ❌ WRONG (original code in tests):
config.workspace_mask = create_fp32_tensor(batch_size * seq_len, seq_len);
// Allocates: [total_len, seq_len] = [8, 4] for batch_size=2, seq_len=4

// ✅ CORRECT (fixed):
config.workspace_mask = create_fp32_tensor(batch_size * seq_len, batch_size * seq_len);
// Allocates: [total_len, total_len] = [8, 8] for batch_size=2, seq_len=4
```

**Explanation**: 
- The `create_combined_batch_mask` function in `AttentionUtils.h` (line 429) writes to `mask[i * total_len + j]`
- This requires a **square mask matrix** of size `[total_len, total_len]` where `total_len = batch_size * seq_len`
- The mask represents attention relationships between **all tokens across all batches**
- Allocating `[total_len, seq_len]` caused out-of-bounds writes for `j >= seq_len`

### Impact

This bug would affect **any code path using batched attention with workspace masks**, including:
- E2E tests (ComprehensiveBatchParity, MultiSequenceBatch, BatchScaling)
- Qwen2Pipeline batched inference
- Any pipeline using GQAAttention::compute_batch()

## Test Infrastructure

### Helper Functions

```cpp
// Tensor creation without TensorFactory (for unit tests)
std::unique_ptr<FP32Tensor> create_fp32_tensor(size_t rows, size_t cols)
{
    return std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols}, 0);
}

// Deterministic initialization
void init_random_like(float *data, int count, int seed);
void init_test_data(float *data, int count, float scale, float offset);

// Detailed comparison with error reporting
struct ComparisonResult {
    bool equal;
    float max_abs_diff;
    float mean_abs_diff;
    int first_mismatch_idx;
    float first_expected;
    float first_actual;
};
ComparisonResult compare_tensors(const float *expected, const float *actual, 
                                 int count, float tolerance);
```

### CMake Integration

```cmake
# tests/v2/CMakeLists.txt
add_executable(v2_test_gqa_attention_batched_path 
    unit/Test__GQAAttention_BatchedPath.cpp)
target_link_libraries(v2_test_gqa_attention_batched_path 
    llaminar2_core GTest::gtest GTest::gtest_main)
add_v2_test(V2_Unit_GQAAttention_BatchedPath
    COMMAND $<TARGET_FILE:v2_test_gqa_attention_batched_path>
    LABELS "V2;Unit;Attention;GQA;Batched;Causal;Padding;HeadBroadcasting"
    MPI_PROCS 1)
```

## Running the Tests

```bash
# Build
cmake --build build_v2 --target v2_test_gqa_attention_batched_path --parallel

# Run all GQA batched tests
ctest --test-dir build_v2 -R "V2_Unit_GQAAttention_BatchedPath" --output-on-failure

# Run specific test
ctest --test-dir build_v2 -R "V2_Unit_GQAAttention_BatchedPath" -V \
  --gtest_filter="GQAAttention_BatchedPath.BatchSize1EquivalentToSingleSequence"
```

## Test Results

```
[==========] Running 6 tests from 1 test suite.
[----------] 6 tests from GQAAttention_BatchedPath
[ RUN      ] GQAAttention_BatchedPath.BasicBatchedCompute
[       OK ] GQAAttention_BatchedPath.BasicBatchedCompute
[ RUN      ] GQAAttention_BatchedPath.BatchSize1EquivalentToSingleSequence
[       OK ] GQAAttention_BatchedPath.BatchSize1EquivalentToSingleSequence
[ RUN      ] GQAAttention_BatchedPath.BatchedEquivalentToSequentialExecution
[       OK ] GQAAttention_BatchedPath.BatchedEquivalentToSequentialExecution
[ RUN      ] GQAAttention_BatchedPath.CausalVsNonCausalMasking
[       OK ] GQAAttention_BatchedPath.CausalVsNonCausalMasking
[ RUN      ] GQAAttention_BatchedPath.PaddingMaskCorrectness
[       OK ] GQAAttention_BatchedPath.PaddingMaskCorrectness
[ RUN      ] GQAAttention_BatchedPath.GQAHeadBroadcasting
[       OK ] GQAAttention_BatchedPath.GQAHeadBroadcasting
[----------] 6 tests from GQAAttention_BatchedPath (480 ms total)

[  PASSED  ] 6 tests.
```

## Next Steps

1. ✅ **Unit tests passing** - Batched attention works correctly when masks are properly allocated
2. **Fix E2E tests**: Update E2E tests to allocate workspace masks with correct dimensions
3. **Fix Qwen2Pipeline**: Ensure pipeline allocates workspace buffers correctly during initialization
4. **Verify**: Re-run E2E tests after fixes to confirm parity

## Key Learnings

1. **ASAN is invaluable**: Immediately caught the buffer overflow that would have been hard to debug otherwise
2. **Unit tests before integration**: These focused tests isolated the issue much faster than E2E debugging
3. **Workspace buffer allocation is critical**: Mask dimensions must match the attention implementation's expectations
4. **Test equivalence at boundaries**: `batch_size=1` equivalence test is crucial for validating batched paths

## Files Modified

- **Created**: `tests/v2/unit/Test__GQAAttention_BatchedPath.cpp` (~630 lines)
- **Modified**: `tests/v2/CMakeLists.txt` (added test registration)

## Documentation

See also:
- `.github/copilot-instructions.md` - V2 testing guidelines
- `changelog/2025-01-XX-e2e-batch-parity-debugging.md` - Full E2E debugging session
- `.github/instructions/llaminar-v2-architecture.instructions.md` - V2 architecture details
