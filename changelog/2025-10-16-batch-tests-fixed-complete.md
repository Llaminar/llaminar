# Batch Test Suite Fixed - All 16/16 Tests Passing

**Date:** October 16, 2025  
**Status:** ✅ **COMPLETE**  
**Author:** David Sanftenberg

## Overview

Fixed critical bugs in MPILinearOperator and test_operator_batch_interfaces.cpp that were causing test failures. All 16 batch tests now passing (100%).

## Issues Fixed

### 1. MPILinearOperator Weight Dimension Validation Bug

**File:** `src/operators/MPILinearOperator.cpp`  
**Line:** 264

**Problem:** Incorrect validation check comparing wrong weight dimensions
```cpp
// BEFORE (WRONG)
if (input->shape()[1] != weight->shape()[0]) {
    LOG_ERROR("Input size " << input->shape()[1] 
              << " doesn't match weight input size " << weight->shape()[0]);
    return false;
}
```

**Fix:** Weight format is `[out_dim, in_dim]`, so check against `weight[1]`
```cpp
// AFTER (CORRECT)
if (input->shape()[1] != weight->shape()[1]) {
    LOG_ERROR("Input size " << input->shape()[1] 
              << " doesn't match weight input size " << weight->shape()[1] 
              << " (weight shape=[" << weight->shape()[0] << "," << weight->shape()[1] << "])");
    return false;
}
```

**Impact:** This bug was rejecting valid inputs with correct dimensions.

### 2. Test File Weight Dimension Convention

**File:** `tests/test_operator_batch_interfaces.cpp`

**Problem:** Tests were creating weights with reversed dimensions
- Tests used: `create2DTensor(hidden_dim, output_dim)` → `[512, 1024]`
- Operators expect: `[output_dim, hidden_dim]` → `[1024, 512]`

**Root Cause:** Weight matrices are stored transposed for efficiency:
```cpp
// From MPILinearBatchOperator.cpp line 82:
// Weight is [out_dim, in_dim]
// For C = A @ W^T computation
```

**Fixes Applied:**
- All linear operator tests: Changed weight creation from `[hidden_dim, output_dim]` to `[output_dim, hidden_dim]`
- RMSNorm tests: Changed from 2D `[1, hidden_dim]` to 1D `[hidden_dim]` weight vectors

### 3. Test Output Pre-allocation Issues

**Problem:** Some tests used `outputs(1)` expecting auto-allocation, but operators require pre-allocated tensors

**Fixes Applied:**
- `LinearDetects3DInput`: Pre-allocate output tensor
- `LinearWithBias2D`: Pre-allocate output tensor  
- `LinearWithBias3D`: Pre-allocate output tensor
- `LinearBackwardCompatible`: Pre-allocate output tensor
- `RMSNormDetects2DInput`: Pre-allocate output tensor
- `RMSNormDetects3DInput`: Pre-allocate output tensor
- `RMSNormBackwardCompatible`: Pre-allocate output tensor (plus missing test body)
- `EmbeddingBackwardCompatible`: Pre-allocate output tensor

### 4. Operator Selection for 3D Inputs

**Problem:** Tests were using legacy `MPILinearOperator` for 3D inputs, but it doesn't support batching

**Fix:** Updated tests to use appropriate operators:
- 2D inputs `[seq, hidden]` → `MPILinearOperator`
- 3D inputs `[batch, seq, hidden]` → `MPILinearBatchOperator`

**Changes:**
- `LinearDetects3DInput`: Changed to `MPILinearBatchOperator`
- `LinearWithBias3D`: Changed to `MPILinearBatchOperator`
- `BatchDimensionPreservedThroughLinear`: Changed to `MPILinearBatchOperator`

## Test Results

### Before Fixes
```
0% tests passed, 1 tests failed out of 1
- Segfaults and validation errors throughout
```

### After Fixes
```
100% tests passed, 0 tests failed out of 1

Test #122: OperatorBatchInterfaceTest .......   Passed    1.91 sec
  ✅ 17/17 individual test cases passing
```

### Full Batch Test Suite
```
100% tests passed, 0 tests failed out of 16

Tests passing:
  ✅ BatchedKVCacheTest (0.02s)
  ✅ AttentionBatchIntegrationTest (2.25s)
  ✅ OperatorBatchInterfaceTest (1.51s)
  ✅ BatchLinearOperatorTest (1.40s)
  ✅ BatchSwiGLUOperatorTest (2.70s)
  ✅ BatchQwenPipelineTest (35.21s)
  ✅ BatchQwenPipelinePerformanceTest (111.25s)
  ✅ BatchQwenPipelineParityTest (92.36s)
  ✅ And 8 more...

Total: 16/16 tests (313.32 seconds)
```

## Files Modified

### Source Code (Bug Fixes)
1. **src/operators/MPILinearOperator.cpp** (line 264)
   - Fixed weight dimension validation check
   - Changed `weight->shape()[0]` to `weight->shape()[1]`

### Test Code (Corrections)
2. **tests/test_operator_batch_interfaces.cpp**
   - Fixed weight dimensions in 10+ tests
   - Added output pre-allocation to 8 tests
   - Changed operator types for 3D input tests
   - Fixed RMSNorm weight from 2D to 1D
   - Completed missing test body for `RMSNormBackwardCompatible`

## Key Learnings

### Weight Matrix Convention
**Important:** All linear operators use transposed weight format `[out_dim, in_dim]` because:
1. Computation is `C = A @ W^T` (A is row-major activations)
2. Storing W^T directly avoids transpose operation
3. Memory layout is optimized for cache access

**When creating weights:**
```cpp
// CORRECT
auto weight = create2DTensor(output_dim, hidden_dim, 0.01f);  // [out, in]

// WRONG
auto weight = create2DTensor(hidden_dim, output_dim, 0.01f);  // [in, out] ❌
```

### Output Tensor Allocation
**All Llaminar operators require pre-allocated output tensors:**
```cpp
// CORRECT
auto output = std::make_shared<SimpleTensor>(std::vector<int>{batch, seq, hidden});
std::vector<std::shared_ptr<TensorBase>> outputs = {output};
bool success = op.execute(inputs, outputs);

// WRONG - Will fail with null tensor error
std::vector<std::shared_ptr<TensorBase>> outputs(1);  // Contains nullptr ❌
```

### Batch vs Non-Batch Operators
- **2D tensors** `[seq, hidden]`: Use `MPILinearOperator`, `MPIRMSNormOperator`
- **3D tensors** `[batch, seq, hidden]`: Use `MPILinearBatchOperator`, `MPIRMSNormOperator` (supports both)

Exception: `MPIRMSNormOperator` handles both 2D and 3D natively (discovered in Day 2).

## Impact on Option C

This completes the **KV cache validation** phase (todo item 6):
- ✅ Created `test_attention_batch_integration.cpp` (5 tests for batch sizes 1, 4, 8, 16, 32)
- ✅ Fixed `test_operator_batch_interfaces.cpp` (17 tests for operator validation)
- ✅ Fixed critical bug in `MPILinearOperator` validation
- ✅ All 16/16 batch tests passing

**Option C Status:** ~90% complete
- ✅ Day 1-2: Batch operator implementation
- ✅ Day 3: KV cache validation and test fixes
- 📋 Remaining: End-to-end benchmarking (~2 hours)

## Next Steps

**Priority:** Complete Option C benchmarking
1. Run `./run_batch_performance.sh` or equivalent benchmark
2. Compare Phase 4.1 vs Option C throughput
3. Measure prefill (expect ~48.5× maintained) and decode (expect 2-3× improvement)
4. Document results and create final Option C completion report

**Estimated Time:** 2 hours

## Conclusion

Successfully debugged and fixed all batch test failures. The issues were:
1. **Validation bug** in MPILinearOperator checking wrong dimension
2. **Test convention error** creating weights in wrong format
3. **Missing output allocation** in several tests
4. **Wrong operator selection** for 3D inputs

All fixes applied cleanly with zero regressions. Batch test suite now 100% passing and ready for final benchmarking phase.
