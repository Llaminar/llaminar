# MPILinearOperator_v2 Refactoring to Use ITensorGemm Directly

**Date**: October 22, 2025  
**Status**: ✅ Complete - All tests passing

## Summary

Successfully refactored `MPILinearOperator_v2` to use the `ITensorGemm` interface directly instead of routing through `adaptiveMatMul`. This simplifies the code path and properly supports distributed quantized weight computation via row-offset parameters.

## Problem

The previous implementation had architectural issues:

1. **Routing through adaptiveMatMul was unnecessary** - Tensors with ITensorGemm support should use it directly
2. **Quantized weights couldn't be distributed** - IQ4_NL weights are compressed blocks that can't be memcpy'd like FP32 weights
3. **Dimension validation failed** - GEMM expected local dimensions but received global weight reference
4. **Tests were failing** - 3/4 tests failing with "FP32 GEMM failed" or exceptions

## Solution

### 1. Extended ITensorGemm Interface

Added `row_offset` and `row_count` parameters to support distributed computation:

```cpp
virtual bool multiply(const float *A, float *C,
                     int m, int n, int k,
                     bool transpose_B = true,
                     float alpha = 1.0f,
                     float beta = 0.0f,
                     int row_offset = 0,      // NEW
                     int row_count = -1) = 0;  // NEW

virtual bool multiply_bf16(const uint16_t *A_bf16, float *C,
                          int m, int n, int k,
                          bool transpose_B = true,
                          float alpha = 1.0f,
                          float beta = 0.0f,
                          int row_offset = 0,      // NEW
                          int row_count = -1);     // NEW
```

**Rationale:**
- MPI ranks need to compute only their portion of output features
- For FP32 weights: Can memcpy rows [offset:offset+count]
- For quantized weights: Must decode only rows [offset:offset+count] on-the-fly

### 2. Updated IQ4_NLQuantizedGemm Implementation

Modified to decode only the specified row range:

```cpp
// Old: Validate against full tensor dimensions
if (tensor_->shape()[0] != expected_rows || ...) {
    return false;
}

// New: Validate row range, decode only requested rows
if (row_offset < 0 || row_offset + row_count > tensor_->shape()[0]) {
    return false;
}

// Decode with offset
tensor_->decode_block_at(row_offset + j, kb, B_block);
```

### 3. Updated FP32Gemm Implementation

Adjusted pointer to subset of rows:

```cpp
// Adjust B pointer to start at the offset row
const float* B_subset = B + row_offset * shape[1];

// Use subset in BLAS call
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
           m, row_count, k,  // Note: row_count instead of n
           alpha,
           A, k,
           B_subset, k,      // Use subset
           beta,
           C, row_count);
```

### 4. Refactored MPILinearOperator_v2

Removed `adaptiveMatMul` dependency, use ITensorGemm directly:

```cpp
// Try ITensorGemm interface first
ITensorGemm* gemm = local_weight->createGemmRaw();
if (gemm && gemm->supports(m, n, k)) {
    // For quantized: use row_offset (global weight reference)
    // For FP32: no offset needed (already distributed subset)
    int row_offset = (local_weight->native_type() == TensorDataType::QUANTIZED) 
                      ? out_offset : 0;
    
    success = gemm->multiply(
        input->data(),
        local_output->data(),
        m, n, k,
        /*transpose_B=*/true,
        /*alpha=*/1.0f,
        /*beta=*/0.0f,
        /*row_offset=*/row_offset,
        /*row_count=*/local_out_dim
    );
    delete gemm;
} else {
    // Fallback to BLAS for FP32 weights
    cblas_sgemm(...);
}
```

## Files Modified

### Core Interface
- **src/ITensorGemm.h**: Added row_offset/row_count parameters to both multiply() methods

### Implementations
- **src/tensors/IQ4_NLTensor.h**:
  - Updated `IQ4_NLQuantizedGemm::multiply()` signature and validation
  - Updated `IQ4_NLQuantizedGemm::multiply_bf16()` signature and validation
  - Changed all `tensor_->decode_block_at(j, ...)` to `tensor_->decode_block_at(row_offset + j, ...)`

- **src/tensors/FP32Tensor.h**:
  - Updated `FP32Gemm::multiply()` signature and validation
  - Adjusted B pointer with `B_subset = B + row_offset * shape[1]`
  - Updated BLAS calls to use `row_count` instead of `n`

### Operator
- **src/operators/MPILinearOperator_v2.cpp**:
  - Removed `#include "../AdaptiveMatmul.h"`
  - Added `#include "../ITensorGemm.h"` and `#include <cblas.h>`
  - Replaced all `adaptiveMatMul()` calls with direct ITensorGemm usage
  - Added row_offset calculation: `(is_quantized) ? out_offset : 0`
  - Maintained BLAS fallback for weights without GEMM support

## Test Results

```
[==========] Running 4 tests from 1 test suite.
[----------] 4 tests from MPILinearOperatorV2IQ4NLTest
[ RUN      ] MPILinearOperatorV2IQ4NLTest.FP32ActivationWithIQ4NLWeight
       OK ] MPILinearOperatorV2IQ4NLTest.FP32ActivationWithIQ4NLWeight (36 ms)
[ RUN      ] MPILinearOperatorV2IQ4NLTest.BF16ActivationWithIQ4NLWeight
       OK ] MPILinearOperatorV2IQ4NLTest.BF16ActivationWithIQ4NLWeight (37 ms)
[ RUN      ] MPILinearOperatorV2IQ4NLTest.WeightCachingBehavior
       OK ] MPILinearOperatorV2IQ4NLTest.WeightCachingBehavior (35 ms)
[ RUN      ] MPILinearOperatorV2IQ4NLTest.VariousSequenceLengths
       OK ] MPILinearOperatorV2IQ4NLTest.VariousSequenceLengths (30 ms)
[----------] 4 tests from MPILinearOperatorV2IQ4NLTest (152 ms total)

[  PASSED  ] 4 tests.
```

**All tests now passing!** ✅

## Technical Benefits

### 1. Simpler Code Path
- **Before**: `operator → adaptiveMatMul → cache check → createGemm → multiply`
- **After**: `operator → createGemm → multiply`
- Removed one layer of indirection

### 2. Proper Distributed Quantized GEMM
- Quantized weights stay in global memory (one copy)
- Each rank decodes only its output feature rows on-the-fly
- No redundant decompression or memory waste

### 3. Clearer Intent
```cpp
// Now it's obvious we're using tensor-specific GEMM
ITensorGemm* gemm = weight->createGemmRaw();
if (gemm) {
    gemm->multiply(A, C, m, n, k, ...);
}
```

### 4. Deprecation Path for adaptiveMatMul
- adaptiveMatMul is still used in legacy code but no longer in v2 operator
- Can be gradually phased out as other operators migrate to ITensorGemm

## Performance Characteristics

### Memory Usage
- **FP32 weights**: Each rank stores only its rows (distributed)
- **Quantized weights**: All ranks reference global tensor (shared), decode on-demand

### Computation Distribution
```
Example: 2 ranks, 896 output features, 896 input features

Rank 0:
  - Computes output features [0:448]
  - For FP32: Uses local weight [448, 896]
  - For IQ4_NL: Decodes rows [0:448] from global weight

Rank 1:
  - Computes output features [448:896]
  - For FP32: Uses local weight [448, 896]
  - For IQ4_NL: Decodes rows [448:896] from global weight
```

### Streaming Dequantization
- IQ4_NL weights decoded in 32-element blocks
- Only requested rows decoded
- No full-weight materialization
- Cache-friendly block processing

## Future Work

1. **Extend to other operators**: Migrate `MPILinearOperator` (v1), `MPILinearBatchOperator`
2. **Remove adaptiveMatMul**: Once all consumers migrated to ITensorGemm
3. **Add more GEMM implementations**: Q6_K, Q8_0, etc.
4. **Optimize row-range decoding**: SIMD optimizations for IQ4_NL decode_block_at

## Conclusion

This refactoring demonstrates the power of the `ITensorGemm` interface:
- **Extensible**: New tensor types just implement the interface
- **Efficient**: Streaming dequantization with row-range selection
- **Clean**: Direct usage without intermediate layers
- **Correct**: All integration tests passing

The row-offset/row-count extension enables proper MPI distribution of quantized matrix operations, which was the missing piece for distributed quantized inference.
