# Q8_1 Block-Aligned K-Slicing Implementation (Phase 6.4)

**Date**: 2025-12-12

## Summary

Implemented Q8_1 block-aligned k-dimension slicing for tensor-parallel GEMM operations. This optimization allows k-sliced GEMM to operate directly on Q8_1 quantized activations without the overhead of FP32 dequantization when the slice boundaries align with Q8_1 block boundaries (multiples of 32 elements).

## Problem Statement

In tensor-parallel GEMM with k-dimension sharding, input activations must be sliced along the k-axis before being passed to each rank's local GEMM operation. Previously, Q8_1 activations were always dequantized to FP32 before slicing, then the sliced FP32 data was used for GEMM. This defeated the bandwidth optimization of Q8_1 quantized activations.

The Q8_1 format uses 32-element blocks (36 bytes each: 2-byte scale, 2-byte sum, 32 int8 values). When k_start and k_size are multiples of 32, we can slice directly at block boundaries without breaking the quantization structure.

## Implementation

### New API

**`Q8_1Tensor::is_k_aligned(size_t k_start)`** (static constexpr)
- Checks if a k-dimension offset is aligned to block boundaries (multiple of 32)
- Zero runtime cost - evaluated at compile time when possible

**`Q8_1Tensor::slice_k_blocks(size_t k_start, size_t k_size)`**
- Creates a new Q8_1Tensor containing only the specified k-dimension slice
- Preserves Q8_1 block structure (copies blocks, not individual elements)
- Returns `nullptr` if k_start or k_size are not 32-aligned
- Returns `nullptr` if slice exceeds tensor bounds

### Files Modified

1. **`src/v2/tensors/Tensors.h`**
   - Added `is_k_aligned()` static constexpr method
   - Added `slice_k_blocks()` method declaration

2. **`src/v2/tensors/Q8_1Tensor.cpp`**
   - Implemented `slice_k_blocks()` (~70 lines)
   - Validates alignment and bounds
   - Copies blocks row-by-row from source to result tensor

3. **`src/v2/pipelines/PipelineBase.cpp`**
   - Updated `project_row_parallel()` to use `slice_k_blocks()` when input is Q8_1 and dimensions are aligned
   - Falls back to FP32 dequantization path if not aligned

### New Test File

**`tests/v2/unit/tensors/Test__Q8_1_BlockAlignedSlicing.cpp`** (537 lines, 13 tests)
- `IsKAligned` - Tests static alignment check function
- `RejectsUnalignedKStart` - Validates rejection of unaligned k_start
- `RejectsUnalignedKSize` - Validates rejection of unaligned k_size  
- `RejectsOutOfBounds` - Validates bounds checking
- `SliceFirstHalf` - Tests slicing first half of columns
- `SliceSecondHalf` - Tests slicing second half of columns
- `SliceMiddleBlocks` - Tests slicing middle blocks
- `SliceSingleBlock` - Tests extracting a single block
- `SliceEntireTensor` - Tests full tensor copy via slicing
- `TensorParallelDimensions` - Tests 2-way tensor parallel split (d_model=896)
- `FourWayTensorParallel` - Tests 4-way tensor parallel split (d_model=4096)
- `BlockIntegrity` - Validates Q8_1 block structure after slicing
- `LargeTensorSlicing` - Performance test with large tensors (512x4096)

## Test Results

```
[==========] Running 13 tests from 1 test suite.
[  PASSED  ] 13 tests.
```

All 139 V2 unit tests pass with no regressions.

## Usage Example

```cpp
// Check if dimensions are aligned for native Q8_1 k-slicing
const int k_local = k / world_size;
const int k_start = k_local * rank;

if (Q8_1Tensor::is_k_aligned(k_start) && Q8_1Tensor::is_k_aligned(k_local)) {
    // Native Q8_1 k-slicing - no FP32 dequantization needed!
    auto input_sliced = input_q8_1->slice_k_blocks(k_start, k_local);
    if (input_sliced) {
        // Use Q8_1 slice directly for GEMM
        gemm_op->execute(input_sliced.get(), weight, output, m, n, k_local);
    }
}
```

## Alignment Requirements

For Qwen2.5 models with tensor parallelism:
- d_model=896 with 2 ranks: k_local=448 ✓ (448 % 32 = 0)
- d_model=4096 with 4 ranks: k_local=1024 ✓ (1024 % 32 = 0)
- d_model=4096 with 8 ranks: k_local=512 ✓ (512 % 32 = 0)

Most practical tensor-parallel configurations will be aligned because hidden dimensions in LLMs are typically multiples of 128 or 256.

## Performance Impact

- **Aligned case**: Avoids FP32 dequantization (saves memory bandwidth and allocation)
- **Unaligned case**: Falls back to existing FP32 path (no regression)
- **Memory**: Sliced tensor is a copy (not a view) since column slicing requires block reorganization

## Future Work

- Add zero-copy view support for row slicing + column slicing combined
- Extend to other quantized formats (Q4_0, Q6_K, etc.) if needed
