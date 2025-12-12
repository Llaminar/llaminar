# MPI Precision-Aware Allreduce Integration Tests

**Date**: 2025-12-12

## Summary

Added comprehensive MPI integration tests for precision-aware allreduce operations across FP32, FP16, BF16, and Q8_1 formats. Also cleaned up duplicate FP16/BF16 conversion functions in test files.

## Changes

### 1. Removed Duplicate Conversion Functions

**Files Modified:**
- `tests/v2/unit/Test__CPUSoftmaxKernelT.cpp`
- `tests/v2/unit/Test__CPUSwiGLUKernelT.cpp`

Both test files had local implementations of FP16/BF16 conversion functions that were duplicates of the robust implementations in `FP16Utils.h` and `SIMDHelpers.h`. The local versions used simple truncation while the shared versions use proper round-to-nearest-even with denormal handling.

**Before:**
```cpp
// Local duplicate implementations (simple truncation)
inline float bf16_to_fp32(uint16_t bf16) { ... }
inline uint16_t fp32_to_bf16(float val) { ... }
inline float fp16_to_fp32(uint16_t h) { ... }
inline uint16_t fp32_to_fp16(float val) { ... }
```

**After:**
```cpp
#include "tensors/FP16Utils.h"
#include "tensors/SIMDHelpers.h"

using simd::fp32_to_bf16;
using simd::bf16_to_fp32;
using fp16_to_fp32;   // From FP16Utils.h (global namespace)
using fp32_to_fp16;   // From FP16Utils.h (global namespace)
```

### 2. New MPI Integration Test

**File Created:**
- `tests/v2/integration/Test__MPI_PrecisionAwareAllreduce.cpp`

**CMake Target:**
- `v2_integration_mpi_precision_allreduce` (requires 2 MPI ranks)

**Tests Added (6 total):**

| Test Name | Description |
|-----------|-------------|
| `FP32_Allreduce_TwoRanks` | Baseline FP32 MPI_Allreduce |
| `FP16_Allreduce_TwoRanks` | FP16 native reduction via MPI_BYTE + SIMD sum |
| `BF16_Allreduce_TwoRanks` | BF16 native reduction via MPI_BYTE + SIMD sum |
| `Q8_1_Allreduce_TwoRanks` | Q8_1 block-native reduction (32 elements) |
| `Q8_1_Allreduce_LargerData` | Q8_1 with 8 blocks (256 elements) |
| `Bandwidth_Comparison` | Verification of bandwidth savings |

### 3. Bandwidth Comparison Results

For 1024 elements:
- **FP32**: 4096 bytes (1.0x baseline)
- **FP16**: 2048 bytes (2x better)
- **BF16**: 2048 bytes (2x better)
- **Q8_1**: 1152 bytes (3.56x better)

## Test Verification

```bash
# Build (Release mode)
cmake --build build_v2_release --target v2_integration_mpi_precision_allreduce --parallel

# Run
mpirun -np 2 --bind-to socket --map-by socket \
  ./build_v2_release/tests/v2/v2_integration_mpi_precision_allreduce

# Results: All 6 tests pass
```

## Phase 6 Status Update

| Phase | Description | Status |
|-------|-------------|--------|
| 6.1 | Q8_1-Native MPI Allreduce | ✅ Complete |
| 6.2 | FP16/BF16-Native MPI Allreduce | ✅ Complete |
| 6.3 | Unified Interface | ⏸️ Deferred |
| 6.4 | Q8_1 Block-Aligned Slicing | ⬜ Not started |

## Files Modified

1. `tests/v2/unit/Test__CPUSoftmaxKernelT.cpp` - Removed duplicates
2. `tests/v2/unit/Test__CPUSwiGLUKernelT.cpp` - Removed duplicates
3. `tests/v2/integration/Test__MPI_PrecisionAwareAllreduce.cpp` - **New file**
4. `tests/v2/CMakeLists.txt` - Added new test target

## Related Files

- `src/v2/utils/MPIContext.h` - Contains allreduce_q8_1_inplace(), allreduce_fp16_inplace(), allreduce_bf16_inplace()
- `src/v2/tensors/SIMDHelpers.h` - Contains simd::fp16_sum_n, simd::bf16_sum_n, simd::q8_1_sum_n
- `src/v2/tensors/FP16Utils.h` - Contains fp16_to_fp32, fp32_to_fp16

## Next Steps

Phase 6.4 (Q8_1 Block-Aligned Slicing) involves:
1. Add `Q8_1Tensor::slice_blocks()` for k-sliced GEMM
2. Update k-sliced path in `project_row_parallel()` to avoid FP32 dequantization when k_start aligns with 32-element boundaries
3. Fallback to FP32 path for unaligned slices
