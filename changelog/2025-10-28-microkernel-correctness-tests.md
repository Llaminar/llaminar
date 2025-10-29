# Micro-Kernel Mathematical Correctness Tests - October 28, 2025

## Summary

Successfully created and validated comprehensive mathematical correctness tests for the V2 micro-kernel GEMM implementation. All 12 tests now pass with excellent numerical accuracy after fixing a critical memory layout bug in the test helper.

## Test Suite Overview

**Location**: `tests/v2/unit/Test__MicroKernelCorrectness.cpp` (661 lines)

**Coverage**: 12 comprehensive test cases validating:
- Basic matrix operations (identity, zero, diagonal)
- Random matrices (small, medium, large)
- Ground truth comparison (OpenBLAS reference)
- Edge cases (1×1, single row, non-tile-multiples)
- Variant agreement (all 1225+ kernel variants produce identical results)
- Numerical stability (large/small value mixing)

## Test Results

**All 12 Tests Pass** ✅

Key numerical accuracy metrics:

1. **IdentityMatrix**: `rel_error=0, max_diff=0` - **Perfect match**
2. **VsOpenBLAS**: `rel_error=4.04e-07, max_diff=2.86e-05` - Well within tolerances

### Full Test List

```
PASS: MicroKernelCorrectness.IdentityMatrix
PASS: MicroKernelCorrectness.ZeroMatrix
PASS: MicroKernelCorrectness.DiagonalMatrix
PASS: MicroKernelCorrectness.RandomMatricesSmall
PASS: MicroKernelCorrectness.RandomMatricesMedium
PASS: MicroKernelCorrectness.VsOpenBLAS
PASS: MicroKernelCorrectness.SingleRow
PASS: MicroKernelCorrectness.TinyMatrix
PASS: MicroKernelCorrectness.NonMultipleOfTileSize
PASS: MicroKernelCorrectness.AllVariantsAgree
PASS: MicroKernelCorrectness.LargeMatrix
PASS: MicroKernelCorrectness.NumericalStability
```

## Critical Bug Fix: FP32BlockDecoder Memory Layout

### Problem

The test helper `FP32BlockDecoder` had a fundamental memory layout bug that caused **catastrophic test failures** (9/12 tests failing with rel_error ~1.7, max_diff ~335).

**Root Cause**: Weight matrices in GEMM are conceptually transposed. The decoder represents B^T (n×k), but the test matrix B is stored as k×n row-major in memory. The decoder must extract **columns** from the k×n matrix, not rows.

### The Bug

**Before (BROKEN)**:
```cpp
class FP32BlockDecoder : public IBlockDecoder {
public:
    FP32BlockDecoder(const float* data, int rows, int cols)
        : data_(data), rows_(rows), cols_(cols) {}
    
    void decode_block_at(size_t row_idx, size_t k_block_offset, float* output) const override {
        // BUG: Reads rows from the matrix
        const float* row_start = data_ + row_idx * cols_;
        size_t k_start = k_block_offset * 32;
        size_t k_end = std::min(k_start + 32, static_cast<size_t>(cols_));
        
        for (size_t k = k_start; k < k_end; ++k) {
            output[k - k_start] = row_start[k];  // Wrong memory access!
        }
    }
    
    size_t decoder_rows() const override { return rows_; }
    size_t decoder_cols() const override { return cols_; }
};

// Test usage:
std::vector<float> B(k * n);  // k rows, n columns, row-major
fill_random(B.data(), k, n, -1.0f, 1.0f);
FP32BlockDecoder decoder(B.data(), n, k);  // Wrong: sets rows_=n, cols_=k
```

**Issue**: 
- B is k×n stored row-major: each row has n elements (stride = n)
- Constructor sets `rows_=n, cols_=k`
- `decode_block_at` reads: `data_ + row_idx * cols_` = `data_ + row_idx * k`
- **Should be**: `data_ + row_idx * n` (wrong stride!)
- Reads completely wrong memory locations → garbage output

### The Fix

**After (CORRECT)**:
```cpp
class FP32BlockDecoder : public IBlockDecoder {
public:
    // Parameters renamed to clarify: k_rows×n_cols matrix, row-major
    FP32BlockDecoder(const float* data, int k_rows, int n_cols)
        : data_(data), k_rows_(k_rows), n_cols_(n_cols) {}
    
    void decode_block_at(size_t row_idx, size_t k_block_offset, float* output) const override {
        // FIXED: Extract column row_idx from the k×n matrix
        // row_idx is the output feature index (column in the k×n matrix)
        // k_block_offset is which 32-element block along k dimension
        size_t k_start = k_block_offset * 32;
        size_t k_end = std::min(k_start + 32, static_cast<size_t>(k_rows_));
        
        // Elements are at: data[i * n_cols + row_idx] for i in [k_start, k_end)
        for (size_t i = k_start; i < k_end; ++i) {
            output[i - k_start] = data_[i * n_cols_ + row_idx];  // Correct!
        }
        
        // Pad remaining with zeros
        for (size_t i = k_end - k_start; i < 32; ++i) {
            output[i] = 0.0f;
        }
    }
    
    size_t decoder_rows() const override { return n_cols_; }  // n output features
    size_t decoder_cols() const override { return k_rows_; }  // k input features each
};

// Test usage (FIXED):
std::vector<float> B(k * n);  // k rows, n columns, row-major
fill_random(B.data(), k, n, -1.0f, 1.0f);
FP32BlockDecoder decoder(B.data(), k, n);  // Correct: k_rows=k, n_cols=n
```

**Key Changes**:
1. Constructor parameters renamed: `(k_rows, n_cols)` instead of `(rows, cols)` for clarity
2. `decode_block_at` now extracts **columns** using correct stride: `data_[i * n_cols_ + row_idx]`
3. Constructor calls updated: `FP32BlockDecoder(B.data(), k, n)` instead of `(B.data(), n, k)`
4. Added extensive documentation explaining the memory layout

## Understanding the Memory Layout

### GEMM Operation
```
C = A × B
A: m × k (activations)
B: k × n (weights, row-major storage)
C: m × n (output)
```

### Weight Matrix B
- **Storage**: k×n row-major (k rows, n columns)
- **Memory layout**: `[B[0,0], B[0,1], ..., B[0,n-1], B[1,0], ..., B[k-1,n-1]]`
- **Element B\[i,j\]** at index: `i * n + j`

### Decoder Perspective
The decoder represents **B^T** (transposed view):
- **Logical shape**: n×k (n output features, k input features each)
- `decoder_rows()` = n (number of output features)
- `decoder_cols()` = k (number of input features per output)
- **To access output feature `row_idx`**: Extract column `row_idx` from original k×n matrix
  - Column elements: `B[0, row_idx], B[1, row_idx], ..., B[k-1, row_idx]`
  - Memory positions: `row_idx, n + row_idx, 2n + row_idx, ..., (k-1)n + row_idx`
  - Formula: `data[i * n + row_idx]` for `i in [0, k)`

## Test Accuracy Thresholds

Tests use different tolerances based on operation complexity:

### Strict Tolerances (Simple Operations)
- **IdentityMatrix**: `rel_error < 1e-5, max_diff < 1e-4`
- **ZeroMatrix**: `rel_error < 1e-6, max_diff < 1e-6`
- **DiagonalMatrix**: `rel_error < 1e-5, max_diff < 1e-4`

### Relaxed Tolerances (Complex Operations)
- **VsOpenBLAS**: `rel_error < 1e-4, max_diff < 1e-3`
- **RandomMatrices**: `rel_error < 1e-4, max_diff < 1e-3`
- **LargeMatrix**: `rel_error < 1e-4, max_diff < 1e-3`

### Special Tests
- **AllVariantsAgree**: Ensures all 1225+ kernel variants produce identical results
- **NumericalStability**: Tests with large/small value mixing (1e6 / 1e-6)

## Performance Notes

Test execution includes auto-tuning for each unique matrix shape:
- Total test time: ~18 seconds (includes auto-tuning overhead)
- Auto-tuning benchmarks 10 top candidates per shape
- Achieved GFLOPS during tests:
  - 64×896×896: ~140 GFLOPS
  - 256×896×896: ~150 GFLOPS
  - 1024×896×896: ~197 GFLOPS

## Files Modified

1. **tests/v2/unit/Test__MicroKernelCorrectness.cpp** (CREATED, 661 lines)
   - 12 comprehensive correctness tests
   - Fixed FP32BlockDecoder memory layout bug
   - Added diagnostic output for key tests

2. **tests/v2/CMakeLists.txt** (MODIFIED)
   - Added test executable with OpenBLAS linkage
   - Configured test labels: `V2;Unit;Kernels;MicroKernel;GEMM;Correctness;NumericalAccuracy;Parity`

## Running the Tests

```bash
# Build tests
cmake --build build_v2_release --target v2_test_microkernel_correctness -j$(nproc)

# Run all tests
cd build_v2_release && ./tests/v2/v2_test_microkernel_correctness

# Run specific tests
./tests/v2/v2_test_microkernel_correctness --gtest_filter='*IdentityMatrix:*VsOpenBLAS'

# Via CTest
cd build_v2_release && ctest -R V2_Unit_MicroKernelCorrectness --output-on-failure
```

## Next Steps

1. **Performance validation**: Compare micro-kernel performance against OpenBLAS across full range of shapes
2. **Quantized testing**: Extend tests to IQ4_NL and other quantized formats
3. **Multi-threading**: Validate correctness with multiple OpenMP threads
4. **Edge cases**: Add tests for extremely large matrices (multi-GB)

## Conclusion

The micro-kernel is now **mathematically validated** with:
- ✅ 12/12 tests passing
- ✅ Perfect accuracy on identity matrices (rel_error=0)
- ✅ Excellent agreement with OpenBLAS (rel_error=4e-07)
- ✅ All 1225+ kernel variants produce identical results
- ✅ Numerical stability confirmed

This provides a solid foundation for:
- Production deployment confidence
- Performance optimization (knowing correctness is preserved)
- Future kernel development (regression testing)
