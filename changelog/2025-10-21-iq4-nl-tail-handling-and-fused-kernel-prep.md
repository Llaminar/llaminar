# IQ4_NL Tail Handling Implementation & Fused Kernel Preparation

**Date**: October 21, 2025  
**Status**: ✅ Complete  
**Test Results**: 3/3 tests passing (FullBlockShapes, TailHandling, Determinism)

## Summary

Implemented comprehensive tail handling strategy for `IQ4_NLTensor` to support non-multiple-of-32 column sizes and prepared infrastructure for fused GEMM kernel integration. This work enables efficient streaming decode with proper padding semantics and provides the foundation for fused decode+matmul operations.

## Key Changes

### 1. Padding Semantics & Block Layout

**Previous Behavior:**
- Constructor calculated blocks globally: `(total_elements + 31) / 32`
- Decode paths used per-row block indexing
- Mismatch caused failures for non-multiple-of-32 columns

**New Behavior:**
- **Per-row block counting**: `blocks_per_row = (cols + 31) / 32`
- **Consistent layout**: Each row has `blocks_per_row` contiguous blocks
- **Explicit padding**: `padded_k()` exposes physical dimension (rounded up)
- **Logical dimension**: `logical_k()` returns actual column count

### 2. New API Methods

Added for fused kernel integration:

```cpp
// Dimension accessors
size_t logical_k() const;   // Actual column count
size_t padded_k() const;    // Rounded up to multiple of 32

// Fused kernel helpers
void decode_block_at(size_t row_idx, size_t k_block_offset, float* output) const;
void decode_tile_blocks(size_t row_start, size_t tile_n, size_t k_block_offset, float* output) const;
```

**Use Cases:**
- `decode_block_at()`: Single-block decode for strategy (1) - low memory footprint
- `decode_tile_blocks()`: Multi-row SoA decode for strategy (2) - better amortization (tile_n ≥ 8)

### 3. Decode Path Improvements

**decode_to_fp32:**
- Fixed buffer overflow when cols not multiple of 32
- Now uses temporary buffer + memcpy for tail blocks
- Handles tail elements correctly: only copies `min(32, remaining_cols)` elements

**decode_to_fp32_microkernel:**
- Removed restriction requiring `cols % 32 == 0`
- Added tail block handling with temporary buffer
- Maintains vectorized path for full blocks, scalar for tail

**decodeRow:**
- Simplified to per-row block layout (no more global indexing)
- Consistent with constructor's block counting
- Handles tail blocks explicitly with memcpy

### 4. Test Infrastructure

**Re-enabled Test**: `NonMultipleOfBlockColumnsTailHandling`
- **Removed**: `GTEST_SKIP()` placeholder
- **Test Coverage**: cols ∈ {10, 50, 63, 100, 127, 200}
- **Validation**: 
  - `logical_k()` and `padded_k()` correctness
  - Decode output matches llama.cpp reference
  - Zero mismatches for all shapes

**Test Strategy:**
- Quantize padded rows (zero-filled beyond logical_k)
- Use llama.cpp functions with multiple-of-32 sizes (required by their API)
- Compare decoded output (first `logical_k` elements) against reference

### 5. Constructor Validation

**Enhanced Error Messages:**
```cpp
"IQ4_NL raw data size mismatch: expected X bytes 
(Y rows × Z blocks/row), got W bytes"
```
Now clearly shows per-row block structure in error messages.

## Technical Details

### Block Layout (Per-Row Paradigm)

For a tensor with shape `[rows=5, cols=100]`:
- `blocks_per_row = (100 + 31) / 32 = 4`
- `padded_k = 4 * 32 = 128`
- `logical_k = 100`
- **Memory layout**: `[block0, block1, block2, block3_tail] × 5 rows`

Each block is 18 bytes (16 nibbles + 2-byte FP16 scale).

### Tail Block Handling

**Block 3** (tail block) contains:
- Elements `[96, 99]` (4 valid elements from logical data)
- Elements `[100, 127]` (28 padding elements, decode to arbitrary values)

**Decode behavior**:
- Decode all 32 elements to temporary buffer
- Copy only first 4 elements `[96, 99]` to output
- Ignore remaining 28 elements (padding)

### Microkernel Path Selection

```cpp
if (env.dequant.iq4_microkernel) {
    // Now handles all column sizes (previously required cols % 32 == 0)
    decode_to_fp32_microkernel(...);
}
```

**Vectorization strategy:**
- Full blocks: AVX512 (2-block unroll) or AVX2 (4-block unroll)
- Tail block: Scalar decode + memcpy of valid elements

## Fused Kernel Integration Roadmap

### Phase 1: Preparation (✅ Complete)
- ✅ Expose `logical_k()` and `padded_k()`
- ✅ Implement `decode_block_at()` for single-block decode
- ✅ Implement `decode_tile_blocks()` for multi-row SoA decode
- ✅ Validate tail handling across diverse column sizes

### Phase 2: Fused GEMM Kernel (Next)
Will implement Option A (transitional):

```cpp
bool adaptiveMatMulIQ4NL(
    const float* A,           // [M, K] activations
    int M, int N_local, int K,
    const IQ4_NLTensor& Wq,   // [N, K] quantized weights (row-major)
    float* C                  // [M, N_local] output
);
```

**Algorithm**:
```cpp
for (k0 = 0; k0 < K; k0 += 32) {
    // Strategy (1): Decode per output row
    for (j = 0; j < N_local; ++j) {
        Wq.decode_block_at(j, k0/32, tmp);
        for (m = 0; m < M; ++m) {
            C[m*N + j] += dot(A[m*K + k0:k0+32], tmp);
        }
    }
    
    // OR Strategy (2): Decode tile for amortization (tile_n ≥ 8)
    for (j0 = 0; j0 < N_local; j0 += tile_n) {
        Wq.decode_tile_blocks(j0, tile_n, k0/32, tile_buf);
        for (j = 0; j < tile_n; ++j) {
            for (m = 0; m < M; ++m) {
                C[m*N + (j0+j)] += dot(A[m*K + k0:k0+32], tile_buf[j*32:(j+1)*32]);
            }
        }
    }
}
```

**Tail masking**: Loop over `k0 < padded_k()`, but only accumulate first `min(32, logical_k - k0)` elements in final block.

### Phase 3: MPILinearOperator Integration
Add hook in `MPILinearOperator::execute()`:

```cpp
auto iq4 = std::dynamic_pointer_cast<IQ4_NLTensor>(global_weight);
if (iq4 && debugEnv().quant.iq4nl_fused) {
    fused_gemm_iq4nl(input->data(), seq_len, local_output_size, input_size,
                     *iq4, local_output->data());
    return true;
}
```

## Test Results

### Before Changes
```
[ RUN      ] IQ4NLMicrokernel.NonMultipleOfBlockColumnsTailHandling
GTEST_SKIP() << "Tail-handling disabled: layout mismatch..."
[  SKIPPED ] (1 ms)
```

### After Changes
```
[==========] Running 3 tests from 1 test suite.
[ RUN      ] IQ4NLMicrokernel.FullBlockShapesMultipleSeeds
[       OK ] IQ4NLMicrokernel.FullBlockShapesMultipleSeeds (39 ms)
[ RUN      ] IQ4NLMicrokernel.NonMultipleOfBlockColumnsTailHandling
[       OK ] IQ4NLMicrokernel.NonMultipleOfBlockColumnsTailHandling (0 ms)
[ RUN      ] IQ4NLMicrokernel.DeterminismSingleShapeMultipleRuns
[       OK ] IQ4NLMicrokernel.DeterminismSingleShapeMultipleRuns (0 ms)
[  PASSED  ] 3 tests.
```

**Test Coverage**:
- ✅ Full-block shapes (32, 256, 896, 3584 columns)
- ✅ Tail shapes (10, 50, 63, 100, 127, 200 columns)
- ✅ Multiple random seeds (1, 42, 1337, 2025)
- ✅ Deterministic output (bitwise identical across runs)

## Performance Notes

### Current Microkernel Performance
From previous benchmarking:
- **Single-token (1×896×896)**: ~6× faster than llama.cpp (microkernel)
- **Large prefill (8192×896×3584)**: Competitive with llama.cpp fused vec-dot

### Expected Fused GEMM Performance
- **Decode elimination**: ~30-40% reduction in memory bandwidth
- **Cache locality**: Decode → accumulate in single pass (better L1 hit rate)
- **Estimated speedup**: 1.2-1.6× vs separate decode+BLAS (initial target)

### Optimization Opportunities (Future)
1. Register tiling: Decode multiple A rows × tile_n outputs simultaneously
2. Prefetch next block scales during accumulation
3. Non-temporal stores for large M (streaming write pattern)
4. Fusion with bias addition (epilogue)

## Files Modified

- **`src/tensors/IQ4_NLTensor.h`**: 
  - Added `logical_k()`, `padded_k()` methods
  - Updated constructor to per-row block counting
  - Fixed decode paths for tail handling
  - Added `decode_block_at()`, `decode_tile_blocks()` helpers
  - Enhanced microkernel to handle non-multiple columns

- **`tests/test_iq4_nl_microkernel.cpp`**:
  - Removed `GTEST_SKIP()` from tail test
  - Implemented comprehensive tail validation
  - Added test coverage for 6 non-multiple column sizes

## Next Steps

1. **Implement fused GEMM kernel** (`QuantGemmIQ4NL.{h,cpp}`)
   - Start with strategy (1): decode per output row
   - Add strategy (2): tiled SoA decode for large N
   - Environment flags: `LLAMINAR_IQ4NL_FUSED`, `LLAMINAR_IQ4NL_TILE_N`

2. **Integrate with MPILinearOperator**
   - Add `dynamic_pointer_cast<IQ4_NLTensor>` detection
   - Hook fused kernel when flag enabled
   - Maintain existing paths (Q8_0, FP32, quantized slab)

3. **Validation suite**
   - Parity test: fused vs decode+BLAS
   - Tail correctness: non-multiple K dimensions
   - MPI distribution: ensure local slicing works

4. **Extend to other IQ formats**
   - Apply same tail handling strategy to IQ4_XS, IQ2_XXS, IQ3_XXS
   - Harmonize block layout across all IQ tensors
   - Share microkernel patterns (nibble expansion, LUT lookup)

## Lessons Learned

1. **Per-row vs global indexing**: Consistent block layout is critical for correctness
2. **Buffer overflow prevention**: Always use temporary buffer + memcpy for tail blocks
3. **llama.cpp API constraints**: Quantize functions require multiple-of-block sizes
4. **Test early, test often**: Tail handling bugs manifest as memory corruption
5. **Document padding semantics**: `logical_k` vs `padded_k` distinction crucial for fused kernels

## Related Documents

- `.github/copilot-instructions.md` - General development guidelines
- `docs/WEIGHT_MATRIX_CONVENTIONS.md` - Matrix layout conventions
- `tests/test_iq4_vs_llamacpp.cpp` - Original IQ4_NL parity tests
- `tests/test_iq4_nl_microkernel.cpp` - Microkernel regression suite

---

**Author**: David Sanftenberg (with Copilot assistance)  
**Branch**: `feature/quantized-tensors`  
**Commit**: Ready for integration
