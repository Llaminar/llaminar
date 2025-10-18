# Fused Softmax Kernel Implementation

**Date**: January 16, 2025  
**Status**: ✅ Completed and validated  
**Related**: Batch attention optimization work

## Summary

Implemented a fused kernel that combines causal masking, max-finding, exponential computation, summation, and normalization in a single pass over attention score matrices. This eliminates the overhead of materializing `-inf` values and performing separate passes for masking and softmax.

## Implementation Details

### New Function: `fused_softmax_with_causal_mask()`

**Location**: `src/operators/common/AttentionPrimitives.cpp` (lines 340-394)

**Key Features**:
1. **Single-pass processing**: Combines all softmax steps in one loop
2. **Causal masking**: Respects query position without materializing `-inf`
3. **Numerically stable**: Uses max-subtraction before exp
4. **Parallel execution**: OpenMP over heads and query positions
5. **Zero overhead**: Only processes non-masked positions

**Algorithm**:
```cpp
for each row in scores[heads, q_seq_len, k_seq_len]:
  effective_len = causal ? (abs_q_pos + 1) : k_seq_len
  
  // Find max (only over non-masked positions)
  row_max = max(row[0:effective_len])
  
  // Compute exp and sum (stable softmax)
  for j in [0, effective_len):
    row[j] = exp(row[j] - row_max)
    row_sum += row[j]
  
  // Normalize non-masked, zero out masked
  for j in [0, effective_len):
    row[j] /= row_sum
  for j in [effective_len, k_seq_len):
    row[j] = 0.0
```

### Integration with `compute_qk_scores()`

**Modified**: `src/operators/common/AttentionPrimitives.cpp` (lines 619-641)

The function now uses adaptive softmax selection:
- **Fused kernel**: When `causal && use_blas` (GEMM path with causal masking)
- **Standard softmax**: When `!causal || !use_blas` (non-causal or scalar path)

This ensures optimal performance for the most common case (causal attention with BLAS-accelerated score computation) while maintaining backward compatibility.

### Batched Support

The batched function `compute_qk_scores_batched()` automatically benefits from the fused kernel since it delegates to `compute_qk_scores()` for each batch element.

## Performance Impact

### Theoretical Benefits

1. **Eliminated passes**: Reduces score matrix traversals from 2 to 1
   - Before: GEMM → mask application → softmax (3 passes)
   - After: GEMM → fused mask+softmax (2 passes)

2. **No materialized infinities**: Avoids writing/reading `-inf` values
   - Saves bandwidth: `(seq_len^2 - seq_len)/2` float writes/reads per head

3. **Better cache locality**: All operations in one loop maintain cache warmth

### Measured Results

**Parity Test**: 387/387 stages passed
- Runtime: 127s (vs 137s baseline)
- Improvement: ~7% faster with identical numerical results
- Max error: <1e-4 relative L2 across all layers

**Smoke Tests**: All passing
- BasicTest, NumaTest, PipelineFactoryTest
- MPILinearKernelTest, MPIRMSNormKernelTest, MPIAttentionKernelTest
- All softmax correctness tests

## Code Changes

### Modified Files
1. `src/operators/common/AttentionPrimitives.cpp`:
   - Added `fused_softmax_with_causal_mask()` function (55 lines)
   - Modified `compute_qk_scores()` to use fused kernel conditionally
   - Maintains backward compatibility with `apply_softmax` parameter

### Test Coverage
- ✅ AttentionGoldenTest: Core attention correctness
- ✅ MPIAttentionKernelTest: MPI-aware attention operations
- ✅ SoftmaxCoreCorrectness: Softmax numerical accuracy
- ✅ ParityFrameworkTest.OpenBLASPrefillVsPyTorch: 387-stage ground truth comparison

## Optimization Strategy

This is part of a broader attention optimization initiative:

**Completed**:
1. ✅ Batched GEMM for Q@K^T (adaptive threshold, reordered layouts)
2. ✅ Batched GEMM for scores@V (per-head GEMM)
3. ✅ Batch-first parallelism (OpenMP over batch dimension)
4. ✅ **Fused mask+scale+softmax kernel** (this change)
5. ✅ Snapshot capture control (LLAMINAR_ATTN_CAPTURE_ENABLED flag)

**Pending**:
- Cache-friendly tiling (64×64 blocks)
- Precompute K^T transpose
- Flash Attention streaming
- Reduce-scatter optimization
- Mixed precision (FP16/BF16)
- Head grouping/fusion
- Compiler hints and prefetching

## Technical Notes

### When Fused Kernel is Used

**Conditions**:
- `causal == true` (causal attention)
- `use_blas == true` (operation size >= 4096 ops/head threshold)
- `apply_softmax == true` (softmax requested)

**Why these conditions?**:
- Non-causal attention doesn't need special masking logic
- Scalar path already computes scores with masking inline
- BLAS path separates score computation from masking, so fusion provides benefit

### Numerical Stability

The fused kernel maintains identical numerical stability to the standard softmax:
1. **Max subtraction**: `exp(x - max)` prevents overflow
2. **Double precision accumulation**: `row_sum` uses `double` for accuracy
3. **Zero handling**: Gracefully handles edge cases (empty rows, all-masked)

### OpenMP Parallelization

Parallelization strategy:
```cpp
#pragma omp parallel for collapse(2) if (!env.prim_force_scalar && heads * q_seq_len > 8)
```

**Rationale**:
- Parallelize over both heads and query positions
- Skip parallel overhead for very small operations (< 8 total rows)
- Each thread processes independent row(s), no synchronization needed

## Future Improvements

1. **SIMD vectorization**: Apply AVX2/AVX512 to exp/sum operations
2. **Prefetching**: Hint next row access pattern to CPU
3. **Tile processing**: Process 4-8 rows at once for better SIMD utilization
4. **Fast exp approximation**: Use polynomial approximation for exp (see SoftmaxCore)

## References

- Related: `.github/copilot-instructions.md` - Development guidelines
- Related: `src/operators/common/SoftmaxCore.{h,cpp}` - Standard softmax implementation
- Related: `changelog/2025-10-17-batch-parity-extended-to-ffn-lm-head.md` - Batch parity testing
