# Stage 2 Refactoring Complete: distributeWeightsByHead()

**Date**: 2025-10-13  
**Author**: David Sanftenberg  
**Status**: ✅ Complete - All parity tests passed

## Summary

Successfully extracted STEP 2 (Weight distribution by head dimension) from `MPIAttentionKernel::execute()` into a dedicated helper function `distributeWeightsByHead()`. This is the second of 9 planned refactoring stages.

## Changes Made

### Files Modified

#### `src/kernels/MPIAttentionKernel.h`
- **Added** `WeightSlices` struct (lines ~58-82)
  - Contains local weight slices (local_wq, local_wk, local_wv, local_wo)
  - Contains local bias slices (local_bq, local_bk, local_bv, may be nullptr)
  - Metadata: copied_global_weights flag

- **Added** `distributeWeightsByHead()` method declaration (line ~282)
  - Signature: `WeightSlices distributeWeightsByHead(const InputSetupResult &setup)`
  - Private helper method taking InputSetupResult from Stage 1

#### `src/kernels/MPIAttentionKernel.cpp`
- **Added** `distributeWeightsByHead()` implementation (lines 685-979, 295 lines)
  - Handles three weight distribution cases:
    1. **Pre-sharded weights**: Use directly (zero-copy)
    2. **Single rank**: Use global weights directly
    3. **Multi-rank with global weights**: Slice and copy by head dimension
  - Row-wise slicing for Q/K/V weights: [local_head_dim, d_model]
  - Column-wise slicing for output weight: [d_model, local_head_dim]
  - Bias handling: use if size > 1, else nullptr
  - Debug logging for bias flow, pointer verification, weight statistics
  - Weight tracing diagnostics (if LLAMINAR_ATTN_TRACE_WEIGHT_SLICING enabled)
  - File dumping for detailed comparison (if verbose mode)

- **Modified** `execute()` method (lines 1058-1076)
  - Replaced ~282 lines of weight distribution code with ~18 lines
  - Calls distributeWeightsByHead(setup)
  - Extracts 8 local variables from WeightSlices struct
  - Clean separation between STEP 2 (refactored) and STEP 3 (untouched)

### Code Metrics

- **Stage 1 result**: execute() reduced to ~2,050 lines
- **Stage 2 reduction**: ~2,050 → ~1,786 lines (reduced by ~264 lines, additional 13%)
- **Cumulative reduction**: Original 2,287 lines → 1,786 lines (**-501 lines, -22%**)
- **Helper method size**: 295 lines
- **Net change Stage 2**: +31 lines overall (struct + method overhead)

## Validation Results

All three parity tests passed with identical results to baseline:

### Test 1: OpenBLAS Prefill vs PyTorch
```
✅ Passed:  387/387
✗ Failed:  0/387
? Missing: 0/387
Duration: 90.2s
```

### Test 2: COSMA Prefill vs PyTorch
```
✅ Passed:  387/387
✗ Failed:  0/387
? Missing: 0/387
Duration: 106.5s
```

### Test 3: True Incremental Decode vs PyTorch
```
✅ Stages passed:  1170 (585 per rank × 2)
✗ Stages failed:  0
Tokens: 3/3 passed
Duration: 33.3s
```

**Total validation**: 1,944 stage comparisons, 100% pass rate

## Technical Details

### Three Weight Distribution Cases

**Case 1: Pre-sharded weights** (most common in production)
- Weights already sliced during model loading (QwenPipeline)
- Zero-copy: directly use input weight pointers
- Biases also pre-sliced
- No runtime overhead

**Case 2: Single rank** (testing/debugging)
- Use full global weights directly
- No slicing needed
- Biases used if size > 1

**Case 3: Multi-rank with global weights** (legacy/fallback)
- Runtime slicing required
- Row-wise for Q/K/V: memcpy [local_head_dim, d_model]
- Column-wise for output: per-row memcpy [d_model, local_head_dim]
- Sets copied_global_weights flag for diagnostics

### Struct Design Rationale

The `WeightSlices` struct was designed to:
1. **Encapsulate distributed weights** - Clear ownership of this rank's weight slices
2. **Support three distribution modes** - Unified return type regardless of case
3. **Track copy status** - copied_global_weights flag for diagnostics
4. **Handle optional biases** - nullptr if not present or size <= 1
5. **Enable Stage 3** - QKV projections need these exact slices

### Debug Instrumentation Preserved

All debug logging from original code preserved:
- **BIAS_FLOW**: Tracks bias presence/size through all three cases
- **PTR_DEBUG**: Verifies pointer assignment in pre-sharded path
- **WQ_VERIFY**: Statistics and file dumps for detailed comparison
- **ATTN_WEIGHT_TRACE**: Per-head weight slice verification with global comparison

Environment flags controlling instrumentation:
- `LLAMINAR_ATTN_VERBOSE`: Enables all debug logging (layer 0 only)
- `LLAMINAR_ATTN_TRACE_WEIGHT_SLICING`: Enables detailed weight tracing

### Variable Extraction Pattern (Stage 2)

```cpp
auto weights = distributeWeightsByHead(setup);

// Extract locals for use in STEP 3-8 (not yet refactored)
auto local_wq = weights.local_wq;
auto local_wk = weights.local_wk;
auto local_wv = weights.local_wv;
auto local_wo = weights.local_wo;
auto local_bq = weights.local_bq;
auto local_bk = weights.local_bk;
auto local_bv = weights.local_bv;
bool copied_global_weights = weights.copied_global_weights;
```

This pattern maintains backward compatibility with STEP 3-8 (not yet refactored).

## Compilation Notes

Build completed successfully with same narrowing warnings as Stage 1 (non-critical, to be addressed in Stage 9 cleanup).

## Next Steps

**Stage 3**: Extract STEP 3 (QKV linear projections) into `computeQKVProjections()` helper method.

**Expected metrics**:
- Lines to extract: ~242 lines (including bias addition and validation)
- Return type: QKVProjectionResult struct (local_q, local_k, local_v tensors)
- Testing protocol: Same 3 parity tests (387, 387, 585)

## Lessons Learned

1. **Zero-copy optimization matters** - Pre-sharded path avoids all runtime slicing overhead
2. **Three-case pattern is clean** - if/else-if/else cleanly separates sharded/single/multi-rank
3. **Debug preservation critical** - Extensive logging helped debug original MPI issues, must preserve
4. **Struct composition works well** - InputSetupResult → WeightSlices flows naturally
5. **Bias handling is subtle** - size > 1 check prevents nullptr dereference, placeholder tensors skipped

## Performance Impact

**Expected**: None - Pure refactoring with no algorithmic changes.

**Measured**:
- OpenBLAS prefill: 90.2s (baseline: ~88-90s)
- COSMA prefill: 106.5s (baseline: ~105-107s)  
- Incremental decode: 33.3s (baseline: ~33s)

All within normal variance. No performance regression detected.

## Architecture Notes

### Pre-sharding Optimization

The most common path (pre-sharded weights) is highly optimized:
- **Zero allocations**: No new tensors created
- **Zero copies**: Direct pointer aliasing
- **Zero overhead**: Single pointer assignment per weight

This optimization was critical for achieving 2 ops/layer in production decode (Phase 4/5 MPI work).

### Memory Layout

Weight slicing respects GGUF canonical format:
- **Q/K/V weights**: [out_features, in_features] = [head_dim, d_model]
  - Row-major layout
  - Row-wise slicing: contiguous memory
  - Simple memcpy per weight matrix
  
- **Output weight**: [in_features, out_features] = [d_model, head_dim]
  - Row-major layout
  - Column-wise slicing: per-row memcpy required
  - Stride calculation: row * (n_head * head_dim) + head_offset * head_dim

### Bias Distribution Strategy

Biases are **pre-sliced during weight loading** (QwenPipeline), not in the attention kernel:
- Avoids hot-path slicing overhead
- Simplifies kernel logic: just use or skip
- Size > 1 check distinguishes real bias from placeholder

---

**Refactoring Progress**: Stages 1-2/9 complete (22% done)  
**Lines refactored**: 614 / ~2,000 (31% done)  
**Execute() reduction**: 501 lines / ~1,500 target (33% done)  
**Cumulative time saved**: ~519 lines of complex logic now self-contained
