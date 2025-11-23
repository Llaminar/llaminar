# E2E Test Investigation Summary - 2025-01-31

## Context
Investigation of E2E test failures in Qwen2E2ECorrectness suite after Phase 1 fusion framework implementation (FusedRMSNormQuantize).

## Root Cause Identified
**INT8Tensor Dequantization Row Count Mismatch**

### Problem
- INT8 activation buffers allocated for `effective_max` rows (maximum batch capacity)
- Only `effective_seq_len` rows contain valid data with corresponding per-row scales
- Dequantization code was iterating over full buffer capacity, accessing garbage scales

### Impact
- **Before Fix**: Exploding activations with max divergence ~7000
- **After Fix**: Normal quantization error with max divergence ~10

## Fixes Applied

### 1. Pipeline Snapshot Capture (Qwen2Pipeline.cpp)
**Lines 527-555 (Attention Block)** and **Lines 864-892 (FFN Block)**

Changed dequantization from full buffer to actual data range:

```cpp
// OLD (WRONG): Dequantize entire buffer
for (int i = 0; i < shape_[0] * d_model_; ++i) { ... }

// NEW (CORRECT): Only dequantize valid rows
for (int r = 0; r < effective_seq_len; ++r) {
    const float row_scale = buffers.normalized_scales[r];
    for (int c = 0; c < d_model_; ++c) {
        const size_t idx = r * d_model_ + c;
        normalized_hidden->mutable_data()[idx] = 
            static_cast<float>(int8_data[idx]) * row_scale;
    }
}
```

### 2. INT8Tensor::data() Implementation (INT8Tensor.cpp)
**Lines 145-180**

Updated to only dequantize rows for which we have scales:

```cpp
// OLD (WRONG): Try to dequantize all rows, fail if insufficient scales
const size_t rows = shape_[0];
if (row_scales_cache_.size() < rows) {
    LOG_ERROR("Insufficient row scales");
    // fallback...
}

// NEW (CORRECT): Dequantize only rows with scales
const size_t rows_to_dequantize = row_scales_cache_.size();
for (size_t r = 0; r < rows_to_dequantize; ++r) { ... }
// Zero out remaining buffer space
```

## Test Results

### Initial Discovery (Before Fix)
```
4/7 tests FAILING with catastrophic divergences:
- layer0_FFN_SWIGLU: max=6483-7047 (explosion!)
- layer9_ATTENTION_RESIDUAL: max_diff=1692
- Pattern: Only batched multi-sequence tests fail
```

### After Pipeline Fix
```
Divergences reduced dramatically (700× improvement):
- layer0_FFN_SWIGLU: max_diff=9.52 (was ~7000)
- layer0_ATTENTION_RESIDUAL: max_diff=0.101 (was ~1700)
- No more "Exploding activations" warnings
```

### Current Status (After INT8Tensor::data() Fix)
```
Test Results: INCONSISTENT
- Initial run: 6/7 passing (BatchScaling only failure with max_diff=0.0011)
- After rebuild: 4/7 failing (same tests as before fix)
- Divergences: max_diff=22, rel_l2=1.39 (worse than after pipeline fix)
```

## Analysis

### What Worked
1. ✅ Eliminated exploding activations (700× improvement)
2. ✅ Fixed buffer overflow in snapshot capture
3. ✅ Fixed INT8Tensor::data() to respect scale count
4. ✅ Removed "Insufficient row scales" error messages

### Remaining Issues
1. ❌ Tests show non-deterministic behavior (pass → fail after rebuild)
2. ❌ Divergences larger than expected (~22 vs ~10)
3. ❌ 4/7 tests still failing in latest run

### Hypothesis
The non-deterministic behavior and larger divergences suggest:
1. **Possible cause**: INT8Tensor::data() zero-fill of unused buffer space may be affecting downstream operations
2. **Alternative**: There may be another code path calling data() that we haven't addressed
3. **Rebuild difference**: E2ERelease build configuration might be caching or using different codepaths

## Test Tolerance Analysis

Test uses **dual criteria** for pass:
```cpp
result.passed = (result.max_abs_diff <= tolerance &&
                 result.rel_l2_norm <= 0.01f);
```

Current failures show:
- `max_abs_diff=22` (way above tolerance=1e-3 or 2e-3)
- `rel_l2_norm=1.39` (way above 0.01 threshold)

This is NOT a tolerance issue - it's a correctness issue.

## Files Modified

1. **src/v2/pipelines/qwen/Qwen2Pipeline.cpp**
   - Lines 527-555: Fixed attention block INT8 dequantization
   - Lines 864-892: Fixed FFN block INT8 dequantization
   - Added debug logging for scale validation

2. **src/v2/tensors/INT8Tensor.cpp**
   - Lines 145-180: Fixed data() to only dequantize rows with scales
   - Added zero-fill for unused buffer space
   - Changed from error fallback to smart partial dequantization

3. **tests/v2/e2e/Test__Qwen2E2ECorrectness.cpp**
   - Line 789: Temporarily increased BatchScaling tolerance to 3e-3 (reverted)

## Next Steps

### Immediate Investigation
1. Determine why test results are non-deterministic (build configuration issue?)
2. Check if INT8Tensor::data() is being called from unexpected locations
3. Verify E2ERelease build flags match expectations (-O3, -march=native, ENABLE_PIPELINE_SNAPSHOTS)

### Code Review Needed
1. Search for all call sites of INT8Tensor::data()
2. Verify no other code paths access INT8 buffers beyond valid rows
3. Check if snapshot framework itself has buffer management issues

### Testing Strategy
1. Run tests multiple times to confirm determinism
2. Add assertions to catch invalid buffer access
3. Use ASAN build to detect any remaining buffer overflows

## Build Configuration Note

**CRITICAL**: E2E tests require `E2ERelease` build type:
```bash
cmake -B build_v2_e2e -S src/v2 -DCMAKE_BUILD_TYPE=E2ERelease
```

Using `Release` or `Release -DENABLE_PIPELINE_SNAPSHOTS=ON` does NOT work correctly!
The E2ERelease build type has specific flags that regular Release lacks.

## Conclusion

We've made significant progress:
- **Root cause identified and fixed** (buffer overflow in dequantization)
- **Numerical explosions eliminated** (700× improvement)
- **Error messages resolved** ("Insufficient row scales")

However, **test results are inconsistent**, suggesting:
- Either there's another bug we haven't found
- Or the build system is not behaving deterministically
- Or INT8Tensor::data() zero-fill is problematic

**Status**: Investigation ongoing - significant progress but not yet resolved.
