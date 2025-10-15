# Stage 1 Refactoring Complete: validateAndSetupInputs()

**Date**: 2025-10-13  
**Author**: David Sanftenberg  
**Status**: ✅ Complete - All parity tests passed

## Summary

Successfully extracted STEP 1 (Input validation and setup) from the massive `MPIAttentionKernel::execute()` method into a dedicated helper function `validateAndSetupInputs()`. This is the first of 9 planned refactoring stages to transform the 2,287-line execute() method into a clean orchestration function.

## Changes Made

### Files Modified

#### `src/kernels/MPIAttentionKernel.h`
- **Added** `InputSetupResult` struct (lines ~12-58)
  - Contains all validated inputs (10 tensor pointers)
  - Parameters (seq_len, d_model, rank, world_size)
  - Mode flags (is_decode_mode, cache_seq_len)
  - Head distribution (local_heads, head_offset, local_kv_heads, kv_head_offset, dimensions)
  - Weight format detection (weights_are_sharded)
  - Early exit control (should_early_exit, early_exit_success)

- **Added** `validateAndSetupInputs()` method declaration (line ~210)
  - Signature: `InputSetupResult validateAndSetupInputs(const std::vector<std::shared_ptr<TensorBase>> &inputs, std::vector<std::shared_ptr<TensorBase>> &outputs)`
  - Private helper method

#### `src/kernels/MPIAttentionKernel.cpp`
- **Added** `validateAndSetupInputs()` implementation (lines 386-667, 281 lines)
  - Extracts and validates all 10 input tensors
  - Detects prefill vs decode mode (based on n_past_)
  - Calculates head distribution via getHeadDistribution()
  - Handles early exit for ranks with no work (local_heads == 0)
  - Validates bias sizes (matches local or global dimensions)
  - Detects weight format (sharded vs full)
  - Validates weight dimensions against expected shapes
  - Runs contract validation (if enabled via LLAMINAR_ATTN_VALIDATE_OUTPUT)
  - Performs health checks for NaN/Inf values
  - Logs detailed input statistics for debugging
  - Broadcasts input tensor via MPI_Bcast
  - Throws std::runtime_error on validation failures

- **Modified** `execute()` method (lines 684-752)
  - Replaced ~330 lines of validation code with ~70 lines
  - Calls validateAndSetupInputs() in try-catch block
  - Handles early exit via should_early_exit flag
  - Extracts 19 local variables from InputSetupResult struct for backward compatibility
  - Clean separation between STEP 1 (refactored) and STEP 2 (untouched)

### Code Metrics

- **Before**: execute() method: 2,287 lines
- **After**: execute() method: ~2,050 lines (reduced by ~237 lines, 10% reduction)
- **Helper method size**: 281 lines
- **Net change**: +44 lines overall (refactored code is slightly more explicit with struct)
- **Readability improvement**: Significant - STEP 1 logic is now self-contained and reusable

## Validation Results

All three parity tests passed with identical results to baseline:

### Test 1: OpenBLAS Prefill vs PyTorch
```
✅ Passed:  387/387
✗ Failed:  0/387
? Missing: 0/387
Duration: 88.2s
```

### Test 2: COSMA Prefill vs PyTorch
```
✅ Passed:  387/387
✗ Failed:  0/387
? Missing: 0/387
Duration: 105.3s
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

### Struct Design Rationale

The `InputSetupResult` struct was designed to:
1. **Encapsulate all validated inputs** - Clear contract of what validation produces
2. **Support early exit** - Separate flag for ranks with no work
3. **Enable reuse** - Struct can be passed to future helper methods
4. **Document intent** - Named fields make code self-documenting
5. **Maintain flexibility** - Easy to extend with additional fields in future

### Error Handling Strategy

- **Validation failures**: Throw `std::runtime_error` with descriptive message
- **Early exit (no work)**: Set `should_early_exit = true`, `early_exit_success = true`
- **Execute() catches exceptions**: Logs error and returns false
- **Preserves original behavior**: All error messages unchanged

### Variable Extraction Pattern

To maintain backward compatibility with STEP 2-8 (not yet refactored), we extract all variables from the struct:

```cpp
auto setup = validateAndSetupInputs(inputs, outputs);
if (setup.should_early_exit) return setup.early_exit_success;

// Extract locals for use in subsequent steps
auto input = setup.input;
auto wq_global = setup.wq_global;
const int seq_len = setup.seq_len;
// ... etc (19 variables total)
```

This pattern will be removed once all stages are refactored and struct members can be accessed directly.

## Compilation Notes

Build completed successfully with only minor narrowing conversion warnings:
- Line 493: `size_t` → `int` in TensorFactory::create_simple() call
- Lines 633-634: Array initialization with size() calls

These warnings are non-critical and can be addressed in Stage 9 cleanup.

## Next Steps

**Stage 2**: Extract STEP 2 (Weight distribution by head dimension) into `distributeWeightsByHead()` helper method.

**Expected metrics**:
- Lines to extract: 744-1025 (~282 lines)
- Return type: WeightSlices struct
- Testing protocol: Same 3 parity tests (387, 387, 585)

## Lessons Learned

1. **Incremental approach works** - One stage at a time with full validation prevents regressions
2. **Struct-based returns are clean** - Much better than multiple out-parameters
3. **Early exit handling is tricky** - Needed separate flag to distinguish "no work" from "failure"
4. **Variable extraction preserves compatibility** - Allows refactoring without touching downstream code
5. **Parity tests catch everything** - Even minor logic changes would show up in 1,944 comparisons

## Performance Impact

**Expected**: None - This is a pure refactoring with no algorithmic changes.

**Measured**: 
- OpenBLAS prefill: 88.2s (baseline: ~88s)
- COSMA prefill: 105.3s (baseline: ~105s)
- Incremental decode: 33.3s (baseline: ~33s)

All within normal variance. No performance regression detected.

---

**Refactoring Progress**: Stage 1/9 complete (11% done)  
**Lines refactored**: 332 / ~2,000 (17% done)  
**Execute() reduction**: 237 lines / ~1,500 target (16% done)
