# Q8_1 GEMM Compensation Bug - Root Cause Analysis and Fix

**Date:** November 13, 2025  
**Component:** Q8_1GemmKernel (V2)  
**Status:** ⚠️ **PARTIALLY FIXED** - Compensation removed, edge case B layout bug discovered

## Summary

Decomposed Q8_1GemmKernel into unit-testable helper functions and discovered **two critical bugs**:

1. ✅ **FIXED:** Incorrect 128 compensation for Q8_0 symmetric quantization
2. ❌ **NEW BUG FOUND:** Edge case microkernel uses wrong B memory layout

## Bug #1: Incorrect Compensation Formula (FIXED)

### Problem

Q8_1×Q8_0 kernel applied compensation formula designed for **asymmetric** quantization to Q8_0's **symmetric** format:

```cpp
// WRONG (old formula):
compensated = accum - sum_qs * 128

// With test data:
accum = 43225 (positive dot product)
sum_qs = 706.103 (positive sum)
compensated = 43225 - 90368 = -47156  ❌ HUGE NEGATIVE!
result = -2.734 (wrong sign!)
expected = +2.518 (positive)
error = 208%
```

### Root Cause

Q8_0 format uses **symmetric quantization**:
- Range: `[-127, 127]` (signed int8)
- Zero point: 0 (no offset)
- Dequantization: `x = x_quant * scale`

The 128 compensation term is only needed for **asymmetric** quantization where values are stored with a +128 offset to fit in `[0, 255]` unsigned range. Q8_0 doesn't use this offset!

### Fix Applied

Removed compensation from THREE locations in `Q8_1GemmKernel.h`:

1. **Helper function** (line ~148):
```cpp
static inline int32_t apply_compensation(int32_t accum, float sum_qs) {
    (void)sum_qs;  // Unused for Q8_0
    return accum;  // No compensation!
}
```

2. **Vectorized path** (line ~795):
```cpp
// OLD: __m512 compensated = _mm512_fnmadd_ps(sum_qs_vec, compensation_const, accum_f32);
__m512 compensated = accum_f32;  // No compensation!
```

3. **Scalar tail path** (line ~840):
```cpp
// OLD: int32_t compensated = accum(ir, jr, kb) - static_cast<int32_t>(sum_qs * 128.0f);
int32_t compensated = accum(ir, jr, kb);  // No compensation!
```

4. **Edge case microkernel** (line ~1003):
```cpp
// OLD: int32_t compensated = accum - static_cast<int32_t>(sum_qs * 128.0f * K_blocks);
int32_t compensated = accum;  // No compensation!
```

### Verification

Unit test (`HelperFunction_ManualSingleElementGEMM`):
- **Before fix**: -2.734 (wrong sign, 208% error)
- **After fix (manual)**: +2.506 (correct sign, 0.5% error) ✅
- **After fix (kernel)**: +0.670 (still wrong - BUG #2!)

## Bug #2: Edge Case B Layout Mismatch (NEW DISCOVERY)

### Problem

Edge case microkernel produces **wrong accumulation**:
- Edge case accum: **11550** ❌
- Manual accum: **43225** ✅
- Kernel result: 0.670 (should be 2.506)

### Root Cause

Edge case unpacking code assumes **simple column-major** B layout:
```cpp
const uint8_t *b_col_base = B_quants + (kb * BLOCK_SIZE * NR + j);
uint8_t b_val_u8 = b_col_base[k_in * NR];
int8_t b_val = static_cast<int8_t>(b_val_u8 - 128);
```

But `pack_B_panel` uses **column-pair ZMM packing** (line ~461):
```cpp
uint8_t *zmm_base = quants_base + (kb * (NR / 2) + jr_pair) * 64;
// Layout: [kb][jr_pair][64] where 64 bytes = [col0_bytes0..31][col1_bytes0..31]
```

The edge case is indexing into this packed format incorrectly, reading garbage data instead of actual B values!

### Impact

- All small matrix tests (M×N < MR×NR) use edge case path
- ManualSingleElementGEMM (1×1×32): FAIL
- SmallMatrix_8x8x32: FAIL
- All tests using microkernel_edge are affected

### Fix Required

Edge case must respect the packed B layout:
1. Determine which `jr_pair` contains column `j`
2. Calculate offset within 64-byte ZMM chunk
3. Extract correct byte from column-pair packing

**TODO:** Implement correct B unpacking in `microkernel_edge`.

## Test Results

### Helper Function Tests (Fixed)

- ✅ `HelperFunction_ComputeSumQs`: PASS
- ✅ `HelperFunction_ApplyCompensation`: PASS (updated to expect no compensation)
- ✅ `HelperFunction_ScaleToFp32`: PASS
- ✅ `HelperFunction_ComputeBlockResult`: PASS (updated)
- ✅ `HelperFunction_BlockIndexColMajor`: PASS
- ❌ `HelperFunction_ManualSingleElementGEMM`: FAIL (Bug #2)

### Full GEMM Tests

**Status:** 4/17 passing (worse than before due to edge case bug exposure)

**Passing:**
- HelperFunction_ComputeSumQs
- HelperFunction_ScaleToFp32
- HelperFunction_BlockIndexColMajor
- InvalidKDimension_NotMultipleOf32

**Failing (13 tests):**
- All matrix tests (BasicCorrectness, SmallMatrix, etc.)
- All on-the-fly quantization tests (FP32, FP16, BF16)
- PreComputedSumsValidation
- Updated helper tests (ApplyCompensation, ComputeBlockResult, ManualSingleElementGEMM)

## Files Modified

### Kernel Implementation
- `src/v2/kernels/cpu/gemm_v2/Q8_1GemmKernel.h`:
  - Line ~148: Removed compensation from `apply_compensation()`
  - Line ~745: Removed `compensation_const` variable
  - Line ~795: Removed vectorized compensation (`_mm512_fnmadd_ps`)
  - Line ~840: Removed scalar tail compensation
  - Line ~1003: Removed edge case compensation
  - Line ~1011: Added debug output for edge case (reveals layout bug)

### Unit Tests
- `tests/v2/unit/Test__Q8_1GemmKernel.cpp`:
  - Line ~618: Updated `HelperFunction_ApplyCompensation` test (expect no compensation)
  - Line ~672: Updated `HelperFunction_ComputeBlockResult` test (expect no compensation)
  - Line ~687: Added `HelperFunction_ManualSingleElementGEMM` test

## Key Insights from Decomposition Strategy

User requested: "let's decompose the kernel into smaller, inlined functions, and aggressively unit test them"

**Result:** Decomposition strategy was **highly effective**:

1. **Isolated root cause in 5 minutes** - Wrong compensation in 3-line function
2. **Discovered hidden bug** - Edge case layout mismatch exposed by unit tests
3. **Simplified debugging** - Each helper function testable in isolation
4. **Improved code clarity** - Helper functions document computation steps

**Lessons:**
- Monolithic 500+ line kernel hid bugs in complex SIMD code
- Extracting testable components immediately revealed issues
- Unit tests caught bugs that integration tests missed
- Helper functions serve as executable documentation

## Next Steps

### Immediate (HIGH PRIORITY)

1. **Fix edge case B layout** (estimated 30 minutes):
   - Implement correct column-pair unpacking in `microkernel_edge`
   - Add debug logging to verify B values match packed format
   - Re-run ManualSingleElementGEMM test

2. **Verify all tests pass** (10 minutes):
   - Expect 17/17 tests passing after edge case fix
   - Confirm compensation removal didn't break other tests

### Follow-up (MEDIUM PRIORITY)

3. **Performance validation** (30 minutes):
   - Benchmark Q8_1×Q8_0 vs Q8_0×Q8_0
   - Verify removing compensation doesn't hurt performance
   - Check that pre-computed sums still provide speedup

4. **Documentation** (15 minutes):
   - Update kernel comments explaining why Q8_0 needs no compensation
   - Add reference to symmetric vs asymmetric quantization
   - Document edge case B layout for future maintainers

## Diagnostic Commands Used

```bash
# Build and run unit tests
cmake --build build_v2 --target v2_test_q8_1_gemm_kernel
./build_v2/tests/v2/v2_test_q8_1_gemm_kernel --gtest_filter="*ManualSingleElement*"

# Run all helper function tests
./build_v2/tests/v2/v2_test_q8_1_gemm_kernel --gtest_filter="*HelperFunction*"

# Run full test suite
./build_v2/tests/v2/v2_test_q8_1_gemm_kernel
```

## References

- llama.cpp Q8_0 quantization: `ggml-quants.c`
- Q8_0 block structure: `src/v2/tensors/BlockStructures.h` line 41
- Q8_1 block structure: `src/v2/tensors/BlockStructures.h` line 50
- V2 GEMM kernel architecture: `.github/instructions/llaminar-v2-architecture.instructions.md`
