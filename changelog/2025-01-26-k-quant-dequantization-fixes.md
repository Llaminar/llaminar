# K-Quant Dequantization Bug Fixes (Q4_K, Q5_K, Q6_K)

**Date**: January 26, 2025  
**Status**: ✅ All K-quant formats now passing equivalency tests  
**Test Results**: 6 PASSED, 4 SKIPPED (no models available)

## Executive Summary

Fixed critical dequantization bugs in Q4_K, Q5_K, and Q6_K tensor formats by carefully matching llama.cpp's reference implementations. All three formats now achieve **bit-exact equivalency** with llama.cpp (0 mismatches, max abs diff = 0).

## Bugs Fixed

### 1. Q6_K Scale Indexing Bug (CRITICAL)

**Symptom**: 2313/4864 mismatches (47.5%), values ~2× too large  
**Root Cause**: Incorrect scale index calculation `sc[is * 2 + offset]` instead of `sc[is + offset]`  
**Impact**: Complete corruption of Q6_K dequantized values

**Fix** (`src/v2/tensors/Q6_KTensor.cpp`):
```cpp
// BEFORE (WRONG - doubled scale indices)
y[l + 0]  = d * sc[is * 2 + 0] * q1;  // is=0: sc[0], is=1: sc[2] ✗
y[l + 32] = d * sc[is * 2 + 2] * q2;  // is=0: sc[2], is=1: sc[4] ✗
y[l + 64] = d * sc[is * 2 + 4] * q3;  // is=0: sc[4], is=1: sc[6] ✗
y[l + 96] = d * sc[is * 2 + 6] * q4;  // is=0: sc[6], is=1: sc[8] ✗

// AFTER (CORRECT - proper interleaved indices)
y[l + 0]  = d * sc[is + 0] * q1;  // is=0: sc[0], is=1: sc[1] ✓
y[l + 32] = d * sc[is + 2] * q2;  // is=0: sc[2], is=1: sc[3] ✓
y[l + 64] = d * sc[is + 4] * q3;  // is=0: sc[4], is=1: sc[5] ✓
y[l + 96] = d * sc[is + 6] * q4;  // is=0: sc[6], is=1: sc[7] ✓
```

**Math**: For `l ∈ [0,31]`, `is = l/16` produces:
- `l ∈ [0,15]`: `is=0` → scales {0, 2, 4, 6}
- `l ∈ [16,31]`: `is=1` → scales {1, 3, 5, 7}

Each half uses 4 interleaved scales from the 8-element scale array.

**Result**: ✅ 0 mismatches (was 2313/4864)

---

### 2. Q4_K Layout and Scale Extraction Bug (CRITICAL)

**Symptom**: Incorrect layout (8 sub-blocks instead of 4 groups), wrong scale/min extraction  
**Root Cause**: Missing `get_scale_min_k4()` helper, incorrect bit manipulation

**Fix** (`src/v2/tensors/Q4_KTensor.cpp`):

1. **Added scale/min extraction helper** (matching llama.cpp exactly):
```cpp
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
{
    if (j < 4)
    {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    }
    else
    {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
```

2. **Rewrote layout** (from 8 sub-blocks to 4 groups of 64):
```cpp
// BEFORE (WRONG - 8 sub-blocks)
for (size_t i = 0; i < 8; ++i) {
    // 32 elements per sub-block
}

// AFTER (CORRECT - 4 groups of 64, 2×32 pattern)
for (size_t i = 0; i < 4; ++i)
{
    // Group 1: 32 elements (lower nibbles)
    for (size_t l = 0; l < 32; ++l) {
        output[offset + l] = d1 * (q[l] & 0xF) - m1_val;
    }
    
    // Group 2: 32 elements (upper nibbles)
    for (size_t l = 0; l < 32; ++l) {
        output[offset + 32 + l] = d2 * (q[l] >> 4) - m2_val;
    }
}
```

**Result**: ✅ 0 mismatches (was failing completely)

---

### 3. Q5_K High Bit Extraction Bug (CRITICAL)

**Symptom**: Incorrect 5-bit value reconstruction  
**Root Cause**: Wrong high bit extraction, missing bit mask tracking

**Fix** (`src/v2/tensors/Q5_KTensor.cpp`):

1. **Added bit mask tracking**:
```cpp
uint8_t u1 = 1;  // Mask for q1/q3 (bits 0,2,4,6 of qh)
uint8_t u2 = 2;  // Mask for q2/q4 (bits 1,3,5,7 of qh)

// Shift masks by 2 every iteration
u1 <<= 2;
u2 <<= 2;
```

2. **Rewrote 5-bit reconstruction**:
```cpp
// BEFORE (WRONG - incorrect high bit extraction)
const int8_t q1 = ((ql[l] & 0xF) | (/* wrong bit */)) - 16;

// AFTER (CORRECT - proper bit masking)
const int8_t q1 = ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - 16;
const int8_t q2 = ((ql[l + 32] & 0xF) + (qh[l] & u2 ? 16 : 0)) - 16;
const int8_t q3 = ((ql[l] >> 4) + (qh[l] & (u1 << 1) ? 16 : 0)) - 16;
const int8_t q4 = ((ql[l + 32] >> 4) + (qh[l] & (u2 << 1) ? 16 : 0)) - 16;
```

**Pattern**: Alternating bit masks extract high bit from correct position in `qh[]` array.

**Result**: ✅ 0 mismatches (was failing completely)

---

## Files Modified

### Core Tensor Implementations
1. **`src/v2/tensors/Q6_KTensor.cpp`** (lines 129-170)
   - Fixed scale indexing: `sc[is + offset]` instead of `sc[is * 2 + offset]`
   
2. **`src/v2/tensors/Q4_KTensor.cpp`** (lines 118-165)
   - Added `get_scale_min_k4()` helper (lines 157-165)
   - Rewrote to 4 groups of 64 elements
   - Fixed scale/min extraction
   
3. **`src/v2/tensors/Q5_KTensor.cpp`** (lines 119-175)
   - Added bit mask tracking (`u1`, `u2`)
   - Rewrote 5-bit value reconstruction
   - Added `get_scale_min_k4()` helper (lines 167-175)

### Header Declarations
4. **`src/v2/tensors/Tensors.h`**
   - Q4_KTensor: Added `get_scale_min_k4()` declaration (line 1329)
   - Q5_KTensor: Added `get_scale_min_k4()` declaration (line 1169)

## Test Results

**Before Fixes**:
```
❌ Q6_K_Equivalency: FAILED (2313/4864 mismatches, 47.5%)
❌ Q4_K_Equivalency: FAILED (completely broken)
❌ Q5_K_Equivalency: FAILED (completely broken)
```

**After Fixes**:
```
✅ Q6_K_Equivalency: PASSED (0 mismatches, 0 max abs diff)
✅ Q4_K_Equivalency: PASSED (0 mismatches, 0 max abs diff)
✅ Q5_K_Equivalency: PASSED (0 mismatches, 0 max abs diff)
```

**Full Test Suite**:
```
[==========] 10 tests from 1 test suite
[  PASSED  ] 6 tests
  ✅ Q8_0_Equivalency
  ✅ IQ4_NL_Equivalency
  ✅ Q4_0_Equivalency
  ✅ Q6_K_Equivalency    ← FIXED
  ✅ Q4_K_Equivalency    ← FIXED
  ✅ Q5_K_Equivalency    ← FIXED

[  SKIPPED ] 4 tests
  ⏭️ Q4_1_Equivalency (no model)
  ⏭️ Q2_K_Equivalency (mixed quantization)
  ⏭️ Q3_K_Equivalency (mixed quantization)
  ⏭️ Q8_K_Equivalency (no model)
```

## Bug Discovery Pattern

All three bugs were discovered by comparing against llama.cpp's reference implementations:

1. **Q6_K**: Studied `dequantize_row_q6_K()` in `ggml-quants.c` (lines 1773-1779)
   - Found scale indexing used `sc[is + offset]` not `sc[is * 2 + offset]`
   
2. **Q4_K**: Studied `dequantize_row_q4_K()` in `ggml-quants.c` (lines 1517-1600)
   - Found `get_scale_min_k4()` helper for scale extraction
   - Observed 4 groups of 64 elements (not 8 sub-blocks)
   
3. **Q5_K**: Studied `dequantize_row_q5_K()` in `ggml-quants.c` (lines 1723-1771)
   - Found bit mask pattern: `u1=1, u2=2` shifting by 2 each iteration
   - Observed 5-bit reconstruction using `qh[l] & mask ? 16 : 0`

## Reference Implementation Study

**llama.cpp Functions Used**:
- `dequantize_row_q6_K()`: Q6_K reference (256 elements/block, 2 halves of 128)
- `dequantize_row_q4_K()`: Q4_K reference (256 elements/block, 4 groups of 64)
- `dequantize_row_q5_K()`: Q5_K reference (256 elements/block, 4 groups of 64)
- `get_scale_min_k4()`: Shared helper for Q4_K/Q5_K scale extraction

## Impact Assessment

**Severity**: CRITICAL (all three formats completely broken)

**Affected Operations**:
- All Q4_K, Q5_K, Q6_K weight dequantization
- GEMM operations using K-quant tensors
- Model inference with K-quant models

**Blast Radius**:
- Models: qwen2.5-0.5b-instruct-q4_k_m.gguf, q5_k_m.gguf, q6_k.gguf
- Layers: All FFN layers using K-quant formats
- Accuracy: Would produce completely incorrect outputs

## Mixed Quantization Discovery

During testing, discovered K-quant "medium" models use **mixed quantization**:

| Model Variant | Attention Layers | FFN Layers | Other |
|---------------|------------------|------------|-------|
| Q2_K_M | IQ4_NL | Q3_K | Q5_0, Q8_0, F32 |
| Q3_K_M | IQ4_NL | Q3_K | Q5_0, Q8_0, F32 |
| Q4_K_M | Q5_0 | Q4_K | Q8_0, F32 |
| Q5_K_M | Q5_1 | Q5_K | Q8_0, F32 |
| Q6_K | Q8_0 | Q6_K | F32 |

**Implication**: Cannot use attention layer weights for K-quant testing. Tests now use FFN weights (`blk.*.ffn_down.weight`) where the format actually exists.

## Testing Strategy

**Test Framework**: `Test__DequantEquivalency.cpp`
- Compares IBlockDecoder vs llama.cpp's `dequantize_row_*()` functions
- Uses 1e-5 tolerance for floating point comparison
- Reports detailed mismatch statistics

**Validation Approach**:
1. Load single row from FFN weight tensor
2. Dequantize using both IBlockDecoder and llama.cpp
3. Compare element-by-element
4. Report mismatches, max absolute/relative differences

**Why FFN Weights?**:
- K-quant models use mixed quantization
- Attention layers often use different format (Q5_0, Q5_1, Q8_0)
- FFN layers (`blk.*.ffn_down.weight`) contain actual Q4_K/Q5_K/Q6_K blocks

## Verification Commands

```bash
# Build test suite
cmake --build build_v2 --target v2_test_dequant_equivalency --parallel

# Run all tests
./build_v2/tests/v2/v2_test_dequant_equivalency

# Run specific K-quant tests
./build_v2/tests/v2/v2_test_dequant_equivalency --gtest_filter="*Q6_K*"
./build_v2/tests/v2/v2_test_dequant_equivalency --gtest_filter="*Q4_K*"
./build_v2/tests/v2/v2_test_dequant_equivalency --gtest_filter="*Q5_K*"
```

## Related Work

**Previous Bug Fixes** (same test framework):
- 2025-01-25: Fixed IQ4_NL lookup table bug (127× scaling error)
- 2025-01-25: Fixed Q4_0 nibble layout bug (67% corruption)

**Test Framework Development**:
- 2025-01-24: Created llama.cpp integration test suite
- 2025-01-24: Integrated GGML library into V2 build system

## Next Steps

1. ✅ All available K-quant formats validated
2. ⏭️ Q2_K/Q3_K: Cannot test (no pure tensors in available models)
3. ⏭️ Q4_1/Q8_K: Cannot test (no models available)
4. 🔄 Remaining IQ formats: IQ1_S, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ4_XS (need models)

## Lessons Learned

1. **Study Reference Implementations**: llama.cpp's code was essential for fixing all three bugs
2. **Bit-Exact Validation**: 0 mismatches confirms perfect implementation
3. **Mixed Quantization**: K-quant model filenames misleading - must inspect tensor types
4. **Scale Indexing Complexity**: K-quant formats use interleaved scale arrays requiring careful indexing
5. **Helper Functions**: Reuse llama.cpp's helpers (e.g., `get_scale_min_k4()`) to ensure correctness

## Conclusion

All three K-quant dequantization bugs fixed by carefully studying and matching llama.cpp's reference implementations. Test suite now validates bit-exact equivalency with 0 mismatches for Q4_K, Q5_K, and Q6_K formats. Combined with previous IQ4_NL and Q4_0 fixes, Llaminar V2 now has **6 validated quantized tensor formats** ready for production use.

**Status**: ✅ **K-QUANT VALIDATION COMPLETE**
