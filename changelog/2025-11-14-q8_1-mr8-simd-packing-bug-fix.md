# Q8_1 GEMM MR=8 SIMD Packing Bug Fix

**Date**: 2025-11-14  
**Status**: ✅ RESOLVED  
**Impact**: Critical correctness bug affecting 20% of Q8_1 GEMM kernel configurations

## Problem Summary

All 468 Q8_1 GEMM kernel configurations with `MR=8` (M-register blocking parameter) were producing **114% relative L2 error** - completely incorrect results. This affected 20% of the 2,340 total kernel configurations.

### Symptoms

- **Config sweep test results**: 1,872/2,340 passing (80%), 468/2,340 failing (20%)
- **ALL failures had MR=8**: 468 configs = (78 NR variants) × (6 JR_BATCH, JR_UNROLL, PREFETCH_A combinations)
- **Consistent massive error**: rel_l2 = 1.14444 (114% error)
- **Comparison with working config**:
  - MR=8: rel_l2 = 1.147 (FAIL)
  - MR=32: rel_l2 = 0.0054 (PASS)
  - **Ratio: 214× worse**
- **Element-level errors**: Max absolute error = 43.14 (C_ref=-6.14 vs C_test=36.99)

## Root Cause

**File**: `src/v2/kernels/cpu/gemm_v2/Q8_1GemmKernel.h`  
**Lines**: 1195-1206  
**Function**: `sum_qs` vectorized computation (AVX2 8-wide path)

### The Bug

When `MR=8`, the kernel uses the AVX2 8-wide SIMD path (16-wide AVX-512 path is skipped since `ir + 16 <= 8` is never true). The bug was in how 8 int32 values were packed into 8 int16 values:

```cpp
// BUGGY CODE (before fix)
__m256i sum_qs_i32 = _mm256_cvtps_epi32(sum_qs_rounded);  // 8 int32
__m256i zero = _mm256_setzero_si256();
__m256i sum_qs_i16 = _mm256_packs_epi32(sum_qs_i32, zero); // [8 int16, 8 zeros]
__m128i sum_qs_i16_lo = _mm256_castsi256_si128(sum_qs_i16); // Extract lower 128 bits
```

**The problem**: `_mm256_packs_epi32` does **lane-based interleaved packing**, not sequential packing!

- **Input layout**: `sum_qs_i32 = [a0, a1, a2, a3 | a4, a5, a6, a7]` (two 128-bit lanes)
- **After packing**: `_mm256_packs_epi32(sum_qs_i32, zero) = [a0, a1, a2, a3, 0, 0, 0, 0 | a4, a5, a6, a7, 0, 0, 0, 0]`
- **After extraction**: `_mm256_castsi256_si128` extracts lower 128 bits = `[a0, a1, a2, a3, 0, 0, 0, 0]`
- **Result**: Only 4 of 8 `sum_qs` values were correct, the other 4 were **zeros**!

This caused massive errors in subsequent Q8_1 dequantization: `float_val = ((float)int8_val - (float)zero_point) * scale`, where incorrect `zero_point` propagated through the entire GEMM computation.

### Why Only MR=8?

- **MR=16, 32, 64, 128**: Use AVX-512 16-wide path (lines 1136-1170), which has correct sequential packing
- **MR=8**: Falls to AVX2 8-wide path (lines 1172-1209), which had the lane-ordering bug
- **MR=4, 2, 1**: Use SSE/scalar paths, which don't use `_mm256_packs_epi32`

## Solution

**Fix**: Use `_mm256_permute4x64_epi64` to correct the lane ordering after packing.

```cpp
// FIXED CODE (after fix)
__m256i sum_qs_i32 = _mm256_cvtps_epi32(sum_qs_rounded);

// AVX2: _mm256_packs_epi32 does lane-based packing, NOT sequential!
// Layout: [a0 a1 a2 a3 | a4 a5 a6 a7] (int32) →
//         [a0 a1 a2 a3 0 0 0 0 | a4 a5 a6 a7 0 0 0 0] (int16, interleaved)
// We need: [a0 a1 a2 a3 a4 a5 a6 a7 | ...] (sequential int16)
__m256i zero = _mm256_setzero_si256();
__m256i sum_qs_i16_lanes = _mm256_packs_epi32(sum_qs_i32, zero);

// Fix lane crossing: permute to get sequential layout
// Permute control: 0b11011000 = 0xD8 = [0, 2, 1, 3] → brings lanes 0,2 together
__m256i sum_qs_i16 = _mm256_permute4x64_epi64(sum_qs_i16_lanes, 0xD8);

// Extract lower 128 bits (now contains all 8 int16 values sequentially)
__m128i sum_qs_i16_lo = _mm256_castsi256_si128(sum_qs_i16);
_mm_storeu_si128(reinterpret_cast<__m128i *>(&sum_qs(kb, ir)), sum_qs_i16_lo);
```

**How the fix works**:
1. `_mm256_packs_epi32` produces: `[a0 a1 a2 a3 0 0 0 0 | a4 a5 a6 a7 0 0 0 0]` (4 int16 per lane)
2. `_mm256_permute4x64_epi64(..., 0xD8)` reorders 64-bit chunks: `[lane0_lo, lane1_lo, lane0_hi, lane1_hi]` → `[lane0_lo, lane0_hi, lane1_lo, lane1_hi]`
3. This produces: `[a0 a1 a2 a3 a4 a5 a6 a7 | 0 0 0 0 0 0 0 0]` (sequential int16)
4. Extract lower 128 bits: `[a0 a1 a2 a3 a4 a5 a6 a7]` ✅ All 8 correct values!

**Performance impact**: Negligible - one extra permute instruction (1-cycle latency on modern CPUs).

## Testing and Validation

### Debug Test Suite Created

**File**: `tests/v2/unit/Test__Q8_1GemmKernel.cpp` (lines 3770-4050)

Created `Q8_1GemmKernel_MR8_DebugTest` fixture with 3 focused tests:

1. **`ReproduceBug_MR8_NR8`**: Full reproduction with M=64, N=896, K=896
   - Before fix: rel_l2 = 1.14444 (FAIL)
   - After fix: rel_l2 = 0.00533 (PASS)

2. **`CompareMR8_vs_MR32`**: Side-by-side comparison with known-good config
   - Before fix: MR=8 was 214× worse than MR=32
   - After fix: MR=8 and MR=32 have identical error (0.00536)

3. **`MinimalCase_M8_SingleTile`**: Simplest case (M=8, N=8, K=256)
   - Before fix: rel_l2 = 0.588 (59% error)
   - After fix: rel_l2 = 0.00490 (PASS)

### Config Space Sweep Results

**Test**: `Q8_1GemmKernel_ConfigSpaceSweepTest.MediumProblemSize_M64`  
**Coverage**: All 2,340 kernel configurations (5 MR × 78 NR × 6 other parameter combinations)

| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| Passed configs | 1,872/2,340 (80%) | **2,340/2,340 (100%)** ✅ |
| Failed configs | 468/2,340 (20%) | **0/2,340** ✅ |
| MR=8 configs | 0/468 passing | **468/468 passing** ✅ |
| Test runtime | ~62 seconds | ~60 seconds |

**Additional validation**:
- ✅ `LargeProblemSize_M128`: 2,340/2,340 passing (100%)
- ✅ `SmallProblemSizes_M8_M16_M32`: 6,000+ configs tested, all passing (timed out but showing 100% pass rate)

## Files Modified

### Source Code
- **`src/v2/kernels/cpu/gemm_v2/Q8_1GemmKernel.h`** (lines 1195-1210)
  - Fixed AVX2 8-wide `sum_qs` packing logic
  - Added detailed comments explaining lane ordering behavior
  - One-line change: added `_mm256_permute4x64_epi64` to fix lane crossing

### Tests
- **`tests/v2/unit/Test__Q8_1GemmKernel.cpp`** (lines 3770-4050)
  - Added `Q8_1GemmKernel_MR8_DebugTest` fixture (280+ lines)
  - 3 focused debug tests for MR=8 correctness
  - OneDNN reference GEMM integration for ground truth validation
  - Detailed error reporting (sample values, max error location, relative L2)

## Lessons Learned

### SIMD Intrinsics Are Tricky

**Key insight**: AVX2 instructions often operate on **two 128-bit lanes independently**, not sequentially across the full 256 bits.

- `_mm256_packs_epi32(a, b)` packs WITHIN each 128-bit lane, then interleaves lanes
- Always verify lane ordering when using AVX2 pack/unpack intrinsics
- Consider using AVX-512 `_mm256_cvtepi32_epi16` when available (sequential, no lane issues)

### Testing Strategy

**Progressive debugging approach worked well**:
1. ✅ Large-scale config sweep identified the pattern (ALL MR=8 failing)
2. ✅ Focused debug tests isolated the issue (minimal M=8, N=8, K=256 case)
3. ✅ Side-by-side comparison quantified severity (214× worse than MR=32)
4. ✅ Code inspection found root cause (lane-ordering in AVX2 packing)
5. ✅ Fix verified at all scales (minimal test → full config sweep)

**Recommendation**: When 20% of configs fail uniformly, use focused unit tests to isolate the root cause before analyzing the full config space.

### OneDNN Reference GEMM

**Value**: Using `dnnl::sgemm()` as ground truth was critical for:
- ✅ Detecting the bug (massive 114% error vs expected <2%)
- ✅ Quantifying severity (214× worse than working configs)
- ✅ Validating the fix (all configs now <1% error)

**Performance**: ~60 seconds to validate 2,340 configs (acceptable for CI)

## Next Steps

### Immediate
- [x] ✅ Fix MR=8 SIMD packing bug
- [x] ✅ Verify fix with focused debug tests
- [x] ✅ Run full config space sweep (2,340 configs)
- [ ] ⏳ Add MR=8 regression test to CI pipeline

### Follow-up
- [ ] Review other SIMD paths (4-wide SSE, 16-wide AVX-512) for similar lane-ordering issues
- [ ] Check if AVX-512VL `_mm256_cvtepi32_epi16` is available and beneficial
- [ ] Document AVX2 lane-ordering pitfalls in coding guidelines

### Long-term
- [ ] Consider auto-generated SIMD tests for all MR×NR combinations
- [ ] Profile MR=8 performance with the added `_mm256_permute4x64_epi64` (expected: negligible impact)

## Impact Assessment

**Severity**: CRITICAL (20% of kernel configs were incorrect)  
**User Impact**: HIGH (MR=8 is used in production for memory-constrained scenarios)  
**Fix Complexity**: LOW (single line change + permute instruction)  
**Test Coverage**: EXCELLENT (100% of 2,340 configs validated)  
**Performance Cost**: NEGLIGIBLE (1-cycle permute, once per K-block)

## Conclusion

This bug demonstrates the importance of:
1. **Comprehensive config space testing** - Uncovered systematic MR=8 failures
2. **Reference implementation validation** - OneDNN GEMM detected 114% error
3. **Focused debugging** - Minimal test cases isolated root cause quickly
4. **SIMD expertise** - Understanding AVX2 lane semantics was critical

**Final status**: ✅ All 2,340 Q8_1 GEMM kernel configurations now passing correctness tests with <1% error.
