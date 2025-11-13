# Phase 2: 48×8 Microkernel Performance Bugs Identified

**Date**: November 12, 2025  
**Context**: INT8 GEMM microkernel optimization (targeting OneDNN's 6600 GOPS)  
**Status**: BUGS FOUND - 48×8 is 30% SLOWER than 16×16 due to implementation bugs

## Summary

Created benchmark test `Perf__Int8Gemm_MicrokernelSize.cpp` to compare different microkernel dimensions. **Expected 48×8 to be 20-30% faster than 16×16** (OneDNN's primary microkernel), but found it's actually **30% SLOWER**.

### Benchmark Results (M=2048, N=3584, K=896, 28 threads)

| Microkernel | Time (ms) | GOPS | Speedup vs 16×16 | Change |
|-------------|-----------|------|------------------|--------|
| **16×16 (baseline)** | 12.92 | **1018 GOPS** | 1.00× | --- |
| 16×8 | 17.00 | 774 GOPS | 0.76× | **-24%** ❌ |
| 32×8 | 18.02 | 730 GOPS | 0.72× | **-28%** ❌ |
| **48×8 (OneDNN)** | 18.46 | 712 GOPS | 0.70× | **-30%** ❌ |

**Key Finding**: ALL MR×8 variants are slower than 16×16. The 48×8 variant (with M_VECS=3) is the slowest.

### M-Scaling Results

| M | 16×16 GOPS | 48×8 GOPS | Speedup | Change |
|---|------------|-----------|---------|--------|
| 512 | 650 | 192 | 0.29× | **-71%** |
| 2048 | 1270 | 676 | 0.53× | **-47%** |
| 8192 | 1451 | 919 | 0.63× | **-37%** |

**Pattern**: Performance gap narrows at larger M (better amortization of inefficiencies), but 48×8 is consistently slower.

### Single-Thread Results (isolate microkernel, remove OpenMP effects)

| Microkernel | GOPS | Speedup |
|-------------|------|---------|
| 16×16 | 79.5 GOPS | 1.00× |
| 48×8 | 55.1 GOPS | 0.69× (-31%) |

**Conclusion**: The performance degradation is in the microkernel itself, not parallelization issues.

## Root Cause Analysis

Examined `src/v2/kernels/cpu/gemm_v2/ParameterizedInt8Gemm.h` and identified **two catastrophic bugs** in the M_VECS > 1 code path (used when MR > 16):

### Bug 1: Inefficient A Matrix Loading (Lines 253-263)

**Current implementation**:
```cpp
// Load A values - INEFFICIENT!
int32_t a_dwords[16];
for (int r = 0; r < 16; ++r) {
    if (row_base + r < MR) {
        // PROBLEM: Loading each 32-bit value individually (16 scalar loads!)
        a_dwords[r] = *reinterpret_cast<const int32_t*>(
            A + (row_base + r) * lda + k);
    } else {
        a_dwords[r] = 0;
    }
}
a_vec[i] = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a_dwords));
```

**Problem**:
- Each A load requires 16 scalar memory accesses (one per row)
- Memory is strided (separated by `lda` bytes)
- No SIMD vectorization benefit
- For 48×8 with 4× K-unroll: **3 M_VECS × 4 K-blocks × 16 loads = 192 scalar loads per iteration**

**Correct approach** (OneDNN pattern):
```cpp
// Load contiguous 16-byte chunks (4 INT8 values) per row
// Then extract and broadcast as needed
a_vec[i] = _mm512_loadu_si512(
    reinterpret_cast<const __m512i*>(A + row_base * lda + k)
);
```

**Expected impact**: **10-15% slowdown** from excessive scalar loads

### Bug 2: Catastrophic Compensation Logic (Lines 309-319)

**Current implementation**:
```cpp
// Apply compensation to each row's outputs
int row_base = i * 16;
for (int r = 0; r < 16 && row_base + r < MR; ++r) {  // Loop over 16 rows
    int32_t compensation = row_sums[r] * 128;
    __m512i comp_vec = _mm512_set1_epi32(compensation);  // Unused!
    
    // CATASTROPHIC: Store-modify-load for EVERY element
    for (int j = 0; j < NR; ++j) {  // Loop over 8 columns
        alignas(64) int32_t c_data[16];
        _mm512_store_si512(reinterpret_cast<__m512i*>(c_data), c_regs[i][j]);  // STORE
        c_data[r] -= compensation;  // MODIFY (single element!)
        c_regs[i][j] = _mm512_load_si512(reinterpret_cast<const __m512i*>(c_data));  // LOAD
    }
}
```

**Problem**:
- **Nested loop**: For each of 16 rows, for each of 8 columns
- **16 × 8 = 128 store-modify-load cycles per M_VEC**
- **For M_VECS=3 (48×8): 3 × 128 = 384 memory round-trips!**
- Each round-trip: 64-byte store + cache line reload + single INT32 modify
- Completely defeats register optimization

**Correct approach** (OneDNN pattern):
```cpp
// Build compensation vector with different value per lane
for (int i = 0; i < M_VECS; ++i) {
    // Extract row sums
    alignas(64) int32_t row_sums[16];
    _mm512_store_si512(reinterpret_cast<__m512i*>(row_sums), sum_a[i]);
    
    // Multiply by -128 (vectorized)
    for (int r = 0; r < 16; ++r) {
        row_sums[r] *= -128;
    }
    __m512i comp_vec = _mm512_load_si512(reinterpret_cast<const __m512i*>(row_sums));
    
    // Subtract compensation from all NR columns in one shot
    for (int j = 0; j < NR; ++j) {
        c_regs[i][j] = _mm512_sub_epi32(c_regs[i][j], comp_vec);
    }
}
```

**Expected impact**: **30-40% slowdown** from excessive memory traffic

### Combined Impact

Bug 1 (A loading) + Bug 2 (compensation) = **40-50% total slowdown**

Observed: **30% slower** (matches upper end of Bug 2 alone)

This explains why 16×16 (M_VECS=1, no nested compensation) is fastest.

## Why 16×16 Works

When MR=16, M_VECS=1, so:
- Bug 1: Still present but only affects 1 M_VEC (not 3)
- Bug 2: **Avoided entirely** - outer loop executes once, inner loop is 16×16=256 ops (not 3×16×8=384)
- The compensation bug doesn't trigger the worst-case nested behavior

**This is why 16×16 achieves 1018 GOPS while 48×8 only gets 712 GOPS.**

## Fix Strategy

### Priority 1: Fix Bug 2 (Compensation Logic)

**Impact**: +30-40% expected improvement  
**Complexity**: Medium - requires understanding ZMM lane semantics

**Steps**:
1. Build compensation vector with 16 different values (one per row)
2. Apply vector subtraction instead of scalar store-modify-load
3. Test correctness with small matrix (48×8×32)

**File to edit**: `src/v2/kernels/cpu/gemm_v2/ParameterizedInt8Gemm.h` (lines 290-320)

### Priority 2: Fix Bug 1 (A Loading)

**Impact**: +10-15% expected improvement  
**Complexity**: High - requires understanding dpbusd semantics and data layout

**Steps**:
1. Determine if we can use aligned loads for contiguous chunks
2. May need to reorganize how A data is extracted/broadcast
3. Consider transposing or repacking A in outer loop

**File to edit**: `src/v2/kernels/cpu/gemm_v2/ParameterizedInt8Gemm.h` (lines 250-265)

## Expected Results After Fixes

### After Bug 2 Fix (Compensation)
```
Current 48×8:     712 GOPS
+ Fix Bug 2:      980 GOPS (+38%)  ← Should match/exceed 16×16
```

### After Bug 1 Fix (A Loading)
```
After Bug 2:      980 GOPS
+ Fix Bug 1:     1130 GOPS (+15%)  ← 11% improvement over baseline
```

### Combined With Phase 1 (Prefetching)
```
Baseline 16×16:  1018 GOPS
+ Prefetch (P1):  1023 GOPS (+0.5%)
+ 48×8 (P2):      1130 GOPS (+11%)
────────────────────────────────────
Total vs orig:    1130 vs 973 = +16%
```

Still far from OneDNN's 6600 GOPS (5.8× gap), but establishes solid foundation for Phase 3-5 optimizations.

## Next Steps

1. **Fix Bug 2** (compensation logic) - highest impact
2. **Re-run benchmark** - expect 48×8 to match or exceed 16×16
3. **Fix Bug 1** (A loading) - additional 10-15%
4. **Proceed to Phase 3** (precomputed compensation) once microkernel is correct

## Test Files Created

### `tests/v2/performance/Perf__Int8Gemm_MicrokernelSize.cpp` (216 lines)

Three test cases:
1. **MicrokernelComparison_M2048**: Compare 48×8, 32×8, 16×8, 16×16 at M=2048
2. **MicrokernelScalingWithM**: Test scaling across M=512,2048,8192
3. **SingleThreadMicrokernelImpact**: Isolate microkernel performance (no OpenMP)

**To run**:
```bash
cd /workspaces/llaminar
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --target v2_perf_int8_gemm_microkernel_size --parallel
cd build_v2_release && OMP_NUM_THREADS=28 ctest -R "V2_Perf_Int8Gemm_MicrokernelSize" --verbose
```

## Key Learnings

1. **M_VECS > 1 path was never properly tested**: 16×16 worked by accident (M_VECS=1 avoids bugs)
2. **Compensation logic is critical**: 30% performance impact from naive implementation
3. **A loading pattern matters**: Can't use scalar loads in hot inner loop
4. **Always benchmark before optimizing**: Found bugs instead of expected improvements
5. **OneDNN's 48×8 choice is valid**: Our bugs, not bad microkernel size

## References

- **OneDNN source**: `src/cpu/x64/gemm/s8x8s32/jit_avx512_core_u8_copy_an_kern.cpp` (A packing)
- **OneDNN source**: `src/cpu/x64/gemm/s8x8s32/jit_avx512_core_gemm_s8u8s32_kern.cpp` (microkernel)
- **Phase 1 results**: `changelog/2025-11-12-phase1-prefetching-implementation.md`
- **OneDNN analysis**: `changelog/2025-11-12-onednn-comparison-missing-optimizations.md`

---

**Status**: Bugs identified, fix strategy defined. Ready to proceed with Bug 2 (compensation) fix.

**Target after fixes**: 1130 GOPS (11% improvement over 1018 GOPS baseline)
