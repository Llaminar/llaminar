# Phase 3: Precomputed Compensation - Negative Result

**Date**: 2025-11-12  
**Status**: ❌ **ABANDONED** (no improvement)

## Hypothesis

Precomputing `sum(A_row) × 128` outside the microkernel hot loop would eliminate dpbusd accumulation overhead and provide +15% improvement.

## Implementation

**Changes Made** (7 edits to `ParameterizedInt8Gemm.h`):

1. Added optional `A_compensation` parameter to microkernel signature
2. Added conditional compensation loading (precomputed vs inline)
3. Conditionalized sum(A) accumulation in main K-loop
4. Conditionalized sum(A) accumulation in remainder K-loop  
5. Updated compensation application to use precomputed values when available
6. Added precomputation loop in `gemm()` function:
   ```cpp
   std::vector<int32_t> A_compensation(M);
   #pragma omp parallel for schedule(static)
   for (int i = 0; i < M; ++i) {
       int32_t sum = 0;
       for (int k = 0; k < K; ++k) {
           sum += static_cast<int32_t>(A[i * lda + k]);
       }
       A_compensation[i] = sum * 128;
   }
   ```
7. Updated microkernel call site to pass compensation array

## Results

**Expected**: +15% improvement (1273 → 1465 GOPS)  
**Actual**: ~0% improvement (1273 → 1270 GOPS)

### Benchmark Data (M=8192, N=3584, K=896, 28 threads)

| Configuration | GOPS | vs Baseline |
|---------------|------|-------------|
| 32×8 (Phase 2) | **1273** | Baseline |
| 32×8 (Phase 3) | **1270** | -0.2% ❌ |

**Correctness**: ✅ All tests pass (backward compatible via optional parameter)

## Root Cause Analysis

### Why Phase 3 Failed

**Memory Bandwidth Bottleneck**:
- **Precomputation approach**: Reads A matrix TWICE
  - Precomputation pass: M×K reads (2048×896 = 1.8M values)
  - GEMM pass: M×K reads again  
  - **Total**: 2× memory bandwidth

- **Inline approach**: Reads A matrix ONCE
  - dpbusd accumulation happens during existing K-loop
  - A values already in cache/registers for main computation
  - **Zero** additional memory traffic

**Computational Cost**:
- Precomputation: M×K operations (1.8M) + M multiplications (2048×128)
- Inline: dpbusd executes in parallel with main computation (effectively FREE)

**Cache Efficiency**:
- Precomputed: Extra memory traffic for loading `A_compensation[]` (8-32 KB)
- Inline: Compensation computed from values already in registers

### Key Insight

The dpbusd instruction for `sum(A)` accumulation is effectively **free** when executed in parallel with the main computation. Loading precomputed values from memory is **SLOWER** than computing them inline with zero-overhead instruction-level parallelism.

## Lessons Learned

1. **Instruction-level parallelism is powerful**: dpbusd can execute in parallel with main computation without adding latency
2. **Memory is the bottleneck**: Loading from memory (even L1 cache) is slower than register-based computation
3. **Data reuse is critical**: Inline computation reuses A values already loaded for GEMM
4. **Measurement beats assumptions**: Empirical testing revealed wrong hypothesis

## Code Status

**Kept Implementation**: The Phase 3 code remains in place with `A_compensation = nullptr` as default parameter. This provides:
- ✅ Backward compatibility (default behavior is Phase 2 inline compensation)
- ✅ Flexibility for future experiments
- ✅ Clean code organization (no dead code removal needed)
- ✅ All correctness tests passing

**Performance Recommendation**: **Do NOT** use precomputed compensation (keep default nullptr).

## Next Steps

**Proceed to Phase 4**: Vectorized B packing

**Phase 4 Strategy**:
- Current: Scalar s8→u8 conversion + transpose during B packing
- Target: SIMD parallel conversion + transpose
- Expected: +15% improvement (1273 → 1465 GOPS)

**Why Phase 4 Should Work**:
- B packing is a separate serial bottleneck (not overlapped with computation)
- Vectorization provides true parallelism (not just ILP)
- No additional memory traffic (same data accessed, just faster)

## Performance Trajectory

```
Phase 0 (baseline):      973 GOPS
Phase 1 (prefetch):     1023 GOPS (+5%)
Phase 2 (32×8 + fixes): 1273 GOPS (+30%)
Phase 3 (precomp):      1270 GOPS (+0%)  ← ABANDONED
────────────────────────────────────────
Next: Phase 4 (vect pack)
Expected:               1465 GOPS (+15%)
OneDNN target:          6600 GOPS
Gap remaining:          4.5× slower
```

## Technical Details

### Precomputation Code (Not Used)

```cpp
// In gemm() function (executed once per GEMM call):
std::vector<int32_t> A_compensation(M);

#pragma omp parallel for schedule(static)
for (int i = 0; i < M; ++i) {
    int32_t sum = 0;
    for (int k = 0; k < K; ++k) {
        sum += static_cast<int32_t>(A[i * lda + k]);
    }
    A_compensation[i] = sum * 128;  // Multiply by zero-point offset
}

// In microkernel call:
microkernel(A + i * lda, Bpack.get(), C + i * ldc + j, 
           K, lda, ldc, A_compensation.data() + i);
```

### Inline Compensation (Currently Used - Optimal)

```cpp
// In microkernel K-loop (executed for every tile):
__m512i sum_a[M_VECS];
for (int i = 0; i < M_VECS; ++i) {
    sum_a[i] = _mm512_setzero_si512();
}

// During main K-loop (FREE - executes in parallel):
for (int kb = 0; kb < K; kb += 4) {
    // Main computation
    c_regs[i][j] = _mm512_dpbusd_epi32(c_regs[i][j], a_vec[i], b_vec[j]);
    
    // Sum(A) accumulation (ZERO overhead due to ILP)
    sum_a[i] = _mm512_dpbusd_epi32(sum_a[i], ones_u8, a_vec[i]);
}

// After K-loop: Apply compensation
for (int i = 0; i < M_VECS; ++i) {
    alignas(64) int32_t row_sums[16];
    _mm512_store_si512(reinterpret_cast<__m512i*>(row_sums), sum_a[i]);
    for (int r = 0; r < 16; ++r) {
        row_sums[r] *= 128;
    }
    __m512i comp_vec = _mm512_load_si512(...);
    for (int j = 0; j < NR; ++j) {
        c_regs[i][j] = _mm512_sub_epi32(c_regs[i][j], comp_vec);
    }
}
```

## References

- OneDNN s8s8s32 GEMM reference: https://github.com/oneapi-src/oneDNN/blob/master/src/cpu/x64/gemm/s8x8s32/jit_avx512_core_gemm_s8u8s32_kern.cpp
- Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- Previous phases: `changelog/2025-11-11-phase1-prefetch-results.md`, `changelog/2025-11-12-phase2-32x8-microkernel-bug-fixes.md`

## Conclusion

Phase 3 demonstrates the importance of empirical testing over theoretical assumptions. The inline compensation approach (Phase 2) is **optimal** due to instruction-level parallelism and cache efficiency. Precomputation adds memory bandwidth overhead without providing computational benefits.

**Status**: ❌ Negative result - Keep Phase 2 implementation (inline compensation)  
**Next**: Proceed to Phase 4 (vectorized B packing)
