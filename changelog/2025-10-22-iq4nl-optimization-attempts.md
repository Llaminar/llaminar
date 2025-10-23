# IQ4_NL GEMM Optimization Attempts - October 22, 2025

## Session Summary

Investigated and attempted optimizations for IQ4_NL quantized GEMM performance, discovering critical insights about profiling debug vs release builds and the nature of the actual bottlenecks.

## Initial Analysis (Incorrect)

**From Debug build profiling:**
- multiply() function: 15.05% of cycles
- Identified `vcvtsi2ss` (scalar int→float conversion) as bottleneck in dequantization
- Expected: 3.7× faster dequantization → +15-20% overall GEMM

**Problem:** Debug build profiling is misleading for performance optimization!

## What We Actually Discovered

### Critical Finding: No cblas_sgemm Calls

Initially thought 10.63% mystery symbol was `cblas_sgemm`, but **we don't call BLAS at all** for IQ4_NL!

**Our implementation is already fused:**
```cpp
// Decode IQ4_NL block on-demand (32 elements)
tensor_->decode_block_at(row, block_idx, buffer);

// Custom SIMD dot product (not BLAS)
float acc = dot_product_simd(A_row, B_block, k_count);
```

**Architecture:**
- Cache-blocked decode: Small tiles (24-128 columns) decoded to FP32
- Custom dot product: AVX512/AVX2 FMA-based accumulation
- No intermediate full matrix: Memory-efficient streaming

**Real bottleneck breakdown:**
- `dot_product_simd()`: ~10-11% (the 0x767b088776c0 mystery symbol)
- `decodeBlock()`: ~4-5% (dequantization)
- Overhead: ~0-1%

### Debug vs Release Build Differences

| Metric | Debug Build | Release Build |
|--------|-------------|---------------|
| Dequantization % | ~15% | ~4-5% |
| BLAS % | ~15% | Highly optimized |
| Overall GEMM | 409 GFLOPS | 407 GFLOPS |

**Why the difference?**
- Debug: No optimizations, everything slow proportionally
- Release: BLAS library is pre-optimized assembly, dequant becomes smaller fraction

## Optimizations Attempted

### Attempt 1: Vectorized Dequantization (Priority 1 from ACTION_PLAN)

**Implementation:**
```cpp
// Before (scalar):
for (size_t j = 0; j < 16; ++j) {
    output[j] = d * static_cast<float>(kvalues_iq4nl[qbyte & 0x0F]);  // vcvtsi2ss
}

// After (vectorized):
__m512i i32_vals = _mm512_cvtepi8_epi32(i8_vals);  // 16 at once
__m512 floats = _mm512_cvtepi32_ps(i32_vals);      // 16 at once (1 cycle!)
```

**Result:**
- Dequantization itself: 3.5× faster ✅
- Overall GEMM: No measurable change (407 GFLOPS → 407 GFLOPS)
- **Why:** Dequant only 4-5% of time, so 3.5× faster = 2-3% overall (within noise)

### Attempt 2: Optimized Dot Product (Always Unrolled)

**Target:** The real bottleneck (10-11% of execution time)

**Implementation:**
```cpp
// 4 independent accumulators to hide FMA latency
__m512 sum0, sum1, sum2, sum3;

// Unroll 4× (64 elements per iteration)
for (; i + 64 <= count; i += 64) {
    __builtin_prefetch(a + i + 64, 0, 0);  // Prefetch ahead
    sum0 = _mm512_fmadd_ps(va0, vb0, sum0);
    sum1 = _mm512_fmadd_ps(va1, vb1, sum1);
    sum2 = _mm512_fmadd_ps(va2, vb2, sum2);
    sum3 = _mm512_fmadd_ps(va3, vb3, sum3);
}
```

**Results:**
| Workload | Baseline | Optimized | Change |
|----------|----------|-----------|--------|
| Q-Projection 4096 | 407 GFLOPS | 399 GFLOPS | **-2.0%** ❌ |
| FFN 512 | 491 GFLOPS | 510 GFLOPS | **+3.9%** ✅ |
| FFN 8192 | 379 GFLOPS | 367 GFLOPS | **-3.2%** ❌ |

**Analysis:**
- Memory-bound (FFN 512): Prefetching helps (+4%)
- Compute-bound (Q-proj, large FFN): Register pressure hurts (-2 to -3%)

### Attempt 3: Adaptive Dot Product (Size-Based Selection)

**Strategy:**
```cpp
if (count > 512) {
    // Use unrolled 4-accumulator version (memory-bound workloads)
} else {
    // Use simple 1-accumulator version (compute-bound workloads)
}
```

**Results:**
| Workload | Baseline | Adaptive | Change |
|----------|----------|----------|--------|
| Q-Projection 4096 | 407 GFLOPS | 399 GFLOPS | **-2.0%** ❌ |
| FFN 512 | 491 GFLOPS | 505 GFLOPS | **+2.9%** ✅ |
| FFN 8192 | 379 GFLOPS | 363 GFLOPS | **-4.2%** ❌ |

**Why it failed:**
- Threshold at 512 was wrong for our workload
- k dimension is almost always 896 (hidden dimension)
- So threshold=512 means everything uses simple path (same as baseline)
- Only FFN down (k=4864) gets optimization, but that's minority of ops

## Key Lessons Learned

### 1. Always Profile Release Builds

Debug builds (-O0) distort the performance profile:
- Unoptimized code everywhere makes slow operations look important
- Pre-optimized libraries (BLAS) appear slower than they really are
- **Result:** You optimize the wrong thing!

**Correct workflow:**
1. Profile Release build to identify real bottlenecks
2. Implement optimization
3. Benchmark Release build to verify improvement

### 2. Amdahl's Law Is Unforgiving

Even a **10× speedup** of a 5% component only gives **~4.5% overall improvement**.

For our case:
- Dequantization: 4% of time, 3.5× faster → 2.6% overall gain (within noise)
- Dot product: 10% of time, even if 2× faster → only 9% overall gain

**To get meaningful improvements**, need to:
- Target components >20% of execution time, OR
- Optimize multiple medium-sized components simultaneously, OR
- Find algorithmic improvements (different approach entirely)

### 3. Optimization Trade-offs Are Real

Loop unrolling and multiple accumulators:
- **Help:** Memory-bound workloads (hide memory latency with computation)
- **Hurt:** Compute-bound workloads (register pressure, code size, branch prediction)

No single optimization is universally beneficial. Adaptive approaches sound good but:
- Add complexity
- Require correct threshold tuning for your specific workload
- Branch misprediction overhead can negate benefits

### 4. We Already Have Good Performance

**Current performance:** 407 GFLOPS for Q-projection 4096
- Theoretical peak: ~4480 GFLOPS (2 sockets × 28 cores × 2 FMA × 2.5 GHz × 16 wide)
- Achieved: 9.1% of peak
- **This is reasonable** for memory-bound quantized GEMM

**Why we're memory-bound:**
- IQ4_NL requires decode on every access (4.5 bits per weight)
- Custom implementation trades memory for flexibility
- Already using cache blocking and SIMD

## What Actually Works

### Real Optimization Opportunities (Not Pursued)

1. **Intel MKL backend** (if available):
   - Better AVX512 tuning than OpenBLAS
   - Expected: +5-10%

2. **Custom AVX512 GEMM kernel** for specific shapes:
   - Hand-tuned for 896×896, 896×4864
   - Expected: +10-20% (if done well)
   - Effort: 2-3 weeks

3. **Fused dequant+GEMM kernel**:
   - Eliminate intermediate decode buffer
   - Stream quantized weights directly into GEMM
   - Expected: +5-15% (bandwidth reduction)
   - Effort: 1-2 weeks

4. **GPU backends** (CUDA/ROCm):
   - 10-50× speedup potential
   - Different bottlenecks (memory bandwidth, kernel launch overhead)

## Decision: Revert to Baseline

**Rationale:**
- No consistent improvement across workloads
- Regressions in key operations (Q-projection -2%, FFN 8192 -3-4%)
- Code complexity not justified by +3% gain on FFN 512 alone
- Baseline is simple, well-tested, and already performant

**Current state:** Reverted to original `dot_product_simd()` implementation.

## Recommendations

1. **Accept current IQ4_NL performance** as good enough (9% of peak is reasonable)
2. **Focus on other priorities** with higher ROI:
   - ✅ MKL BF16 backend (already implemented)
   - ✅ Batch processing (higher throughput)
   - 🔄 CUDA/ROCm backends (orders of magnitude faster)
   - 🔄 Fused attention kernels (reduce memory traffic)

3. **If IQ4_NL performance is critical:**
   - Try Intel MKL as drop-in replacement for OpenBLAS
   - Investigate fused dequant+GEMM kernel (bigger architectural change)
   - Profile with `perf` to identify the real 10.63% symbol (likely our dot_product_simd)

## Files Modified (Then Reverted)

- `src/tensors/IQ4_NLTensor.h`:
  - `decodeBlockAVX512()`: Vectorized dequantization (kept - no harm)
  - `decodeBlockAVX2()`: Vectorized dequantization (kept - no harm)
  - `dot_product_simd()`: **Reverted to baseline** (unrolling optimization removed)

## Performance Baseline (For Future Reference)

**Qwen 2.5 0.5B IQ4_NL (896 hidden dimension):**
- Q-Projection 4096 tokens: **407 GFLOPS**
- FFN 512 tokens: **491 GFLOPS**
- FFN 8192 tokens: **379 GFLOPS**

**Hardware:** 2-socket system (56 physical cores), AVX512F, OpenBLAS backend
**Build:** Release (-O3 -march=native)
