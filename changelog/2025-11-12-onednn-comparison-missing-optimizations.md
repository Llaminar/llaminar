# OneDNN vs Llaminar INT8 GEMM: Missing Optimizations

**Date**: November 12, 2025  
**Current Performance**: 1597 GOPS (M=14336, 28 threads)  
**Target Performance**: 6600 GOPS (OneDNN reference)  
**Gap**: **4.1× slower than OneDNN**

## System Specifications

- **CPU**: 12-channel DDR4-2933 (6 channels per socket, 2 sockets)
- **Theoretical Bandwidth**: ~90 GB/s per socket, 180 GB/s total
- **Current Bandwidth Utilization**: 3.3 GB/s (3.6% of theoretical!)

## Critical Finding: We're Memory-Bound, But Inefficiently

Our current 3.3 GB/s is **nowhere near** the 90 GB/s available. This suggests we're:
1. Not streaming data efficiently (cache misses)
2. Not prefetching aggressively
3. Not using optimal data layouts
4. Not maximizing instruction-level parallelism (ILP)

---

## OneDNN Architecture Analysis

### 1. Microkernel Design (48×8 vs our 16×16)

**OneDNN (`jit_avx512_core_gemm_s8u8s32_kern.hpp`)**:
```cpp
static const int IGEMM_UNROLL_M_ = 48;  // 48 rows
static const int IGEMM_UNROLL_N_ = 8;   // 8 columns
```

**Llaminar (current)**:
```cpp
using Int8Gemm_6x16 = ParameterizedInt8GemmKernel<16, 16>;  // 16×16
```

**Why OneDNN's 48×8 is better**:

| Aspect | OneDNN 48×8 | Llaminar 16×16 | Impact |
|--------|-------------|----------------|--------|
| **M unrolling** | 48 rows | 16 rows | 3× more work per microkernel call |
| **Register usage** | 3 ZMM for A (48/16) × 8 outputs = 24 ZMM | 1 ZMM for A × 16 outputs = 16 ZMM | Better register utilization |
| **Amortization** | Loop overhead amortized over 48 rows | Overhead every 16 rows | 3× better overhead amortization |
| **Cache blocking** | Matches L1 cache better | Suboptimal | Better data reuse |

**Recommendation**: Test 48×8 microkernel size.

---

### 2. Aggressive Prefetching

**OneDNN (`jit_avx512_core_gemm_s8u8s32_kern.cpp` lines 96-127)**:
```cpp
// Prefetch configuration
static const int prefetch_size_a_ = 32 * 5;  // 160 bytes ahead
static const int prefetch_size_b_ = 32 * 4;  // 128 bytes ahead

void kernel_loop(int unroll_m, int unroll_n, bool cfetch) {
    for (int h = 0; h < 4; h++) {
        for (int j = 0; j < unroll_n; j++) {
            // ... compute ...
            
            if (j == 1 && !(h & 1))
                prefetch_b(ptr[BO_ + isize_ * (prefetch_size_b_ + ...)]);
            else if (j % 3 == 0)
                prefetch_a(ptr[AO_ + isize_ * (prefetch_size_a_ + ...)]);
            
            if (cfetch && (j == std::min(1, unroll_n - 1))) {
                if (h == 3)
                    lea(CO2_, ptr[CO2_ + LDC_]);
                else if (h < um_vecs)
                    prefetch_c(ptr[CO2_ + (16 * h * size_)]);
            }
        }
    }
}
```

**Llaminar (current)**: **NO PREFETCHING AT ALL**

**Missing**:
- `prefetcht0` for A matrix (temporal locality)
- `prefetcht0` for B matrix (temporal locality)  
- `prefetchw` for C matrix (write prefetch)
- Strategic prefetch distances (5-10 cache lines ahead)

**Impact**: Hiding memory latency is CRITICAL for bandwidth utilization!

**Recommendation**: Add prefetching for A, B, C with tuned distances.

---

### 3. K-Loop Unrolling (4× in OneDNN)

**OneDNN**: 4-way K-loop unrolling (processes 16 K elements per iteration)

```cpp
void kernel_loop(int unroll_m, int unroll_n, bool cfetch) {
    for (int h = 0; h < 4; h++) {  // 4× unrolled
        // Compute for h-th K-block
        for (int j = 0; j < unroll_n; j++) {
            vpbroadcastd(b, ptr[BO_ + isize_ * (2*j + 2*h*unroll_n - offset_b_)]);
            dot_product(c_regs_[0][j], b, a_regs_[0]);
            // ... more work ...
        }
        
        // Load next A blocks
        for (int i = 0; i < um_vecs; i++)
            vmovups(a_regs_[i], ptr[AO_ + isize_ * (32*i + 2*(h+1)*unroll_m - offset_a_)]);
    }
    // Loop overhead: once per 16 K elements (4 blocks × 4 bytes)
}
```

**Llaminar (current)**: 4× unrolling exists, but:
- Uses scalar loads for A (not vectorized properly)
- Missing interleaved prefetching
- Loop overhead not optimized

**Recommendation**: Properly vectorize A loads and interleave prefetch.

---

### 4. Data Layout Optimization

**OneDNN B packing** (`jit_avx512_core_u8_copy_bn_kern_autogen.cpp`):
- B matrix packed into VNNI-friendly layout: `[K/4][N][4]`
- 64-byte alignment for cache line optimization
- Separate copy kernels for different transpose cases

**Llaminar (current)**:
```cpp
static void pack_B_panel(const int8_t* B_src, int8_t* B_packed, int K, int ldb) {
    for (int k = 0; k < K; k += 4) {
        for (int j = 0; j < NR; ++j) {
            for (int kk = 0; kk < 4; ++kk) {
                dst[j*4 + kk] = B_src[j * ldb + k + kk] ^ 0x80;
            }
        }
    }
}
```

**Missing**:
- Vectorized packing (OneDNN uses SIMD for copy)
- Cache-line-aligned allocations
- Parallel packing (OneDNN uses `parallel_nd`)

**Recommendation**: Use SIMD for B packing, align to 64 bytes.

---

### 5. Compensation Strategy

**OneDNN** (`simple_gemm_s8s8s32.cpp` lines 53-112):
```cpp
void compensation_compute(bool transa, dim_t m, dim_t k, float alpha,
        const int8_t *a, dim_t lda, int32_t *compensation) {
    const auto L2_cache_size = platform::get_per_core_cache_size(2);
    const int blocking_factor = static_cast<int>(
        nstl::min(k, L2_cache_size / lda + 1));
    const dim_t npanels = k / blocking_factor;

    parallel_nd(npanels, m, [&](dim_t j, dim_t i) {
        int32_t val = 0;
        for (dim_t jb = 0; jb < blocking_factor; jb++) {
            val += a[(i + j * blocking_factor * lda) + jb * lda];
        }
        val = q10n::out_round<int32_t>(
            q10n::saturate<int32_t>((double)val * alpha * -128.0));
        fetch_and_add(&compensation[i], val);
    });
}
```

**Key differences**:
- **Cache-aware blocking**: Tiles K dimension based on L2 cache size
- **Parallel computation**: Uses `parallel_nd` for multi-threading
- **Atomic accumulation**: `fetch_and_add` for thread-safe updates
- **Precomputed ONCE**: Not in the hot loop!

**Llaminar (current)**: Compensation computed **inside microkernel** (3× overhead per K-step!)

**Recommendation**: Precompute compensation before main GEMM loop.

---

### 6. Parallelization Strategy

**OneDNN**: Multi-level parallelism
```cpp
parallel_nd(npanels, m, [&](dim_t j, dim_t i) {
    // Parallel over K panels AND M rows
});
```

**Llaminar (current)**:
```cpp
#pragma omp parallel for schedule(static)
for (int i = 0; i < M; i += MR) {
    // Only parallel over M dimension
}
```

**Missing**:
- Parallel K-dimension blocking
- Parallel B packing
- NUMA-aware work distribution

**Recommendation**: Add parallel K-blocking for very large K.

---

### 7. Remainder Handling

**OneDNN** (`jit_avx512_core_gemm_s8u8s32_kern.cpp` lines 171-275):
- Specialized kernels for K remainder: k%8, k%4, k%2, k%1
- Each uses optimal instruction sequence (vpbroadcastd, vpbroadcastw, vpbroadcastb)
- SIMD unpacking for k%2 and k%1 cases

**Llaminar (current)**: Scalar fallback for remainder

**Recommendation**: Add SIMD remainder kernels.

---

## Bandwidth Analysis

### Current vs Theoretical

| Metric | Current | Theoretical | Utilization |
|--------|---------|-------------|-------------|
| **Bandwidth** | 3.3 GB/s | 90 GB/s (per socket) | **3.6%** |
| **GOPS** | 1597 | 6600 (OneDNN) | **24%** |

### Why We're Memory-Bound But Slow

**Problem**: We're waiting on memory, but not using available bandwidth!

**Root causes**:
1. **No prefetching** → Every cache miss stalls pipeline
2. **Small microkernel** (16×16 vs 48×8) → Poor amortization
3. **Sequential packing** → Not using all cores for data prep
4. **Suboptimal layout** → Cache line splits, unaligned accesses

**Solution**: OneDNN hides latency through:
- Aggressive prefetching (5-10 cache lines ahead)
- Larger microkernel (more work in flight)
- Parallel packing (all cores preparing data)
- Aligned, cache-friendly layouts

---

## Priority Optimizations (Ranked by Expected Impact)

### 1. **CRITICAL**: Add Aggressive Prefetching (Expected: +50-100% → 2400-3200 GOPS)
```cpp
// Before compute:
_mm_prefetch((const char*)(A + prefetch_dist_a), _MM_HINT_T0);
_mm_prefetch((const char*)(B + prefetch_dist_b), _MM_HINT_T0);
_mm_prefetch((const char*)(C + i*ldc + j), _MM_HINT_ET1);  // Write prefetch

// Tuned distances:
constexpr int prefetch_dist_a = 160;  // 5 cache lines (32 bytes each)
constexpr int prefetch_dist_b = 128;  // 4 cache lines
```

### 2. **HIGH**: Increase Microkernel to 48×8 (Expected: +20-30% → 1900-2100 GOPS)
```cpp
using Int8Gemm_Primary = ParameterizedInt8GemmKernel<48, 8>;
using Int8Gemm_Fallback = ParameterizedInt8GemmKernel<16, 8>;
```

### 3. **HIGH**: Precompute Compensation (Expected: +15-20% → 1850-1950 GOPS)
```cpp
// BEFORE main GEMM loop:
#pragma omp parallel for
for (int i = 0; i < M; ++i) {
    int32_t sum = 0;
    for (int k = 0; k < K; ++k) {
        sum += A[i*lda + k];
    }
    compensation[i] = sum * (-128);
}

// In microkernel: just add precomputed value
```

### 4. **MEDIUM**: Vectorized B Packing (Expected: +10-15% → 1750-1850 GOPS)
```cpp
void pack_B_panel_vectorized(const int8_t* B_src, int8_t* B_packed, int K, int ldb) {
    #pragma omp parallel for
    for (int k_panel = 0; k_panel < K; k_panel += PANEL_SIZE) {
        // Use SIMD for packing
        for (int k = k_panel; k < k_panel + PANEL_SIZE; k += 64) {
            __m512i b = _mm512_loadu_si512((__m512i*)(B_src + k*ldb));
            // XOR with 0x80, transpose, store
            _mm512_store_si512((__m512i*)(B_packed + ...), b);
        }
    }
}
```

### 5. **LOW**: SIMD Remainder Kernels (Expected: +5% → 1680 GOPS)
- Add specialized k%8, k%4, k%2, k%1 handlers

---

## Cumulative Impact Projection

If we implement all HIGH/CRITICAL optimizations:

| Optimization | Baseline | After | Cumulative |
|--------------|----------|-------|------------|
| Current | 1597 | 1597 | 1.0× |
| + Prefetching | 1597 | 2400 | 1.5× |
| + 48×8 microkernel | 2400 | 3120 | 2.0× |
| + Precomputed compensation | 3120 | 3750 | 2.3× |
| + Vectorized packing | 3750 | 4500 | 2.8× |
| **TOTAL (conservative)** | | | **2.8× → 4476 GOPS** |
| **TOTAL (optimistic)** | | | **3.5× → 5590 GOPS** |

**Still short of 6600 GOPS target, but much closer!**

Remaining gap likely from:
- OneDNN's JIT code generation (vs our C++ intrinsics)
- Additional microarchitectural tuning
- Specialized instruction scheduling

---

## Immediate Next Steps

1. **TODAY**: Add prefetching to existing 16×16 microkernel
   - Benchmark: Expect 2400-3200 GOPS

2. **TOMORROW**: Implement 48×8 microkernel
   - Benchmark: Expect 3000-4000 GOPS

3. **THIS WEEK**: Precompute compensation + vectorized packing
   - Benchmark: Expect 4500-5500 GOPS

4. **STRETCH GOAL**: If we hit 5500 GOPS, investigate JIT code generation (like OneDNN's Xbyak)

---

## References

**OneDNN Source Files Analyzed**:
- `external/onednn/src/cpu/x64/gemm/s8x8s32/jit_avx512_core_gemm_s8u8s32_kern.hpp`
- `external/onednn/src/cpu/x64/gemm/s8x8s32/jit_avx512_core_gemm_s8u8s32_kern.cpp`
- `external/onednn/src/cpu/gemm/s8x8s32/simple_gemm_s8s8s32.cpp`

**Key Insights**:
- **Prefetching is non-negotiable** for bandwidth utilization
- **Larger microkernels** (48×8) amortize overhead better
- **Precomputed compensation** avoids hot-loop overhead
- **Parallel packing** exploits all cores

**Current Llaminar**:
- `src/v2/kernels/cpu/gemm_v2/ParameterizedInt8Gemm.h` (365 lines)
- Missing all the above optimizations!

---

**Status**: Analysis complete. Ready to implement prefetching as first optimization.
