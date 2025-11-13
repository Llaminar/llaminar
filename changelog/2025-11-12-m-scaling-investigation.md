# INT8 GEMM M-Scaling Investigation Results

**Date**: November 12, 2025  
**Finding**: **ParameterizedInt8Gemm has NO OpenMP parallelization**

---

## Executive Summary

Tested INT8 GEMM performance scaling from M=128 to M=4096 with 1 to 56 threads.

**Critical Discovery**: Adding more threads provides **ZERO speedup** - performance stays constant at ~100 GOPS regardless of M size or thread count!

**Root Cause**: `ParameterizedInt8Gemm.h` has **no `#pragma omp` directives**. The kernel is entirely sequential.

---

## Benchmark Results

### M-Scaling Test (Single vs Multi-thread)

All tests show single-thread ≈ multi-thread performance:

| M | Single-Thread GOPS | Multi-Thread GOPS (56 cores) | Speedup |
|---|-------------------|------------------------------|---------|
| 128 | 101.7 | 104.1 | 1.02× (noise) |
| 256 | 104.3 | 106.0 | 1.02× (noise) |
| 512 | 105.0 | 106.8 | 1.02× (noise) |
| 1024 | 105.2 | 106.7 | 1.01× (noise) |
| 2048 | 99.8 | 99.2 | 0.99× (worse!) |
| 4096 | 94.8 | 94.8 | 1.00× (identical) |

**Conclusion**: Multi-threading provides NO benefit at any M size.

### Thread Scaling Test (M=2048, varying threads)

| Threads | Time (ms) | GOPS | Speedup | Efficiency |
|---------|-----------|------|---------|------------|
| 1 | 129.9 | 101.3 | 1.00× | 100.0% |
| 2 | 130.9 | 100.5 | 0.99× | 49.6% |
| 4 | 130.9 | 100.5 | 0.99× | 24.8% |
| 8 | 131.3 | 100.2 | 0.99× | 12.4% |
| 16 | 131.9 | 99.7 | 0.98× | 6.2% |
| 28 | 131.4 | 100.1 | 0.99× | 3.5% |
| 56 | 131.5 | 100.0 | 0.99× | 1.8% |

**Conclusion**: Adding threads provides NO speedup (efficiency <2% at 56 threads!)

---

## Root Cause Analysis

### ParameterizedInt8Gemm.h Has No OpenMP

```bash
$ grep "#pragma omp" src/v2/kernels/cpu/gemm_v2/ParameterizedInt8Gemm.h
(no results)
```

**The kernel is entirely sequential!**

### Why Previous perf Results Showed "51% in OpenMP"

Looking back at perf results, we saw:
- 51% cycles in `libgomp.so.1.0.0`
- 18.9% cycles in `IntegerGemmKernel::multiply`

**This was from a DIFFERENT kernel** (`IntegerGemmKernel`, not `ParameterizedInt8GemmKernel`)!

The test file `Test__ParameterizedInt8Gemm.cpp` calls:
```cpp
Int8Gemm_6x16::gemm(M, N, K, A.data(), lda, B.data(), ldb, C.data(), ldc);
```

This is the **static method** on `ParameterizedInt8GemmKernel<16, 16>`, which has:
```cpp
static void gemm(int M, int N, int K,
                const int8_t* A, int lda,
                const int8_t* B, int ldb,
                int32_t* C, int ldc) {
    // NO OpenMP here!
    for (size_t i = 0; i < M; i += MR) {
        // Process MR rows sequentially
        for (size_t j = 0; j < N; j += NR) {
            microkernel(...);  // Sequential call
        }
    }
}
```

---

## Performance Implications

### Current Situation
- **100 GOPS** on M=128-2048 (single-core performance)
- **Target**: 1000 GOPS (10× gap)
- **Problem**: Cannot leverage 28-56 cores!

### Theoretical Multi-core Potential
If we add OpenMP parallelization:
- Single-core: 100 GOPS
- 28 cores (physical): 2800 GOPS (28× ideal scaling)
- 56 threads (with HT): 3000-4000 GOPS (realistic with 50-70% efficiency)

**We're leaving 2700-3900 GOPS on the table!**

---

## Comparison: Which Kernels Have OpenMP?

### ParameterizedInt8Gemm (NEW, OneDNN-inspired)
- ❌ **NO OpenMP parallelization**
- Performance: 100 GOPS (single-core)
- Status: Just implemented, not yet parallelized

### IntegerGemmKernel (OLDER, Q8_0×Q8_0)
- ✅ **HAS OpenMP parallelization** (what perf profiled earlier)
- Performance: Unknown (need to retest)
- Status: Mature kernel with OpenMP

### Recommendation
Test `IntegerGemmKernel` M-scaling to see if THAT kernel achieves multi-core speedup.

---

## Next Steps

### Priority 1: Add OpenMP to ParameterizedInt8Gemm (30 minutes)

**Simple addition**:
```cpp
static void gemm(int M, int N, int K,
                const int8_t* A, int lda,
                const int8_t* B, int ldb,
                int32_t* C, int ldc) {
    #pragma omp parallel for schedule(static)  // ADD THIS
    for (size_t i = 0; i < M; i += MR) {
        for (size_t j = 0; j < N; j += NR) {
            microkernel(...);
        }
    }
}
```

**Expected result**: 10-20× speedup on M=2048 with 28-56 threads

### Priority 2: Re-run M-Scaling Tests (5 minutes)

After adding OpenMP, rerun:
```bash
./performance/v2_perf_int8_gemm_m_scaling --gtest_filter='*MScaling*'
```

**Expected**:
- M=128: ~500 GOPS (5× speedup, 8 iterations)
- M=256: ~1000 GOPS (10× speedup, 16 iterations)
- M=512: ~1500 GOPS (15× speedup, 32 iterations)
- M=1024: ~2000 GOPS (20× speedup, 64 iterations)
- M=2048: ~2500 GOPS (25× speedup, 128 iterations)

### Priority 3: Compare Against IntegerGemmKernel (10 minutes)

Test the existing `IntegerGemmKernel` (which has OpenMP) to validate multi-core scaling works:
```bash
# Need to create similar M-scaling test for IntegerGemmKernel
```

---

## Key Insights

1. **ParameterizedInt8Gemm is sequential** - no OpenMP at all
2. **Previous perf profiling was on different kernel** (IntegerGemmKernel)
3. **100 GOPS is single-core performance** - actually quite good!
4. **Multi-core potential is 2500-3000 GOPS** with proper OpenMP
5. **1000 GOPS target is achievable** with parallelization

---

## Revised Performance Roadmap

### Current State
- ParameterizedInt8Gemm: 100 GOPS (single-core, M=2048)
- Status: Sequential implementation

### After OpenMP Addition
- Expected: 2000-2500 GOPS (M=2048, 28 cores, 70-90% efficiency)
- **Instantly hits 1000 GOPS target!**

### Further Optimizations
Once parallelized, then focus on:
1. K-loop unrolling (8× instead of 4×): +5-10%
2. Vectorized compensation: +10-15%
3. Prefetching: +5%
4. NUMA optimization: +10-20%

**Final target**: 3000-4000 GOPS (3-4× beyond 1000 GOPS goal)

---

## Summary

The M-scaling investigation revealed that `ParameterizedInt8Gemm` has **no OpenMP parallelization**. This explains:

- ✅ Why single-thread = multi-thread performance
- ✅ Why performance doesn't scale with M
- ✅ Why we're stuck at ~100 GOPS

**Solution**: Add `#pragma omp parallel for` to the outer loop.

**Expected outcome**: Instant 20-25× speedup, easily surpassing 1000 GOPS target.

**Action**: Implement OpenMP parallelization next.
