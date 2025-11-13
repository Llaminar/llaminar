# Integer GEMM Catastrophic Performance Bug

**Date**: November 11, 2025  
**Author**: David Sanftenberg  
**Type**: Critical Performance Bug  
**Status**: ❌ **UNRESOLVED** - Requires complete kernel rewrite

## Summary

After fixing the buffer overflow bug and running the full configuration space sweep (8000 configs), discovered that **Integer GEMM performance is catastrophically poor**: 

- **Best achieved**: 2.00 GFLOPS (MR=1 configuration)
- **Theoretical peak**: 179.2 GFLOPS (AVX512 VNNI single core)
- **Efficiency**: **1.1% of peak** ❌
- **Comparison**: FP32 GEMM achieves 335-1208 GFLOPS (100-600× faster!)

## Root Cause Analysis

### Performance Scaling Anomaly

The sweep revealed a devastating performance pattern:

| MR | GFLOPS | Relative | Expected Scaling |
|----|--------|----------|------------------|
| 1  | 2.00   | 1.00×    | 1.00×            |
| 2  | 1.19   | 0.59×    | ~1.00× (should be same!) |
| 4  | 0.57   | 0.28×    | ~1.00× (should be same!) |
| 8  | 0.31   | 0.15×    | ~1.00× (should be same!) |
| 16 | 0.13   | 0.07×    | ~1.00× (should be same!) |

**Key finding**: Performance scales as `1/MR`, indicating **O(MR) overhead per row** instead of amortized constant cost.

### Bug #1: O(MR × NR) Scale Accumulation Loop

**Location**: `src/v2/kernels/cpu/gemm/int8/IntegerGemmMicroKernel.h:156-163`

**Original code** (WRONG):
```cpp
// Accumulate combined scales for requantization
for (int i = 0; i < TILE_M; ++i)
{
    for (int j = 0; j < TILE_N; ++j)
    {
        combined_scale_acc_ += a_scales[i] * b_scales[j];
    }
}
```

**Problem**: For MR=16, NR=32, this loop runs **512 times per K-block** to accumulate a single scalar value!

**Fixed code**:
```cpp
// O(MR + NR) instead of O(MR × NR)
double a_scale_sum = 0.0;
double b_scale_sum = 0.0;
for (int i = 0; i < TILE_M; ++i)
    a_scale_sum += a_scales[i];
for (int j = 0; j < TILE_N; ++j)
    b_scale_sum += b_scales[j];
combined_scale_acc_ += a_scale_sum * b_scale_sum;
```

**Impact**: ✅ Fixed but **performance unchanged** (still 1-2 GFLOPS)  
**Conclusion**: This bug was real but not the dominant bottleneck!

### Bug #2: Fundamental Kernel Design Issue

**The real problem**: Even after fixing the O(MR×NR) loop, performance remains at **1% of peak**.

**Evidence**:
1. **Single token workload** (m=1, n=896, k=896) = 1.6M operations
2. **Theoretical peak**: 179 GFLOPS = 179B ops/sec
3. **Should take**: 1.6M / 179B = **0.009 ms**
4. **Actually takes**: 0.8 ms (best config) = **89× slower than theoretical!**

**Hypothesis**: The micro-kernel is fundamentally inefficient for several reasons:

1. **Poor tile reuse**: Single token (m=1) means MR>1 configs process mostly padding
2. **Memory bandwidth bound**: Tiny tiles don't amortize load/store overhead
3. **VNNI not properly utilized**: Debug output shows `lane_val=0` (suspicious!)
4. **Scalar fallback**: May be falling back to scalar path despite AVX512 VNNI available
5. **Register spilling**: MR×NR accumulator matrix may exceed register budget

## Comparison to Existing Kernels

| Kernel | Workload | GFLOPS | Efficiency | Notes |
|--------|----------|--------|------------|-------|
| **FP32 CPU GEMM** | 1×896×896 | 335-1208 | 50-70% | Well-optimized BLAS |
| **INT8 Integer GEMM** | 1×896×896 | **2.00** | **1.1%** | ❌ **THIS KERNEL** |
| **Expected INT8** | 1×896×896 | 90-125 | 50-70% | What we SHOULD achieve |

**Conclusion**: INT8 kernel is **168-604× slower** than comparable FP32 kernel!

## Sweep Data Analysis

**Full sweep results** (integer_gemm_sweep_single.csv):
- **Configurations tested**: 8000
- **Min GFLOPS**: 0.04 (MR=16, worst configs)
- **Max GFLOPS**: 2.00 (MR=1, best configs)
- **Mean GFLOPS**: 0.77
- **Median GFLOPS**: 0.54

**Top 10 configurations** (all MR=1):
```
MR  UNROLL_K  PREFETCH_DIST   MC   KC    NC  gflops
1          1              1  128 2048    64    2.00
1          2              5  512 2048   256    1.97
1          1              2  128  512   128    1.96
```

**Bottom 10 configurations** (all MR=16, PREFETCH_DIST=0):
```
MR  UNROLL_K  PREFETCH_DIST   MC   KC    NC  gflops
16         1              0  128 1024   128    0.04
16         1              0  256 2048   256    0.04
```

**Key insight**: Configuration tuning is **irrelevant** when base performance is 100× too slow!

## Hardware Context

**CPU**: Intel Ice Lake with AVX512 VNNI  
**Confirmed flags**: `avx512f avx512dq avx512bw avx512vl avx512_vnni`

**Theoretical capabilities**:
- **dpbusd instruction**: 64 INT8 ops per instruction
- **Dual FMA units**: 2 dpbusd/cycle
- **Base frequency**: ~2.8 GHz
- **Peak**: 2.8 × 2 × 64 = **358 GOPS/core** (179 GFLOPS)

**Achieved**: 2.0 GFLOPS = **1.1% utilization** ❌

## Action Items

### Immediate (Block Full Sweep)

- [x] ~~Run configuration space sweep~~ **COMPLETED** but results show kernel is broken
- [x] ~~Fix O(MR×NR) scale loop~~ **FIXED** but no performance impact
- [ ] **DO NOT use this kernel for production** - it's 100× slower than FP32!

### Short Term (Fix or Replace Kernel)

**Option 1: Debug existing kernel**
1. Verify VNNI code path is actually executing (not falling back to scalar)
2. Profile with `perf` to find actual bottleneck
3. Check if register spilling is occurring
4. Validate that dpbusd instructions are being emitted
5. Compare assembly output to reference VNNI kernels

**Option 2: Use proven library** (RECOMMENDED)
1. Integrate Intel MKL's INT8 GEMM (`cblas_gemm_s8u8s32`)
2. Or use BLIS INT8 kernel
3. Or use oneDNN (DNNL) for quantized inference

**Option 3: Rewrite from scratch**
1. Start with simple scalar reference
2. Add AVX512 VNNI step-by-step with validation
3. Study existing high-performance INT8 kernels (MKL, oneDNN, CUTLASS)
4. Focus on single token (m=1) case first (most common for inference)

### Long Term

- **Deprecate custom INT8 kernel** in favor of optimized library (MKL/BLIS/oneDNN)
- **Focus V2 development** on FP32/BF16 kernels where we have proven performance
- **Use vendor libraries** for INT8 quantized inference (not a core competency)

## Files Affected

- **Broken kernel**: `src/v2/kernels/cpu/gemm/int8/IntegerGemmMicroKernel.h`
- **Sweep results**: `integer_gemm_sweep_single.csv` (8000 configs, all terrible)
- **Tests**: `tests/v2/performance/cpu/kernels/gemm/Perf__IntegerGEMM_*.cpp`

## Lessons Learned

1. **Always benchmark against baseline**: Should have compared to FP32 immediately
2. **Theoretical analysis first**: 1% efficiency is a red flag for fundamental design issues
3. **Configuration tuning is pointless** if base kernel is 100× too slow
4. **Use vendor libraries for complex SIMD**: AVX512 VNNI is hard to get right
5. **Test incrementally**: Should have validated VNNI code path independently

## Recommendation

**🛑 STOP using Integer GEMM kernel for production**

**Instead**:
1. Use OpenBLAS/MKL FP32 GEMM (335-1208 GFLOPS proven)
2. If quantization needed, use MKL's `cblas_gemm_s8u8s32` (industry-tested)
3. Focus V2 development on FP32/BF16 where we have working kernels

The custom INT8 kernel is a **failed experiment** - it would require complete rewrite to achieve acceptable performance.

---

**Status**: Configuration sweep completed successfully, but revealed kernel is fundamentally broken and **100-600× slower** than FP32 baseline. Not worth fixing - use vendor libraries instead.
