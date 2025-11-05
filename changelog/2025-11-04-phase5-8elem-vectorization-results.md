# Phase 5 JIT Kernel: 8-Element Vectorization Results

**Date**: November 4, 2025  
**Session**: NCU profiling and vectorization optimization  
**Author**: David Sanftenberg

## Executive Summary

**MIXED RESULTS**: 8-element vectorization improves performance by **+5.44%** (9.34 TFLOPS vs 8.86 baseline), but **global memory coalescing remains completely broken** (81% excessive sectors, identical to baseline).

## Performance Improvement ✅

**Parity Test Results** (`v2_test_phase5_parity --gtest_filter=Phase5ParityTest.Phase5A_Baseline_Config`):

| Configuration | TFLOPS | Improvement | Status |
|---------------|--------|-------------|--------|
| **Baseline (vectorize_a=1, scalar)** | 8.86 | - | ✅ |
| Failed (vectorize_a=4, float4) | 8.83 | 0% | ❌ |
| **New (vectorize_a=8, 2×float4)** | **9.34** | **+5.44%** | ✅ |

- **Test Status**: ✅ PASS (parity within ±15% tolerance)
- **JIT Config**: `p5_64_64_64_sub16_mma2x2_buf2_thr128_swz333_vec8`
- **Build Time**: ~13 seconds

## Memory Coalescing Issue Persists ❌

**NCU Profiling Results** (phase5_jit_vec8.ncu-rep, 70 kernel passes):

```
Section: Source Counters
OPT   Est. Speedup: 59.75%
      This kernel has uncoalesced global accesses resulting in a total of 
      8,644,608 excessive sectors (81% of the total 10,665,984 sectors)
      
Section: Memory Workload Analysis Tables
OPT   Est. Speedup: 46.21%
      The memory access pattern for global loads from L1TEX might not be optimal. 
      On average, only 5.8 of the 32 bytes transmitted per sector are utilized by 
      each thread. Check the Source Counters section for uncoalesced global loads.
```

**Comparison with Baseline**:

| Metric | Baseline (vec=1) | 8-Element (vec=8) | Change |
|--------|------------------|-------------------|---------|
| **Excessive Sectors** | 7,038,976 / 9,060,352 (78%) | 8,644,608 / 10,665,984 (81%) | **+3% WORSE** |
| **Bytes Used per Sector** | 6.9 / 32 (21.56%) | 5.8 / 32 (18.13%) | **-3.4% WORSE** |
| **Load Efficiency** | 21.56% | 18.13% | **DEGRADED** |
| **Shared Memory Bank Conflicts** | 4.2M / 11.2M (38%) | 4.2M / 11.2M (38%) | No change |

**CRITICAL**: Vectorization made memory coalescing **slightly worse**, not better!

## Root Cause Analysis

### Why 8-Element Vectorization Improved Performance Despite Worse Coalescing

**Hypothesis**: The 5.44% improvement comes from:
1. **Reduced instruction count**: 2 float4 loads + 8 half_t stores vs 8 scalar float loads + 8 half_t stores
2. **Better register utilization**: Vector loads may reduce register pressure
3. **Instruction-level parallelism**: Fewer memory instructions to schedule

**However**: The swizzle pattern still breaks coalescing, so bandwidth waste remains.

### The Real Problem: Global → Shared Memory Access Pattern

**From CuTe Blog** (https://leimao.github.io/blog/CuTe-Swizzle/):
> "MBase must equal log2(vector_size) for contiguous access"

**Our Implementation**:
- ✅ **MBase = 3** (Swizzle<3,3,3>)
- ✅ **vector_size = 8** (log2(8) = 3)
- ✅ **Mathematics**: MBase matches vector size perfectly

**What We Missed**: The blog post guarantees contiguous **shared memory writes**, NOT contiguous **global memory reads**!

**The Issue**:
```cpp
// Global A-matrix layout (IQ4_NL quantized, row-major)
// A[m * K + k] where K = 896 (not power-of-2!)

// Our access pattern (8-element vectorized):
for (int vec_idx = tid; vec_idx < VEC_GROUPS; vec_idx += THREADS_PER_BLOCK) {
    int linear_idx = vec_idx * VEC_WIDTH;
    int m = linear_idx / TILE_K;  // m = linear_idx / 64
    int k_base = (linear_idx % TILE_K) & ~7;  // k_base aligned to 8
    
    int gm = blockIdx.x * TILE_M + m;
    int gk = k_offset + k_base;
    
    // Global read (THIS IS WHERE COALESCING BREAKS):
    // Threads access A[gm * 896 + gk] with different 'gm' values
    // Stride between threads: NOT contiguous due to K=896
    float4 vec4_lo = *reinterpret_cast<const float4*>(&A[gm * K + gk]);
}
```

**Threads' Global Memory Access Pattern**:
- Thread 0: `A[0 * 896 + k]`
- Thread 1: `A[1 * 896 + k]` (offset by 896 elements = 3584 bytes)
- Thread 2: `A[2 * 896 + k]` (offset by 1792 elements = 7168 bytes)
- ...
- Thread 31: `A[31 * 896 + k]` (offset by 27776 elements)

**Result**: Each 128-byte cache line is accessed by ONLY 1 thread (128 bytes / 4 bytes per float = 32 floats, but threads stride by 896 floats). This is **catastrophically uncoalesced**.

### Why Swizzle Doesn't Help Global Reads

**Swizzle<3,3,3> optimizes**:
- ✅ **Shared memory writes**: Prevents bank conflicts when writing to smem
- ✅ **Shared memory reads**: Prevents bank conflicts when reading from smem
- ❌ **Global memory reads**: Does NOT change the global access pattern

**Swizzle operates on the DESTINATION (smem), not the SOURCE (gmem)!**

## The Fix We Actually Need

**Option 1: Change Memory Layout (Most Effective)** ⭐

```cpp
// Current: Row-major A[M][K]
// Access: A[m * K + k]  (stride-K between threads)

// Proposed: Column-major A[K][M] or blocked layout
// Access: A[k * M + m]  (stride-1 between threads - COALESCED!)

// OR: Use cuBLAS/cuTe's memory format expectations
```

**Expected Gain**: 45-60% speedup (eliminates 81% bandwidth waste)

**Option 2: Transpose/Reformat A-Matrix Before GEMM**

```cpp
// Transpose A: [M, K] → [K, M] in global memory
// Then threads access A_T[k][m] with contiguous 'm' - COALESCED!
```

**Expected Gain**: 30-50% speedup (transpose overhead, but better than current)

**Option 3: Use cuBLAS for IQ4_NL Dequant (If Supported)**

cuBLAS may have optimized kernels for quantized formats that handle coalescing properly.

## Implementation Status

**What We Changed** (`src/v2/kernels/cuda/CudaGemmKernelTemplatePhase5.h`):

1. **Added 8-Element Vectorization** (Lines 175-222 prologue, 356-403 prefetch):
   ```cpp
   #if VECTORIZE_A == 8
       // Load 2×float4 from global (256 bits total)
       float4 vec4_lo = *reinterpret_cast<const float4*>(&A[gm * K + gk]);
       float4 vec4_hi = *reinterpret_cast<const float4*>(&A[gm * K + gk + 4]);
       
       // Write 8×half_t to shared (contiguous for MBase=3)
       sA_write(m, k_base + 0) = half_t(__float2half(vec4_lo.x));
       // ... through k_base + 7
   #elif VECTORIZE_A == 4
       // Previous 4-element implementation
   #else
       // Scalar fallback
   #endif
   ```

2. **Updated Configuration** (`src/v2/kernels/cuda/CudaGemmConfigPhase5.h`):
   ```cpp
   // Changed default: vectorize_a = 1 → 8
   int vectorize_a = 8;  // Default to 8 (matches Swizzle<3,3,3> MBase=3)
   ```

3. **Documentation Updates**:
   - Template header: Added vectorize_a=8 option with MBase=3 note
   - Config header: Comprehensive documentation about MBase matching

**Files Modified**:
- `src/v2/kernels/cuda/CudaGemmKernelTemplatePhase5.h` (~100 lines added)
- `src/v2/kernels/cuda/CudaGemmConfigPhase5.h` (5 lines changed)
- `changelog/2025-11-04-phase5-vectorization-attempt-analysis.md` (updated with fix details)

## Lessons Learned

### Misconceptions Corrected

1. **WRONG**: "MBase = log2(vector_size) guarantees global memory coalescing"
   - **RIGHT**: MBase only affects **shared memory bank conflict avoidance**

2. **WRONG**: "Vectorized loads automatically improve coalescing"
   - **RIGHT**: Coalescing depends on **access pattern stride**, not load width

3. **WRONG**: "CuTe swizzle optimizes all memory accesses"
   - **RIGHT**: Swizzle optimizes **shared memory layout**, not global memory access

### What We Should Have Done

1. **Analyze global access pattern FIRST**: Check stride between threads in global memory
2. **Use NCU's "Global Access Pattern" view**: Visualize thread divergence
3. **Test with simpler layout**: Column-major or transposed to verify coalescing fix

## Next Steps

### Immediate Actions

1. **✅ Document findings** (this file)
2. **Test Option 2**: Transpose A-matrix before GEMM
   - Add A_T buffer in shared memory
   - Transpose during prologue load
   - Measure NCU coalescing metrics

3. **Test Option 1**: Change A-matrix layout to column-major
   - Requires model loader changes
   - Most impactful but larger refactor

### Long-Term Optimization

4. **Profile cuBLAS**: Compare against cuBLAS GEMM for IQ4_NL
5. **Investigate CuTe Copy Atoms**: Use CuTe's built-in copy operations (may handle coalescing)
6. **Consider Mixed Precision**: FP16 A-matrix (if dequant can happen earlier)

## Conclusion

**Success**: 8-element vectorization provides a modest **+5.44% speedup** without breaking correctness.

**Failure**: Global memory coalescing remains **81% inefficient** due to row-major stride-K access pattern.

**Path Forward**: Fixing coalescing (via transpose or layout change) could yield an additional **45-60% speedup** on top of current 9.34 TFLOPS → **13-15 TFLOPS target**.

**Key Insight**: CuTe swizzle optimizes shared memory, not global memory. We need to fix the **source** (global A-matrix layout), not just the **destination** (shared memory).

---

**Files Generated This Session**:
- `changelog/2025-11-04-phase5-jit-ncu-profiling-analysis.md` (initial profiling)
- `changelog/2025-11-04-phase5-vectorization-attempt-analysis.md` (CuTe insights)
- `changelog/2025-11-04-phase5-8elem-vectorization-results.md` (this file - final summary)

**NCU Reports**:
- `build_v2_release/tests/v2/phase5_jit_profile.ncu-rep` (baseline scalar)
- `build_v2_release/tests/v2/phase5_jit_vectorized.ncu-rep` (4-element, failed)
- `build_v2_release/tests/v2/phase5_jit_vec8.ncu-rep` (8-element, mixed results)
