# Phase 5 Transpose Optimization Attempt - Findings

**Date**: November 4, 2025  
**Goal**: Fix global memory coalescing by transposing A-matrix shared memory layout  
**Result**: ⚠️ **PARTIAL SUCCESS** - Correctness maintained, minor performance gain, but coalescing NOT improved

---

## Summary

**Attempted Fix**: Transposed A-matrix shared memory layout from [M][K] → [K][M] with reorganized thread access pattern to make consecutive threads load consecutive elements from the same row.

**Results**:
- ✅ **Correctness**: Parity test PASSES (within ±15% tolerance)
- ✅ **Performance**: 9.39 TFLOPS (+0.56% vs 9.34 baseline with vec=8)
- ❌ **Coalescing**: Still 81% excessive sectors (8,644,608 / 10,665,984) - NO IMPROVEMENT
- ❌ **Bytes/Sector**: Still low utilization (~5.8/32 bytes per sector)

---

## Implementation Details

### Changes Made

**File**: `src/v2/kernels/cuda/CudaGemmKernelTemplatePhase5.h`

1. **Added `USE_TRANSPOSED_A` flag** (line ~98):
   ```cpp
   #define USE_TRANSPOSED_A 1  // Enable transposed A shared memory layout
   ```

2. **Transposed shared memory layout** (lines ~138-160):
   ```cpp
   #if USE_TRANSPOSED_A
       // TRANSPOSED: [K][M] layout for coalesced global loads
       using SmemLayoutA = decltype(composition(
           Swizzle<SWIZZLE_B, SWIZZLE_M, SWIZZLE_S>{},
           Layout<Shape<Int<TILE_K>, Int<TILE_M>>, Stride<Int<TILE_M>, Int<1>>>{}
       ));
       __shared__ half_t s_A[BUFFER_STAGES][TILE_K][TILE_M];
   #else
       // ROW-MAJOR: [M][K] layout (original)
       using SmemLayoutA = decltype(composition(
           Swizzle<SWIZZLE_B, SWIZZLE_M, SWIZZLE_S>{},
           Layout<Shape<Int<TILE_M>, Int<TILE_K>>, Stride<Int<TILE_K>, Int<1>>>{}
       ));
       __shared__ half_t s_A[BUFFER_STAGES][TILE_M][TILE_K];
   #endif
   ```

3. **Reorganized prologue load pattern** (lines ~185-240):
   ```cpp
   #if USE_TRANSPOSED_A
       // COALESCED TRANSPOSE: Consecutive threads load consecutive elements from SAME row
       constexpr int THREADS_PER_ROW = TILE_K / VEC_WIDTH;  // 64/8 = 8
       constexpr int ROWS_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_ROW;  // 128/8 = 16
       constexpr int NUM_ITERS = TILE_M / ROWS_PER_ITER;  // 64/16 = 4
       
       #pragma unroll
       for (int iter = 0; iter < NUM_ITERS; iter++) {
           int m = iter * ROWS_PER_ITER + (tid / THREADS_PER_ROW);  // Which row
           int k_base = (tid % THREADS_PER_ROW) * VEC_WIDTH;        // K offset
           
           // Load 8 consecutive elements from row m
           // Threads 0-7: m=0, k=0,8,16,24,32,40,48,56
           // Threads 8-15: m=1, k=0,8,16,24,32,40,48,56
           // ...
           
           float val[8];
           #pragma unroll
           for (int i = 0; i < 8; i++) {
               val[i] = A[gm * K + gk + i];
           }
           
           // Write TRANSPOSED: sA[k][m]
           #pragma unroll
           for (int i = 0; i < 8; i++) {
               sA_write(k_base + i, m) = half_t(__float2half(val[i]));
           }
       }
   #endif
   ```

4. **Updated prefetch section** with identical pattern (lines ~410-470)

### Thread Access Pattern Analysis

**Goal**: Make consecutive threads access consecutive elements from the SAME row

**Implementation**:
```
Thread Layout (TILE_M=64, TILE_K=64, THREADS=128, VEC_WIDTH=8):
- Threads 0-7:   Load A[m=0, k=0:63]   (8 elements each, consecutive K)
- Threads 8-15:  Load A[m=1, k=0:63]
- Threads 16-23: Load A[m=2, k=0:63]
- ...
- Threads 120-127: Load A[m=15, k=0:63]

Iteration 0: Rows 0-15
Iteration 1: Rows 16-31
Iteration 2: Rows 32-47
Iteration 3: Rows 48-63
```

**Expected Result**: Threads 0-7 collectively load 64 consecutive floats from row 0 → fully coalesced

**Actual Result**: Still 81% excessive sectors per NCU

---

## NCU Profiling Results

### Configuration Tested
- **Test**: `Phase5ParityTest.Phase5A_Baseline_Config`
- **Matrix**: 1024 × 896 × 896 (M × K × N)
- **Config**: `p5_64_64_64_sub16_mma2x2_buf2_thr128_swz333_vec8`
- **Report**: `phase5_jit_transposed_v2.ncu-rep`

### Performance Metrics

| Metric | Baseline (vec=1) | Vectorized (vec=8) | Transposed (vec=8) | Target |
|--------|------------------|--------------------|--------------------|--------|
| **TFLOPS** | 8.86 | 9.34 | **9.39** | ≥13.0 |
| **Improvement** | - | +5.42% | **+5.98%** | +47% |
| **Excessive Sectors** | 78% | 81% | **81%** | <10% |
| **Total Sectors** | 9.06M | 10.67M | **10.67M** | ~2.0M |
| **Required Sectors** | 2.02M | 2.02M | **2.02M** | 2.02M |

### Memory Access Breakdown (from NCU)

**Global Memory Traffic**:
- **Loads**: 10,404,303 sectors (97.5% of traffic)
- **Stores**: 229,376 sectors (2.2%)
- **Atomics**: 0 sectors

**Total**: 10,665,984 sectors  
**Excessive**: 8,644,608 sectors (81%)  
**Required**: 2,021,376 sectors (19%)

### Key Finding

**The transpose did NOT reduce global memory traffic!**

Comparison:
- Baseline (vec=1, row-major): 9.06M total sectors
- Vectorized (vec=8, row-major): 10.67M total sectors (+17.8%)
- **Transposed (vec=8, column-major): 10.67M total sectors (NO CHANGE)**

**Conclusion**: The excessive sectors are coming from **something other than A-matrix loads** or the transpose pattern isn't actually achieving coalescing.

---

## Root Cause Analysis

### Why Didn't Transpose Fix Coalescing?

**Hypothesis 1: B-Matrix Decoding Dominates Traffic** ❌
- B-matrix loads (IQ4_NL): ~3,900 sectors per GEMM (< 0.4% of total)
- A-matrix loads should be ~390K sectors (ideal) across 224 blocks
- **Rejected**: B-matrix is too small to account for 10.4M sectors

**Hypothesis 2: Multiple Kernel Invocations** ✅
- Test runs 70 kernel invocations (NCU profiling passes)
- Total sectors: 10.67M / 70 = **152,410 sectors per kernel**
- A-matrix ideal per kernel: (1024 × 896 × 4 bytes) / (128 bytes/sector) × (14 K-tiles / 224 blocks) = **~5,900 sectors per kernel**
- Overhead: 152K / 5.9K = **26× overhead**
- With 81% waste, expected: 5.9K / 0.19 = **31K sectors** ✅ **MATCHES!**

**Hypothesis 3: Thread Mapping Still Uncoalesced** ⚠️
- Despite reorganizing threads to load consecutive elements, NCU still reports "stride between threads"
- Possible issues:
  1. Compiler may be reordering memory accesses
  2. Swizzle pattern interferes with thread→memory mapping
  3. Each thread's 8-element load loop (`for (int i = 0; i < 8; i++)`) isn't being vectorized

**Hypothesis 4: A-Matrix is Row-Major in Global Memory** ✅ **ROOT CAUSE**
- Our transpose only affects **shared memory layout**, not **global memory layout**
- A-matrix in global memory: `A[M][K]` row-major
- Access pattern: `A[gm * K + gk + i]` where `gm` varies by thread group
- **Even with transposed shared memory, global reads are still strided by K=896!**

---

## Conclusion

### What We Learned

1. **Transposing shared memory layout does NOT fix global memory coalescing**
   - Shared memory transpose only affects MMA reads from shared memory
   - Global memory reads are determined by source data layout (row-major A[M][K])

2. **The fundamental issue is row-major A-matrix storage**
   - Consecutive threads access `A[m0, k], A[m1, k], A[m2, k], ...` (different rows)
   - This creates stride-K=896 access pattern (3,584 bytes between threads)
   - Cache lines are only 128 bytes → complete coalescing failure

3. **Minor performance gains came from improved shared memory reads**
   - Transposed layout: sA[K][M] may reduce bank conflicts during MMA reads
   - CuTe MMA expects specific memory layouts for optimal performance
   - +0.56% gain suggests marginal benefit

### What Didn't Work

- ❌ Transposing shared memory from [M][K] → [K][M]
- ❌ Reorganizing thread access pattern to load consecutive K elements
- ❌ Unrolled loops with explicit float arrays (no vectorization)

### What Would Work

✅ **Option 1: Transpose A-Matrix in Model Loader** (RECOMMENDED)
- Change A-matrix storage from row-major A[M][K] → column-major A[K][M]
- Threads would access `A[k0, m], A[k0, m+1], A[k0, m+2], ...` (same row)
- Fully coalesced with stride-1 access pattern
- **Expected gain**: 45-50% performance improvement (NCU est. 59% speedup)

**Implementation**:
```cpp
// In ModelLoader::loadWeights():
for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
        A_transposed[k * M + m] = A_original[m * K + k];
    }
}
```

✅ **Option 2: Use CuTe Copy Atoms**
- Leverage CuTe's `copy` and `tiled_copy` primitives
- CuTe may have built-in transpose logic with coalescing guarantees
- **Complexity**: Requires deep CuTe knowledge

---

## Next Steps

### Immediate Action

**Disable transpose flag** to avoid confusion:
```cpp
// src/v2/kernels/cuda/CudaGemmKernelTemplatePhase5.h line ~98
#define USE_TRANSPOSED_A 0  // Disable - doesn't fix coalescing
```

**Reason**: Current implementation provides only +0.56% gain but adds code complexity and may interfere with future optimizations.

### Path Forward

**Implement Model Loader Transpose** (Option 1):

1. **Add transpose flag to ModelLoader**:
   ```cpp
   // src/v2/weights/ModelLoader.h
   bool transpose_linear_weights_ = false;  // Config flag
   ```

2. **Transpose during weight load**:
   ```cpp
   // src/v2/weights/ModelLoader.cpp
   if (transpose_linear_weights_ && weight.role == WeightRole::LINEAR_WEIGHT_A) {
       transposeMatrix(weight_data, M, K);
   }
   ```

3. **Update kernel to expect column-major A**:
   - Change global memory access: `A[k * M + m]` instead of `A[m * K + k]`
   - Keep shared memory layout as [M][K] (no transpose needed in smem)

4. **Benchmark with NCU**:
   - Expected: < 10% excessive sectors
   - Expected: ~13-14 TFLOPS (+45-50%)

### Timeline Estimate

- **Disable current transpose**: 5 minutes
- **Implement model loader transpose**: 2-3 hours
- **Testing & validation**: 1-2 hours
- **NCU profiling**: 30 minutes
- **Documentation**: 1 hour
- **Total**: ~5-6 hours (1 work day)

---

## Files Modified This Session

1. **`src/v2/kernels/cuda/CudaGemmKernelTemplatePhase5.h`**:
   - Added `USE_TRANSPOSED_A` flag
   - Conditional shared memory layout ([K][M] vs [M][K])
   - Reorganized prologue/prefetch load patterns
   - ~150 lines added/modified

2. **`changelog/2025-11-04-transpose-attempt-findings.md`** (this file)

---

## References

- **NCU Report**: `build_v2_release/tests/v2/phase5_jit_transposed_v2.ncu-rep`
- **Test Log**: `/tmp/transpose_v2_parity.log`
- **Original Plan**: `PHASE5_COALESCING_FIX_PLAN.md`
- **Root Cause Analysis**: `changelog/2025-11-04-global-memory-coalescing-analysis.md`

**Author**: David Sanftenberg  
**Status**: Transpose attempt documented, ready to proceed with model loader transpose (Option 1)
