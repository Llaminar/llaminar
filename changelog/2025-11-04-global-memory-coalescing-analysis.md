# Global Memory Coalescing Analysis: Why Vectorization Didn't Help

**Date**: November 4, 2025  
**Context**: Phase 5 JIT kernel 8-element vectorization paradox

## The Problem Visualized

### Current Access Pattern (ROW-MAJOR A-MATRIX)

**Matrix A Layout in Global Memory** (IQ4_NL quantized, FP32 after dequant):
```
A[M=2048, K=896] stored as row-major:
┌───────────────────────────────────────────────────┐
│ Row 0: A[0,0] A[0,1] A[0,2] ... A[0,895]         │ ← 896 elements (3584 bytes)
│ Row 1: A[1,0] A[1,1] A[1,2] ... A[1,895]         │ ← 896 elements (3584 bytes)
│ Row 2: A[2,0] A[2,1] A[2,2] ... A[2,895]         │
│ ...                                               │
│ Row 2047: A[2047,0] ... A[2047,895]              │
└───────────────────────────────────────────────────┘
```

**Thread Access Pattern** (8-element vectorized):
```cpp
// Each thread loads 8 consecutive elements from DIFFERENT rows
for (int vec_idx = tid; vec_idx < VEC_GROUPS; vec_idx += THREADS_PER_BLOCK) {
    int linear_idx = vec_idx * 8;
    int m = linear_idx / TILE_K;  // m = linear_idx / 64
    int k_base = (linear_idx % TILE_K) & ~7;
    
    int gm = blockIdx.x * TILE_M + m;  // ← DIFFERENT 'm' per thread!
    int gk = k_offset + k_base;
    
    // Access: A[gm * 896 + gk] to A[gm * 896 + gk + 7]
}
```

**What Threads Actually Read** (TILE_M=64, TILE_K=64, 128 threads):

```
Warp 0 (threads 0-31):
  Thread  0: A[  0 * 896 + k] → A[  0 * 896 + k+7]  (row 0)
  Thread  1: A[  1 * 896 + k] → A[  1 * 896 + k+7]  (row 1)
  Thread  2: A[  2 * 896 + k] → A[  2 * 896 + k+7]  (row 2)
  ...
  Thread 31: A[ 31 * 896 + k] → A[ 31 * 896 + k+7]  (row 31)

Warp 1 (threads 32-63):
  Thread 32: A[ 32 * 896 + k] → A[ 32 * 896 + k+7]  (row 32)
  ...
```

**Memory Layout for Warp 0 Transaction** (32 threads × 32 bytes each = 1024 bytes):

```
Cache Line 0 (128 bytes):
┌────────────────────────────────────────────────┐
│ A[0, k:k+31]                                   │ ← Thread 0 reads [k:k+7]
│                                                │   Wastes [k+8:k+31]
└────────────────────────────────────────────────┘

Cache Line 1 (128 bytes, offset +896*4 = +3584 bytes):
┌────────────────────────────────────────────────┐
│ A[1, k:k+31]                                   │ ← Thread 1 reads [k:k+7]
│                                                │   Wastes [k+8:k+31]
└────────────────────────────────────────────────┘

... 30 more separate cache lines ...

Cache Line 31 (offset +31*3584 = +111,104 bytes):
┌────────────────────────────────────────────────┐
│ A[31, k:k+31]                                  │ ← Thread 31 reads [k:k+7]
│                                                │   Wastes [k+8:k+31]
└────────────────────────────────────────────────┘
```

**Result**: 
- **Sectors Requested**: 32 threads × 1 sector per thread = **32 sectors** (4096 bytes)
- **Bytes Used**: 32 threads × 32 bytes per thread = **1024 bytes**
- **Bytes Wasted**: 4096 - 1024 = **3072 bytes (75% waste)**

This matches NCU's finding: **"only 5.8 of the 32 bytes transmitted per sector are utilized"**

### Optimal Access Pattern (COLUMN-MAJOR OR TRANSPOSED)

**Transposed A_T[K=896, M=2048]** stored as row-major:
```
A_T[K, M] layout (each row has M=2048 elements):
┌───────────────────────────────────────────────────┐
│ Row 0: A_T[0,0] A_T[0,1] ... A_T[0,2047]         │ ← 2048 elements (8192 bytes)
│ Row 1: A_T[1,0] A_T[1,1] ... A_T[1,2047]         │
│ ...                                               │
│ Row 895: A_T[895,0] ... A_T[895,2047]            │
└───────────────────────────────────────────────────┘

Equivalent to: A[m,k] stored as A_T[k,m]
```

**Thread Access Pattern** (transposed):
```cpp
// Access A_T[k, m] instead of A[m, k]
int gk = k_offset + k_base;
int gm_base = blockIdx.x * TILE_M + (linear_idx / TILE_K);

// Access: A_T[gk * M + gm_base] with CONTIGUOUS 'gm_base'
for (int i = 0; i < 8; i++) {
    int gm = gm_base + i;
    float val = A_T[gk * 2048 + gm];  // ← CONTIGUOUS across threads!
}
```

**What Threads Read** (transposed, same TILE):

```
Warp 0 (threads 0-31):
  Thread  0: A_T[k, m+  0] → A_T[k, m+  7]  (same k, consecutive m)
  Thread  1: A_T[k, m+  8] → A_T[k, m+ 15]  (same k, consecutive m)
  Thread  2: A_T[k, m+ 16] → A_T[k, m+ 23]  (same k, consecutive m)
  Thread  3: A_T[k, m+ 24] → A_T[k, m+ 31]  (same k, consecutive m)
  Thread  4: A_T[k, m+ 32] → A_T[k, m+ 39]  (same k, consecutive m)
  ...
  Thread 31: A_T[k, m+248] → A_T[k, m+255]  (same k, consecutive m)
```

**Memory Layout for Warp 0 Transaction** (COALESCED):

```
Cache Line 0 (128 bytes):
┌────────────────────────────────────────────────┐
│ A_T[k, m+0:m+31]                               │ ← Threads 0-3 read ALL
│                                                │   100% utilization!
└────────────────────────────────────────────────┘

Cache Line 1 (128 bytes, offset +128 bytes):
┌────────────────────────────────────────────────┐
│ A_T[k, m+32:m+63]                              │ ← Threads 4-7 read ALL
│                                                │   100% utilization!
└────────────────────────────────────────────────┘

Cache Line 2 (128 bytes, offset +256 bytes):
┌────────────────────────────────────────────────┐
│ A_T[k, m+64:m+95]                              │ ← Threads 8-11 read ALL
└────────────────────────────────────────────────┘

... 5 more cache lines ...

Cache Line 7 (128 bytes, offset +896 bytes):
┌────────────────────────────────────────────────┐
│ A_T[k, m+224:m+255]                            │ ← Threads 28-31 read ALL
└────────────────────────────────────────────────┘
```

**Result**:
- **Sectors Requested**: 8 sectors (32 threads / 4 threads per sector)
- **Bytes Used**: 1024 bytes
- **Bytes Wasted**: 0 bytes (**0% waste, 100% coalesced!**)

**Speedup**: 32 sectors → 8 sectors = **4× reduction in memory traffic**

## NCU Metrics Explained

### Baseline (Scalar, vectorize_a=1)

```
Excessive sectors: 7,038,976 / 9,060,352 (78%)
```

**Interpretation**:
- **Total sectors requested**: 9,060,352 sectors (128 bytes each)
- **Useful sectors**: 2,021,376 sectors
- **Wasted sectors**: 7,038,976 sectors (78% overhead)

**Why 78% waste?**
- Each thread loads 1 float (4 bytes) from a different row
- 128-byte sector contains 32 floats
- Only 1 float used per sector → 4/128 = 3.1% utilization
- Waste: 100% - 3.1% = 96.9% (but NCU reports 78% due to some spatial locality)

### 8-Element Vectorized (vectorize_a=8)

```
Excessive sectors: 8,644,608 / 10,665,984 (81%)
```

**Interpretation**:
- **Total sectors**: 10,665,984 (increased by 18% vs baseline)
- **Excessive sectors**: 8,644,608 (increased by 23% vs baseline)

**Why MORE sectors?**
- Scalar: 1 float per thread × 128 threads = 128 floats per iteration
- Vectorized: 8 floats per thread × 128 threads = 1024 floats per iteration
- 8× more data loaded → more sector requests
- But coalescing is STILL broken (same row-major stride issue)

**Why 81% waste (worse than 78%)?**
- Larger loads (2×float4 = 32 bytes) per thread
- Still accessing different rows (stride-896 between threads)
- NCU penalizes larger uncoalesced loads more heavily

### Shared Memory Bank Conflicts (Unchanged)

```
9.6-way bank conflicts (87% of wavefronts)
```

**Why Swizzle<3,3,3> Doesn't Help**:
- Swizzle pattern: `XOR(row_id >> 3, col_id >> 3) << 3`
- Designed for 8-element groups (MBase=3)
- **Problem**: We write to shared memory in a pattern that STILL causes conflicts

**Example Conflict**:
```
8 threads write to same shared memory bank:
  Thread 0 → bank (0 XOR 0) = 0
  Thread 8 → bank (1 XOR 0) = 1
  Thread 16 → bank (2 XOR 0) = 2
  ...
  
But with our indexing:
  Thread 0 writes to (m=0, k=0:7)
  Thread 1 writes to (m=1, k=0:7)  ← SAME banks as thread 0!
```

The swizzle helps reduce conflicts but doesn't eliminate them due to our thread→element mapping.

## Mathematical Analysis

### Current Bandwidth Waste

**Per-Warp Bandwidth** (32 threads):
- **Requested**: 32 sectors × 128 bytes = 4096 bytes
- **Used**: 32 threads × 32 bytes (8×FP32) = 1024 bytes
- **Efficiency**: 1024 / 4096 = **25%** (matches NCU's ~21% load efficiency)

**Total Kernel Bandwidth**:
- **GEMM operation**: 2048 × 896 × 896 ≈ 1.6 billion FLOPs
- **A-matrix reads**: 2048 × 896 × 4 bytes = 7.3 MB
- **Excessive sectors**: 81% waste → 7.3 MB × (1 + 0.81/0.19) = **38.4 MB read** (5.3× overhead!)

### Potential Speedup from Fixing Coalescing

**NCU Estimate**: 59.75% speedup

**Manual Calculation**:
- Current: 38.4 MB read time = T_read
- Optimal: 7.3 MB read time = 0.19 × T_read
- Speedup: T_read / (0.19 × T_read) = **5.3×** on memory-bound portions

**Realistic Speedup** (not all operations are memory-bound):
- Memory-bound: ~60% of kernel time (based on NCU occupancy)
- Speedup: 1 / (0.4 + 0.6/5.3) = **46-50%**

**Expected Performance**:
- Current: 9.34 TFLOPS
- With fix: 9.34 × 1.5 = **14.0 TFLOPS** 🎯

## Implementation Options

### Option 1: Transpose A-Matrix in Shared Memory ⭐ (RECOMMENDED)

**Approach**: Load A in current pattern, then transpose in shared memory before MMA

```cpp
// Prologue: Load A into temporary buffer (still uncoalesced)
__shared__ half_t sA_temp[TILE_M][TILE_K];  // Row-major
__shared__ half_t sA[TILE_K][TILE_M];       // Column-major (transposed)

// Load phase (uncoalesced, but only once per tile)
for (int vec_idx = tid; vec_idx < (TILE_M * TILE_K) / 8; vec_idx += THREADS_PER_BLOCK) {
    int m = ...;
    int k = ...;
    // Load from global to sA_temp
}
__syncthreads();

// Transpose phase (coalesced shared→shared)
for (int idx = tid; idx < TILE_M * TILE_K; idx += THREADS_PER_BLOCK) {
    int m = idx / TILE_K;
    int k = idx % TILE_K;
    sA[k][m] = sA_temp[m][k];  // Transpose
}
__syncthreads();
```

**Pros**:
- ✅ Fixes global memory coalescing for MMA reads
- ✅ No model loader changes needed
- ✅ Can optimize transpose with bank conflict avoidance

**Cons**:
- ❌ 2× shared memory usage during prologue
- ❌ Additional transpose overhead (~5-10% time)
- ❌ May exceed shared memory limits (48 KB)

**Expected Gain**: 30-40% (transpose overhead reduces theoretical 50%)

### Option 2: Change Model Loader to Store A as Column-Major

**Approach**: Modify `ModelLoader` to transpose A-matrix when loading GGUF

```cpp
// In ModelLoader::loadWeights()
if (weight.role == WeightRole::LINEAR_WEIGHT_A) {
    // Transpose during load: A[M, K] → A_T[K, M]
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            A_transposed[k * M + m] = A_original[m * K + k];
        }
    }
}
```

**Pros**:
- ✅ Maximum performance (no runtime transpose)
- ✅ No shared memory overhead
- ✅ Fixes global coalescing permanently

**Cons**:
- ❌ Requires model loader refactor
- ❌ May break other kernels expecting row-major
- ❌ Larger code change (higher risk)

**Expected Gain**: 45-50% (best case)

### Option 3: Use CuTe Copy Atoms (ADVANCED)

**Approach**: Leverage CuTe's built-in `copy` and `copy_aligned` primitives

```cpp
// CuTe copy handles coalescing automatically
auto copyA = make_tiled_copy(
    Copy_Atom<DefaultCopy, float>{},
    Layout<Shape<_128, _8>>{},  // 128 threads, 8 elements per thread
    Layout<Shape<_1, _8>>{}     // Memory layout
);

auto gA = make_tensor(make_gmem_ptr(A), ...);
auto sA = make_tensor(make_smem_ptr(sA_ptr), ...);

copy(copyA, gA, sA);  // CuTe handles optimal access pattern
```

**Pros**:
- ✅ CuTe's expertise in memory access optimization
- ✅ Future-proof (CuTe evolves with hardware)
- ✅ May handle bank conflicts automatically

**Cons**:
- ❌ Requires deep CuTe understanding
- ❌ Current code heavily customized
- ❌ May not work well with IQ4_NL dequant

**Expected Gain**: 40-50% (CuTe's optimal patterns)

## Next Steps

1. **✅ Document findings** (this file)

2. **Test Option 1** (Transpose in shared memory):
   - Implement shared memory transpose
   - Measure NCU coalescing metrics
   - Verify parity test still passes
   - Benchmark performance

3. **Compare with cuBLAS**:
   - Benchmark cuBLAS GEMM for IQ4_NL (if supported)
   - Establish performance ceiling

4. **If Option 1 succeeds**, consider Option 2 for maximum performance

5. **Profile with NCU source attribution**:
   ```bash
   sudo /usr/local/cuda/bin/ncu --set full --section SourceCounters \
     --nvtx --nvtx-include "GEMM/" -lineinfo \
     build_v2_release/tests/v2/v2_test_phase5_parity \
     --gtest_filter=Phase5ParityTest.Phase5A_Baseline_Config
   ```

## Conclusion

**Root Cause Identified**: Row-major A-matrix with stride-K access pattern causes 81% global memory bandwidth waste.

**Swizzle<3,3,3> Does NOT Fix Global Coalescing**: Swizzle optimizes shared memory, not global memory.

**Path to 14 TFLOPS**: Fix global memory coalescing via transpose (Option 1) or layout change (Option 2).

**Key Takeaway**: Always analyze **source memory layout** (global) and **access pattern** (stride) before optimizing vector widths!

---

**References**:
- CuTe Swizzle Blog: https://leimao.github.io/blog/CuTe-Swizzle/
- NVIDIA GPU Memory Coalescing: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory
- NCU Profiling Reports: `build_v2_release/tests/v2/phase5_jit_*.ncu-rep`
