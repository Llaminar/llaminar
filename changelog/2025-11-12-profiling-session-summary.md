# INT8 GEMM Profiling Session Summary

**Date**: November 12, 2025  
**Goal**: Identify bottleneck preventing progress from 102 GOPS → 1000 GOPS target  
**Method**: `perf stat` performance counter analysis

---

## Key Findings

### ✅ Profiling Complete - Bottleneck Identified

**Performance counters (M=128, N=3584, K=896 workload)**:
```
IPC:              1.45 instructions/cycle    ← Good compute utilization
L1 cache misses:  5.15%                      ← NOT memory-bound
LLC cache misses: 0.43%                      ← Excellent cache behavior
Branch mispred:   0.49%                      ← Predictable loops
```

**Single-threaded vs Multi-threaded**:
- Single-thread: 103.6 GOPS
- Multi-thread: 102.4 GOPS
- **Conclusion**: OpenMP provides NO benefit for M=128 workload!

**Root cause**: M=128 with MR=16 → only 8 iterations total
- Not enough work to parallelize effectively
- Thread synchronization overhead ≈ computation benefit
- Need larger M (1024-2048) for meaningful parallelization

---

## OneDNN Investigation Results

**Question**: Can we use OneDNN's s8s8s32 GEMM for Q8_0×Q8_0 instead of custom microkernel?

**Answer**: ❌ **No - Fundamental incompatibility**

**Why**:
- OneDNN API: `C = alpha * (A - ao) × (B - bo) + beta * C + co`
  - Single `alpha` scale for entire matrix
  - Designed for uniform quantization (neural networks)

- Q8_0 format: `C[i,j] = Σ_k (scale[i,k/32] × scale[j,k/32]) × (A_q[i,k] × B_q[j,k])`
  - Different scale product for each 32-element block
  - Cannot express with single `alpha` parameter

**Conclusion**: Custom microkernel with fused scale handling is necessary.

---

## Performance Analysis

### Current State
- **102.4 GOPS** (16×16 microkernel, M=128, Release mode)
- **Target**: 1000 GOPS (9.8× gap)

### Why NOT Memory-Bound
- L1 miss rate: 5.15% (good, <10% is ideal)
- LLC miss rate: 0.43% (excellent, <1% is great)
- Memory bandwidth: 600 MB/s measured (system capable of 100 GB/s)
- **Conclusion**: Plenty of headroom

### Why NOT Compute-Bound
- IPC: 1.45 (theoretical max ~4-6 for AVX512)
- Single-core theoretical peak: 320 GOPS (2× dpbusd @ 2.5 GHz)
- Achieving: 3.6 GOPS/core = **1.1% of single-core peak**
- **Conclusion**: Massive optimization opportunity

### Actual Bottleneck
**For M=128 workload**: Insufficient parallelization granularity
- 8 iterations × 103 MFLOPs/iter = not enough work per thread
- Single-threaded (103.6 GOPS) ≈ Multi-threaded (102.4 GOPS)

**For larger workloads (M=1024-2048)**: Unknown - needs testing!
- 64-128 iterations → better thread utilization expected
- Hypothesis: Will see better multi-core scaling

---

## Prioritized Action Plan

### Priority 1: Test Larger Workloads (15 minutes)
**Why**: M=128 is too small for effective parallelization

**Tests**:
1. M=1024, N=3584, K=896 (realistic FFN layer)
2. M=2048, N=3584, K=896 (large FFN layer)
3. Compare single-thread vs multi-thread

**Expected**: 5-10× multi-thread speedup on larger M

**Implementation**:
```cpp
// Add to Test__ParameterizedInt8Gemm.cpp
TEST(Test__ParameterizedInt8Gemm, DISABLED_Benchmark_ExtraLargeMatrix) {
    const int M = 2048, N = 3584, K = 896;
    // ... benchmark code ...
}
```

### Priority 2: K-Loop Unrolling (1-2 hours)
**Why**: Reduce loop overhead, improve instruction scheduling

**Current**: 4× unroll (process 16 K-elements per iteration)

**Target**: 8× unroll (process 32 K-elements)

**Expected**: 5-10% improvement

**Implementation**:
```cpp
// In microkernel, change:
for (int k = 0; k < K; k += 16) { ... }
// To:
for (int k = 0; k < K; k += 32) { 
    // Process 8 blocks instead of 4
}
```

### Priority 3: Vectorized Compensation (2-3 hours)
**Why**: Reduce 256 store/load pairs per microkernel invocation

**Current**: Scalar per-row compensation
```cpp
for (int r = 0; r < 16; ++r) {
    int32_t comp = row_sums[r] * 128;
    // Apply to row r
}
```

**Target**: SIMD compensation in registers
```cpp
__m512i comp_vec = _mm512_mullo_epi32(row_sums_vec, _mm512_set1_epi32(128));
// Apply using shuffle/permute before storing C
```

**Expected**: 10-15% improvement

**Alternative**: Precompute column sums (OneDNN strategy)
```cpp
// Before K-loop:
int32_t col_sums[N];
for (int j = 0; j < N; ++j) col_sums[j] = sum_k B[k,j];

// After K-loop:
C[i,j] -= row_sum[i] * col_sums[j] * 128;
```

### Priority 4: Prefetching (30 minutes)
**Why**: Hide memory latency (though cache miss rate already low)

**Implementation**:
```cpp
_mm_prefetch((const char*)(B_packed + kb + 2), _MM_HINT_T0);
_mm_prefetch((const char*)(A + (i+MR)*K), _MM_HINT_T0);
```

**Expected**: ~5% improvement

### Priority 5: Assembly Analysis (1 hour)
**Why**: Check for register spilling, verify dpbusd usage

**Method**:
```bash
objdump -d build_v2_release/tests/v2/v2_test_parameterized_int8_gemm | \
  grep -A 200 "microkernel" > microkernel_asm.txt
```

**Look for**:
- Register spills to stack
- dpbusd instruction count
- Unnecessary memory operations

---

## Realistic Performance Targets

### Optimistic Scenario
| Optimization | Performance | Cumulative |
|--------------|-------------|------------|
| Baseline (M=128) | 102 GOPS | 1.0× |
| Larger M (M=2048, multi-core) | 500 GOPS | 4.9× |
| K-loop 8× unroll | 550 GOPS | 5.4× |
| Vectorized compensation | 633 GOPS | 6.2× |
| Prefetching | 665 GOPS | 6.5× |
| Assembly tuning | 730 GOPS | 7.2× |

**Realistic target**: **600-800 GOPS** (with all optimizations)

### Conservative Scenario
- M=2048 multi-core: 300 GOPS (3× baseline)
- Microkernel opts: 400 GOPS (1.3× improvement)

**Conservative target**: **400-500 GOPS**

### Why 1000 GOPS is Hard
**Theoretical single-core peak**: 320 GOPS
- Need 3-4 cores worth of work
- Multi-core scaling limited by:
  - Memory bandwidth (not currently an issue)
  - Cache coherence traffic
  - Thread synchronization
  - Load balancing

**Path to 1000 GOPS**:
1. ✅ Larger workloads (M=2048+): 4-5× from parallelization
2. ✅ Microkernel optimization: 1.5× from unrolling + compensation
3. ❓ Multi-socket NUMA optimization: 1.2-1.5×
4. ❓ Further microkernel tuning: 1.1-1.2×

**Verdict**: 1000 GOPS achievable but requires all optimizations + large M

---

## Summary

### Completed Work
- ✅ Implemented parameterized microkernels (48×8, 32×8, 16×8, 16×16)
- ✅ Validated correctness (all sizes exact match vs reference)
- ✅ Profiled with perf (identified NOT memory-bound, NOT compute-saturated)
- ✅ Investigated OneDNN (confirmed incompatible with Q8_0 per-block scales)
- ✅ Single-thread vs multi-thread analysis (OpenMP ineffective for M=128)

### Key Insights
1. **16×16 is optimal** for current architecture (wider N > taller M)
2. **NOT memory-bound**: 5.15% L1 miss rate, plenty of bandwidth
3. **NOT branch limited**: 0.49% misprediction rate
4. **M=128 too small**: Need M=1024-2048 for effective parallelization
5. **OneDNN incompatible**: Must use custom microkernel for Q8_0

### Next Session Focus
1. **Immediate**: Test M=1024 and M=2048 workloads
2. **High priority**: K-loop 8× unrolling
3. **Medium priority**: Vectorized compensation
4. **Validation**: Re-profile after each optimization

### Performance Gap
- Current: 102 GOPS (M=128)
- Target: 1000 GOPS
- Gap: 9.8×
- Achievable with optimizations: 6-8× (600-800 GOPS)
- Remaining gap: Requires larger problem sizes or further microkernel innovation
