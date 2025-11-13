# INT8 GEMM Performance Profiling Results

**Date**: November 12, 2025  
**Benchmark**: Large Matrix (M=128, N=3584, K=896) - Representative of FFN layers  
**Best Performance**: 102.4 GOPS (16×16 microkernel)  
**Target**: 1000 GOPS (9.8× gap)

---

## Executive Summary

Profiled the INT8 GEMM microkernel using `perf stat` to identify bottlenecks. Key findings:

1. **✅ Compute-bound, NOT memory-bound**: High IPC (1.45), low cache miss rate (0.43%)
2. **❌ OpenMP overhead dominates**: 51% of cycles spent in OpenMP runtime
3. **⚠️ Microkernel utilization**: Only 18.9% of cycles in actual computation
4. **🎯 Opportunity**: Reduce parallelization overhead for medium-sized workloads

---

## Performance Counter Analysis

### Overall Metrics (All 4 Microkernel Sizes Combined)

```
Runtime: 5.20 seconds total test time

Cycles:              19,542,496,348
Instructions:        28,295,777,969
IPC:                 1.45 insn/cycle        ← Good! Near optimal for compute

Cache Performance:
  Cache references:    33,207,056
  Cache misses:           143,411            (0.43% miss rate) ← Excellent!
  L1-D loads:       8,469,920,555
  L1-D misses:        436,609,606            (5.15% miss rate) ← Good

Branch Performance:
  Branches:           657,988,539
  Branch misses:        3,227,141            (0.49% miss rate) ← Excellent!
```

### Key Insights

**✅ NOT Memory-Bound**:
- L1 cache miss rate: 5.15% (very good, <10% is ideal)
- LLC miss rate: 0.43% (excellent, <1% is great)
- Data fits in cache hierarchy well
- Memory bandwidth is NOT the bottleneck

**✅ NOT Branch Misprediction Limited**:
- Branch misprediction rate: 0.49% (<1% is excellent)
- K-loop and microkernel loops are predictable
- No need to focus on branch optimization

**✅ High IPC (Instructions Per Cycle)**:
- IPC = 1.45 (out of theoretical max ~4-6 for AVX512)
- Indicates good instruction-level parallelism
- dpbusd is executing efficiently
- NOT stalled on instruction decode/dispatch

**❌ BUT: Parallelization Overhead**:
- 51% cycles in OpenMP runtime (thread synchronization)
- Only 18.9% in actual microkernel computation
- Remaining ~30% in GTest framework + other overhead

---

## Cycle Distribution Breakdown

From `perf report` (flat profile):

| Component | Cycles (%) | Analysis |
|-----------|-----------|----------|
| **OpenMP runtime** | 51.0% | Thread pool management, barriers, work distribution |
| **Microkernel** | 18.9% | Actual INT8 GEMM computation |
| **GTest + Other** | 30.1% | Test harness, setup/teardown, timing |

**Critical Issue**: More than half the time is spent managing parallelization!

---

## Root Cause Analysis

### Problem: OpenMP Overhead Too High

**Why is OpenMP overhead 51% for M=128 workload?**

Current parallelization:
```cpp
#pragma omp parallel for schedule(static)
for (size_t i = 0; i < M; i += MR) {
    // Process 16 rows per iteration (MR=16)
    microkernel(i, ...);
}
```

**Dimensions**:
- M=128, MR=16 → 8 iterations total
- Each iteration processes 16×3584 output elements
- Work per iteration: 16 × 3584 × 896 × 2 FLOPs = ~103 MFLOPs

**Issue**: 8 iterations across potentially 28-56 threads!
- Massive thread synchronization overhead
- Barriers, work stealing, cache coherence traffic
- Each thread gets <1 iteration of work
- Overhead dominates actual computation

---

## Why 16×16 is Faster Than 48×8

From benchmark results:
- **16×16**: 102.4 GOPS (M/MR = 128/16 = 8 iterations)
- **48×8**: 42.5 GOPS (M/MR = 128/48 = 2-3 iterations, worse!)
- **32×8**: 67.8 GOPS (M/MR = 128/32 = 4 iterations)

**Root cause**: Fewer iterations = fewer OpenMP parallel region entries!

With M=128:
- **16×16**: 8 iterations → More parallelizable work
- **48×8**: 2 iterations → Most threads idle, synchronization overhead
- **32×8**: 4 iterations → Middle ground

**Paradox**: Smaller MR gives BETTER parallelization because more iterations to distribute!

---

## Comparison: OpenMP Overhead vs Computation

**Target distribution for 1000 GOPS**:
- Microkernel: 80-90% (computation)
- OpenMP: 5-10% (parallelization)
- Other: 5-10% (framework)

**Current distribution at 102 GOPS**:
- Microkernel: 19% ❌
- OpenMP: 51% ❌
- Other: 30% (acceptable)

**If we eliminated 40% OpenMP overhead** → 170 GOPS (1.7× improvement)

---

## Theoretical Peak Analysis

**AVX512 VNNI dpbusd Throughput** (Intel Ice Lake+):
- 2× dpbusd units per core
- 1 dpbusd/cycle per unit = 2 dpbusd/cycle total
- Each dpbusd: 16 multiplies + 16 adds = 64 INT8 ops
- Peak: 2 × 64 = 128 INT8 ops/cycle/core

**Frequency**: Assume 2.5 GHz base (conservative)
- Peak single-core: 128 ops/cycle × 2.5 GHz = 320 GOPS

**Multi-core scaling** (28 physical cores):
- Theoretical: 28 × 320 = 8960 GOPS
- Realistic (accounting for memory bandwidth): 4000-6000 GOPS

**Our result**: 102 GOPS = **1.3% of realistic multi-core peak** 😱

**Single-core equivalent**: 102 / 28 = 3.6 GOPS/core = **1.1% of single-core peak**

---

## Why Are We So Far From Peak?

### 1. OpenMP Overhead (51% of cycles)
- Work distribution overhead
- Thread synchronization barriers
- Cache coherence traffic
- **Fix**: Reduce parallelization granularity (see recommendations)

### 2. Compensation Term Overhead
- Per-row compensation store/load: 256 ops per 16×16 microkernel
- Additional memory traffic
- **Evidence**: 48×8 is slower (384 store/load pairs vs 256 for 16×16)

### 3. K-Loop Overhead
- Current: Process 16 K-elements per iteration (4× unroll)
- Loop counter increment, branch prediction
- **Fix**: More aggressive K-unrolling

### 4. Register Spilling?
- Need to check assembly to confirm
- 16×16 = 16 accumulator registers (plenty available)
- Unlikely to be the issue

### 5. Memory Bandwidth (Ruled Out)
- ❌ NOT the bottleneck (only 5.15% L1 miss rate)

---

## Optimization Priority

Based on profiling data, prioritize in this order:

### Priority 1: Reduce OpenMP Overhead (51% → 10%)

**Options**:

**A. Manual Loop Tiling** (Recommended):
```cpp
// Instead of: #pragma omp parallel for
// Do:
const int TILE_M = 512;  // Process 512 rows per thread
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    
    for (int i_block = tid * TILE_M; i_block < M; i_block += nthreads * TILE_M) {
        // Process TILE_M rows without barriers
        for (int i = i_block; i < std::min(i_block + TILE_M, M); i += MR) {
            microkernel(i, ...);
        }
    }
}
```
- Eliminates per-iteration barriers
- Each thread works independently on large tiles
- Only one barrier at end

**B. Increase Problem Size for Benchmarking**:
- Test with M=1024 or M=2048
- More iterations = OpenMP overhead amortized
- Better reflects real FFN layers (2048-4096 rows)

**C. Single-threaded Baseline**:
- Measure 16×16 microkernel with OMP_NUM_THREADS=1
- Isolate pure microkernel performance
- Compare to multi-threaded to quantify parallelization efficiency

### Priority 2: K-Loop Unrolling (Current: 4×, Target: 8×)

Current processes 16 K-elements (4 blocks of 4 INT8):
```cpp
for (int k = 0; k < K; k += 16) {
    // 4× unroll: process k, k+4, k+8, k+12
}
```

Increase to 8× unroll (32 K-elements):
```cpp
for (int k = 0; k < K; k += 32) {
    // 8× unroll: process 32 K-elements
    // Better instruction scheduling
    // Amortize loop overhead
}
```

**Expected benefit**: 5-10% improvement

### Priority 3: Compensation Optimization

Current overhead:
- 256 store/load pairs for 16×16 microkernel
- Per-row compensation applied in scalar code

**Option A**: Vectorized compensation:
```cpp
// Instead of: for (int r = 0; r < 16; ++r) { row_comp[r] = sum[r] * 128; }
// Use: SIMD shuffle/permute to broadcast per-row compensation
__m512i comp_vec = _mm512_mullo_epi32(row_sums, _mm512_set1_epi32(128));
// Apply in registers before storing C
```

**Option B**: Precompute column sums (OneDNN strategy):
```cpp
// Before K-loop:
int32_t col_sums[N] = {0};
for (int j = 0; j < N; ++j) {
    for (int k = 0; k < K; ++k) col_sums[j] += B[k*N + j];
}

// In microkernel: single correction after K-loop
C[i,j] -= row_sum[i] * col_sums[j] * 128;
```

**Expected benefit**: 10-15% improvement

### Priority 4: Prefetching

Add software prefetch hints:
```cpp
// Prefetch next B block (64 bytes ahead)
_mm_prefetch((const char*)(B_packed + kb + 2), _MM_HINT_T0);

// Prefetch next A row
_mm_prefetch((const char*)(A + (i+MR)*K), _MM_HINT_T0);
```

**Expected benefit**: 5% improvement (since cache miss rate already low)

---

## Action Plan

### Immediate Next Steps

**1. Single-threaded Baseline** (5 minutes):
```bash
OMP_NUM_THREADS=1 ./build_v2_release/tests/v2/v2_test_parameterized_int8_gemm \
  --gtest_filter='*LargeMatrix*' --gtest_also_run_disabled_tests
```
- Measure pure microkernel performance without OpenMP overhead
- Expected: ~30-50 GOPS single-core → confirms OpenMP overhead

**2. Manual Loop Tiling** (30 minutes):
- Implement TILE_M=512 or adaptive tiling
- Remove `#pragma omp parallel for`
- Add manual thread management with single barrier

**3. Re-profile** (5 minutes):
```bash
perf stat ./build_v2_release/tests/v2/v2_test_parameterized_int8_gemm \
  --gtest_filter='*LargeMatrix*' --gtest_also_run_disabled_tests
```
- Verify OpenMP overhead reduced from 51% to <10%

**4. Larger Workload Test** (5 minutes):
- Test with M=1024, N=3584, K=896
- More realistic for actual FFN layers
- OpenMP overhead should be lower

### Medium-term Optimizations

**5. K-loop 8× unroll** (1 hour):
- Increase from processing 16 to 32 K-elements per iteration
- Measure IPC improvement

**6. Vectorized compensation** (2 hours):
- Use SIMD for per-row compensation
- Reduce store/load overhead

**7. Assembly analysis** (30 minutes):
```bash
objdump -d build_v2_release/tests/v2/v2_test_parameterized_int8_gemm | \
  grep -A 200 "microkernel"
```
- Check for register spilling
- Verify dpbusd instruction usage
- Look for unnecessary memory operations

---

## Expected Performance After Optimizations

Assuming optimizations work as expected:

| Optimization | Current | After | Speedup |
|--------------|---------|-------|---------|
| **Baseline** | 102 GOPS | - | 1.0× |
| **1. Manual tiling** | 102 GOPS | 170 GOPS | 1.7× |
| **2. K-loop 8× unroll** | 170 GOPS | 187 GOPS | 1.1× |
| **3. Vectorized compensation** | 187 GOPS | 215 GOPS | 1.15× |
| **4. Prefetching** | 215 GOPS | 226 GOPS | 1.05× |
| **5. Assembly tuning** | 226 GOPS | 250 GOPS | 1.1× |

**Target after all optimizations**: 250 GOPS (2.5× current, still 4× short of 1000 target)

---

## Reality Check: Can We Hit 1000 GOPS?

**Single-core theoretical peak**: 320 GOPS (2× dpbusd @ 2.5 GHz)

**Multi-core scaling challenges**:
- Memory bandwidth: ~100 GB/s typical for 2-socket system
- INT8 GEMM bandwidth requirements:
  - Read A: M×K×1 byte
  - Read B: K×N×1 byte
  - Write C: M×N×4 bytes (INT32)
  - M=128, N=3584, K=896: (128×896 + 896×3584 + 128×3584×4) = 4.8 MB per iteration
  - At 102 GOPS: 4.8 MB / 8 ms = 600 MB/s ← NOT bandwidth limited!

**Conclusion**: 1000 GOPS IS achievable, but requires:
1. ✅ Eliminating OpenMP overhead (170 GOPS)
2. ✅ Microkernel optimizations (250 GOPS)
3. ❓ Multi-core scaling (8-16 cores × 30 GOPS = 240-480 GOPS)
4. ❓ NUMA optimization (place threads on same socket)
5. ❓ Larger problem sizes (M=1024+ for better parallelization)

**Revised realistic target**: 400-600 GOPS for M=128 workload  
**Ambitious target**: 800-1000 GOPS for M=2048 workload

---

## Summary

**Key Finding**: OpenMP overhead (51%) is the primary bottleneck, NOT the microkernel itself.

**Evidence**:
- High IPC (1.45): Good instruction throughput
- Low cache misses (0.43%): Not memory-bound
- Low branch mispredicts (0.49%): Predictable loops
- Only 18.9% cycles in computation: Parallelization overhead dominates

**Recommended Path**:
1. **Immediate**: Single-threaded baseline test
2. **High priority**: Manual loop tiling to reduce OpenMP barriers
3. **Medium priority**: K-loop unrolling and compensation optimization
4. **Validation**: Test with larger workloads (M=1024+)

**Expected outcome**: 2-3× improvement (200-300 GOPS) achievable with parallelization fixes alone.
