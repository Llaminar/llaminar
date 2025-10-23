# IQ4_NL Vectorized Dequantization Optimization - Complete

**Date**: October 22, 2025  
**Status**: ✅ Implemented and Verified

## Summary

Successfully implemented **Priority 1** optimization from the ACTION_PLAN: Vectorized integer-to-float conversion for IQ4_NL dequantization.

## Implementation

**File**: `src/tensors/IQ4_NLTensor.h`

### AVX512 Path (lines 390-419)
- Replaced scalar `vcvtsi2ss` with `_mm512_cvtepi32_ps`
- Processes 16 values at once instead of 1
- Speedup: **3.5× faster** in isolation (microbenchmark)

### AVX2 Path (lines 436-478)
- Replaced scalar conversion with `_mm_cvtepi32_ps`  
- Processes 4 values at a time
- Speedup: **3.5× faster** in isolation

## Performance Impact

### Micro-benchmark (dequantization only)
- **Before**: ~224 cycles per 32-element block
- **After**: ~60 cycles per 32-element block
- **Speedup**: 3.7×

### Macro-benchmark (end-to-end GEMM)
- **Before**: 394-407 GFLOPS (baseline)
- **After**: 394-407 GFLOPS (within noise)
- **Impact**: None measurable

## Analysis: Why No Performance Gain?

**Root Cause**: Dequantization is only **4-5% of total execution time** in Release builds.

From profiling (`profile_results/perf_20251022_151138.data`):
- Total samples: 155,438
- Dequantization: ~6,000-8,000 samples (4-5%)
- **Even 10× faster dequantization** → only 4% overall gain

**Amdahl's Law**: Optimizing 4% of execution by 3.7× → 3% overall speedup (lost in measurement noise)

## Key Learnings

### 1. **Debug vs Release Profiling is Critical**
- **Debug build**: Dequantization appeared as 30-40% of time (misleading!)
- **Release build**: Dequantization is 4-5% of time (reality)
- **Lesson**: Always profile Release builds for optimization decisions

### 2. **Real Bottlenecks (from Release profiling)**
- **OpenMP overhead**: 30% of execution time
- **dot_product_simd**: 18-20% of execution time  
- **Unknown symbol 0x7ba58a6606c0**: 10.8% (needs investigation)
- **Dequantization**: 4-5%

### 3. **OpenMP Optimization Complexity**
Attempted to reduce 30% OpenMP overhead by combining nested `#pragma omp parallel` + `#pragma omp for`:
- **Result**: Severe regression (394 → 219 GFLOPS on Q-Projection 4096)
- **Root Cause**: Thread-local storage semantics - nested structure was intentional
- **Lesson**: OpenMP optimizations require deep semantic understanding

## Recommendations

### Keep This Optimization
Despite no measurable performance gain, we should **keep the vectorized dequantization** because:
1. **No downsides**: Code is cleaner, no regressions
2. **Future-proofing**: If dequantization becomes bottleneck in other workloads
3. **Best practices**: SIMD is the right way to do bulk conversions

### Next Priorities

**Priority 1: Investigate Unknown Hotspot (10.8%)**
```bash
perf script -i profile_results/perf_20251022_151138.data | \
    grep '7ba58a6606c0' | head -20
```
Likely candidates:
- Inlined `dot_product_simd` (AVX512 FMA loop)
- OpenBLAS internal functions
- Tiled accumulation

**Priority 2: OpenMP Thread Affinity Tuning**
- Current: Default OpenMP scheduling
- Try: Explicit thread pinning, different chunk sizes
- Expected: 5-10% gain if we reduce barriers/synchronization

**Priority 3: Fused Dequant+GEMM Kernel**
- Current: Decode to buffer → GEMM
- Proposed: Fused decode+accumulate (no intermediate buffer)
- Expected: 10-15% gain from eliminated memory traffic

**Deferred**: FP16 scale pre-conversion (only 1-2% potential gain)

## Files Modified

- `src/tensors/IQ4_NLTensor.h`: 
  - `decodeBlockAVX512()` (lines 390-419)
  - `decodeBlockAVX2()` (lines 436-478)

## Benchmarks

**Test command**:
```bash
mpirun -np 2 --bind-to socket --map-by socket \
    ./build_release/benchmark_iq4nl_gemm
```

**Results** (FP32 path):
- Q-Projection 4096: 394 GFLOPS (unchanged)
- FFN 512: 525 GFLOPS (unchanged)
- FFN 8192: 384 GFLOPS (unchanged)

## Conclusion

This optimization is a **technical success** (3.7× faster dequantization) but a **performance non-event** (no measurable GEMM speedup). The key insight is that **profiling Release builds** is essential - Debug profiling led us to optimize a non-bottleneck.

**Status**: Optimization complete, focus shifting to real bottlenecks (OpenMP overhead, unknown 10.8% hotspot).
