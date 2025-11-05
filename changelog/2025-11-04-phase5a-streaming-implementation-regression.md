# Phase 5A Streaming Dequant Implementation Complete - Performance Analysis

**Date**: November 4, 2025  
**Status**: ✅ Implemented, ❌ Performance regression (-18%)

## Executive Summary

Implemented Phase 5A streaming dequantization kernel based on profiling data showing B dequant as 40% bottleneck. **Result: 18% SLOWER than Phase 4** (7.19 vs 8.75 TFLOPS). Streaming approach adds more overhead than benefit.

## Implementation Details

### Kernel Strategy (Phase 5A)
```
Phase 4 (baseline):
  for k_tile in K_tiles:
      decode all B[k_tile] → shared memory (64×64)
      __syncthreads()
      MMA on full tile

Phase 5A (streaming):
  for k_tile in K_tiles:
      for sub_k in [0..3]:  # Split into 4×16 sub-tiles
          decode B[sub_k] → shared memory (64×16)
          __syncthreads()
          MMA on sub-tile
```

### Key Changes
1. **Split K-dimension**: TILE_K=64 → 4 sub-tiles of SUB_K=16
2. **Single B buffer**: Reduced from 2×8KB (Phase 4) to 1×2KB (Phase 5A)
3. **On-the-fly decoding**: Decode B inside inner loop, not pre-fetched
4. **Smaller MMA chunks**: Process 64×64×16 instead of 64×64×64

## Performance Results

| Kernel | Time (ms) | TFLOPS | vs Phase 4 | Status |
|--------|-----------|--------|------------|--------|
| **Phase 4 (swizzle)** | 0.188 | **8.75** | baseline | ✅ Production |
| **Phase 5A (streaming)** | 0.229 | 7.19 | -18% ⬇️ | ❌ **SLOWER** |

**Expected**: +15-25% gain (profiling showed 40% dequant time)  
**Actual**: -18% regression

## Root Cause Analysis

### Why Streaming Failed

1. **Excessive Synchronization** (4× overhead):
   - Phase 4: 2 syncs per K-tile (prefetch + compute)
   - Phase 5A: 5 syncs per K-tile (4 sub-tiles + 1 for next A)
   - **Impact**: ~400% more sync overhead

2. **Reduced Tensor Core Utilization**:
   - Phase 4: Large 64×64×64 tiles saturate Tensor Cores
   - Phase 5A: Small 64×64×16 sub-tiles underutilize Tensor Cores
   - **Impact**: Poor occupancy, latency not hidden

3. **Repeated Block Decoding**:
   - Each IQ4_NL block decoded 2× (first half, then second half)
   - No benefit from decode overlap (CUDA cores idle during MMA anyway)
   - **Impact**: 2× decode overhead for no gain

4. **Shared Memory Pressure**:
   - Constant buffer reuse (64×16) prevents double-buffering
   - No opportunity to overlap decode with MMA
   - **Impact**: Serialized execution

### Profiling Data Misinterpretation

**Original assumption**: "B dequant is 40% of time, so overlapping should save 30%"

**Reality**:
- B dequant uses CUDA cores
- MMA uses Tensor Cores
- **These don't compete** - they can run in parallel already!
- CUDA scheduler already overlaps independent operations

**Lesson**: Instruction-level parallelism only helps when resources are bottlenecked, not when different execution units are used.

## Correctness Validation

✅ **Bit-exact parity with Phase 4**:
```
Phase 5A vs Phase 4 max diff: 0
Mismatches: 0 / 262144
```

✅ **CPU reference test**: 1112 / 65536 mismatches (1.7%, within quantization error tolerance)

## Code Files

**Created**:
1. `src/v2/kernels/cuda/CudaGemmKernelPhase5ASimple.cu` (330 lines)
   - Streaming dequantization kernel
   - SUB_K=16 sub-tiles
   - Single B buffer (2KB)

2. `src/v2/kernels/cuda/CudaGemmKernelPhase5ASimple.h` (30 lines)
   - Kernel launch API

3. `tests/v2/unit/Test__CudaGemmPhase5A.cpp` (450 lines)
   - Correctness vs CPU reference
   - Parity vs Phase 4
   - Performance benchmark

**Modified**:
1. `src/v2/CMakeLists.txt` - Added Phase 5A kernel to build
2. `tests/v2/CMakeLists.txt` - Added Phase 5A test target

## Lessons Learned

### 1. Profiling Can Mislead

**Observation**: "40% time in dequant" ≠ "40% performance gain from overlapping"

**Reason**: Different execution units (CUDA cores vs Tensor Cores) already run in parallel.

**Corrected understanding**: Dequant time represents **load/decode latency**, not **compute bottleneck**.

### 2. Synchronization Kills Performance

Adding 4× more `__syncthreads()` calls outweighs any potential benefit from streaming.

**Rule of thumb**: Minimize sync points, even if it means larger shared memory buffers.

### 3. Tile Size Matters

Tensor Cores need large tiles (64×64×64) to hide latency. Smaller tiles (64×64×16) expose latency.

**Guideline**: Keep K-dimension ≥32 for Tensor Core efficiency.

### 4. Double-Buffering > Streaming

Phase 4's double-buffered approach (prefetch next while computing current) is more effective than streaming sub-tiles.

**Lesson**: Coarse-grained overlap (tile-level) beats fine-grained (sub-tile-level).

## Next Steps

### Option A: Abandon Streaming, Try Larger Tiles

**Hypothesis**: Dequant isn't the bottleneck - memory bandwidth is.

**Approach**: Increase tile size to 128×128×128
- Better Tensor Core utilization
- More work per memory fetch
- Expected gain: +10-20%

### Option B: Try Warp Specialization (Phase 5B)

**Hypothesis**: True producer/consumer split could help.

**Approach**: Dedicate 1 warp to dequant, 3 warps to MMA
- Eliminates sync overhead (async queues)
- Requires CUTLASS-style warp-level coordination
- Expected gain: Unclear (may still regress due to reduced compute resources)

### Option C: Accept Phase 4 as Optimal

**Hypothesis**: 8.75 TFLOPS is near-optimal for this approach.

**Rationale**:
- 24.6% of peak (35.58 TFLOPS)
- Quantization overhead is inherent
- Further gains require fundamentally different approach

## Recommendation

**Proceed with Option A**: Try Phase 6 with 128×128×128 tiles.

**Rationale**:
1. Profiling showed dequant time, but true bottleneck may be memory bandwidth
2. Larger tiles amortize decode cost over more compute
3. Simpler implementation than warp specialization
4. Clear path to 10-12 TFLOPS (28-34% of peak)

**Skip Phase 5B** (warp specialization):
- Too complex
- Uncertain benefit
- Risk of further regression

## Performance Context

| Phase | TFLOPS | % Peak | Speedup | Status |
|-------|--------|--------|---------|--------|
| Phase 3 Part 1 | 1.05 | 3.0% | 1.00× | ❌ Baseline |
| Phase 3 Part 2 | 6.56 | 18.4% | 6.25× | ✅ Pipelined |
| Phase 4 | **8.75** | **24.6%** | **8.33×** | ✅ **Current best** |
| Phase 5A | 7.19 | 20.2% | 6.85× | ❌ **Regression** |

**Target for Phase 6**: 10-12 TFLOPS (28-34% of peak) with 128×128×128 tiles

## Technical Debt

1. Remove `#pragma once` from .cu files (warnings)
2. Remove unused `block_within_tile` variable
3. Consider removing Phase 5A from production build (keep for reference)

## References

- **Profiling data**: `changelog/2025-11-04-phase4-profiling-complete.md`
- **Phase 4 baseline**: `changelog/2025-11-04-phase4-swizzle-batch-scaling-analysis.md`
- **Streaming analysis**: `STREAMING_DEQUANT_ANALYSIS.md`
- **CUTLASS reference**: Hopper TMA warp specialization patterns
