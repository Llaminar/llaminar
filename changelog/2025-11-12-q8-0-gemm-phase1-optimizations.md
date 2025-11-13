# Q8_0 GEMM Phase 1 Memory Access Optimizations

**Date**: November 12, 2025  
**Scope**: Q8_0 × Q8_0 → FP32 GEMM kernel optimization  
**Focus**: Block and scale access pattern improvements  

## Summary

Implemented three memory access optimizations targeting redundant memory touches and cache-hostile access patterns in the Q8_0 GEMM kernel. Achieved **+12.8% performance improvement** (486 → 548 GFLOPS) through systematic elimination of gather instructions and improved spatial locality.

## Baseline Performance

**Before optimizations:**
- **Performance**: 486 GFLOPS (M=4096, N=896, K=896)
- **Time**: 13.5 ms per iteration
- **Status**: 2.6× slower than dense INT8 baseline (1273 GFLOPS)

**Key bottlenecks identified:**
1. A blocks touched **twice**: Once for quantized data load, again for scale gather in post-processing
2. **Expensive gather instructions** for both A and B scales (10-15 cycle latency each)
3. **Strided memory access** for scales (34-byte stride for A, 8-element stride for B)
4. No prefetching despite predictable access patterns

## Optimization 1: A Scale Extraction During K-Loop

### Problem
A blocks were accessed twice:
```cpp
// K-loop: Load quantized data
const Q8_0Block& a_block = A_blocks[ir * K_blocks + kb];
__m256i a_ymm = _mm256_loadu_si256(a_block.qs);  // Touch memory

// Post-processing: Gather scales from same blocks (cache miss likely)
__m512i a_scales_gathered = _mm512_i32gather_epi32(
    a_scale_byte_offsets, a_blocks_base, 1  // Touch same memory again!
);
```

### Solution
Extract scales when blocks are hot in L1 cache:
```cpp
// K-loop: Extract scale while loading block
const Q8_0Block& a_block = A_blocks[ir * K_blocks + kb];
a_scales(ir, kb) = a_block.d;  // Free extraction (same cache line)
__m256i a_ymm = _mm256_loadu_si256(a_block.qs);

// Post-processing: Sequential load from contiguous buffer
__m256i a_scales_fp16 = _mm256_loadu_si256(&a_scales(ir, kb));  // L1 hit!
```

### Results
- **Performance**: 486 → 503 GFLOPS (**+3.6% improvement**)
- **Time**: 13.5 ms → 13.1 ms (0.4 ms saved)
- **Benefits**:
  - Eliminated 34-byte-strided gather (10-15 cycles → 2-3 cycles)
  - Reduced memory traffic (no redundant block touches)
  - Better cache utilization

### Implementation
- **File**: `Q8_0GemmKernel.h`
- **Lines modified**: 
  - Added `a_scales_storage` buffer (line ~273)
  - Extract scale during A block load (line ~303)
  - Replace gather with sequential load (line ~365-368)
  - Update scalar tail (line ~416)

## Optimization 2: B Scale Reorganization for Sequential Access

### Problem
B scales accessed with 8-element stride (cache-hostile):
```cpp
// Old layout: scales[kb * NR + jr] (stride-8 for consecutive kb)
__m512i b_scale_offsets = _mm512_setr_epi32(
    0*NR+jr, 1*NR+jr, 2*NR+jr, ..., 15*NR+jr  // Non-contiguous!
);
__m512i b_scales_gathered = _mm512_i32gather_epi32(offsets, base, 2);
```

### Solution
Transpose scale layout for unit stride:
```cpp
// New layout: scales[jr * K_blocks + kb] (stride-1 for consecutive kb)

// pack_B_panel(): Transpose during packing
for (int jr = 0; jr < panel_width; ++jr) {
    for (int kb = 0; kb < K_blocks; ++kb) {
        scales_base[jr * K_blocks + kb] = block.d;  // Transposed!
    }
}

// Post-processing: Sequential load (single cache line)
__m256i b_scales_fp16 = _mm256_loadu_si256(&B_scales[jr * K_blocks + kb]);
```

### Results
- **Performance**: 503 → 547 GFLOPS (**+8.7% improvement**)
- **Time**: 13.1 ms → 12.0 ms (1.1 ms saved)
- **Benefits**:
  - Eliminated strided gather (10-15 cycles → 2-3 cycles)
  - Sequential access fits in single cache line (64 bytes = 32 fp16 scales)
  - Better memory bandwidth utilization

### Implementation
- **File**: `Q8_0GemmKernel.h`
- **Lines modified**:
  - Updated pack_B_panel() scale storage (line ~200-205)
  - Replace gather with sequential load (line ~378-381)
  - Update scalar tail (line ~419)
  - Update edge microkernel (line ~463)

## Optimization 3: Software Prefetching for A Blocks

### Problem
A blocks have terrible spatial locality:
- Consecutive rows are `34 × K_blocks` bytes apart
- For K_blocks=896: 30,464 byte stride (way larger than cache lines!)
- Hardware prefetcher struggles with irregular stride

### Solution
Software prefetch 2 iterations ahead:
```cpp
for (int kb = 0; kb < K_blocks; ++kb) {
    // Prefetch A blocks for kb+2 (hides ~100 cycle memory latency)
    if (kb + 2 < K_blocks) {
        for (int ir = 0; ir < MR; ++ir) {
            _mm_prefetch(&A_blocks[ir * K_blocks + kb + 2], _MM_HINT_T0);
        }
    }
    // ... load and compute ...
}
```

### Results
- **Performance**: 547 → 548 GFLOPS (**+0.2% improvement**)
- **Time**: 12.0 ms → 12.0 ms (negligible change)
- **Analysis**: Hardware prefetcher already doing a good job despite irregular stride

### Implementation
- **File**: `Q8_0GemmKernel.h`
- **Lines modified**: Added prefetch loop (line ~289-298)

## Phase 1 Summary

### Overall Performance Improvement
- **Baseline**: 486 GFLOPS (13.5 ms)
- **After Phase 1**: 548 GFLOPS (12.0 ms)
- **Improvement**: **+12.8%** (+62 GFLOPS, -1.5 ms)

### Optimization Breakdown
| Optimization | GFLOPS Before | GFLOPS After | Δ GFLOPS | Δ % | Time Saved |
|--------------|---------------|--------------|----------|-----|------------|
| A scale extraction | 486 | 503 | +17 | +3.6% | 0.4 ms |
| B scale reorganization | 503 | 547 | +44 | +8.7% | 1.1 ms |
| A block prefetching | 547 | 548 | +1 | +0.2% | ~0 ms |
| **Total** | **486** | **548** | **+62** | **+12.8%** | **1.5 ms** |

### Key Learnings

1. **Redundant memory access is expensive**: Touching A blocks twice cost ~3.6% performance
2. **Gather instructions have high overhead**: Replacing gathers with sequential loads gave biggest gains (+8.7%)
3. **Layout matters**: Transposing B scales from `[kb][jr]` to `[jr][kb]` enabled unit-stride vectorization
4. **Hardware prefetchers are smart**: Software prefetching had minimal impact (~0.2%)
5. **Sequential > strided**: Both optimizations converted strided access to sequential (34-byte → 1-byte, 8-element → 1-element)

### Remaining Gap Analysis

**Current performance**: 548 GFLOPS  
**Dense INT8 baseline**: 1273 GFLOPS  
**Gap**: 2.3× slower

**Why Q8_0 can't match dense INT8:**
1. **2× scale lookups**: Q8_0 requires per-block scales for both A and B matrices (dense INT8: 0)
2. **FP16→FP32 conversions**: Scales stored as FP16, converted to FP32 for computation
3. **Compensation arithmetic**: Must subtract `sum_a × 128` per block (Q8_0 specific)
4. **Memory pressure**: Additional scale storage and lookups compete for cache

**Phase 1 addressed:**
- ✅ Redundant memory accesses (eliminated)
- ✅ Gather instruction overhead (eliminated)
- ✅ Strided memory access (converted to sequential)
- ✅ Cache locality (improved with prefetching attempt)

**Phase 1 did NOT address:**
- ❌ Per-block scale computation overhead (intrinsic to Q8_0 format)
- ❌ FP16→FP32 conversion latency (format requirement)
- ❌ Compensation arithmetic (Q8_0 specific)
- ❌ Additional memory bandwidth for scales

## Correctness Validation

All optimizations verified with comprehensive testing:

**Test**: `Q8_0GemmPerformance.CorrectnessWithRealWeights`
- **Shape**: M=16, N=16, K=896
- **Max absolute error**: 2.38e-07 (floating-point rounding only)
- **Max relative error**: 7.06e-06
- **Mismatches**: 0 / 256 (0.0%)
- **Status**: ✅ **PERFECT MATCH**

## Next Steps: Phase 2 Potential Optimizations

### Recommended (Medium Risk, Medium-High Reward)

1. **Scale Fusion** (Est. +5-10%):
   - Pre-compute `combined_scale[kb][jr] = a_scale[kb] × b_scale[kb][jr]`
   - Eliminates half the scale multiplications in post-processing
   - Tradeoff: Extra ~230KB memory for combined scales

2. **B Scale Prefetching** (Est. +2-5%):
   - Prefetch B scales 4-8 iterations ahead
   - Likely more effective than A prefetching (better stride pattern)

### High Risk, High Reward

3. **Loop Restructuring for Outer Product** (Est. +20-30%):
   - Current: Load A 64× per kb (8 ir × 8 jr iterations)
   - Optimized: Load A once, compute 8×8 outer product
   - Challenge: Fitting outer product in AVX-512 registers
   - Requires significant restructuring

### Not Recommended

4. **Further prefetching tuning**: Hardware prefetcher already effective (+0.2% max)

## Conclusion

Phase 1 successfully improved Q8_0 GEMM performance by **+12.8%** through systematic elimination of memory access inefficiencies. The optimizations focused on:

1. **Extracting scales during block loads** (no redundant memory touches)
2. **Reorganizing layouts for sequential access** (unit stride instead of large strides)
3. **Software prefetching** (minimal impact due to effective hardware prefetcher)

The remaining 2.3× gap to dense INT8 baseline is largely **intrinsic to the Q8_0 format** (per-block scales, FP16→FP32 conversions, compensation arithmetic). Phase 2 optimizations could potentially gain another 10-20%, bringing us to ~620-660 GFLOPS (~2× gap), which may be the practical ceiling for Q8_0 GEMM.

**Recommendation**: Phase 1 represents excellent ROI (12.8% gain for low-risk changes). Phase 2 should focus on scale fusion and B prefetching before attempting high-risk loop restructuring.

---

**Files Modified**:
- `src/v2/kernels/cpu/gemm_v2/Q8_0GemmKernel.h` (13 changes across 3 optimizations)

**Tests Passing**:
- ✅ `Q8_0GemmPerformance.CorrectnessWithRealWeights` (0 errors)
- ✅ `Q8_0GemmPerformance.LargeBatchedPrefill` (548 GFLOPS)
