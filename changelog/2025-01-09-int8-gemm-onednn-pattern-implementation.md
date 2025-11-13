# INT8 GEMM OneDNN Pattern Implementation - Session Summary

**Date**: January 9, 2025  
**Focus**: Implementing OneDNN's signed×signed INT8 GEMM pattern with vpdpbusd compensation

## Session Objectives

Mirror OneDNN's INT8 GEMM implementation to handle signed×signed INT8 multiplication using AVX512 VNNI vpdpbusd instruction (which natively supports unsigned×signed only).

## Key Discoveries

### OneDNN's s8s8s32 Strategy

**Pattern Found in Source**:
- **Kernel**: `external/onednn/src/cpu/x64/gemm/s8x8s32/jit_avx512_core_gemm_s8u8s32_kern.cpp`
- **GEMV Reference**: `jit_avx512_core_kernel_gemv_s8x8s32_kern.cpp`
- **Copy Routine**: `jit_avx512_core_u8_copy_bn_kern_autogen.cpp`

**Core Technique**:
1. Convert signed B matrix to unsigned via `XOR 0x80` (flips sign bit)
2. Use `vpdpbusd(acc, B_unsigned, A_signed)` for computation
3. Apply compensation: `result -= sum(A_row) × 128`

**Mathematical Basis**:
```
A(s8) × (B(s8) XOR 0x80) = A(s8) × B(u8) = A×B + A×128
Therefore: correct_result = result - sum(A) × 128
```

## Implementation Evolution

### Attempt 1: Per-Iteration XOR + Vector Compensation (169 GOPS)
```cpp
// XOR B every k-iteration
__m512i b_unsigned = _mm512_xor_si512(b_packed, xor_mask);
c0 = _mm512_dpbusd_epi32(c0, b_unsigned, a0);

// Compensate via second dpbusd
__m512i comp0 = _mm512_dpbusd_epi32(zeros, ones_u8, a0);
comp0 = _mm512_slli_epi32(comp0, 7);  // *128
c0 = _mm512_sub_epi32(c0, comp0);
```

**Problem**: 2× dpbusd per k-iteration (main + compensation) → limited to ~170 GOPS

### Attempt 2: Packing-Time Conversion + Scalar Compensation (85 GOPS)
```cpp
// XOR during pack_B_panel_vnni
if (convert_s8_to_u8) val ^= 0x80;

// Scalar sum accumulation in k-loop
for (int i = 0; i < 4; ++i) {
    sum_a0 += a0_bytes[i];  // ×6 rows ×4 bytes = 24 extractions/iteration
}

// Post-loop compensation
comp0 = _mm512_set1_epi32(sum_a0 * 128);
```

**Problem**: Scalar byte extraction killed performance (24 ops per k-iteration) → **SEVERE REGRESSION**

### Attempt 3: Packing-Time Conversion + Vector Sum Accumulation (176 GOPS)
```cpp
// XOR during packing (one-time cost)
pack_B_panel_vnni(B, Bpack, K, ldb, convert_s8_to_u8=true);

// Vector sum accumulation in k-loop
sum_a0 = _mm512_dpbusd_epi32(sum_a0, ones_u8, a0);

// Post-loop horizontal reduction + compensation
int32_t total_sum = _mm512_reduce_add_epi32(sum_a0) / 16;
comp0 = _mm512_set1_epi32(total_sum * 128);
```

**Result**: Similar to Attempt 1 (still 2× dpbusd per k-iteration)

### Attempt 4: 2× K-Loop Unrolling (270 GOPS)
```cpp
// Process 8 K elements (2 blocks) per iteration
for (; k + 7 < K; k += 8) {
    // Block 0
    c0 = _mm512_dpbusd_epi32(c0, b0, a0_vec0);
    sum_a0 = _mm512_dpbusd_epi32(sum_a0, ones_u8, a0_vec0);
    
    // Block 1
    c0 = _mm512_dpbusd_epi32(c0, b1, a0_vec1);
    sum_a0 = _mm512_dpbusd_epi32(sum_a0, ones_u8, a0_vec1);
}
```

**Result**: **60% improvement** over baseline via reduced loop overhead and better ILP

### Attempt 5: 4× K-Loop Unrolling (311 GOPS) ✅ Current Best
```cpp
// Process 16 K elements (4 blocks) per iteration
for (; k + 15 < K; k += 16) {
    // Load all 4 B blocks
    __m512i b0 = _mm512_loadu_si512(...);
    __m512i b1 = _mm512_loadu_si512(...);
    __m512i b2 = _mm512_loadu_si512(...);
    __m512i b3 = _mm512_loadu_si512(...);
    
    // 4× {dpbusd for main compute, dpbusd for sum accumulation}
    // ... (24 dpbusd operations total per iteration)
}
```

**Result**: **84% improvement** over initial implementation, **15% over 2× unrolling**

## Performance Progression

| Approach | GOPS | Speedup | Time (ms) | Notes |
|----------|------|---------|-----------|-------|
| Per-Iteration Vector Comp | 169 | 1.0× | 211 | Baseline (correct) |
| **Packing + Scalar Sum** | **85** | **0.5×** | **418** | **REGRESSION** |
| Packing + Vector Sum | 176 | 1.04× | 202 | Minimal improvement |
| 2× K-Loop Unroll | 270 | 1.60× | 132 | Significant gain |
| **4× K-Loop Unroll** | **311** | **1.84×** | **115** | **Current Best** |

**Benchmark**: FFN Down 4096×896×4864 (35.7 billion FLOPs)

## Key Lessons Learned

### 1. Vector vs Scalar Operations
- **Scalar byte extraction is catastrophic**: 24 byte extractions per k-iteration → 56% performance loss
- **Stay in vector domain**: Even if it means 2× dpbusd, vector ops dominate scalar manipulation

### 2. K-Loop Unrolling Benefits
- **2× unrolling**: +60% performance (270 GOPS)
- **4× unrolling**: +15% more (311 GOPS)
- **Diminishing returns**: Further unrolling unlikely to help significantly

### 3. Fundamental Bottleneck: 2× dpbusd Overhead
- **Root cause**: Signed×signed problem requires compensation proportional to K
- **Unavoidable cost**: Every k-iteration needs 2× dpbusd (main + sum accumulation)
- **Impact**: Theoretical 50% overhead (confirmed: 311 GOPS vs ~600+ theoretical peak)

### 4. OneDNN's Advantage
- **6610 GOPS** (21× faster than our 311 GOPS)
- **Likely optimizations**:
  - Much larger microkernels (24×16, 48×16 vs our 6×16)
  - Precomputed offset arrays outside hot kernel
  - Highly optimized instruction scheduling
  - Multi-level tiling and blocking strategies

## Current Implementation

**File**: `src/v2/kernels/cpu/gemm_v2/RegisterBlockedInt8Gemm.h`

**Key Components**:
1. **pack_B_panel_vnni**: Converts B from column-major to [K/4][16][4] with XOR 0x80 conversion
2. **microkernel_6x16_dpbusd**: 
   - 4× unrolled K-loop (processes 16 K elements per iteration)
   - Vector sum accumulation for compensation
   - Post-loop horizontal reduction and compensation application
3. **gemm**: Top-level driver with N-outer, M-inner tiling

**Architecture**:
- Microkernel: 6×16 (96 INT32 outputs in 6 ZMM registers)
- B packing: [K/4][16][4] layout (64 bytes per K-block)
- Unrolling: 4× K-blocks per iteration
- Compensation: Post-loop horizontal sum + broadcast subtract

## Test Results

**Correctness**: ✅ All tests passing (max_diff=0)
- Single tile 6×16
- Multiple tiles 18×48
- Edge case 7×17

**Performance**: ❌ Failing 1000 GOPS target
- Single token (1×896×4096): 9.6 GOPS
- Medium batch (128×896×4096): ~205 GOPS
- FFN Down (4096×896×4864): **311 GOPS** (target: 1000 GOPS)

## Remaining Challenges

### 1. 3.2× Gap to Phase 1 Target (1000 GOPS)
Current **311 GOPS** vs target **1000 GOPS**

**Potential Next Steps**:
- ❓ Larger microkernel (12×16, 24×16) - may amortize fixed costs better
- ❓ Separate offset precomputation (like OneDNN) - avoid 2× dpbusd in hot loop
- ❓ JIT code generation - optimal instruction scheduling
- ❓ Multi-level tiling (L1/L2/L3 cache hierarchy)

### 2. 21× Gap to OneDNN (6610 GOPS)
Our implementation is fundamentally simpler than OneDNN's production code

**Known OneDNN Advantages**:
- Industrial-strength JIT assembly generation
- Decades of optimization refinement
- Full offset precomputation infrastructure
- Advanced blocking and prefetching strategies

## Next Actions

### Option A: Continue Phase 1 Optimizations
- Implement larger microkernels (12×16, 24×16)
- Add offset precomputation to avoid 2× dpbusd
- Profile to identify remaining bottlenecks

### Option B: Move to Phase 2
- Accept 311 GOPS as Phase 1 result
- Integrate into full pipeline
- Measure end-to-end impact before further optimization

### Option C: Hybrid Approach
- Use current 311 GOPS implementation as baseline
- Investigate OneDNN's actual offset precomputation code
- Consider whether 3× speedup (vs 6×+ target) justifies continued Phase 1 work

## Code Statistics

- **Lines changed**: ~150 lines in RegisterBlockedInt8Gemm.h
- **Compensation strategies tested**: 3 (per-iteration, scalar post-loop, vector post-loop)
- **Unrolling levels tested**: 3 (1×, 2×, 4×)
- **Correctness iterations**: 5+ (all passing)
- **Performance attempts**: 5

## Conclusion

Successfully implemented OneDNN's signed-to-unsigned conversion pattern with compensation, achieving **1.84× speedup** (311 GOPS vs 169 GOPS baseline) through aggressive K-loop unrolling. However, the fundamental 2× dpbusd overhead (main computation + sum accumulation for compensation) remains a bottleneck, leaving us **3.2× short** of the Phase 1 target.

**Key insight**: Vector operations dominate scalar operations so completely that even "redundant" vector dpbusd operations (2× per k-iteration) outperform "optimal" scalar compensation by **2-3×**. This underscores the importance of staying in the vector domain at all costs.

**Recommendation**: Before investing more time in Phase 1 micro-optimization, profile the full pipeline to determine whether 311 GOPS (vs 472 GOPS FP32 baseline) provides sufficient end-to-end speedup to justify the quantization complexity.
