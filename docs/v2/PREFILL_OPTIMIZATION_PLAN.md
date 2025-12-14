# JIT Fused Attention Prefill Optimization Plan

**Date**: December 14, 2025  
**Author**: David Sanftenberg  
**Status**: Planning

## Executive Summary

The prefill kernel currently achieves ~26 GFLOPS vs. decode's ~87 GFLOPS (3.4x slower).
The bottleneck is **not** in K/V cache reuse (that's working correctly), but in:
1. Per-query scalar operations instead of vectorized tile-wide operations
2. Excessive memory traffic in the V accumulation loop
3. Missed register blocking opportunities

## Current Performance

| Benchmark | Time | GFLOPS | Notes |
|-----------|------|--------|-------|
| Qwen7B Decode (seq=1, kv=4096) | 0.97ms | 87 | Baseline |
| Qwen7B Prefill (seq=128, kv=128) | 136ms | 26 | 3.4x slower |
| Qwen14B/32B Decode | 2.23ms | 61 | - |
| Qwen14B/32B Prefill | 288ms | 24 | - |

**Expected prefill performance**: With proper vectorization, prefill should achieve
~60-70 GFLOPS (similar to decode) since the memory access pattern is actually
better (K/V cache reuse across queries).

## Bottleneck Analysis

### 1. Scalar Softmax Operations (CRITICAL)

**Location**: `emit_prefill_tile_attention`, lines 870-920

**Problem**: Each query in the tile does 2 scalar exp() calls sequentially:
```cpp
for (int q = 0; q < q_tile_size_; ++q) {
    emit_prefill_exp_scalar(xmm_corr, xmm_corr);   // exp(old_max - new_max)
    emit_prefill_exp_scalar(xmm_weight, xmm_weight); // exp(score - new_max)
}
```

**Cost**: 
- Each exp() is ~15 instructions (polynomial evaluation)
- 8 queries × 2 exp × 15 instructions = **240 scalar instructions per KV position**

**Solution**: Vectorize exp() across all queries in the tile:
- Pack 8 scores into one ZMM register
- Do one vectorized exp512() call instead of 8 scalar calls
- Unpack results back to per-query state

**Expected speedup**: ~4-8x for softmax phase

### 2. Memory-Heavy V Accumulation (CRITICAL)

**Location**: `emit_prefill_tile_attention`, lines 950-1000

**Problem**: For each query, each V block:
```cpp
for (int b = 0; b < num_blocks; ++b) {
    for (int q = 0; q < q_tile_size_; ++q) {
        vmovups(zmm_ctx_lo, ptr[rsp + ...]);  // LOAD context
        vmovups(zmm_ctx_hi, ptr[rsp + ...]);
        vmulps(...);  // context *= corr
        vfmadd231ps(...);  // context += weight * V
        vmovups(ptr[rsp + ...], zmm_ctx_lo);  // STORE context
        vmovups(ptr[rsp + ...], zmm_ctx_hi);
    }
}
```

**Cost**:
- 8 queries × 4 blocks × (2 loads + 2 stores + 4 FMAs) = **128 memory ops + 128 FMAs per KV**
- Memory bandwidth limited, not compute limited

**Solution**: Keep context in registers during KV loop:
- Allocate zmm0-zmm15 for context (2 per query × 8 queries = 16 ZMM)
- Only load context once at tile start, store once at tile end
- Use corr/weight from zmm26-31 (already broadcasted)

**Expected speedup**: ~2-3x for V accumulation phase

### 3. Redundant Q Data Loading (MODERATE)

**Location**: `emit_prefill_tile_attention`, lines 800-850

**Problem**: Q blocks are loaded from stack every KV iteration:
```cpp
for (int b = 0; b < num_blocks; ++b) {
    for (int q = 0; q < q_tile_size_; ++q) {
        vmovdqu8(ymm_q, ptr[rsp + q_off + 4]);  // LOAD Q every time
        vpxord(ymm_q, ymm_q, Ymm(zmm_128().getIdx()));
        vpdpbusd(ymm_dot, ymm_q, ymm_k);
    }
}
```

**Cost**: 8 queries × 4 blocks × 1 load = **32 loads per KV position**

**Solution**: The code already has `use_q_registers = false` commented out.
Re-enable and fix the register allocation conflict:
- Use zmm8-zmm15 for Q data (8 queries × 1 YMM for head_dim=128/4blocks)
- Actually this won't fit for 4 blocks. Instead, load Q once per head,
  keep in stack, and use explicit unroll with register reuse.

**Expected speedup**: ~10-20% for Q*K phase

### 4. Constant Reloading (MINOR)

**Location**: `emit_prefill_tile_attention`, line 843

**Problem**: `16.0f` constant loaded via mov every query:
```cpp
mov(eax, 0x41800000);
vmovd(xmm_corr_low, eax);
vbroadcastss(ymm_corr, xmm_corr_low);
```

**Solution**: Pre-broadcast 16.0f to a dedicated ZMM at kernel init:
```cpp
// In generate_prefill_kernel(), add:
load_constant_f32(zmm_16(), 16.0f);  // New constant register
```

**Expected speedup**: ~5% for Q*K phase

### 5. Branch-Heavy Tile Bounds Checks (MINOR)

**Location**: `emit_prefill_tile_attention`, multiple locations

**Problem**: Conditional branches for each query:
```cpp
cmp(reg_q_local_start, q);
jg(skip_label, T_NEAR);
cmp(r8, q);
jle(skip_label, T_NEAR);
```

**Solution**: Use masked operations (k-masks) instead of branches:
- Compute mask once at tile start
- Use `vmovups(zmm | k1, ...);` for conditional stores

**Expected speedup**: ~5-10% (branch misprediction elimination)

## Optimization Priority

| Priority | Optimization | Estimated Speedup | Complexity |
|----------|--------------|-------------------|------------|
| P0 | Vectorize softmax across queries | 2-3x | Medium |
| P0 | Register-block context during KV loop | 1.5-2x | High |
| P1 | Pre-load Q data into registers | 1.1-1.2x | Low |
| P1 | Pre-broadcast constants | 1.05x | Low |
| P2 | K-masked tile bounds | 1.05-1.1x | Medium |

**Combined expected speedup**: 3-5x (from 26 GFLOPS to 70-100 GFLOPS)

## Implementation Plan

### Phase 1: Quick Wins (Est. 1.3x speedup)

1. Pre-broadcast 16.0f constant at kernel init
2. Move constant loading out of inner loops
3. Fix `use_q_registers` for small head_dim models

### Phase 2: Vectorized Softmax (Est. 2x speedup)

1. Add `emit_prefill_tile_softmax_vectorized()`:
   - Pack scores into ZMM: `vinsertf32x4` from xmm0-xmm7 into zmm0
   - Pack old_max values similarly
   - Vectorized max, exp, sum operations
   - Unpack corr/weight back to per-query registers

2. Modify `emit_prefill_tile_attention()` to call vectorized version

### Phase 3: Register-Blocked Context (Est. 1.5x speedup)

1. Restructure prefill kernel:
   - Allocate zmm0-zmm15 for context (not accumulators)
   - Keep context in registers during entire KV loop
   - Only spill to memory between tiles

2. Requires register pressure analysis:
   - Context: 8 queries × 2 ZMM = 16 registers
   - V data: 2 ZMM
   - Softmax state: 4 ZMM (max, sum, corr, weight per query - can share)
   - K data: 1 YMM
   - Constants: 6 ZMM (scale, log2e, exp_min, one, 128, 16)
   - Total: ~24 ZMM needed, we have 32 → feasible

### Phase 4: Masked Operations (Est. 1.1x speedup)

1. Compute tile validity mask at tile start
2. Replace conditional jumps with k-masked stores
3. Eliminate branch mispredictions

## Testing Strategy

1. **Correctness**: Run all `Test__JitFusedAttentionWo_Correctness` tests after each phase
2. **Performance**: Run `Perf__FusedAttentionWo_Tuning` benchmark
3. **Target**: Prefill GFLOPS ≥ 0.7 × Decode GFLOPS (~60 GFLOPS)

## Appendix: Register Allocation for Phase 3

```
ZMM Register Map (Phase 3 - Register-Blocked Context):

Context (per-query, persist during KV loop):
  zmm0  = ctx_q0_lo (16 floats)
  zmm1  = ctx_q0_hi (16 floats)
  zmm2  = ctx_q1_lo
  zmm3  = ctx_q1_hi
  ...
  zmm14 = ctx_q7_lo
  zmm15 = ctx_q7_hi

Softmax state (shared scratch, recycled):
  zmm16 = scores_packed (8 scores for vectorized softmax)
  zmm17 = max_packed
  zmm18 = corr_packed
  zmm19 = weight_packed

K/V data:
  zmm20 = K data (YMM portion)
  zmm21 = V_lo (16 floats)
  zmm22 = V_hi (16 floats)

Q data (load from stack):
  zmm23 = Q data (YMM portion)

Scratch:
  zmm24-25 = temporary

Constants:
  zmm26 = scale (1/sqrt(d))
  zmm27 = log2e
  zmm28 = exp_min (-87)
  zmm29 = one
  zmm30 = 128 (0x80808080)
  zmm31 = 16.0f
```

## References

- [JitFusedAttentionWo.h](../../src/v2/kernels/cpu/jit/q8_1/JitFusedAttentionWo.h)
- [Perf__FusedAttentionWo_Tuning.cpp](../../tests/v2/performance/kernels/cpu/attention/Perf__FusedAttentionWo_Tuning.cpp)
- FlashAttention paper (tiling strategy)
- Intel Intrinsics Guide (AVX-512 instruction reference)
