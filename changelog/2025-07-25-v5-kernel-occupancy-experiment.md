# V5 Wide-Tile Kernel: Occupancy Experiment

**Date**: 2025-07-25  
**Scope**: `src/v2/kernels/rocm/ROCmQuantisedGemmKernel_CK.hip`, `ROCmQuantisedGemmKernel.cpp`, `DebugEnv.h`, `Perf__ROCmPrefillDispatchComparison.cpp`

## Summary

Designed and implemented V5 wide-tile GEMM kernel (`qgemm_int8_vnni_wide_tile_v5_kernel`) targeting CK's occupancy profile: **128 VGPRs, 0 spills, 2 waves/SIMD** — vs V4's 256 VGPRs + 118 spills at 1 wave/SIMD.

**Key result**: V5 achieves the target occupancy but does NOT close the performance gap to CK on 7B FFN_Up. This rules out register pressure and occupancy as root causes of the native-vs-CK gap.

## ISA Resource Comparison

| Kernel | VGPRs | Spills | Scratch | Waves/SIMD |
|--------|-------|--------|---------|------------|
| V1/N64/KT8 | 66 | 0 | 0 | 3 |
| V1/N128/KT8 | 113 | 0 | 0 | 2 |
| V3/N64/KT8 | 194 | 0 | 0 | 1 |
| V4/N128/KT8 | 256 | 118 | 240B | 1 |
| **V5/N128/KT8** | **117** | **0** | **0B** | **2** |
| V4/N128/KT16 | 256 | 705 | 1480B | 1 |
| **V5/N128/KT16** | **128** | **14** | **60B** | **2** |
| CK/N128/K16 | 128 | 0 | 0B | 2 |

## V5 Design

### What changed from V4
1. **`__launch_bounds__(256, 2)`** — forces compiler to target ≤128 VGPRs for 2 waves/SIMD
2. **Single-buffered inner loop** — 20 operand VGPRs (a_reg[4] + b_reg[16]) vs V4's implicit 160+ from unrolling
3. **`#pragma clang loop unroll(disable)`** on kk loop — prevents the compiler from unrolling the K-tile loop, which was the root cause of V4's VGPR explosion
4. **2-wave occupancy handles latency hiding** — instead of software pipelining within one wave

### Iteration: ABBA pipeline (rev1) → single-buffered (rev2)

**Rev1 (ABBA dual-buffer)**: Used two operand register sets (40 VGPRs) with alternating load/compute and inline asm `s_waitcnt lgkmcnt(N)`. Result: 128 VGPRs, **386 spills**, 608B scratch. The dual buffers pushed live set to ~132 VGPRs, over the 128 budget.

**Rev2 (single-buffered)**: Single register set (20 VGPRs) + `#pragma clang loop unroll(disable)`. Result: **117 VGPRs, 0 spills, 0 scratch**. Relies on 2-wave occupancy for latency hiding instead of explicit software pipelining.

## Benchmark Results (MI50/MI60 gfx906)

### FFN_Up (target shapes — where CK leads)

| Shape | CK(ms) | V5/KT8(ms) | V5/KT16(ms) | Best Native | vs CK |
|-------|--------|------------|-------------|-------------|-------|
| 0.5B (4864×896) | 1.926 | 2.013 | 2.009 | V2/KT16 1.995 | 0.966× |
| 3B (11008×2048) | 4.422 | 4.864 | **4.863** | **V5/KT16** | 0.909× |
| 7B (18944×3584) | 7.330 | 8.726 | 8.725 | V4/KT16 8.723 | 0.840× |

V5/KT16 wins 3B FFN_Up but all native kernels converge to ~8.72ms on 7B.

### LM_Head (bandwidth-bound — all kernels similar)

| Shape | CK(ms) | V5/KT8(ms) | Best Native |
|-------|--------|------------|-------------|
| 0.5B (151936×896) | 45.353 | 45.628 | V1/N64 45.569 |
| 3B (151936×2048) | 47.469 | 47.744 | V1/N128 47.712 |
| 7B (151936×3584) | 49.315 | 49.736 | V1/N64 49.547 |

V5 is competitive on LM_Head — essentially tied with V1.

### Attention & FFN_Down (where native already beats CK)

V5 is ~2-10% slower than V1/V3 on these shapes. The runtime kk loop (no unrolling) and larger N_TILE=128 don't help when K is small relative to N.

## Key Finding

**The 7B FFN_Up performance gap (~19% to CK) is NOT caused by register pressure or occupancy.**

Evidence:
- V1/N64/KT8: 3 waves, 66 VGPRs, 0 spills → 8.730ms
- V3/N64/KT8: 1 wave, 194 VGPRs, 0 spills → 8.726ms
- V4/N128/KT8: 1 wave, 256 VGPRs, 118 spills → 8.724ms
- **V5/N128/KT8: 2 waves, 117 VGPRs, 0 spills → 8.726ms**

All converge to ~8.72ms regardless of occupancy (1-3 waves), VGPR usage (66-256), or spill count (0-118). The gap is in CK's instruction-level scheduling — its fully-unrolled inner loop with hand-optimized dual-issue patterns across ds_read and v_dot4 instructions.

## Files Changed

| File | Change |
|------|--------|
| `src/v2/kernels/rocm/ROCmQuantisedGemmKernel_CK.hip` | V5 kernel template + dispatch function |
| `src/v2/kernels/rocm/ROCmQuantisedGemmKernel.cpp` | V5 extern declaration + production dispatch wiring |
| `src/v2/utils/DebugEnv.h` | `LLAMINAR_ROCM_WIDE_TILE_V5` env var |
| `tests/v2/performance/kernels/rocm/Perf__ROCmPrefillDispatchComparison.cpp` | V5/KT8 + V5/KT16 benchmark entries |

## Next Steps

To close the remaining ~19% gap on 7B FFN_Up, the only remaining approach is **inline assembly for the full inner loop** — replacing the C++ kk loop with hand-written AMDGPU ISA that precisely controls:
1. `ds_read_b128` / `v_dot4_i32_i8` interleaving (dual-issue scheduling)
2. `s_waitcnt lgkmcnt(N)` placement (precise partial waits)
3. Register allocation (no compiler interference)

This is the approach CK uses and is the only differentiator left.
