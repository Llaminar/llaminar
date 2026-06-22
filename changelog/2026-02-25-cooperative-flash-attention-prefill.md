# Wavefront-Cooperative Flash Attention 2 Prefill Kernel

**Date**: 2026-02-25
**Impact**: Prefill throughput +49% (764 → 1138 tok/s), FA2 kernel 2.41× faster

## Summary

Replaced the double-buffered tiled FA2 prefill kernel with a wavefront-cooperative
design adapted from the decode kernel's architecture. Lanes own output dimensions
(not score elements), eliminating K/V tiles and the score matrix from LDS entirely.

## Problem

The tiled prefill kernel used 58KB of LDS (Q tile + 2×K tile + 2×V tile + score
matrix), limiting to 1 block/CU = 4 waves/CU = 10% occupancy. The kernel was
execution-latency-bound with no ability to hide memory latency via wavefront
switching.

## Solution

- **Architecture**: Each wavefront independently handles 8 Q rows against all KV
  positions. Only Q tile in LDS (~8KB). K/V loaded per-position from HBM with L2
  cache reuse across wavefronts.
- **Template parameters**: `flash_attention_2_cooperative_mi50_kernel<Q_PER_WAVE, DIMS_PER_LANE>`
  ensures compile-time loop bounds for register promotion of O_lane/m_i/l_i arrays.
- **Occupancy**: 3 waves/SIMD = 12 waves/CU (3× the tiled kernel)

## Key Discovery: Scratch Memory Spills

Initial non-template implementation used runtime-variable loop indices to access
per-Q-row accumulator arrays (`O_lane[qi][di]`, `m_i[qi]`, `l_i[qi]`). The hipcc
compiler placed these in private memory (432 bytes scratch), causing a **2.3× regression**
(35.93ms avg per call vs 15.67ms baseline).

The fix: template on `Q_PER_WAVE` and `DIMS_PER_LANE` with `#pragma unroll` on all
inner loops. This enables the compiler to register-promote the arrays. Final ISA:
84 VGPRs, 5 spills, 24B scratch (vs 30 VGPRs, 0 spills, 432B scratch without template).

## Performance Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| FA2 kernel avg | 15.67 ms | 6.49 ms | **-59%** (2.41×) |
| FA2 total GPU time | 1755 ms | 727 ms | **-59%** |
| FA2 % GPU time | 14.3% | 6.5% | -7.8pp |
| Prefill throughput | 764 tok/s | 1138 tok/s | **+49%** |
| Decode throughput | 58.53 tok/s | 58.45 tok/s | unchanged |
| Kernel VGPRs | 36 | 84 | +48 |
| Kernel occupancy (waves/CU) | 4 (10%) | 12 (30%) | 3× |

## Files Modified

| File | Changes |
|------|---------|
| `src/v2/kernels/rocm/attention/ROCmFlashAttentionKernels.hip` | New template cooperative kernel + launch dispatch |
| `docs/v2/projects/2026-02/flash-attention-cooperative-redesign.project.md` | Updated with actual results |

## ISA Resources (template <8, 2>)

| Resource | Value |
|----------|-------|
| VGPRs | 84 |
| SGPRs | 88 |
| VGPR spill count | 5 |
| Private segment | 24 bytes |
| `__launch_bounds__` | (256, 3) |

## Correctness

- Greedy sampling produces coherent output (counting: "6, 7, 8, 9, 10")
- Minor token divergence vs tiled kernel at low-confidence positions (expected:
  cooperative uses FP32 K from HBM vs FP16 K from LDS in tiled kernel)
- Tiled kernel preserved as fallback for head_dim < 64
