# GEMV KB Heuristic Tuning (INT8 VNNI, gfx906)

**Date**: 2026-02-25
**Files Changed**:
- `src/v2/kernels/rocm/ROCmGemvKernel.hip` — Unified KB heuristic for grid_kpar dispatch
- `tests/v2/performance/kernels/rocm/Perf__ROCmGemvKernel.cpp` — Added AllShapes AutoSweep test, fixed BW constant

## Summary

Optimized the KB (K-parallel block count) heuristic for the `grid_kpar` GEMV kernel path on AMD MI50 (gfx906). The previous heuristic used separate formulas for large-K vs small-K shapes and had a special-case `is_squareish_shape` override. The new unified heuristic targets 8 waves/CU and rounds KB to the nearest factor of `k_groups` for balanced workload distribution.

## Methodology

1. **ISA analysis**: Extracted gfx906 code object from binary, disassembled all GEMV kernels. Confirmed max occupancy (4 waves/SIMD, 16 VGPRs) and well-pipelined inner loops — bottleneck is dispatch parameters, not instruction quality.

2. **Exhaustive TN×KB sweep**: Ran all 7 Qwen7B decode shapes across TN={128,256} × KB={4,6,8,10,12,14,16,20,24,28,32,40,48,56}. New `Benchmark_INT8VNNI_AllShapes_AutoSweep` test (gated by `LLAMINAR_RUN_INT8_ALLSHAPES_AUTOSWEEP=1`).

3. **Key finding**: ~8 waves/CU is the sweet spot for all shapes. More waves adds atomicAdd contention per N-tile without improving BW utilization. KB values that don't evenly divide `k_groups` cause tail effects (uneven work distribution).

## Heuristic Changes

### Old Heuristic
- K > 8192: `kb = max(4, min(32, K/512))` — FFN Down got KB=32
- K ≤ 8192: Target 64 kgrp/wave, min 10 waves/CU — FFN Gate/Up got KB=14
- `is_squareish_shape` override: Forces TN=256, KB=16 for Q/Wo

### New Unified Heuristic
```
target_blocks = 8 * 60 (NUM_CUS)  = 480
kb_raw = ceil(target_blocks / grid_n)
kb_max = k_groups / 16  (min inner loop length)
kb = nearest_factor_of(k_groups, min(kb_raw, kb_max))
```

Rounds to nearest factor of k_groups, biasing toward lower values (fewer atomicAdds).

### Per-Shape KB Changes

| Shape | N | K | Old KB | New KB | Old waves/CU | New waves/CU |
|-------|---|---|--------|--------|-------------|-------------|
| Q proj | 3584 | 3584 | 16 (squareish) | 16 (auto) | 7.5 | 7.5 |
| K proj | 512 | 3584 | 56 | 56 | 3.7 | 3.7 |
| V proj | 512 | 3584 | 56 | 56 | 3.7 | 3.7 |
| Wo proj | 3584 | 3584 | 16 (squareish) | 16 (auto) | 7.5 | 7.5 |
| FFN Gate | 18944 | 3584 | **14** | **4** | 34.5 | **9.9** |
| FFN Up | 18944 | 3584 | **14** | **4** | 34.5 | **9.9** |
| FFN Down | 3584 | 18944 | **32** | **16** | 14.9 | **7.5** |

## Performance Results

**GEMV min times (kernel-only, excluding quantize/scale)**:

| Shape | Old (ms) | New (ms) | Change |
|-------|---------|---------|--------|
| Q proj | 0.029 | 0.029 | — |
| K proj | 0.014 | 0.014 | — |
| V proj | 0.014 | 0.014 | — |
| Wo proj | 0.029 | 0.028 | -3.4% |
| FFN Gate | 0.095 | 0.091 | **-4.2%** |
| FFN Up | 0.095 | 0.090 | **-5.3%** |
| FFN Down | 0.091 | 0.090 | -1.1% |
| LM Head | 0.640 | 0.640 | — |

**End-to-end (28 layers + LM Head)**:
- Old: 20.180 ms → 49.6 tok/s (GEMV-only)
- New: 19.876 ms → 50.3 tok/s (GEMV-only)
- **Improvement: +1.5% throughput**

**Correctness**: All 3 correctness tests pass (Qwen05B, Qwen7B, BiasFusion). Cosine similarity >0.9999 for all shapes.

## Code Changes

### ROCmGemvKernel.hip
- Removed `is_squareish_shape` detection and overrides (variable, TN override, KB override)
- Removed K>8192 special-case path
- Replaced dual-formula KB heuristic with unified formula:
  - Target 8 waves/CU → `kb_raw = ceil(480/grid_n)`
  - Cap at `k_groups/16` for minimum inner loop length
  - Round to nearest factor of `k_groups` (bias lower)
- Preserved `int8_tn_override` and `int8_kb_override` atomics for runtime tuning

### Perf__ROCmGemvKernel.cpp
- Added `Benchmark_INT8VNNI_AllShapes_AutoSweep` test: sweeps all 7 Qwen7B decode shapes × TN={128,256} × 14 KB values. Gated by `LLAMINAR_RUN_INT8_ALLSHAPES_AUTOSWEEP=1`.
- Fixed hardcoded HBM2 peak BW from 480 GB/s to 1000 GB/s (MI50 32GB spec).

## Analysis Notes

- **Q/Wo (3584×3584, 12.8 MB)**: Launch-overhead limited at ~45% peak BW. Performance is flat across all TN/KB combinations. Cannot improve with dispatch tuning.
- **K/V (512×3584, 1.8 MB)**: Too small to saturate HBM. Only reaches 13% peak. Also unable to be improved.
- **FFN Gate/Up (18944×3584, 68 MB)**: Primary optimization target. Reducing KB from 14→4 reduces atomicAdd contention by 3.5×. BW improves from ~72% to ~75%.
- **FFN Down (3584×18944, 68 MB)**: Reducing KB from 32→16 halves atomicAdd contention. BW improves from ~75% to ~76%.
- **LM Head (152064×3584, 545 MB)**: Uses wide kernel (no atomics), already at 85% peak. Not affected by this change.

## Remaining Ceiling Analysis

The ~76% peak BW ceiling for FFN shapes appears to be limited by:
1. **AtomicAdd overhead**: Even KB=4 requires 4 atomic reductions per N-tile
2. **MemsetAsync**: ~2-3 μs per dispatch for output zeroing
3. **3-kernel pipeline**: quantize + GEMV + scale = 3 launches per GEMV call
4. **Non-coalesced B access**: Stride of 2K bytes between threads (one cache line per lane)

Further improvements would require architectural changes: two-pass reduction, kernel fusion, or weight layout optimization.
