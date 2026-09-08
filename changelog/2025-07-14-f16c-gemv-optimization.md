# F16C Hardware FP16 + 4-Block Unroll GEMV Optimization

**Date**: 2025-07-14
**Impact**: CPU decode throughput +7.6% (10.79 → 11.61 tok/s)

## Summary

Optimized the Q8_0 native GEMV hot path (96.7% of decode time) with two key improvements:

1. **Hardware F16C conversion**: Replaced software `fp16_to_fp32()` (~20 instructions with branches)
   with hardware `_mm_cvtph_ps` (2 instructions). Called 112× per row in the inner loop.

2. **4-block unroll with 8 FMA accumulators**: Doubled ILP from 4 to 8 independent FMA chains,
   better saturating AVX-512 FMA throughput.

3. **Fused multi-projection GEMV**: Single OMP region for QKV (3 projections) and GateUp
   (2 projections), eliminating 84 OMP fork/join barriers per token across 28 layers.

## Files Changed

- `src/v2/kernels/cpu/gemm/Q8_0NativeGemv.h` — New file: F16C + 4-block unrolled GEMV
  with `gemv_dot_row_q8_0()` shared helper, `q8_0_native_gemv()` single-projection,
  and `q8_0_native_gemv_fused()` multi-projection path.

- `src/v2/kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h` — Updated dispatch: M=1 Q8_0
  uses `q8_0_native_gemv`, fused path uses `FusedProjectionDesc` stack array.

## Performance Results

### Decode Throughput (pp4+tg20, Qwen2.5-7B-Q8_0, 28 cores, NUMA-bound)

| Engine | tok/s | vs Llaminar |
|--------|-------|-------------|
| **Llaminar (after)** | **11.61** | — |
| ik_llama.cpp (AVX-512+VNNI) | 11.81 | +1.7% |
| Llaminar (before) | 10.79 | -7.1% |
| llama.cpp mainline (AVX-512) | 8.80 | -24.2% |
| llama.cpp mainline (AVX2) | 8.21 | -29.3% |

### Per-Stage Kernel Timing (decode, ms/tok)

| Stage | Before | After | Δ |
|-------|--------|-------|---|
| GEMM_FUSED_GATE_UP | 45.78 | 42.97 | -6.1% |
| GEMM (Wo+Down) | 29.34 | 27.85 | -5.1% |
| GEMM_FUSED_QKV | 6.91 | 5.68 | -17.8% |
| LM_HEAD | 6.41 | 6.11 | -4.7% |
| **TOTAL KERNEL** | **90.91** | **84.92** | **-6.6%** |

### Bandwidth Improvement

- Overall: ~84 GB/s → ~89 GB/s (+6%)
- Best stage (LM_HEAD): ~95 GB/s

## Correctness

RMSE=0 across all decode steps (bit-perfect match with software FP16 path).

## Approaches Explored and Rejected

- **VNNI integer dot products**: Regressed -4.8% on Cascade Lake. Root cause: AVX-512
  dispatches 512-bit ops as 2×256-bit µops, so FP32 FMA throughput matches VNNI. Plus
  VNNI adds activation quantization and correction overhead.

- **No software prefetch**: -2.8% regression. Hardware prefetcher alone insufficient for
  34-byte Q8_0 block stride.

- **Two-level prefetch (L1+L2)**: Neutral — no improvement over single-level L1 prefetch.

## Remaining Gap Analysis

The 1.7% gap to ik_llama is framework overhead (stage graph executor, virtual dispatch,
tensor resolution per stage), not kernel performance. Closing this requires architectural
changes to the stage execution model, not GEMV micro-optimization.
