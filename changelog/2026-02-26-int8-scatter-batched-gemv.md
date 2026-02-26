# INT8-Input Batched Scatter GEMV

**Date**: 2026-02-26
**Branch**: tensor-parallel
**Component**: ROCm GEMV decode kernels (gfx906)

## Summary

Replaces the FP32 scatter GEMV path with INT8-input scatter kernels that take
pre-quantized activations, and adds batched dispatch that combines multiple
projections (Q/K/V, Gate/Up) into single kernel launches.

**Key improvements over FP32 scatter**:
1. **Quantize once**: Activations are quantized INT8 once per layer step instead 
   of re-quantizing per projection (7×/layer → 1×/layer)
2. **1 byte/element reads**: INT8 activations vs 4-byte FP32 → 4× less activation 
   bandwidth
3. **Batched launches**: QKV = 6→2 kernel launches; Gate/Up = 4→2 launches
4. **Better occupancy**: Small projections (K/V: 4 blocks) batched with Q (28 blocks) 
   gives 36 total blocks for better CU utilization

**Key improvements over old 3-kernel path** (quantize→GEMV→applyScaling):
1. **Scatter+reduce pipeline**: Better pipelining via k-parallel decomposition
2. **No atomicAdd contention**: Plain stores to partial buffer
3. **Hybrid self-reduce**: Auto-selects self-reducing single-kernel for small N

## Changes

### New Kernels (`src/v2/kernels/rocm/ROCmGemvKernel.hip`)

| Kernel | Description |
|--------|-------------|
| `gemv_int8_scatter_vnni_kernel_t` | Individual INT8 scatter (no LDS, 0 bytes shared, max occupancy) |
| `gemv_int8_scatter_selfreduce_vnni_kernel_t` | Individual self-reducing variant (last-to-arrive reduction) |
| `gemv_int8_scatter_batched_vnni_kernel_t` | Batched scatter for multiple projections in 1 launch |
| `gemv_reduce_scale_batched_kernel_t` | Batched reduce for multiple projections in 1 launch |

### New Dispatch Functions

| Function | Description |
|----------|-------------|
| `rocmGemv_int8_scatter_vnni()` | Individual dispatch with hybrid (self-reduce for N<16 blocks, 2-kernel otherwise) |
| `rocmGemv_int8_scatter_batched_vnni()` | Batched dispatch: builds `GemvBatchArgs` struct, always 2-kernel path |

### `GemvBatchArgs` Struct

~400 bytes passed by value as kernel arg (within HIP 4KB limit). Holds:
- Per-projection device pointers: `d_B[8]`, `d_C[8]`, `d_scales_B[8]`, `d_bias[8]`
- Per-projection N, grid_n_offset (prefix sum), n_offset (prefix sum for partial buffer)
- `total_grid_n`, `total_N`, `num_batches`

Concatenated N partial buffer layout: `partial[kb * total_N + n_offset[batch] + local_n]`

### Production Integration (`src/v2/kernels/rocm/ROCmQuantisedGemmKernel.cpp`)

**`multiply_fused_tensor`** (main multi-projection decode path):
- Always quantizes activations for M=1 (removed conditional skip)
- VNNI projections collected into batch arrays during the projection loop
- Single batched dispatch after loop (QKV batched, Gate/Up batched)
- Native-VNNI (Q4/IQ4) projections dispatched individually (as before)
- Batch overflow (>8 projections) handled with individual INT8 scatter fallback

**`multiply_tensor`**, **`multiply_fp32_to_fp32`**, **`multiply_fp32_to_fp32_with_bias`**:
- Replaced FP32 scatter with: quantize → individual INT8 scatter
- Removed `isFusedGemvEnabled()` gate check
- Ungated: INT8 scatter is now the default M=1 path (no env var needed)

### Gate Removal

- Removed `isFusedGemvEnabled()` method from `ROCmQuantisedGemmKernel`
- Removed `fused_gemv` field from `DebugEnv::ROCmConfig`
- Removed `LLAMINAR_FUSED_GEMV` environment variable parsing
- INT8 scatter is now the unconditional default for M=1 VNNI decode

## Qwen2.5-7B Batch Geometry

| Group | Projections | N values | grid_n total | Reduce blocks |
|-------|------------|----------|--------------|---------------|
| QKV | Q, K, V | 3584, 512, 512 | 28+4+4=36 | 36 |
| Gate+Up | Gate, Up | 18944, 18944 | 148+148=296 | 296 |
| Down | Down | 3584 | 28 | 28 (individual) |
| Wo | Wo | 3584 | 28 | 28 (individual) |

## Expected Performance (ROCm, MI50, gfx906)

Baseline: 60.2 tok/s (old 3-kernel path, `LLAMINAR_FUSED_GEMV` OFF)
Target: Improvement from:
- Quantize-once (1×/layer vs 7×/layer)
- Scatter pipeline (same benefit as +30% FP32 scatter demonstrated earlier)
- Batched launches (6→2 for QKV, 4→2 for Gate/Up)
- No FP32 re-quantization bandwidth waste

## Next Steps

- E2E benchmark on MI50 to measure actual tok/s improvement
- Perf test harness for INT8 scatter correctness (individual + batched)
