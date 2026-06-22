# GEMV TP Dispatch Unification (Phase 1)

**Date**: 2026-03-11  
**Component**: ROCm GEMV INT8 VNNI Kernel  
**Impact**: Improved decode throughput at TP-sharded dimensions

## Summary

Unified the GEMV blockwise dispatch to route all vec4 N≥128 shapes through the
high-performance LDS k-reduce kernel. Previously, three separate dispatch paths
existed with an N≥2048 threshold that excluded TP-sharded shapes from the best path.

## Changes

### `src/v2/kernels/rocm/gemm/ROCmGemvKernel_INT8_VNNI.hip`

- **Absorbed tiny-KV path into unified LDS k-reduce**: Removed the separate tiny-KV
  dispatch block (~40 lines). The LDS k-reduce kernel has no N-dimension constraint.
- **Lowered QWO-LDS threshold**: `N >= 2048` → `N >= 128` (GRID_KPAR_TILE_N).
  Renamed `use_qwo_lds_kreduce_blockwise` → `use_lds_kreduce_blockwise`.
- **Added KB cap to occupancy selector**: `select_blockwise_qwo_outer_splits` now caps
  KB at `act_blocks / wk` to prevent excessive splits at small grid_n. Without cap,
  grid_n=1 produced KB=60 with mostly empty wave-shards and severe atomicAdd contention.
- Grid-kpar and scatter paths are now unreachable for blockwise vec4 shapes (left as
  dead-code safety net).

### `tests/v2/performance/kernels/rocm/Perf__ROCmTPScaling.cpp`

- Updated `classifyGemvDispatch()` to reflect unified dispatch: removed tiny-KV, QWO,
  grid-kpar, scatter classifications → single "LDS-kred" for all N≥128 vec4 shapes.

### `docs/v2/projects/2026-03/ROCM_GEMV_TP_TUNING_PLAN.md`

- Created project plan for the full 4-phase TP tuning effort.
- Updated Phase 1 status to complete with measured results.

## Results

### Per-Shape Micro-Benchmarks (GEMV TP Scaling Test)

| Model | TP | Before Eff% | After Eff% | Delta |
|-------|----|------------|-----------|-------|
| Qwen-7B | 2 | 67.5% | 75.1% | +7.6pp |
| Qwen-7B | 4 | 46.0% | 50.8% | +4.8pp |
| Qwen-3B | 2 | 62.3% | 68.3% | +6.0pp |

Key shape: Q proj 7B TP=2 (1792×3584): **44µs → 25.9µs** (grid-kpar → LDS-kred)

### End-to-End Benchmark (Qwen-7B Q8_0, 128 decode tokens)

| Config | Decode tok/s |
|--------|-------------|
| Single (rocm:0) | 74.50 |
| TP=2 (rocm:0,1) | 82.79 |

No regression vs pre-change baseline (82.91). The ~4% per-token GEMV improvement
is within the benchmark's ~10% run-to-run variance.

### Correctness

- Single-device inference: ✅ correct output
- TP=2 inference: ✅ correct output
- GEMM unit tests: ✅ 3/3 passed
- Build: ✅ 1233/1233, zero warnings
