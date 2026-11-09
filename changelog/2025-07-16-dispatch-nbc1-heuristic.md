# VNNI Dispatch Heuristic: n_block_chunks=1 for FFN + GEMM

**Date**: 2025-07-16

## Summary

Changed the NativeQuantised tile dispatch heuristic to use `n_block_chunks=1` for both
GEMV (decode, M=1) FFN shapes and GEMM (prefill, M>1) non-tiled paths. This was
validated by a comprehensive dispatch parameter sweep.

## Root Cause

The previous heuristic computed `n_block_chunks` using L2 cache sizing formulas
that yielded nbc=2 for FFN shapes. The theory was that processing 2 N-chunks per
task (128 columns) would improve L2 residency for scale/compensation data during
K-dimension streaming.

The sweep revealed this theory doesn't hold: each N-chunk is processed independently
with full-K streaming, so finer N-granularity (nbc=1 = 64 columns per task) gives
better thread load balance without any cache penalty.

## Changes

- **`CPUNativeQuantisedTileConfig.h`**:
  - GEMV FFN case: `nbc = min(chunks_per_task, max_chunks_l2)` → `nbc = 1`
  - GEMM non-tiled else-branch: removed `target_tasks/m_tasks/calc_chunks` formula → `nbc = 1`
  - Added `LLAMINAR_CPU_VNNI_*` env-var overrides for future sweep testing (GEMV + GEMM paths)
  - K-parallel `MIN_BPR` now overridable via `LLAMINAR_CPU_VNNI_MIN_BPR_K_PARALLEL`

- **`DebugEnv.h`**: Added `CPUVNNIConfig` struct with 5 env-var overrides
  (`n_block_chunks`, `k_tile_blocks`, `m_unroll`, `k_tiles`, `min_bpr_k_parallel`)

- **`KernelProfiler.h`**: Added `recordParallel()` for correct OMP thread-time accounting

- **Sweep test**: `Perf__CPUNativeQuantised_DispatchSweep.cpp` — 5 test cases covering
  Q8_0/Q4_0 decode + prefill across all Qwen 7B shapes

## Micro-benchmark Results (Q8_0 decode, M=1, 28 threads)

| Shape | Old (μs) | New (μs) | Improvement |
|-------|----------|----------|-------------|
| FFN_Gate (18944×3584) | 200.2 | 175.3 | **+14.2%** |
| FFN_Up (18944×3584) | 201.4 | 175.7 | **+14.6%** |
| FFN_Down (3584×18944) | 215.8 | 177.9 | **+17.6%** |
| Q_proj (3584×3584) | 24.1 | 17.8 | **+26.2%** |
| Wo_proj (3584×3584) | 21.6 | 16.6 | **+23.1%** |

Default-vs-best gap reduced from 8-14% to <1% for FFN shapes.

## End-to-end Results (596-token prefill, 128-token decode, Qwen 7B Q8_0)

| Metric | nbc=2 (old) | nbc=1 (new) | Delta |
|--------|-------------|-------------|-------|
| Prefill | ~211 tok/s | ~235 tok/s | **+11%** |
| Decode | 4.29 tok/s | 4.31 tok/s | ~flat |

Note: Prefill numbers have ~15% run-to-run variance on shared VM; micro-benchmarks
are the authoritative measurement.

## Test Results

- 322/322 unit tests pass
- Sweep test confirms defaults are near-optimal (0-1% gap vs best for FFN shapes)
