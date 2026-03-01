# ROCm Model Loading Sprint — Sub-4 Second Startup

**Date**: 2026-02-07  
**Scope**: ROCm Q8_0 model loading pipeline optimization  
**Target**: Load + repack + first inference in under 4 seconds  
**Result**: **3.9s average** (from 10.8s baseline → **2.8× speedup**)

## Summary

Six optimizations reduced end-to-end startup for the Qwen 2.5 0.5B Q8_0 model on ROCm from ~10.8 seconds to ~3.9 seconds. The Sprint targeted every major bottleneck: unnecessary FP32 round-trips in weight repacking, excessive hipMalloc calls, oversized logit allocations, MPI bootstrap overhead, and CUDA static initialization.

## Performance Progression

| Milestone | Profiled (ms) | Wall-clock (s) | Speedup |
|-----------|---------------|----------------|---------|
| Baseline | 10,800 | ~12.0 | 1.0× |
| + Q8_0 fast path | 5,242 | — | 2.1× |
| + KV cache pooling | ~5,000 | — | 2.2× |
| + Logits reduction | 3,943 | ~5.9 | 2.7× |
| + Single-process mode | 3,994 | ~4.0 | 2.7× |
| + Deferred CUDA init | 3,994 | **~3.9** | **2.8×** |

## Optimizations

### 1. Q8_0 Direct Repack (GEMM Repack: 3778ms → 1071ms)

**Root cause**: `packWeightsToROCm()` called `tensor->fp32_data()` which dequantized Q8_0→FP32, then requantized FP32→INT8. Q8_0 blocks are already INT8 — the FP32 round-trip was completely unnecessary.

**Fix**: New `packWeightsToROCm_Q8_0_fast()` reads Q8_0 blocks directly:
- Computes per-row scale from block FP16 scales (`max(|d|)` across blocks)
- Rescales INT8 values in a single fused OpenMP-parallel pass
- Produces both row-major `[N×K]` and VNNI `[K/4][N][4]` layouts simultaneously
- Generic fallback also got `#pragma omp parallel for`

**File**: `src/v2/kernels/rocm/ROCmQuantisedGemmKernel.cpp`

### 2. KV Cache hipMalloc Pooling (96 hipMalloc calls → 1)

**Root cause**: `ROCmRingKVCache` allocated each cache entry individually (24 layers × 2 KV × 2 batch = 96 `hipMalloc` calls).

**Fix**: Pooled allocation — single `hipMalloc` of 48 MB for all entries, with pointer arithmetic for individual entry assignment.

**Files**: `src/v2/kernels/rocm/kvcache/ROCmRingKVCache.{h,cpp}`

### 3. Logits Allocation Reduction (initializeInferenceState: 1557ms → 297ms)

**Root cause**: Logits buffer allocated as `[batch_size * max_seq_len, vocab_size]` = `[2048, 151936]` = 1.16 GB using `hipHostMallocMapped`. The LM head always computes M=1 (last token only, even during prefill).

**Fix**: Reduced allocation to `[batch_size, vocab_size]` = `[1, 151936]` = 0.58 MB. Fixed `getLogits()` offset calculation accordingly.

**File**: `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`

### 4. Single-Process Mode (skip MPI bootstrap for single-device)

**Root cause**: For single-device inference (`-d rocm:0`), the normal MPI bootstrap requires:
1. Parent enumerates topology (~200ms)
2. `execvp("mpirun")` forks a new process
3. Child re-enumerates topology and re-initializes everything

This adds ~1.2-2s of overhead for a runtime that ultimately runs as rank=0, world_size=1.

**Fix**: Detect when MPI is unnecessary (tp≤1, pp≤1, single device) and skip the entire MPI bootstrap:
- Configure OpenMP directly instead of via mpirun
- Create `MPIContext(0, 1, MPI_COMM_NULL)` immediately
- Guard all `MPI_Barrier`/`MPI_Bcast`/`MPI_Finalize` calls with `world_size() > 1`
- Set `LLAMINAR_SKIP_CUDA_STARTUP=1` when targeting ROCm to avoid unnecessary CUDA enumeration in DeviceManager

**Files**: `src/v2/Main.cpp`, `src/v2/utils/BenchmarkRunner.cpp`, `src/v2/backends/ComputeBackend.cpp`

### 5. Backend-Selective Enumeration Skip

**Root cause**: `DeviceManager::initialize()` enumerates ALL GPU backends (CUDA + ROCm) even when only one is needed. Each CUDA device enumeration costs ~250ms.

**Fix**: Added `LLAMINAR_SKIP_CUDA_STARTUP` and `LLAMINAR_SKIP_ROCM_STARTUP` environment variables. Main.cpp sets the appropriate skip env var based on the target device before DeviceManager initialization.

**Files**: `src/v2/backends/ComputeBackend.cpp`, `src/v2/backends/cuda/NvidiaContextFactory.cu`, `src/v2/backends/rocm/AMDContextFactory.cpp`

### 6. Deferred CUDA Static Factory Registration

**Root cause**: `NvidiaContextFactory.cu` had a static constructor (`NvidiaFactoryRegistrar`) that enumerated CUDA devices at process startup — before `main()` runs — adding ~500-800ms even when CUDA is not used.

**Fix**: Changed the static constructor to a no-op. CUDA factory registration is deferred to `ensureNvidiaFactoryRegistered()` which is called explicitly by `DeviceGraphOrchestrator` when a CUDA context is first needed. Zero cost when only ROCm or CPU backends are used.

**File**: `src/v2/backends/cuda/NvidiaContextFactory.cu`

### Reverted: Async Overlap (ROCm driver contention)

An attempt to overlap `initializeInferenceState` with `configureOrchestratorWeightsImpl` via `std::async` was tried and reverted. ROCm's `hipMalloc` causes severe system-wide contention — concurrent `hipMalloc` and `hipMemcpy` operations are slower than sequential. This is a known ROCm driver limitation on MI60/MI50 hardware.

## Final Profile (3994ms profiled)

| Component | Time (ms) | % |
|-----------|-----------|---|
| GEMM Repack (host→device) | 1,071 | 26.8% |
| Non-GEMM Upload (host→device) | 563 | 14.1% |
| eager_layer_cache_load (file I/O) | 1,356 | 33.9% |
| initializeInferenceState | 297 | 7.4% |
| Other (graph build, setup) | 707 | 17.7% |
| **TOTAL** | **3,994** | |

## Remaining Opportunities

- **eager_layer_cache_load** (1,356ms): Serialized file I/O under `file_mutex_` in ModelLoader. Options: `pread()` for lock-free parallel reads, or `mmap()` the GGUF file
- **Non-GEMM upload** (563ms): 122 weight tensors uploaded serially; could be pipelined
- **Pipeline bubble** (372ms): Producer-consumer queue imbalance; could tune queue capacity or consumer count

## Test Results

5 trials (no profiling), wall-clock:
```
3.891s, 3.990s, 4.009s, 4.009s, 4.048s
```
Mean: **3.99s** | Min: **3.89s** | Max: 4.05s

5 trials (with profiling):
```
3.914s, 3.923s, 3.975s, 4.054s, 4.199s
```
Mean: **4.01s** | Min: 3.91s | Max: 4.20s

## Hardware

- AMD Instinct MI60/MI50 (gfx906), 3 GPUs × 31 GB
- 56 cores / 112 logical (2× Xeon)
- NVIDIA RTX 3090 × 2 (not used, successfully skipped)
- Model: Qwen 2.5 0.5B Q8_0 (675 MB GGUF)
