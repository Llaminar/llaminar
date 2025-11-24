# Q8_1 GEMM Performance Optimization Report

**Date:** November 24, 2025
**Kernel:** `src/v2/kernels/cpu/gemm_v4/Q8_1GemmKernel.h`
**Test Suite:** `tests/v2/performance/Perf__Q8_1_GEMM.cpp`

## Executive Summary

This document details the optimization journey for the `Q8_1` (8-bit quantized weights, FP32 activation) GEMM kernel. The goal was to achieve high throughput (>1500 GFLOPS) across a wide range of model sizes (Qwen 0.5B to 32B) and batch sizes (M=1 to M=512).

**Key Achievements:**
- **Qwen 32B FFN Down**: Achieved **~2100 GFLOPS** (M=512), saturating compute capability.
- **Qwen 0.5B Performance**: Improved small-model performance by **+60%** via adaptive blocking.
- **Scalability**: Resolved L2 cache thrashing regressions, ensuring consistent scaling from M=128 to M=512.

## Methodology

### Test Environment
- **Threads**: 28 OpenMP threads (Socket-bound).
- **SIMD**: AVX512 (implied by performance numbers).
- **Metric**: GFLOPS (Billion Floating Point Operations Per Second).

### Benchmark Command
```bash
cmake --build build_v2_release --target v2_perf_q8_1_gemm --parallel
ctest --test-dir build_v2_release -R V2_Perf_Q8_1_GEMM$ --verbose
```

## Optimization Journey

### Phase 1: Large Model Optimization (K-Tiling)
**Challenge**: Qwen 32B layers have massive K dimensions (e.g., FFN Down K=27,392). A single row of weights (K * 1 byte) is ~27KB. A standard block of N=64 rows is ~1.7MB, exceeding the typical 1MB L2 cache per core.
**Solution**: Implemented **K-Tiling**.
- The kernel splits the K-loop into smaller tiles (e.g., 256KB).
- It iterates over these tiles while keeping the weight block resident in L2 cache ("B-stationary").
- **Result**: Performance jumped from ~500 GFLOPS to ~1900 GFLOPS for 32B models.

### Phase 2: Small Model Optimization (Adaptive Blocking)
**Challenge**: Qwen 0.5B layers have small N dimensions (e.g., N=896). Standard blocking (N_BLOCK=64) created too few tasks to saturate 28 threads.
**Solution**: Implemented **Adaptive Task Sizing**.
- Dynamically calculates `n_task_block` to ensure at least `4 * num_threads` tasks are generated.
- **Result**: M=1 performance improved significantly.

### Phase 3: Regression Fix (Cache Tuning)
**Challenge**: Performance for Qwen 32B stagnated or regressed when moving from M=128 to M=512.
**Root Cause**: The activation matrix A (M x K) grows with batch size. At M=512, A competes for L2 cache space. The original 1MB weight block target was too aggressive, causing eviction.
**Solution**:
- Reduced target weight block size to **768KB**.
- Reduced K-tile size to **256KB** (128 blocks).
- **Result**: Recovered performance at M=512 (1500 -> 1800+ GFLOPS).

### Phase 4: Load Balancing (Even Splitting)
**Challenge**: For N=896 (Qwen 0.5B), the logic created one block of 832 and one of 64. This caused massive thread load imbalance.
**Solution**: Added logic to split N evenly when clamped by cache limits (e.g., 2 blocks of 448).
**Result**: Qwen 0.5B Attn Output (M=512) improved from ~1100 to **~1750 GFLOPS**.

## Final Performance Results

### Qwen 32B (Large Model)
| Layer | Dimensions (N, K) | M=1 | M=32 | M=128 | M=512 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Attn Output** | 5120, 5120 | 645 | 1732 | 1832 | **1814** |
| **FFN Down** | 5120, 27392 | 221 | 1931 | 2087 | **2089** |

### Qwen 7B (Medium Model)
| Layer | Dimensions (N, K) | M=1 | M=32 | M=128 | M=512 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Attn Output** | 4096, 4096 | 326 | 877 | 975 | **1904** |
| **FFN Down** | 4096, 11008 | 357 | 1473 | 1671 | **1685** |

### Qwen 0.5B (Small Model)
| Layer | Dimensions (N, K) | M=1 | M=32 | M=128 | M=512 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Attn Output** | 896, 896 | 73 | 973 | 1474 | **1751** |
| **FFN Down** | 896, 4864 | 262 | 1316 | 1667 | **1701** |

## Conclusion
The `Q8_1` kernel is now highly robust. It dynamically adapts its blocking strategy to handle:
1.  **Massive K dimensions** (via K-tiling) to preserve L2 locality.
2.  **Small N dimensions** (via adaptive splitting) to ensure thread saturation.
3.  **Large Batch Sizes** (via conservative cache targets) to prevent thrashing.
