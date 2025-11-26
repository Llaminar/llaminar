# Q8_1 GEMM Performance Optimization Report

**Date:** November 25, 2025
**Kernel:** `src/v2/kernels/cpu/gemm_v4/Q8_1GemmKernel.h`
**Test Suites:** 
- `tests/v2/performance/Perf__Q8_1_GEMM.cpp`
- `tests/v2/performance/Perf__Q8_0_GEMM.cpp`
- `tests/v2/performance/Perf__Q4_0_GEMM.cpp`

## Executive Summary

This document details the optimization journey for the `Q8_1` (8-bit quantized weights, FP32 activation) GEMM kernel. The goal was to achieve high throughput (>1500 GFLOPS) across a wide range of model sizes (Qwen 0.5B to 32B) and batch sizes (M=1 to M=512).

**Key Achievements:**
- **Qwen 32B FFN Down**: Achieved **~2100 GFLOPS** (M=512), saturating compute capability.
- **Qwen 0.5B Performance**: Improved small-model performance by **+60%** via adaptive blocking.
- **Scalability**: Resolved L2 cache thrashing regressions, ensuring consistent scaling from M=128 to M=512.
- **Generic Packing**: Successfully integrated `IINT8Unpackable` support, enabling Q4_0 and Q8_0 inference with minimal overhead.

## Methodology

### Test Environment
- **Threads**: 28 OpenMP threads (Socket-bound).
- **SIMD**: AVX512 (implied by performance numbers).
- **Metric**: GFLOPS (Billion Floating Point Operations Per Second).

### Benchmark Command
```bash
cmake --build build_v2_release --target v2_perf_q8_1_gemm v2_perf_q8_0_gemm v2_perf_q4_0_gemm --parallel
ctest --test-dir build_v2_release -R "V2_Perf_Q.*_GEMM$" --verbose
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

### Phase 5: Online Softmax Fusion
**Challenge**: In Attention layers, the GEMM output (Attention Scores) is immediately followed by a Softmax operation. Writing the large GEMM output (M x N) to memory and then reading it back for Softmax incurs significant memory bandwidth overhead, especially for small batches (latency-critical) or large contexts.
**Solution**: Implemented **Online Softmax Fusion**.
- Fused the Softmax calculation directly into the GEMM kernel's accumulation loop.
- Computes local max and sum-of-exponentials during the final reduction.
- Avoids writing the raw GEMM output to memory entirely when `do_softmax=true`.
- **Result**: Significant latency reduction for single-token and small-batch inference.

**Performance Impact (M=1 to M=32):**
| Batch Size (M) | Baseline (ms) | Fused (ms) | Speedup | Context |
| :--- | :--- | :--- | :--- | :--- |
| **M=1** | 0.109 | **0.082** | **1.33x** | Single Token Decode |
| **M=2** | 0.110 | **0.088** | **1.25x** | Small Batch |
| **M=32** | 0.566 | **0.516** | **1.10x** | Batch Prefill |

*Note: Benchmarked with N=4096, K=4096 on 2-socket system.*

## Final Performance Results

### Q8_1 GEMM (Native Format)
| Model | Layer | Dimensions (N, K) | M=1 | M=32 | M=128 | M=512 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 32B** | Attn Output | 5120, 5120 | 603 | 990 | 1550 | **1796** |
| **Qwen 32B** | FFN Down | 5120, 27392 | 161 | 1553 | 2002 | **2075** |
| **Qwen 7B** | Attn Output | 4096, 4096 | 580 | 1577 | 1731 | **1810** |
| **Qwen 7B** | FFN Down | 4096, 11008 | 257 | 1055 | 1693 | **1700** |
| **Qwen 0.5B** | Attn Output | 896, 896 | 70 | 940 | 1425 | **1718** |
| **Qwen 0.5B** | FFN Down | 896, 4864 | 249 | 1298 | 1652 | **1675** |

### Q8_0 GEMM (Generic Packing)
| Model | Layer | Dimensions (N, K) | M=1 | M=32 | M=128 | M=512 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 32B** | Attn Output | 5120, 5120 | 595 | 1665 | 1757 | **1793** |
| **Qwen 32B** | FFN Down | 5120, 27392 | 167 | 1612 | 1895 | **2028** |
| **Qwen 7B** | Attn Output | 4096, 4096 | 329 | 984 | 1348 | **1858** |
| **Qwen 7B** | FFN Down | 4096, 11008 | 258 | 1116 | 1712 | **1694** |
| **Qwen 0.5B** | Attn Output | 896, 896 | 55 | 889 | 1441 | **1690** |
| **Qwen 0.5B** | FFN Down | 896, 4864 | 219 | 1304 | 1656 | **1662** |

### Q4_0 GEMM (Generic Packing)
| Model | Layer | Dimensions (N, K) | M=1 | M=32 | M=128 | M=512 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 32B** | Attn Output | 5120, 5120 | 632 | 1595 | 1851 | **1884** |
| **Qwen 32B** | FFN Down | 5120, 27392 | 168 | 1533 | 1755 | **1838** |
| **Qwen 0.5B** | Attn Output | 896, 896 | 62 | 916 | 1450 | **1723** |
| **Qwen 0.5B** | FFN Down | 896, 4864 | 250 | 1277 | 1594 | **1398** |

## Conclusion
The `Q8_1` kernel is now highly robust. It dynamically adapts its blocking strategy to handle:
1.  **Massive K dimensions** (via K-tiling) to preserve L2 locality.
2.  **Small N dimensions** (via adaptive splitting) to ensure thread saturation.
3.  **Large Batch Sizes** (via conservative cache targets) to prevent thrashing.
4.  **Multiple Quantization Formats** (via `IINT8Unpackable`) with minimal overhead.

### Phase 5: Online Softmax Fusion
**Challenge**: In Attention layers, the GEMM output (Attention Scores) is immediately followed by a Softmax operation. Writing the large GEMM output (M x N) to memory and then reading it back for Softmax incurs significant memory bandwidth overhead, especially for small batches (latency-critical) or large contexts.
**Solution**: Implemented **Online Softmax Fusion**.
- Fused the Softmax calculation directly into the GEMM kernel's accumulation loop.
- Computes local max and sum-of-exponentials during the final reduction.
- Avoids writing the raw GEMM output to memory entirely when `do_softmax=true`.
- **Result**: Significant latency reduction for single-token and small-batch inference.

**Performance Impact (M=1 to M=32):**
| Batch Size (M) | Baseline (ms) | Fused (ms) | Speedup | Context |
| :--- | :--- | :--- | :--- | :--- |
| **M=1** | 0.109 | **0.082** | **1.33x** | Single Token Decode |
| **M=2** | 0.110 | **0.088** | **1.25x** | Small Batch |
| **M=32** | 0.566 | **0.516** | **1.10x** | Batch Prefill |

*Note: Benchmarked with N=4096, K=4096 on 2-socket system.*
