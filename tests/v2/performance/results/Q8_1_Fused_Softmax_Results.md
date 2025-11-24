# Q8_1 GEMM Fused Online Softmax Performance Results

## Benchmark Configuration
- **Date**: 2025-10-24
- **Hardware**: 2 Sockets, 28 Cores/Socket (Simulated/Detected)
- **Build**: Release (AVX512/AVX2 enabled)
- **Comparison**: 
  - **Baseline**: Q8_1 GEMM + OpenMP Standalone Softmax
  - **Fused**: Q8_1 GEMM with Online Softmax (Fused Kernel)

## Results

| Batch Size (M) | N    | K    | Baseline (ms) | Fused (ms) | Speedup |
|----------------|------|------|---------------|------------|---------|
| 1              | 4096 | 4096 | 0.109         | 0.082      | **1.33x** |
| 2              | 4096 | 4096 | 0.110         | 0.088      | **1.25x** |
| 32             | 4096 | 4096 | 0.566         | 0.516      | **1.10x** |

## Analysis
- **Latency Critical (M=1)**: Significant 33% speedup. This is crucial for token generation latency.
- **Small Batch (M=2)**: Strong 25% speedup.
- **Throughput (M=32)**: Moderate 10% speedup. As compute intensity increases (O(M*N*K)), the O(M*N) Softmax cost becomes less dominant, but the fused kernel still provides a measurable benefit by avoiding a memory round-trip for the output matrix C.

## Conclusion
The fused Online Softmax implementation provides substantial performance benefits for low-latency inference scenarios (M=1, M=2) and consistent improvements for larger batches. The implementation is verified correct and performant.
