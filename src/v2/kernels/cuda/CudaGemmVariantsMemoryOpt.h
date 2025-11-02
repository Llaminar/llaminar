/**
 * @file CudaGemmVariantsMemoryOpt.h
 * @brief Memory-optimized CUDA GEMM kernel variants (Phase 1 optimizations)
 *
 * Optimizations: Coalesced memory access, vectorized loads, shared memory padding.
 * Performance: Target 6,000-9,000 GFLOPS (2-3× speedup over baseline).
 *
 * @author David Sanftenberg
 * @date November 1, 2025
 */

#pragma once

#include "CudaGemmConfig.h"
#include "IQ4_NL_BlockDecoder.h"
#include <cuda_runtime.h>

namespace llaminar2
{
    namespace cuda
    {

        /**
         * @brief Launch optimized IQ4_NL GEMM kernel with Phase 1 improvements
         *
         * OPTIMIZATIONS:
         * - ✅ Coalesced memory access (2-3× faster loads)
         * - ✅ Vectorized float4 loads (4× load throughput)
         * - ✅ Shared memory padding (+1 to avoid bank conflicts)
         * - ✅ True TRANSPOSE_SMEM implementation
         *
         * EXPECTED SPEEDUP: 2-3× over baseline
         *
         * @param A         Activation matrix [m × k] (FP32)
         * @param B_blocks  Quantized weight blocks (IQ4_NL format)
         * @param C         Output matrix [m × n] (FP32)
         * @param m         Batch size / rows
         * @param n         Output features
         * @param k         Input features
         * @param config    Kernel configuration (tile sizes, thread counts, etc.)
         * @param stream    CUDA stream (nullptr for default stream)
         *
         * @return cudaSuccess or error code
         */
        cudaError_t launchIQ4NLGemmVariantOptimized(
            const float *A,
            const IQ4_NLBlock *B_blocks,
            float *C,
            int m, int n, int k,
            const CudaGemmConfig &config,
            cudaStream_t stream);

    } // namespace cuda
} // namespace llaminar2
