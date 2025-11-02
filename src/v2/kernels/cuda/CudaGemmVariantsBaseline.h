/**
 * @file CudaGemmVariantsBaseline.h
 * @brief Baseline CUDA GEMM kernel variants (no special optimizations)
 *
 * Standard template-based GEMM kernels with configurable tile sizes.
 * Performance: ~3,000 GFLOPS baseline.
 *
 * @author David Sanftenberg
 * @date October 31, 2025
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
         * @brief Launch IQ4_NL GEMM kernel with specific configuration
         *
         * Dispatches to appropriate template-instantiated kernel variant based on config.
         *
         * @param A Activation matrix [m × k] FP32
         * @param B_blocks Weight matrix blocks [n × k/32] IQ4_NL
         * @param C Output matrix [m × n] FP32
         * @param m Number of rows in A
         * @param n Number of rows in B
         * @param k Number of columns in A (must be multiple of 32)
         * @param config Kernel configuration parameters
         * @param stream CUDA stream for async execution
         * @return cudaError_t Error code
         */
        cudaError_t launchIQ4NLGemmVariant(
            const float *A,
            const IQ4_NLBlock *B_blocks,
            float *C,
            int m, int n, int k,
            const CudaGemmConfig &config,
            cudaStream_t stream = 0);

    } // namespace cuda
} // namespace llaminar2
