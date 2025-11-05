/**
 * @file CudaGemmVariantsTensorCore.h
 * @brief Tensor Core GEMM kernel variants using CuTe (Phase 3+)
 *
 * Comprehensive variant dispatcher supporting full tile configuration space.
 * Integrated with auto-tuning framework for optimal kernel selection.
 *
 * @author David Sanftenberg
 * @date November 1, 2025
 */

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "CudaGemmConfig.h"

namespace llaminar2
{
    namespace cuda
    {

        // Forward declaration
        struct IQ4_NLBlock;

        /**
         * @brief Launch Tensor Core variant of IQ4_NL GEMM kernel
         *
         * Comprehensive implementation supporting ~75 tile configurations:
         * - TILE_M: {16, 32, 64, 128, 256}
         * - TILE_N: {16, 32, 64, 128, 256}
         * - TILE_K: {16, 32, 64}
         *
         * Phase 3 optimizations:
         * - 32×64×16 optimal for single-token decode (m=32) → 2,348 GFLOPS
         * - 64×64×16 good for small batches
         * - 128×128×16 optimal for large prefill
         *
         * @param A Input activation matrix [m × k] (FP32)
         * @param B_blocks Quantized weight blocks [n × (k/32)]
         * @param C Output matrix [m × n] (FP32)
         * @param m Number of rows in A and C
         * @param n Number of columns in B and C
         * @param k Number of columns in A (must be multiple of 32)
         * @param config Kernel configuration (tile dimensions must be multiples of 16)
         * @param stream CUDA stream (nullptr = default)
         * @return cudaSuccess on success, cudaErrorInvalidConfiguration if variant not instantiated
         */
        cudaError_t launchIQ4NLGemmVariantTensorCore(
            const float *A,
            const IQ4_NLBlock *B_blocks,
            float *C,
            int m, int n, int k,
            const CudaGemmConfig &config,
            cudaStream_t stream = nullptr);

        /**
         * @brief Get all available Tensor Core configurations
         *
         * Returns list of all explicitly instantiated configurations.
         * Used by auto-tuner to enumerate search space.
         *
         * Includes:
         * - 25 configs with TILE_K=16 (optimal for sm_80)
         * - 16 configs with TILE_K=32 (alternative)
         * - 12 configs with TILE_K=64 (experimental)
         *
         * Total: ~53 configurations
         *
         * @return Vector of valid Tensor Core configurations
         */
        std::vector<CudaGemmConfig> getTensorCoreConfigs();

    } // namespace cuda
} // namespace llaminar2
