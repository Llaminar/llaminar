/**
 * @file CudaGemmVariantsTensorCore.cu
 * @brief Phase 2 Tensor Core CUDA GEMM kernel launcher (CuTe API)
 *
 * @author David Sanftenberg
 * @date November 1, 2025
 *
 * ✅ **UPDATED**: Now uses CUTLASS CuTe template API (modern approach)
 *
 * PURPOSE: Launch dispatcher for Tensor Core kernels via CuTe
 *
 * PHASE 2 OPTIMIZATIONS:
 * - Tensor Core MMA via SM80_16x8x16_F32F16F16F32_TN
 * - Mixed precision: FP16 compute, FP32 accumulation
 * - CuTe template abstractions for cleaner code
 * - Target: 4-6× speedup over Phase 1 (425 → 1,700-2,550 GFLOPS)
 */

#include "CudaGemmVariantsTensorCore.h"
#include "CudaGemmKernel.cuh"
#include "CudaGemmConfig.h"
#include "IQ4_NL_BlockDecoder.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace llaminar2
{
    namespace cuda
    {

        /**
         * @brief Launch dispatcher for CuTe-based Tensor Core IQ4_NL kernel
         *
         * COMPREHENSIVE IMPLEMENTATION (Phase 3+):
         * - Supports full configuration space (TILE_M × TILE_N × TILE_K)
         * - Integrated with auto-tuning framework
         * - Explicitly instantiated variants for all valid tile combinations
         *
         * Configuration space:
         * - TILE_M: {16, 32, 64, 128, 256} (Phase 3: 32 optimal for m=32 decode)
         * - TILE_N: {16, 32, 64, 128, 256}
         * - TILE_K: {16, 32, 64} (16 is optimal for sm_80)
         *
         * Total variants: ~75 configurations
         *
         * @param A Input activation matrix [m × k] (FP32)
         * @param B_blocks Quantized weight blocks [n × (k/32)]
         * @param C Output matrix [m × n] (FP32)
         * @param m Number of rows in A and C
         * @param n Number of columns in B and C
         * @param k Number of columns in A and rows in B
         * @param config Kernel configuration (tile_m, tile_n, tile_k)
         * @param stream CUDA stream (nullptr = default stream)
         * @return cudaError_t Success or error code
         */
        cudaError_t launchIQ4NLGemmVariantTensorCore(
            const float *A,
            const IQ4_NLBlock *B_blocks,
            float *C,
            int m, int n, int k,
            const CudaGemmConfig &config,
            cudaStream_t stream)
        {
            // Validate inputs
            if (!A || !B_blocks || !C || m <= 0 || n <= 0 || k <= 0)
            {
                return cudaErrorInvalidValue;
            }

            if (k % 32 != 0)
            {
                return cudaErrorInvalidValue; // IQ4_NL block size is 32
            }

            // Create IQ4_NL decoder
            const int num_k_blocks = k / 32;
            IQ4_NL_Decoder<IQ4_NLBlock> decoder(B_blocks, n, num_k_blocks);

// Macro to reduce boilerplate for template instantiation
#define LAUNCH_TENSORCORE(TM, TN, TK)                                                   \
    if (config.tile_m == TM && config.tile_n == TN && config.tile_k == TK)              \
    {                                                                                   \
        return launchQuantizedGemmCuTe<float, IQ4_NL_Decoder<IQ4_NLBlock>, TM, TN, TK>( \
            A, C, m, n, k, decoder, stream);                                            \
    }

            // ============================================================================
            // COMPREHENSIVE TILE CONFIGURATION SPACE
            // ============================================================================
            // Based on Phase 3 findings and model size requirements:
            // - Small tiles (16, 32) good for single-token decode (m=1-32)
            // - Medium tiles (64) good for small batches (m=32-128)
            // - Large tiles (128, 256) good for prefill and large batches (m≥128)

            // TILE_K = 16 (OPTIMAL for sm_80 - matches MMA instruction)
            // -------------------------------------------------------------------------
            // Small tiles - optimal for single-token decode (0.5B, 4B, 7B, 14B)
            LAUNCH_TENSORCORE(16, 16, 16);
            LAUNCH_TENSORCORE(16, 32, 16);
            LAUNCH_TENSORCORE(16, 64, 16);
            LAUNCH_TENSORCORE(16, 128, 16);
            LAUNCH_TENSORCORE(16, 256, 16);

            LAUNCH_TENSORCORE(32, 16, 16);
            LAUNCH_TENSORCORE(32, 32, 16);
            LAUNCH_TENSORCORE(32, 64, 16); // Phase 3 winner for m=32!
            LAUNCH_TENSORCORE(32, 128, 16);
            LAUNCH_TENSORCORE(32, 256, 16);

            // Medium tiles - good for small batches
            LAUNCH_TENSORCORE(64, 16, 16);
            LAUNCH_TENSORCORE(64, 32, 16);
            LAUNCH_TENSORCORE(64, 64, 16); // Phase 2.5 baseline
            LAUNCH_TENSORCORE(64, 128, 16);
            LAUNCH_TENSORCORE(64, 256, 16);

            // Large tiles - good for prefill and large batches
            LAUNCH_TENSORCORE(128, 16, 16);
            LAUNCH_TENSORCORE(128, 32, 16);
            LAUNCH_TENSORCORE(128, 64, 16);
            LAUNCH_TENSORCORE(128, 128, 16);
            LAUNCH_TENSORCORE(128, 256, 16);

            LAUNCH_TENSORCORE(256, 16, 16);
            LAUNCH_TENSORCORE(256, 32, 16);
            LAUNCH_TENSORCORE(256, 64, 16);
            LAUNCH_TENSORCORE(256, 128, 16);
            LAUNCH_TENSORCORE(256, 256, 16);

            // TILE_K = 32 (ALTERNATIVE - more K-tiles, may help large matrices)
            // -------------------------------------------------------------------------
            LAUNCH_TENSORCORE(16, 16, 32);
            LAUNCH_TENSORCORE(16, 32, 32);
            LAUNCH_TENSORCORE(16, 64, 32);
            LAUNCH_TENSORCORE(16, 128, 32);

            LAUNCH_TENSORCORE(32, 16, 32);
            LAUNCH_TENSORCORE(32, 32, 32);
            LAUNCH_TENSORCORE(32, 64, 32); // Phase 3: worse than K=16
            LAUNCH_TENSORCORE(32, 128, 32);

            LAUNCH_TENSORCORE(64, 16, 32);
            LAUNCH_TENSORCORE(64, 32, 32);
            LAUNCH_TENSORCORE(64, 64, 32);
            LAUNCH_TENSORCORE(64, 128, 32);

            LAUNCH_TENSORCORE(128, 16, 32);
            LAUNCH_TENSORCORE(128, 32, 32);
            LAUNCH_TENSORCORE(128, 64, 32);
            LAUNCH_TENSORCORE(128, 128, 32);

            // TILE_K = 64 (EXPERIMENTAL - fewer K-tiles, larger shared memory)
            // -------------------------------------------------------------------------
            LAUNCH_TENSORCORE(16, 16, 64);
            LAUNCH_TENSORCORE(16, 32, 64);
            LAUNCH_TENSORCORE(16, 64, 64);

            LAUNCH_TENSORCORE(32, 16, 64);
            LAUNCH_TENSORCORE(32, 32, 64);
            LAUNCH_TENSORCORE(32, 64, 64);

            LAUNCH_TENSORCORE(64, 16, 64);
            LAUNCH_TENSORCORE(64, 32, 64);
            LAUNCH_TENSORCORE(64, 64, 64); // Large shared memory (>48KB, may not fit sm_70)

            LAUNCH_TENSORCORE(128, 16, 64);
            LAUNCH_TENSORCORE(128, 32, 64);
            LAUNCH_TENSORCORE(128, 64, 64);

#undef LAUNCH_TENSORCORE

            // If no variant matched, return error
            return cudaErrorInvalidConfiguration;
        }

        /**
         * @brief Get all available Tensor Core configurations
         *
         * Returns list of all explicitly instantiated configurations.
         * Used by auto-tuner to enumerate search space.
         *
         * @return Vector of valid configurations
         */
        std::vector<CudaGemmConfig> getTensorCoreConfigs()
        {
            std::vector<CudaGemmConfig> configs;

            // Helper lambda to add configuration
            auto add_config = [&](int tm, int tn, int tk)
            {
                CudaGemmConfig cfg;
                cfg.tile_m = tm;
                cfg.tile_n = tn;
                cfg.tile_k = tk;
                // Note: threads_m, threads_n, work_m, work_n are determined by CuTe kernel
                // These fields are not used for Tensor Core variants (CuTe handles layout)
                cfg.threads_m = 0;
                cfg.threads_n = 0;
                cfg.work_per_thread_m = 0;
                cfg.work_per_thread_n = 0;
                cfg.prefetch_stages = 0;
                cfg.transpose_smem = false;
                cfg.vectorize_load = 0;
                configs.push_back(cfg);
            };

            // Add all TILE_K=16 configurations (OPTIMAL)
            for (int tm : {16, 32, 64, 128, 256})
            {
                for (int tn : {16, 32, 64, 128, 256})
                {
                    add_config(tm, tn, 16);
                }
            }

            // Add selected TILE_K=32 configurations
            for (int tm : {16, 32, 64, 128})
            {
                for (int tn : {16, 32, 64, 128})
                {
                    add_config(tm, tn, 32);
                }
            }

            // Add selected TILE_K=64 configurations (small tiles only to avoid excessive shared memory)
            for (int tm : {16, 32, 64, 128})
            {
                for (int tn : {16, 32, 64})
                {
                    add_config(tm, tn, 64);
                }
            }

            return configs;
        }

    } // namespace cuda
} // namespace llaminar2
