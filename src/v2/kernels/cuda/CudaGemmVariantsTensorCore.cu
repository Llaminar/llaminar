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
#include <cute/arch/mma_sm80.hpp> // For SM80 MMA atom types

namespace llaminar2
{
    namespace cuda
    {
        using namespace cute; // For SM80_16x8x16_F32F16F16F32_TN

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

            // ============================================================================
            // ATOM-AWARE KERNEL DISPATCH
            // ============================================================================
            // The configuration now includes atom_type and atom_layout_* parameters
            // We dispatch to different kernel instantiations based on these settings
            //
            // Atom types:
            //   0 = SM80_16x8x16_F32F16F16F32_TN (K=16, more efficient for larger K)
            //   1 = SM80_16x8x8_F32F16F16F32_TN  (K=8,  smaller footprint)
            //
            // Atom layouts:
            //   1×1×1 = Single atom (16×8 or 16×8 output per K-slice)
            //   2×2×1 = 4 atoms (32×16 output per K-slice) - was hardcoded default
            //   4×4×1 = 16 atoms (64×32 output per K-slice)

// Helper macros for different atom configurations
#define LAUNCH_ATOM_16x8x16(AM, AN, AK, TM, TN, TK)                                                                                   \
    if (config.atom_type == 0 && config.atom_layout_m == AM && config.atom_layout_n == AN && config.atom_layout_k == AK &&           \
        config.tile_m == TM && config.tile_n == TN && config.tile_k == TK)                                                           \
    {                                                                                                                                \
        return launchQuantizedGemmCuTe<float, SM80_16x8x16_F32F16F16F32_TN, AM, AN, AK, IQ4_NL_Decoder<IQ4_NLBlock>, TM, TN, TK>( \
            A, C, m, n, k, decoder, stream);                                                                                         \
    }

#define LAUNCH_ATOM_16x8x8(AM, AN, AK, TM, TN, TK)                                                                                  \
    if (config.atom_type == 1 && config.atom_layout_m == AM && config.atom_layout_n == AN && config.atom_layout_k == AK &&          \
        config.tile_m == TM && config.tile_n == TN && config.tile_k == TK)                                                          \
    {                                                                                                                               \
        return launchQuantizedGemmCuTe<float, SM80_16x8x8_F32F16F16F32_TN, AM, AN, AK, IQ4_NL_Decoder<IQ4_NLBlock>, TM, TN, TK>( \
            A, C, m, n, k, decoder, stream);                                                                                        \
    }

// Convenience macro that tries both atom types with given layout and tile size
#define LAUNCH_TENSORCORE(AM, AN, AK, TM, TN, TK) \
    LAUNCH_ATOM_16x8x16(AM, AN, AK, TM, TN, TK)    \
    LAUNCH_ATOM_16x8x8(AM, AN, AK, TM, TN, TK)

            // ============================================================================
            // ATOM-AWARE TILE CONFIGURATION SPACE
            // ============================================================================
            // Systematically generated for all atom layout × tile size combinations
            // Total configs: 3 atom layouts × 53 tile sizes = 159 kernel instantiations
            // Each instantiates 2 atom types → 318 total kernel variants

            // Atom layout 1×1×1 (Small: 16×8 output tile per atom)
            // -------------------------------------------------------------------------
            LAUNCH_TENSORCORE(1, 1, 1, 16, 16, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 16, 32, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 16, 64, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 16, 128, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 16, 256, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 32, 16, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 32, 32, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 32, 64, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 32, 128, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 32, 256, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 64, 16, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 64, 32, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 64, 64, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 64, 128, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 64, 256, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 128, 16, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 128, 32, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 128, 64, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 128, 128, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 128, 256, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 256, 16, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 256, 32, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 256, 64, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 256, 128, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 256, 256, 16);
            LAUNCH_TENSORCORE(1, 1, 1, 16, 16, 32);
            LAUNCH_TENSORCORE(1, 1, 1, 16, 32, 32);
            LAUNCH_TENSORCORE(1, 1, 1, 16, 64, 32);
            LAUNCH_TENSORCORE(1, 1, 1, 16, 128, 32);
            LAUNCH_TENSORCORE(1, 1, 1, 32, 16, 32);
            LAUNCH_TENSORCORE(1, 1, 1, 32, 32, 32);
            LAUNCH_TENSORCORE(1, 1, 1, 32, 64, 32);
            LAUNCH_TENSORCORE(1, 1, 1, 32, 128, 32);
            LAUNCH_TENSORCORE(1, 1, 1, 64, 16, 32);
            LAUNCH_TENSORCORE(1, 1, 1, 64, 32, 32);
            LAUNCH_TENSORCORE(1, 1, 1, 64, 64, 32);
            LAUNCH_TENSORCORE(1, 1, 1, 64, 128, 32);
            LAUNCH_TENSORCORE(1, 1, 1, 128, 16, 32);
            LAUNCH_TENSORCORE(1, 1, 1, 128, 32, 32);
            LAUNCH_TENSORCORE(1, 1, 1, 128, 64, 32);
            LAUNCH_TENSORCORE(1, 1, 1, 128, 128, 32);
            LAUNCH_TENSORCORE(1, 1, 1, 16, 16, 64);
            LAUNCH_TENSORCORE(1, 1, 1, 16, 32, 64);
            LAUNCH_TENSORCORE(1, 1, 1, 16, 64, 64);
            LAUNCH_TENSORCORE(1, 1, 1, 32, 16, 64);
            LAUNCH_TENSORCORE(1, 1, 1, 32, 32, 64);
            LAUNCH_TENSORCORE(1, 1, 1, 32, 64, 64);
            LAUNCH_TENSORCORE(1, 1, 1, 64, 16, 64);
            LAUNCH_TENSORCORE(1, 1, 1, 64, 32, 64);
            LAUNCH_TENSORCORE(1, 1, 1, 64, 64, 64);
            LAUNCH_TENSORCORE(1, 1, 1, 128, 16, 64);
            LAUNCH_TENSORCORE(1, 1, 1, 128, 32, 64);
            LAUNCH_TENSORCORE(1, 1, 1, 128, 64, 64);

            // Atom layout 2×2×1 (Medium: 32×16 output tile per 4 atoms) - Original default
            // -------------------------------------------------------------------------
            LAUNCH_TENSORCORE(2, 2, 1, 16, 16, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 16, 32, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 16, 64, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 16, 128, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 16, 256, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 32, 16, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 32, 32, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 32, 64, 16); // Phase 3 winner for m=32
            LAUNCH_TENSORCORE(2, 2, 1, 32, 128, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 32, 256, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 64, 16, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 64, 32, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 64, 64, 16); // Phase 2.5 baseline
            LAUNCH_TENSORCORE(2, 2, 1, 64, 128, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 64, 256, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 128, 16, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 128, 32, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 128, 64, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 128, 128, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 128, 256, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 256, 16, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 256, 32, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 256, 64, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 256, 128, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 256, 256, 16);
            LAUNCH_TENSORCORE(2, 2, 1, 16, 16, 32);
            LAUNCH_TENSORCORE(2, 2, 1, 16, 32, 32);
            LAUNCH_TENSORCORE(2, 2, 1, 16, 64, 32);
            LAUNCH_TENSORCORE(2, 2, 1, 16, 128, 32);
            LAUNCH_TENSORCORE(2, 2, 1, 32, 16, 32);
            LAUNCH_TENSORCORE(2, 2, 1, 32, 32, 32);
            LAUNCH_TENSORCORE(2, 2, 1, 32, 64, 32);
            LAUNCH_TENSORCORE(2, 2, 1, 32, 128, 32);
            LAUNCH_TENSORCORE(2, 2, 1, 64, 16, 32);
            LAUNCH_TENSORCORE(2, 2, 1, 64, 32, 32);
            LAUNCH_TENSORCORE(2, 2, 1, 64, 64, 32);
            LAUNCH_TENSORCORE(2, 2, 1, 64, 128, 32);
            LAUNCH_TENSORCORE(2, 2, 1, 128, 16, 32);
            LAUNCH_TENSORCORE(2, 2, 1, 128, 32, 32);
            LAUNCH_TENSORCORE(2, 2, 1, 128, 64, 32);
            LAUNCH_TENSORCORE(2, 2, 1, 128, 128, 32);
            LAUNCH_TENSORCORE(2, 2, 1, 16, 16, 64);
            LAUNCH_TENSORCORE(2, 2, 1, 16, 32, 64);
            LAUNCH_TENSORCORE(2, 2, 1, 16, 64, 64);
            LAUNCH_TENSORCORE(2, 2, 1, 32, 16, 64);
            LAUNCH_TENSORCORE(2, 2, 1, 32, 32, 64);
            LAUNCH_TENSORCORE(2, 2, 1, 32, 64, 64);
            LAUNCH_TENSORCORE(2, 2, 1, 64, 16, 64);
            LAUNCH_TENSORCORE(2, 2, 1, 64, 32, 64);
            LAUNCH_TENSORCORE(2, 2, 1, 64, 64, 64);
            LAUNCH_TENSORCORE(2, 2, 1, 128, 16, 64);
            LAUNCH_TENSORCORE(2, 2, 1, 128, 32, 64);
            LAUNCH_TENSORCORE(2, 2, 1, 128, 64, 64);

            // Atom layout 4×4×1 (Large: 64×32 output tile per 16 atoms)
            // -------------------------------------------------------------------------
            LAUNCH_TENSORCORE(4, 4, 1, 16, 16, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 16, 32, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 16, 64, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 16, 128, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 16, 256, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 32, 16, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 32, 32, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 32, 64, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 32, 128, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 32, 256, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 64, 16, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 64, 32, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 64, 64, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 64, 128, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 64, 256, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 128, 16, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 128, 32, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 128, 64, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 128, 128, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 128, 256, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 256, 16, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 256, 32, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 256, 64, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 256, 128, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 256, 256, 16);
            LAUNCH_TENSORCORE(4, 4, 1, 16, 16, 32);
            LAUNCH_TENSORCORE(4, 4, 1, 16, 32, 32);
            LAUNCH_TENSORCORE(4, 4, 1, 16, 64, 32);
            LAUNCH_TENSORCORE(4, 4, 1, 16, 128, 32);
            LAUNCH_TENSORCORE(4, 4, 1, 32, 16, 32);
            LAUNCH_TENSORCORE(4, 4, 1, 32, 32, 32);
            LAUNCH_TENSORCORE(4, 4, 1, 32, 64, 32);
            LAUNCH_TENSORCORE(4, 4, 1, 32, 128, 32);
            LAUNCH_TENSORCORE(4, 4, 1, 64, 16, 32);
            LAUNCH_TENSORCORE(4, 4, 1, 64, 32, 32);
            LAUNCH_TENSORCORE(4, 4, 1, 64, 64, 32);
            LAUNCH_TENSORCORE(4, 4, 1, 64, 128, 32);
            LAUNCH_TENSORCORE(4, 4, 1, 128, 16, 32);
            LAUNCH_TENSORCORE(4, 4, 1, 128, 32, 32);
            LAUNCH_TENSORCORE(4, 4, 1, 128, 64, 32);
            LAUNCH_TENSORCORE(4, 4, 1, 128, 128, 32);
            LAUNCH_TENSORCORE(4, 4, 1, 16, 16, 64);
            LAUNCH_TENSORCORE(4, 4, 1, 16, 32, 64);
            LAUNCH_TENSORCORE(4, 4, 1, 16, 64, 64);
            LAUNCH_TENSORCORE(4, 4, 1, 32, 16, 64);
            LAUNCH_TENSORCORE(4, 4, 1, 32, 32, 64);
            LAUNCH_TENSORCORE(4, 4, 1, 32, 64, 64);
            LAUNCH_TENSORCORE(4, 4, 1, 64, 16, 64);
            LAUNCH_TENSORCORE(4, 4, 1, 64, 32, 64);
            LAUNCH_TENSORCORE(4, 4, 1, 64, 64, 64);
            LAUNCH_TENSORCORE(4, 4, 1, 128, 16, 64);
            LAUNCH_TENSORCORE(4, 4, 1, 128, 32, 64);
            LAUNCH_TENSORCORE(4, 4, 1, 128, 64, 64);

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
