#pragma once

#include <cstdlib>
#include <string>

/**
 * @file DebugEnv.h
 * @brief Runtime configuration via environment variables for v2 architecture
 * @author David Sanftenberg
 *
 * Minimal stub for IQ4_NL migration. Full implementation pending.
 * For now, provides tile size configuration and basic feature flags.
 */

namespace llaminar2
{

    /**
     * @brief Dequantization configuration group
     */
    struct DequantConfig
    {
        // IQ4_NL tile sizes (tuned via LLAMINAR_IQ4_M_TILE and LLAMINAR_IQ4_N_TILE)
        size_t iq4_m_tile = 64; ///< Default: 64 (optimal from tile sweep)
        size_t iq4_n_tile = 32; ///< Default: 32 (optimal from tile sweep)

        // Experimental features
        bool iq4_microkernel = false;     ///< Multi-block vectorized decode (experimental)
        bool iq4_direct_decode = false;   ///< Direct decode path (vs cache-blocked, experimental)
        bool iq4_gemm_microkernel = true; ///< GEMM microkernel optimization (CANONICAL - enabled by default)

        // Override tile sizes (0 = use adaptive defaults)
        int iq4_override_m_tile = 0;      ///< Override M tile size (FP32)
        int iq4_override_n_tile = 0;      ///< Override N tile size (FP32)
        int iq4_override_m_tile_bf16 = 0; ///< Override M tile size (BF16)
        int iq4_override_n_tile_bf16 = 0; ///< Override N tile size (BF16)

        // SIMD path control for testing (LLAMINAR_DEQUANT_SIMD_PATH)
        // Values: "auto" (default), "scalar", "avx2", "avx512"
        std::string simd_path = "auto"; ///< Force specific SIMD path for testing

        DequantConfig()
        {
            // Parse environment variables
            const char *m_tile_env = std::getenv("LLAMINAR_IQ4_M_TILE");
            if (m_tile_env)
            {
                iq4_m_tile = static_cast<size_t>(std::atoi(m_tile_env));
            }

            const char *n_tile_env = std::getenv("LLAMINAR_IQ4_N_TILE");
            if (n_tile_env)
            {
                iq4_n_tile = static_cast<size_t>(std::atoi(n_tile_env));
            }

            const char *microkernel_env = std::getenv("LLAMINAR_IQ4_MICROKERNEL");
            if (microkernel_env)
            {
                iq4_microkernel = (std::atoi(microkernel_env) != 0);
            }

            const char *direct_decode_env = std::getenv("LLAMINAR_IQ4_DIRECT_DECODE");
            if (direct_decode_env)
            {
                iq4_direct_decode = (std::atoi(direct_decode_env) != 0);
            }

            const char *gemm_micro_env = std::getenv("LLAMINAR_IQ4_GEMM_MICROKERNEL");
            if (gemm_micro_env)
            {
                iq4_gemm_microkernel = (std::atoi(gemm_micro_env) != 0);
            }

            // SIMD path control for testing
            const char *simd_path_env = std::getenv("LLAMINAR_DEQUANT_SIMD_PATH");
            if (simd_path_env)
            {
                simd_path = simd_path_env;
            }
        }
    };

    /**
     * @brief GEMM kernel configuration group
     */
    struct GemmConfig
    {
        // Q8_1 GEMM compensation strategy
        bool use_sa_compensation = false; ///< Use sA-based compensation (vs sum_qs-based, experimental)

        // Q8_1 GEMM microkernel variant selection
        bool use_dense_dpbusd = false; ///< Use dense dpbusd (accumulate across K-blocks, experimental)

        // Cache blocking tuning parameters
        float gemm_l2_limit_pct = 0.9f;          ///< L2 cache limit percentage (default: 0.9)
        float gemm_k_tile_threshold_pct = 0.75f; ///< K-tiling threshold percentage (default: 0.75)
        float gemm_target_b_size_pct = 0.5f;     ///< Target B-tile size percentage (default: 0.5)

        // Advanced tuning parameters
        int gemm_oversubscription_factor = 4;  ///< Task oversubscription factor (default: 4)
        int gemm_min_block_size = 65536;       ///< Minimum block size in bytes (default: 64KB)
        int gemm_k_tile_min_blocks = 32;       ///< Minimum K-tile blocks (default: 32)
        int gemm_k_tile_max_blocks = 256;      ///< Maximum K-tile blocks (default: 256)
        float gemm_l3_share_pct = 0.9f;        ///< L3 cache share percentage (default: 0.9)
        int gemm_m_task_granularity = 2;       ///< M-dimension task granularity (default: 2)
        int gemm_m_unroll_factor = 2;          ///< M-dimension loop unroll factor (default: 2)
        int gemm_quant_parallel_threshold = 0; ///< Threshold for parallel quantization (0=auto)

        GemmConfig()
        {
            reload();
        }

        void reload()
        {
            const char *sa_comp_env = std::getenv("LLAMINAR_USE_SA_COMPENSATION");
            if (sa_comp_env)
            {
                use_sa_compensation = (std::atoi(sa_comp_env) != 0);
            }

            const char *dense_env = std::getenv("LLAMINAR_USE_DENSE_DPBUSD");
            if (dense_env)
            {
                use_dense_dpbusd = (std::atoi(dense_env) != 0);
            }

            const char *l2_limit_env = std::getenv("LLAMINAR_GEMM_L2_LIMIT_PCT");
            if (l2_limit_env)
            {
                gemm_l2_limit_pct = std::strtof(l2_limit_env, nullptr);
            }

            const char *k_tile_env = std::getenv("LLAMINAR_GEMM_K_TILE_THRESHOLD_PCT");
            if (k_tile_env)
            {
                gemm_k_tile_threshold_pct = std::strtof(k_tile_env, nullptr);
            }

            const char *target_b_env = std::getenv("LLAMINAR_GEMM_TARGET_B_SIZE_PCT");
            if (target_b_env)
            {
                gemm_target_b_size_pct = std::strtof(target_b_env, nullptr);
            }

            const char *oversub_env = std::getenv("LLAMINAR_GEMM_OVERSUBSCRIPTION_FACTOR");
            if (oversub_env)
                gemm_oversubscription_factor = std::atoi(oversub_env);

            const char *min_block_env = std::getenv("LLAMINAR_GEMM_MIN_BLOCK_SIZE");
            if (min_block_env)
                gemm_min_block_size = std::atoi(min_block_env);

            const char *k_min_env = std::getenv("LLAMINAR_GEMM_K_TILE_MIN_BLOCKS");
            if (k_min_env)
                gemm_k_tile_min_blocks = std::atoi(k_min_env);

            const char *k_max_env = std::getenv("LLAMINAR_GEMM_K_TILE_MAX_BLOCKS");
            if (k_max_env)
                gemm_k_tile_max_blocks = std::atoi(k_max_env);

            const char *l3_share_env = std::getenv("LLAMINAR_GEMM_L3_SHARE_PCT");
            if (l3_share_env)
                gemm_l3_share_pct = std::strtof(l3_share_env, nullptr);

            const char *m_gran_env = std::getenv("LLAMINAR_GEMM_M_TASK_GRANULARITY");
            if (m_gran_env)
                gemm_m_task_granularity = std::atoi(m_gran_env);

            const char *m_unroll_env = std::getenv("LLAMINAR_GEMM_M_UNROLL_FACTOR");
            if (m_unroll_env)
                gemm_m_unroll_factor = std::atoi(m_unroll_env);

            const char *quant_thresh_env = std::getenv("LLAMINAR_GEMM_QUANT_PARALLEL_THRESHOLD");
            if (quant_thresh_env)
                gemm_quant_parallel_threshold = std::atoi(quant_thresh_env);
        }
    }; /**
        * @brief Global debug environment snapshot
        */
    struct DebugEnv
    {
        DequantConfig dequant;
        GemmConfig gemm;

        // Add more config groups as needed:
        // AttentionConfig attention;
        // NumaConfig numa;
        // MPIConfig mpi;

        DebugEnv() = default;

        void reload()
        {
            gemm.reload();
        }
    };

    /**
     * @brief Access global debug environment (mutable for testing/reloading)
     */
    inline DebugEnv &mutableDebugEnv()
    {
        static DebugEnv env;
        return env;
    }

    /**
     * @brief Access global debug environment (read-only)
     */
    inline const DebugEnv &debugEnv()
    {
        return mutableDebugEnv();
    }

} // namespace llaminar2
