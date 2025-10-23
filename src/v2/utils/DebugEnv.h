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

        // Experimental features (disabled by default)
        bool iq4_microkernel = false;      ///< Multi-block vectorized decode
        bool iq4_direct_decode = false;    ///< Direct decode path (vs cache-blocked)
        bool iq4_gemm_microkernel = false; ///< GEMM microkernel optimization

        // Override tile sizes (0 = use adaptive defaults)
        int iq4_override_m_tile = 0;      ///< Override M tile size (FP32)
        int iq4_override_n_tile = 0;      ///< Override N tile size (FP32)
        int iq4_override_m_tile_bf16 = 0; ///< Override M tile size (BF16)
        int iq4_override_n_tile_bf16 = 0; ///< Override N tile size (BF16)

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
        }
    };

    /**
     * @brief Global debug environment snapshot
     */
    struct DebugEnv
    {
        DequantConfig dequant;

        // Add more config groups as needed:
        // AttentionConfig attention;
        // NumaConfig numa;
        // MPIConfig mpi;

        DebugEnv() = default;
    };

    /**
     * @brief Access global debug environment (lazy static initialization)
     */
    inline const DebugEnv &debugEnv()
    {
        static DebugEnv env;
        return env;
    }

} // namespace llaminar2
