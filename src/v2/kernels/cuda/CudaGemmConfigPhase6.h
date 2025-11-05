/**
 * @file CudaGemmConfigPhase6.h
 * @brief Configuration for Phase 6 Int8 DP4A GEMM kernel
 *
 * Simplified config compared to Phase 5 - no MMA, no swizzle, no buffering complexity.
 *
 * @author David Sanftenberg
 * @date November 5, 2025
 */

#pragma once

#include <string>
#include <sstream>

namespace llaminar2
{
    namespace cuda
    {

        /**
         * @brief Phase 6 kernel configuration
         *
         * Much simpler than Phase 5 - just tile sizes and thread count.
         */
        struct CudaGemmConfigPhase6
        {
            int tile_m;
            int tile_n;
            int tile_k;
            int threads_per_block;

            CudaGemmConfigPhase6(
                int tile_m = 64,
                int tile_n = 64,
                int tile_k = 64,
                int threads_per_block = 256)
                : tile_m(tile_m), tile_n(tile_n), tile_k(tile_k), threads_per_block(threads_per_block)
            {
            }

            /**
             * @brief Generate unique configuration ID for JIT cache
             */
            std::string config_id() const
            {
                std::ostringstream oss;
                oss << "phase6_int8_"
                    << "tm" << tile_m
                    << "_tn" << tile_n
                    << "_tk" << tile_k
                    << "_t" << threads_per_block;
                return oss.str();
            }

            /**
             * @brief Check if configuration is valid
             */
            bool is_valid() const
            {
                // Tile dimensions must be positive and divisible by 32 (IQ4_NL block size)
                if (tile_m <= 0 || tile_n <= 0 || tile_k <= 0)
                    return false;
                if (tile_k % 32 != 0)
                    return false; // Must align with IQ4_NL blocks

                // Thread count must be reasonable
                if (threads_per_block < 32 || threads_per_block > 1024)
                    return false;
                if (threads_per_block % 32 != 0)
                    return false; // Must be warp-aligned

                // Tile must be divisible by thread count for simple partitioning
                if ((tile_m * tile_n) % threads_per_block != 0)
                    return false;

                return true;
            }
        };

        /**
         * @brief Get default Phase 6 configuration
         */
        inline CudaGemmConfigPhase6 get_default_phase6_config()
        {
            return CudaGemmConfigPhase6(
                64, // tile_m
                64, // tile_n
                64, // tile_k
                128 // threads_per_block (64×64 = 4096 elements, 128 threads = 32 elements/thread)
            );
        }

        /**
         * @brief Get Phase 6 configuration for specific problem size
         *
         * Simple heuristic - can be tuned later
         */
        inline CudaGemmConfigPhase6 get_phase6_config_for_size(int M, int N, int K)
        {
            // For now, just use default
            // TODO: Add auto-tuning based on problem size
            return get_default_phase6_config();
        }

    } // namespace cuda
} // namespace llaminar2
