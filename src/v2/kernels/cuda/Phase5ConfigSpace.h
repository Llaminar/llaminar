/**
 * @file Phase5ConfigSpace.h
 * @brief Configuration space for Phase 5 auto-tuning
 *
 * Based on profiling analysis showing:
 * - Shared memory limits occupancy (32KB → 3 blocks/SM → 22% occupancy)
 * - Tensor Core utilization only 6.29% (severely underutilized)
 * - Need to explore buffer stages vs occupancy tradeoff
 *
 * Key parameters to sweep:
 * 1. Buffer stages (1/2/3) - trades memory for latency hiding
 * 2. Tile sizes - larger tiles = better Tensor Core util, more memory
 * 3. Sub-tile size (SUB_K) - streaming granularity
 *
 * @author David Sanftenberg
 * @date November 4, 2025
 */

#pragma once

#include <vector>
#include <string>

namespace llaminar2
{
    namespace cuda
    {

        /**
         * @brief Phase 5 GEMM configuration
         */
        struct Phase5GemmConfig
        {
            // Tile dimensions
            int tile_m;
            int tile_n;
            int tile_k;

            // Streaming sub-tile size (16, 32, or tile_k for no streaming)
            int sub_k;

            // CuTe atom layout multipliers
            int mma_m; // 1, 2, 4
            int mma_n; // 1, 2, 4

            // Buffering strategy: 1 (single), 2 (double), 3 (triple)
            int buffer_stages;

            // Thread block size (derived from MMA layout)
            int threads_per_block; // 32, 64, 128, 256

            // Swizzle parameters (for shared memory bank conflict avoidance)
            int swizzle_b; // BBits parameter (0-6 for TILE_K=64)
            int swizzle_m; // MBase parameter (0-6)
            int swizzle_s; // SShift parameter (0-6)

            /**
             * @brief Compute shared memory usage in bytes
             */
            size_t shared_memory_bytes() const
            {
                // s_A: buffer_stages × tile_m × tile_k × sizeof(__half)
                // s_B_decoded: buffer_stages × tile_n × tile_k × sizeof(__half)
                size_t smem_A = buffer_stages * tile_m * tile_k * 2; // FP16
                size_t smem_B = buffer_stages * tile_n * tile_k * 2; // FP16
                return smem_A + smem_B;
            }

            /**
             * @brief Estimate theoretical occupancy (blocks per SM)
             *
             * Based on A100 limits:
             * - 164 KB shared memory per SM
             * - 2048 threads per SM (64 warps)
             * - 32 blocks per SM (max)
             */
            int estimate_occupancy_blocks_per_sm(int max_smem_per_sm = 164 * 1024) const
            {
                size_t smem = shared_memory_bytes();
                int smem_limit = max_smem_per_sm / smem;

                int thread_limit = 2048 / threads_per_block;

                int occupancy = std::min(smem_limit, thread_limit);
                return std::min(occupancy, 32); // Hardware max
            }

            /**
             * @brief Generate unique config ID string
             */
            std::string config_id() const
            {
                char buf[256];
                snprintf(buf, sizeof(buf),
                         "p5_%d_%d_%d_sub%d_mma%dx%d_buf%d_thr%d_swz%d%d%d",
                         tile_m, tile_n, tile_k,
                         sub_k,
                         mma_m, mma_n,
                         buffer_stages,
                         threads_per_block,
                         swizzle_b, swizzle_m, swizzle_s);
                return std::string(buf);
            }

            /**
             * @brief Validate configuration
             */
            bool is_valid(int max_smem_per_block = 48 * 1024) const
            {
                // Check shared memory limits
                if (shared_memory_bytes() > max_smem_per_block)
                    return false;

                // Check thread limits
                if (threads_per_block > 1024)
                    return false;

                // Check tile divisibility
                if (tile_k % sub_k != 0)
                    return false;
                if (tile_k % 32 != 0)
                    return false; // Must align with IQ4_NL block size

                // Check MMA atom compatibility (16x8x16 for SM80)
                if (tile_m % (16 * mma_m) != 0)
                    return false;
                if (tile_n % (8 * mma_n) != 0)
                    return false;
                if (tile_k % 16 != 0)
                    return false; // MMA K dimension

                // Check buffer stages
                if (buffer_stages < 1 || buffer_stages > 3)
                    return false;

                return true;
            }
        };

        /**
         * @brief Generate comprehensive Phase 5 configuration space
         *
         * Strategy based on profiling insights:
         * 1. Explore buffer stages (1/2/3) to find occupancy sweet spot
         * 2. Test larger tiles (128x128) for better Tensor Core saturation
         * 3. Try different streaming granularities (SUB_K)
         */
        inline std::vector<Phase5GemmConfig> generate_phase5_config_space()
        {
            std::vector<Phase5GemmConfig> configs;

            // Tile sizes to explore
            std::vector<int> tile_m_sizes = {32, 64, 128, 256};
            std::vector<int> tile_n_sizes = {32, 64, 128, 256};
            std::vector<int> tile_k_sizes = {32, 64, 128};

            // Streaming granularity
            std::vector<int> sub_k_sizes = {16, 32, 64, 128}; // 128 = no streaming if tile_k=128

            // MMA atom layouts (1x1, 2x2, 4x4)
            std::vector<std::pair<int, int>> mma_layouts = {
                {1, 1}, // 32 threads
                {2, 2}, // 128 threads
                {4, 4}  // 512 threads (may exceed occupancy limits)
            };

            // Buffer stages: Key parameter!
            std::vector<int> buffer_stages_options = {1, 2, 3};

            // Swizzle (keep constant for now)
            int swizzle_b = 3, swizzle_m = 3, swizzle_s = 3;

            for (int tile_m : tile_m_sizes)
            {
                for (int tile_n : tile_n_sizes)
                {
                    for (int tile_k : tile_k_sizes)
                    {
                        for (int sub_k : sub_k_sizes)
                        {
                            // Skip if sub_k > tile_k
                            if (sub_k > tile_k)
                                continue;

                            for (auto [mma_m, mma_n] : mma_layouts)
                            {
                                // Calculate threads
                                int threads_per_block = 32 * mma_m * mma_n; // 32 threads per 1x1 atom

                                for (int buffer_stages : buffer_stages_options)
                                {
                                    Phase5GemmConfig config = {
                                        tile_m, tile_n, tile_k,
                                        sub_k,
                                        mma_m, mma_n,
                                        buffer_stages,
                                        threads_per_block,
                                        swizzle_b, swizzle_m, swizzle_s};

                                    if (config.is_valid())
                                    {
                                        configs.push_back(config);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return configs;
        }

        /**
         * @brief Generate focused config space for quick sweeps
         *
         * Fewer configs for faster iteration during development
         */
        inline std::vector<Phase5GemmConfig> generate_phase5_config_space_focused()
        {
            std::vector<Phase5GemmConfig> configs;

            // Focus on promising configurations based on profiling
            struct FocusedConfig
            {
                int tile_m, tile_n, tile_k, sub_k;
                int mma_m, mma_n;
                int buffer_stages;
                const char *rationale;
            };

            std::vector<FocusedConfig> focused = {
                // Baseline: Optimal configuration (validated +54% perf with SUB_K=64)
                {64, 64, 64, 64, 2, 2, 2, "Optimal: SUB_K=64 enables coalescing, double buffer"},

                // Single buffering with streaming (low memory pressure)
                {64, 64, 64, 16, 2, 2, 1, "Streaming 16-elem sub-tiles, single buffer"},
                {64, 64, 64, 32, 2, 2, 1, "Streaming 32-elem sub-tiles, single buffer"},

                // Double buffering variations
                {64, 64, 64, 16, 2, 2, 2, "Legacy Phase 5A with SUB_K=16, double buffer"},
                {64, 64, 64, 32, 2, 2, 2, "SUB_K=32 (+31% perf), double buffer"},

                // Larger tiles with single buffer (better Tensor Core saturation)
                {128, 128, 64, 16, 2, 2, 1, "Large tiles, streaming, single buffer"},
                {128, 128, 64, 32, 2, 2, 1, "Large tiles, moderate streaming"},
                {128, 128, 64, 64, 2, 2, 1, "Large tiles, no streaming"},

                // Larger tiles with double buffer (if memory allows)
                {128, 128, 64, 16, 1, 1, 2, "Large tiles, fewer threads, double buffer"},
                {128, 128, 64, 32, 1, 1, 2, "Large tiles, moderate streaming, double buffer"},

                // Very large tiles (maximum Tensor Core work per block)
                {256, 256, 64, 32, 1, 1, 1, "Huge tiles, minimal threads, single buffer"},
            };

            for (const auto &fc : focused)
            {
                Phase5GemmConfig config = {
                    fc.tile_m, fc.tile_n, fc.tile_k,
                    fc.sub_k,
                    fc.mma_m, fc.mma_n,
                    fc.buffer_stages,
                    32 * fc.mma_m * fc.mma_n, // threads
                    3, 3, 3                   // swizzle
                };

                if (config.is_valid())
                {
                    configs.push_back(config);
                }
            }

            return configs;
        }

        /**
         * @brief Generate swizzle parameter sweep for fixed tile configuration
         *
         * For TILE_K=64 with FP16 elements, there are 7 valid swizzle configurations:
         * - Swizzle<B,M,S> where B+M=6 (log2(64)=6) and S=B (symmetric pattern)
         * - This gives: <0,6,0>, <1,5,1>, <2,4,2>, <3,3,3>, <4,2,4>, <5,1,5>, <6,0,6>
         *
         * @param base_config Base configuration (tile sizes, MMA layout, etc.)
         * @return Vector of 7 configurations with different swizzle parameters
         */
        inline std::vector<Phase5GemmConfig> generate_swizzle_sweep(const Phase5GemmConfig &base_config)
        {
            std::vector<Phase5GemmConfig> configs;

            // Compute log2(TILE_K) for this configuration
            int tile_k_log2 = 0;
            int k = base_config.tile_k;
            while (k > 1)
            {
                k >>= 1;
                tile_k_log2++;
            }

            // Verify TILE_K is power of 2
            if ((1 << tile_k_log2) != base_config.tile_k)
            {
                fprintf(stderr, "ERROR: TILE_K=%d is not a power of 2, swizzle requires power-of-2 row size\n",
                        base_config.tile_k);
                return configs; // Empty vector
            }

            // Generate all valid swizzle configurations
            // For each M from 0 to tile_k_log2, we get B=tile_k_log2-M and S=B
            for (int M = 0; M <= tile_k_log2; ++M)
            {
                int B = tile_k_log2 - M;
                int S = B; // Symmetric pattern

                Phase5GemmConfig config = base_config;
                config.swizzle_b = B;
                config.swizzle_m = M;
                config.swizzle_s = S;

                configs.push_back(config);
            }

            return configs;
        }

        /**
         * @brief Generate focused swizzle sweep for current Phase 5 baseline
         *
         * Quick test of all 7 swizzle configurations on the current 64x64x64 config
         */
        inline std::vector<Phase5GemmConfig> generate_swizzle_sweep_baseline()
        {
            // Current Phase 5 baseline configuration
            Phase5GemmConfig baseline = {
                64, 64, 64, // tile_m, tile_n, tile_k
                16,         // sub_k (streaming with 16-element sub-tiles)
                2, 2,       // mma_m, mma_n (2x2 atom layout = 128 threads)
                2,          // buffer_stages (double buffering)
                128,        // threads_per_block
                3, 3, 3     // swizzle (current, will be replaced in sweep)
            };

            return generate_swizzle_sweep(baseline);
        }

    } // namespace cuda
} // namespace llaminar2
