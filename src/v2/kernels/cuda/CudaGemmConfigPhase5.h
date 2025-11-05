/**
 * @file CudaGemmConfigPhase5.h
 * @brief Configuration for Phase 5 JIT-compiled GEMM kernels
 *
 * Extends base configuration with Phase 5-specific parameters:
 * - Buffer stages (single/double/triple buffering)
 * - Streaming sub-tile size (SUB_K)
 * - CuTe atom layout multipliers
 *
 * DEFAULT CONFIGURATION (as of Nov 4, 2025):
 * - SUB_K = 64 (enables coalesced memory access for IQ4_NL)
 * - Validated improvement: +54.3% performance vs SUB_K=16
 * - Achieves ~12.98 TFLOPS on RTX 3090 for 1024×896×896 GEMM
 * - Coalescing enabled: K_BLOCKS_IN_SUB_K = 2 (row-major thread traversal)
 *
 * @author David Sanftenberg
 * @date November 4, 2025
 */

#pragma once

#include <string>
#include <cstddef>
#include <algorithm>

namespace llaminar2
{
    namespace cuda
    {

        /**
         * @brief Phase 5 GEMM configuration with buffering and streaming parameters
         */
        struct CudaGemmConfigPhase5
        {
            // Tile dimensions
            int tile_m = 64;
            int tile_n = 64;
            int tile_k = 64;

            // Streaming sub-tile size (16, 32, 64, or tile_k for no streaming)
            // Default: 64 (enables coalesced memory access for IQ4_NL, validated +54% perf)
            int sub_k = 64;

            // CuTe atom layout multipliers (1, 2, 4)
            int mma_m = 2;
            int mma_n = 2;

            // Buffering strategy: 1 (single), 2 (double), 3 (triple)
            int buffer_stages = 2;

            // Thread block size (derived from MMA layout)
            // For SM80_16x8x16_F32F16F16F32_TN: 32 threads per 1×1 atom
            int threads_per_block = 128; // 32 * 2 * 2

            // Swizzle parameters for bank conflict avoidance
            int swizzle_b = 3;
            int swizzle_m = 3;
            int swizzle_s = 3;

            // Vectorization width for A-matrix global loads (1, 2, 4, 8)
            // 1=scalar, 2=float2, 4=float4, 8=2xfloat4 (256-bit)
            // NOTE: For Swizzle<3,3,3> (MBase=3), use vectorize_a=8 for contiguous writes
            // For Swizzle<4,2,4> (MBase=2), use vectorize_a=4
            int vectorize_a = 8; // Default to 8 (matches Swizzle<3,3,3> MBase=3)

            // B matrix layout control (NCU-guided coalescing optimization)
            // false = Row-major [N][K_blocks] (default, uncoalesced - 8.7/32 bytes)
            // true = Transposed [K_blocks][N] (coalesced - expect 28-32/32 bytes)
            // NCU analysis shows 62% speedup potential from this optimization
            bool transpose_b = false;

            /**
             * @brief Default constructor
             */
            CudaGemmConfigPhase5() = default;

            /**
             * @brief Construct with all parameters
             */
            CudaGemmConfigPhase5(
                int tile_m, int tile_n, int tile_k,
                int sub_k,
                int mma_m, int mma_n,
                int buffer_stages,
                int threads_per_block = 0, // Auto-calculate if 0
                int swizzle_b = 3, int swizzle_m = 3, int swizzle_s = 3,
                int vectorize_a = 8,      // Default to 8 (matches Swizzle<3,3,3>)
                bool transpose_b = false) // Default to row-major
                : tile_m(tile_m), tile_n(tile_n), tile_k(tile_k),
                  sub_k(sub_k),
                  mma_m(mma_m), mma_n(mma_n),
                  buffer_stages(buffer_stages),
                  threads_per_block(threads_per_block > 0 ? threads_per_block : 32 * mma_m * mma_n),
                  swizzle_b(swizzle_b), swizzle_m(swizzle_m), swizzle_s(swizzle_s),
                  vectorize_a(vectorize_a),
                  transpose_b(transpose_b)
            {
            }

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
                int smem_limit = (smem > 0) ? (max_smem_per_sm / smem) : 32;
                int thread_limit = 2048 / threads_per_block;
                int occupancy = std::min(smem_limit, thread_limit);
                return std::min(occupancy, 32); // Hardware max
            }

            /**
             * @brief Generate unique config ID string for caching
             */
            std::string config_id() const
            {
                char buf[256];
                snprintf(buf, sizeof(buf),
                         "p5_%d_%d_%d_sub%d_mma%dx%d_buf%d_thr%d_swz%d%d%d_vec%d_tb%d",
                         tile_m, tile_n, tile_k,
                         sub_k,
                         mma_m, mma_n,
                         buffer_stages,
                         threads_per_block,
                         swizzle_b, swizzle_m, swizzle_s,
                         vectorize_a,
                         transpose_b ? 1 : 0); // tb1 = transposed, tb0 = row-major
                return std::string(buf);
            }

            /**
             * @brief Validate configuration constraints
             */
            bool is_valid(int max_smem_per_block = 48 * 1024) const
            {
                // Check shared memory limits
                if (shared_memory_bytes() > static_cast<size_t>(max_smem_per_block))
                    return false;

                // Check thread limits
                if (threads_per_block > 1024 || threads_per_block < 32)
                    return false;

                // Check tile divisibility by sub-tile
                if (tile_k % sub_k != 0)
                    return false;

                // Must align with IQ4_NL block size (32 elements)
                if (tile_k % 32 != 0)
                    return false;

                // Check MMA atom compatibility (16x8x16 for SM80)
                if (tile_m % (16 * mma_m) != 0)
                    return false;
                if (tile_n % (8 * mma_n) != 0)
                    return false;
                if (tile_k % 16 != 0)
                    return false; // MMA K dimension

                // Check sub_k alignment with MMA K
                if (sub_k % 16 != 0)
                    return false;

                // Check buffer stages
                if (buffer_stages < 1 || buffer_stages > 3)
                    return false;

                // Check MMA multipliers
                if (mma_m < 1 || mma_m > 4)
                    return false;
                if (mma_n < 1 || mma_n > 4)
                    return false;

                // Check threads matches MMA layout
                int expected_threads = 32 * mma_m * mma_n;
                if (threads_per_block != expected_threads)
                    return false;

                return true;
            }

            /**
             * @brief Equality operator for use in maps/caches
             */
            bool operator==(const CudaGemmConfigPhase5 &other) const
            {
                return tile_m == other.tile_m &&
                       tile_n == other.tile_n &&
                       tile_k == other.tile_k &&
                       sub_k == other.sub_k &&
                       mma_m == other.mma_m &&
                       mma_n == other.mma_n &&
                       buffer_stages == other.buffer_stages &&
                       threads_per_block == other.threads_per_block &&
                       swizzle_b == other.swizzle_b &&
                       swizzle_m == other.swizzle_m &&
                       swizzle_s == other.swizzle_s;
            }

            /**
             * @brief Less-than operator for ordered containers
             */
            bool operator<(const CudaGemmConfigPhase5 &other) const
            {
                if (tile_m != other.tile_m)
                    return tile_m < other.tile_m;
                if (tile_n != other.tile_n)
                    return tile_n < other.tile_n;
                if (tile_k != other.tile_k)
                    return tile_k < other.tile_k;
                if (sub_k != other.sub_k)
                    return sub_k < other.sub_k;
                if (mma_m != other.mma_m)
                    return mma_m < other.mma_m;
                if (mma_n != other.mma_n)
                    return mma_n < other.mma_n;
                if (buffer_stages != other.buffer_stages)
                    return buffer_stages < other.buffer_stages;
                if (threads_per_block != other.threads_per_block)
                    return threads_per_block < other.threads_per_block;
                if (swizzle_b != other.swizzle_b)
                    return swizzle_b < other.swizzle_b;
                if (swizzle_m != other.swizzle_m)
                    return swizzle_m < other.swizzle_m;
                return swizzle_s < other.swizzle_s;
            }
        };

    } // namespace cuda
} // namespace llaminar2
