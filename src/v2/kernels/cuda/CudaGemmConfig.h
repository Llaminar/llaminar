/**
 * @file CudaGemmConfig.h
 * @brief Configuration parameters for CUDA GEMM kernel variants
 *
 * Defines the parameter space for adaptive CUDA kernel selection:
 * - Thread block dimensions (TILE_M × TILE_N)
 * - Shared memory strategies (prefetch, double-buffering)
 * - Register tiling
 * - Memory access patterns
 *
 * @author David Sanftenberg
 * @date October 31, 2025
 */

#pragma once

#include <string>
#include <sstream>
#include <cstdint>

namespace llaminar2
{
    namespace cuda
    {

        /**
         * @brief Configuration for a CUDA GEMM kernel variant
         *
         * Parameter ranges (exploded to ~100-200 variants):
         * - TILE_M: {8, 16, 32, 64, 128} - Thread block rows
         * - TILE_N: {8, 16, 32, 64, 128} - Thread block columns
         * - TILE_K: {16, 32, 64} - K-dimension tile size
         * - THREADS_M: {4, 8, 16} - Threads per row
         * - THREADS_N: {4, 8, 16} - Threads per column
         * - WORK_PER_THREAD_M: {1, 2, 4, 8} - Elements computed per thread (M dim)
         * - WORK_PER_THREAD_N: {1, 2, 4, 8} - Elements computed per thread (N dim)
         * - PREFETCH: {0, 1, 2} - Double/triple buffering stages
         * - TRANSPOSE_SMEM: {false, true} - Shared memory layout optimization
         * - ATOM_TYPE: {0, 1} - MMA atom instruction type (0=16x8x16, 1=16x8x8)
         * - ATOM_LAYOUT_M/N/K: {1, 2, 4} - How many atoms to tile together
         */
        struct CudaGemmConfig
        {
            // Thread block dimensions
            int tile_m; // Output tile rows (e.g., 16, 32, 64)
            int tile_n; // Output tile columns (e.g., 16, 32, 64)
            int tile_k; // K-dimension tile (e.g., 32, 64)

            // Thread block configuration
            int threads_m; // Threads in M dimension (e.g., 8, 16)
            int threads_n; // Threads in N dimension (e.g., 8, 16)

            // Work per thread (register tiling)
            int work_per_thread_m; // Elements per thread in M (e.g., 2, 4, 8)
            int work_per_thread_n; // Elements per thread in N (e.g., 2, 4, 8)

            // Memory optimizations
            int prefetch_stages; // 0 = no prefetch, 1 = double-buffer, 2 = triple-buffer
            bool transpose_smem; // Transpose shared memory to reduce bank conflicts
            int vectorize_load;  // Vectorized load width: 1=scalar, 2=float2, 4=float4

            // Tensor Core atom configuration (NEW - enables true configuration diversity)
            int atom_type;     // 0 = SM80_16x8x16 (K=16), 1 = SM80_16x8x8 (K=8)
            int atom_layout_m; // Atoms in M dimension (1, 2, 4)
            int atom_layout_n; // Atoms in N dimension (1, 2, 4)
            int atom_layout_k; // Atoms in K dimension (always 1 for SM80)

            /**
             * @brief Validate configuration consistency
             *
             * Ensures:
             * - tile_m == threads_m * work_per_thread_m
             * - tile_n == threads_n * work_per_thread_n
             * - Shared memory requirements fit within device limits
             */
            bool isValid() const
            {
                if (tile_m != threads_m * work_per_thread_m)
                    return false;
                if (tile_n != threads_n * work_per_thread_n)
                    return false;
                if (tile_k % 32 != 0)
                    return false; // Must align with IQ4_NL block size
                if (threads_m * threads_n > 1024)
                    return false; // Max threads per block

                // Shared memory check: 2 tiles (A and B) + prefetch buffers
                const int smem_per_stage = tile_m * tile_k + tile_n * tile_k;
                const int total_smem = smem_per_stage * (1 + prefetch_stages) * sizeof(float);
                if (total_smem > 48 * 1024)
                    return false; // 48KB typical shared memory limit

                return true;
            }

            /**
             * @brief Get unique identifier for this configuration
             */
            std::string id() const
            {
                std::ostringstream oss;
                oss << "tile_" << tile_m << "x" << tile_n << "x" << tile_k
                    << "_threads_" << threads_m << "x" << threads_n
                    << "_work_" << work_per_thread_m << "x" << work_per_thread_n
                    << "_prefetch_" << prefetch_stages
                    << "_transpose_" << (transpose_smem ? 1 : 0)
                    << "_vec_" << vectorize_load
                    << "_atom_" << (atom_type == 0 ? "16x8x16" : "16x8x8")
                    << "_layout_" << atom_layout_m << "x" << atom_layout_n << "x" << atom_layout_k;
                return oss.str();
            }

            /**
             * @brief Comparison operator for caching
             */
            bool operator==(const CudaGemmConfig &other) const
            {
                return tile_m == other.tile_m &&
                       tile_n == other.tile_n &&
                       tile_k == other.tile_k &&
                       threads_m == other.threads_m &&
                       threads_n == other.threads_n &&
                       work_per_thread_m == other.work_per_thread_m &&
                       work_per_thread_n == other.work_per_thread_n &&
                       prefetch_stages == other.prefetch_stages &&
                       transpose_smem == other.transpose_smem &&
                       vectorize_load == other.vectorize_load;
            }

            /**
             * @brief Compute FNV-1a hash for lookup table
             *
             * Includes both problem size (m,n,k) and config parameters.
             * Uses FNV-1a hash algorithm for fast, collision-resistant hashing.
             * This must match the Python implementation in train_cuda_heuristic.py.
             */
            uint64_t hash(int m, int n, int k) const
            {
                // FNV-1a constants
                uint64_t h = 14695981039346656037ULL;
                const uint64_t fnv_prime = 1099511628211ULL;

                auto mix = [&h, fnv_prime](uint32_t v)
                {
                    h ^= static_cast<uint64_t>(v);
                    h *= fnv_prime;
                };

                // Hash problem size FIRST (critical for distinguishing different tests)
                mix(static_cast<uint32_t>(m));
                mix(static_cast<uint32_t>(n));
                mix(static_cast<uint32_t>(k));

                // Then hash config parameters in order
                mix(static_cast<uint32_t>(tile_m));
                mix(static_cast<uint32_t>(tile_n));
                mix(static_cast<uint32_t>(tile_k));
                mix(static_cast<uint32_t>(threads_m));
                mix(static_cast<uint32_t>(threads_n));
                mix(static_cast<uint32_t>(work_per_thread_m));
                mix(static_cast<uint32_t>(work_per_thread_n));
                mix(static_cast<uint32_t>(prefetch_stages));
                mix(static_cast<uint32_t>(transpose_smem ? 1 : 0));
                mix(static_cast<uint32_t>(vectorize_load));

                return h;
            }

            /**
             * @brief Estimate register pressure
             *
             * Returns number of registers needed per thread.
             * CUDA limit: 255 registers/thread, but high usage reduces occupancy.
             */
            int estimateRegisterPressure() const
            {
                // Accumulators: work_per_thread_m * work_per_thread_n floats
                int accumulator_regs = work_per_thread_m * work_per_thread_n;

                // A/B fragments for GEMM loop
                int fragment_regs = work_per_thread_m + work_per_thread_n;

                // Loop counters, pointers, temps
                int overhead_regs = 10;

                return accumulator_regs + fragment_regs + overhead_regs;
            }

            /**
             * @brief Estimate theoretical occupancy (0.0 to 1.0)
             *
             * Based on register usage and shared memory consumption.
             */
            float estimateOccupancy() const
            {
                const int threads_per_block = threads_m * threads_n;
                const int reg_per_thread = estimateRegisterPressure();
                const int smem_per_block = (tile_m * tile_k + tile_n * tile_k) *
                                           (1 + prefetch_stages) * sizeof(float);

                // SM limits (Ampere architecture - RTX 3090)
                const int max_threads_per_sm = 1536;
                const int max_blocks_per_sm = 16;
                const int max_regs_per_sm = 65536;
                const int max_smem_per_sm = 102400; // 100KB

                // Blocks limited by threads
                int blocks_by_threads = max_threads_per_sm / threads_per_block;

                // Blocks limited by registers
                int regs_per_block = threads_per_block * reg_per_thread;
                int blocks_by_regs = max_regs_per_sm / regs_per_block;

                // Blocks limited by shared memory
                int blocks_by_smem = max_smem_per_sm / smem_per_block;

                // Take minimum
                int active_blocks = blocks_by_threads;
                if (blocks_by_regs < active_blocks)
                    active_blocks = blocks_by_regs;
                if (blocks_by_smem < active_blocks)
                    active_blocks = blocks_by_smem;
                if (max_blocks_per_sm < active_blocks)
                    active_blocks = max_blocks_per_sm;

                // Occupancy = active_threads / max_threads
                float occupancy = static_cast<float>(active_blocks * threads_per_block) /
                                  max_threads_per_sm;

                return std::min(occupancy, 1.0f);
            }
        };

        /**
         * @brief Predefined "good" configurations for heuristic selection
         */
        namespace presets
        {
            // Small matrices (m,n < 128): Low occupancy, minimize latency
            inline CudaGemmConfig small()
            {
                return CudaGemmConfig{
                    .tile_m = 16, .tile_n = 16, .tile_k = 32, .threads_m = 8, .threads_n = 8, .work_per_thread_m = 2, .work_per_thread_n = 2, .prefetch_stages = 0, .transpose_smem = false, .vectorize_load = 1};
            }

            // Medium matrices (128 <= m,n < 512): Balance occupancy and compute
            inline CudaGemmConfig medium()
            {
                return CudaGemmConfig{
                    .tile_m = 32, .tile_n = 32, .tile_k = 32, .threads_m = 8, .threads_n = 8, .work_per_thread_m = 4, .work_per_thread_n = 4, .prefetch_stages = 1, .transpose_smem = true, .vectorize_load = 4};
            }

            // Large matrices (m,n >= 512): Maximize throughput
            inline CudaGemmConfig large()
            {
                return CudaGemmConfig{
                    .tile_m = 64, .tile_n = 64, .tile_k = 32, .threads_m = 16, .threads_n = 16, .work_per_thread_m = 4, .work_per_thread_n = 4, .prefetch_stages = 2, .transpose_smem = true, .vectorize_load = 4};
            }

            // Tall matrices (m >> n): Optimize for column reuse
            inline CudaGemmConfig tall()
            {
                return CudaGemmConfig{
                    .tile_m = 64, .tile_n = 16, .tile_k = 32, .threads_m = 16, .threads_n = 8, .work_per_thread_m = 4, .work_per_thread_n = 2, .prefetch_stages = 1, .transpose_smem = false, .vectorize_load = 2};
            }

            // Wide matrices (n >> m): Optimize for row reuse
            inline CudaGemmConfig wide()
            {
                return CudaGemmConfig{
                    .tile_m = 16, .tile_n = 64, .tile_k = 32, .threads_m = 8, .threads_n = 16, .work_per_thread_m = 2, .work_per_thread_n = 4, .prefetch_stages = 1, .transpose_smem = true, .vectorize_load = 4};
            }
        }

    } // namespace cuda
} // namespace llaminar2
