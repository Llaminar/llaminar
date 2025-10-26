/**
 * @file QuantizedGemm.cpp
 * @brief Generic quantized GEMM implementation
 *
 * @author David Sanftenberg
 */

#include "QuantizedGemm.h"
#include <iostream>
#include <cstring>

namespace llaminar2
{

    bool QuantizedGemmKernel::multiply(
        const float *A, float *C,
        int m, int n, int k,
        bool transpose_B,
        float alpha, float beta,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        // Unused parameters (future: MPI coordination, GPU support)
        (void)mpi_ctx;
        (void)device_idx;

        if (!decoder_)
        {
            return false;
        }

        // Validate dimensions
        int expected_cols = transpose_B ? k : n;
        if (static_cast<int>(decoder_->decoder_cols()) != expected_cols)
        {
            return false; // Dimension mismatch
        }

        // Strategy selection based on batch size
        if (m >= 2 && m <= 16)
        {
            return multiply_cache_blocked(A, C, m, n, k, alpha, beta);
        }
        else
        {
            return multiply_row_wise(A, C, m, n, k, alpha, beta);
        }
    }

    bool QuantizedGemmKernel::multiply_cache_blocked(
        const float *A, float *C,
        int m, int n, int k,
        float alpha, float beta)
    {
        const size_t BLOCK_SIZE = decoder_->block_size();
        const int num_k_blocks = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Cache-blocked algorithm: decode block, use immediately across all m rows
        // Optimal for small batches (m ∈ [2,16]) - keeps 32-element blocks hot in L1

#pragma omp parallel for schedule(static) if (n > 128)
        for (int j = 0; j < n; ++j)
        {
            // Per-thread accumulator (small: m≤16)
            float acc[16] = {0};

            // Process one K-block at a time
            for (int kb = 0; kb < num_k_blocks; ++kb)
            {
                size_t k_start = kb * BLOCK_SIZE;
                size_t k_count = std::min(BLOCK_SIZE, static_cast<size_t>(k) - k_start);

                // Decode weight block to FP32 (stays in L1 cache)
                alignas(64) float B_block[64]; // Max block size (adjust if needed)
                decoder_->decode_block_at(j, kb, B_block);

                // Immediately use block for all M rows (hot in cache)
                for (int i = 0; i < m; ++i)
                {
                    const float *A_row = A + i * k + k_start;
                    acc[i] += dot_product_simd(A_row, B_block, k_count);
                }
            }

            // Write accumulated results
            for (int i = 0; i < m; ++i)
            {
                size_t c_idx = i * n + j;
                C[c_idx] = alpha * acc[i] + beta * C[c_idx];
            }
        }

        return true;
    }

    bool QuantizedGemmKernel::multiply_row_wise(
        const float *A, float *C,
        int m, int n, int k,
        float alpha, float beta)
    {
        const size_t BLOCK_SIZE = decoder_->block_size();
        const int num_k_blocks = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Row-wise algorithm with hybrid adaptive cache blocking: optimal for m>16
        // Dynamically adjust tile sizes based on problem characteristics
        const auto &env = debugEnv();
        int M_TILE, N_TILE;

        // Compute aspect ratio to determine workload characteristics
        // ratio > 1.5: Compute-bound (tall/square) - Q-projection (896×896)
        // ratio < 0.7: Memory-bound (wide) - FFN (896→4864)
        const float aspect_ratio = static_cast<float>(n) / static_cast<float>(m > 0 ? m : 1);
        const bool is_wide_output = aspect_ratio > 2.0f;                     // FFN-like: 4864/2048 = 2.37
        const bool is_square = aspect_ratio >= 0.5f && aspect_ratio <= 2.0f; // Q-proj-like

        // Check for tile size overrides (LLAMINAR_IQ4_M_TILE, LLAMINAR_IQ4_N_TILE)
        if (env.dequant.iq4_override_m_tile > 0 && env.dequant.iq4_override_n_tile > 0)
        {
            M_TILE = env.dequant.iq4_override_m_tile;
            N_TILE = env.dequant.iq4_override_n_tile;
        }
        else if (is_wide_output)
        {
            // MEMORY-BOUND PATH (FFN: wide output, limited by bandwidth)
            // Empirically validated (tile sweep Oct 2025): 64×32 optimal for FFN
            // Results: FFN-Batch-16: 262 GFLOPS, FFN-Batch-256: 451 GFLOPS (97% of 96×96 optimal)
            // Strategy: Unified 64×32 for simplicity; large batches (m≥256) may tune to 96×96 via env var

            if (m >= 256)
            {
                // Large batch: 64×32 achieves 97% of 96×96 optimal (451 vs 463 GFLOPS)
                // For peak: LLAMINAR_IQ4_M_TILE=96 LLAMINAR_IQ4_N_TILE=96
                M_TILE = 64;
                N_TILE = 32; // Universal optimal
            }
            else
            {
                // Small batch: 64×32 empirically optimal (262 GFLOPS)
                M_TILE = 64;
                N_TILE = 32;
            }
        }
        else if (is_square)
        {
            // COMPUTE-BOUND PATH (Q-proj: square matrix, limited by compute)
            // Strategy: Empirically validated tiling (tile sweep Oct 2025)
            // Results: 32×32 achieves 314 GFLOPS for m=1024 (+34% vs 64×64's 234 GFLOPS)
            //          64×32 achieves 352 GFLOPS for m=4096
            // Target: (M_TILE + N_TILE) * k * 4 ≤ 192KB total

            if (m >= 4096 || n >= 4096)
            {
                M_TILE = 64;
                N_TILE = 32; // Empirically optimal for large Q-proj (350 GFLOPS)
            }
            else if (m >= 2048 || n >= 2048)
            {
                M_TILE = 64;
                N_TILE = 32; // Universal optimal configuration
            }
            else if (m >= 1024 || n >= 1024)
            {
                M_TILE = 32;
                N_TILE = 32; // Empirically optimal for Q-proj-1024 (314 GFLOPS)
            }
            else if (m >= 512 || n >= 512)
            {
                M_TILE = 96;
                N_TILE = 96; // Standard
            }
            else
            {
                M_TILE = 128;
                N_TILE = 128; // Minimize overhead for small ops
            }
        }
        else
        {
            // TALL MATRIX PATH (n < m/2: more rows than columns)
            // Strategy: Small N_TILE, very large M_TILE for maximum reuse

            if (m >= 4096)
            {
                M_TILE = 64;
                N_TILE = 24;
            }
            else if (m >= 2048)
            {
                M_TILE = 96;
                N_TILE = 32;
            }
            else
            {
                M_TILE = 128;
                N_TILE = 48;
            }
        }

#pragma omp parallel
        {
            // Thread-local buffer for N_TILE B columns (decode multiple at once)
            // CRITICAL: Align to 64 bytes for AVX-512 _mm512_load_ps intrinsics
            // std::vector does NOT guarantee alignment, so we use aligned_alloc
            size_t tile_size = k * N_TILE;
            void *tile_ptr = aligned_alloc(64, tile_size * sizeof(float) + 64); // Extra space for alignment
            if (!tile_ptr)
            {
                throw std::bad_alloc();
            }
            float *B_tile = reinterpret_cast<float *>(tile_ptr);

#pragma omp for schedule(dynamic)
            for (int jj = 0; jj < n; jj += N_TILE)
            {
                int n_block = std::min(N_TILE, n - jj);

                // Decode N_TILE columns of B at once for better memory access pattern
                if (env.dequant.iq4_gemm_microkernel && n_block >= 4)
                {
                    // MICROKERNEL PATH: Decode multiple columns in vectorized batches (4 at a time)
                    // This reduces loop overhead and improves instruction pipelining
                    int j_vec = 0;
                    for (; j_vec + 4 <= n_block; j_vec += 4)
                    {
                        // Decode 4 columns together - k-blocks are outer loop for better locality
                        for (int kb = 0; kb < num_k_blocks; ++kb)
                        {
                            size_t k_start = kb * BLOCK_SIZE;
                            // Unroll 4 columns per k-block
                            for (int jv = 0; jv < 4; ++jv)
                            {
                                int j = jj + j_vec + jv;
                                float *B_col = B_tile + (j_vec + jv) * k;
                                decoder_->decode_block_at(j, kb, B_col + k_start);
                            }
                        }
                    }
                    // Handle remaining columns (< 4) with standard path
                    for (; j_vec < n_block; ++j_vec)
                    {
                        int j = jj + j_vec;
                        float *B_col = B_tile + j_vec * k;
                        for (int kb = 0; kb < num_k_blocks; ++kb)
                        {
                            size_t k_start = kb * BLOCK_SIZE;
                            decoder_->decode_block_at(j, kb, B_col + k_start);
                        }
                    }
                }
                else
                {
                    // STANDARD PATH: Decode one column at a time
                    for (int j_local = 0; j_local < n_block; ++j_local)
                    {
                        int j = jj + j_local;
                        float *B_col = B_tile + j_local * k;

                        for (int kb = 0; kb < num_k_blocks; ++kb)
                        {
                            size_t k_start = kb * BLOCK_SIZE;
                            decoder_->decode_block_at(j, kb, B_col + k_start);
                        }
                    }
                }

                // Now process all M×N_block with decoded tile
                for (int ii = 0; ii < m; ii += M_TILE)
                {
                    int m_block = std::min(M_TILE, m - ii);

                    // Prefetch next M-tile of A for better cache behavior
                    if (ii + M_TILE < m)
                    {
                        const float *next_A = A + (ii + M_TILE) * k;
                        for (int pf = 0; pf < std::min(M_TILE, m - ii - M_TILE); pf += 8)
                        {
                            __builtin_prefetch(next_A + pf * k, 0, 1); // Prefetch with low temporal locality
                        }
                    }

                    // Compute M_TILE × N_block outputs
                    for (int i_local = 0; i_local < m_block; ++i_local)
                    {
                        int i = ii + i_local;
                        const float *A_row = A + i * k;

                        for (int j_local = 0; j_local < n_block; ++j_local)
                        {
                            int j = jj + j_local;
                            const float *B_col = B_tile + j_local * k;

                            float acc = dot_product_simd(A_row, B_col, k);
                            size_t c_idx = i * n + j;
                            C[c_idx] = alpha * acc + beta * C[c_idx];
                        }
                    }
                }
            }

            // Free aligned allocation before thread exits
            free(tile_ptr);
        }

        return true;
    }

} // namespace llaminar2
