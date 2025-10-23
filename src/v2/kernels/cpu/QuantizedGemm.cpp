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

        // Adaptive tiling based on problem shape
        const auto &env = debugEnv();
        int M_TILE, N_TILE;

        // Check for manual tile size override
        if (env.dequant.iq4_override_m_tile > 0 && env.dequant.iq4_override_n_tile > 0)
        {
            M_TILE = env.dequant.iq4_override_m_tile;
            N_TILE = env.dequant.iq4_override_n_tile;
        }
        else
        {
            // Adaptive defaults based on aspect ratio
            const float aspect_ratio = static_cast<float>(n) / static_cast<float>(m > 0 ? m : 1);
            const bool is_wide_output = aspect_ratio > 2.0f;                     // FFN-like
            const bool is_square = aspect_ratio >= 0.5f && aspect_ratio <= 2.0f; // Q-proj-like

            if (is_wide_output)
            {
                // Memory-bound path (FFN: wide output)
                M_TILE = 64;
                N_TILE = 32; // Empirically optimal
            }
            else if (is_square)
            {
                // Compute-bound path (Q-proj: square matrix)
                if (m >= 2048 || n >= 2048)
                {
                    M_TILE = 64;
                    N_TILE = 32; // Universal optimal
                }
                else if (m >= 512 || n >= 512)
                {
                    M_TILE = 96;
                    N_TILE = 96;
                }
                else
                {
                    M_TILE = 128;
                    N_TILE = 128; // Minimize overhead for small ops
                }
            }
            else
            {
                // Tall matrix path
                M_TILE = 64;
                N_TILE = 24;
            }
        }

#pragma omp parallel
        {
            // Thread-local buffer for N_TILE decoded columns
            std::vector<float> B_tile(k * N_TILE);

#pragma omp for schedule(dynamic)
            for (int jj = 0; jj < n; jj += N_TILE)
            {
                int n_block = std::min(N_TILE, n - jj);

                // Decode N_TILE columns of B (weights)
                for (int j_local = 0; j_local < n_block; ++j_local)
                {
                    int j = jj + j_local;
                    float *B_col = B_tile.data() + j_local * k;

                    // Decode all K-blocks for this column
                    for (int kb = 0; kb < num_k_blocks; ++kb)
                    {
                        size_t k_start = kb * BLOCK_SIZE;
                        decoder_->decode_block_at(j, kb, B_col + k_start);
                    }
                }

                // Process M×N_block with decoded tile
                for (int ii = 0; ii < m; ii += M_TILE)
                {
                    int m_block = std::min(M_TILE, m - ii);

                    // Compute M_block × N_block outputs
                    for (int i_local = 0; i_local < m_block; ++i_local)
                    {
                        int i = ii + i_local;
                        const float *A_row = A + i * k;

                        for (int j_local = 0; j_local < n_block; ++j_local)
                        {
                            int j = jj + j_local;
                            const float *B_col = B_tile.data() + j_local * k;

                            float acc = dot_product_simd(A_row, B_col, k);
                            size_t c_idx = i * n + j;
                            C[c_idx] = alpha * acc + beta * C[c_idx];
                        }
                    }
                }
            }
        }

        return true;
    }

} // namespace llaminar2
