/**
 * @file Q16DotProductRef.cpp
 * @brief Reference implementation of Q16_1 dot product microkernel
 */

#include "Q16DotProductRef.h"
#include <algorithm>
#include <cmath>

namespace llaminar2::kernels::q16_1::microkernels
{

    // ============================================================================
    // Helper: Block-wise scaled dot product
    // ============================================================================

    /**
     * @brief Compute scaled dot product between two Q16_1 vectors
     *
     * Handles block-wise scale factors: actual_dot = Σ(int_dot_block × d_q × d_k)
     */
    static float compute_scaled_dot(
        const Q16_1Block *Q_blocks,
        const Q16_1Block *K_blocks,
        int q_row,
        int k_row,
        int q_start_col,
        int k_start_col,
        int head_dim,
        int q_blocks_per_row,
        int kv_blocks_per_row)
    {
        constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE; // 32

        float actual_dot = 0.0f;
        int num_blocks = (head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int b = 0; b < num_blocks; ++b)
        {
            int q_col_base = q_start_col + b * BLOCK_SIZE;
            int k_col_base = k_start_col + b * BLOCK_SIZE;

            int q_block_idx = q_col_base / BLOCK_SIZE;
            int k_block_idx = k_col_base / BLOCK_SIZE;

            const Q16_1Block &q_block = Q_blocks[q_row * q_blocks_per_row + q_block_idx];
            const Q16_1Block &k_block = K_blocks[k_row * kv_blocks_per_row + k_block_idx];

            // Block dot product (int32 accumulator matches AVX512 VNNI behavior)
            int32_t block_dot = 0;
            int elems_in_block = std::min(BLOCK_SIZE, head_dim - b * BLOCK_SIZE);

            for (int i = 0; i < elems_in_block; ++i)
            {
                int16_t q_val = q_block.qs[i];
                int16_t k_val = k_block.qs[i];
                block_dot += static_cast<int32_t>(q_val) * static_cast<int32_t>(k_val);
            }

            // Scale by both block factors
            actual_dot += static_cast<float>(block_dot) * q_block.d * k_block.d;
        }

        return actual_dot;
    }

    // ============================================================================
    // Single Dot Product
    // ============================================================================

    float q16_dot_product_single(
        const Q16DotProductParams &params,
        int q_row,
        int k_row)
    {
        int q_start_col = params.q_head * params.head_dim;
        int k_start_col = params.kv_head * params.head_dim;

        float dot = compute_scaled_dot(
            params.Q, params.K,
            q_row, k_row,
            q_start_col, k_start_col,
            params.head_dim,
            params.q_blocks_per_row,
            params.kv_blocks_per_row);

        return dot * params.attention_scale;
    }

    // ============================================================================
    // GEMV Variant (Decode)
    // ============================================================================

    void q16_dot_product_gemv(
        const Q16DotProductParams &params,
        int q_row,
        int k_start,
        int k_count,
        float *scores)
    {
        int q_start_col = params.q_head * params.head_dim;
        int k_start_col = params.kv_head * params.head_dim;

        for (int kv = 0; kv < k_count; ++kv)
        {
            int k_row = k_start + kv;

            float dot = compute_scaled_dot(
                params.Q, params.K,
                q_row, k_row,
                q_start_col, k_start_col,
                params.head_dim,
                params.q_blocks_per_row,
                params.kv_blocks_per_row);

            scores[kv] = dot * params.attention_scale;
        }
    }

    // ============================================================================
    // GEMM Variant (Prefill)
    // ============================================================================

    void q16_dot_product_gemm(
        const Q16DotProductParams &params,
        int q_start,
        int q_count,
        int k_start,
        int k_count,
        float *scores)
    {
        int q_start_col = params.q_head * params.head_dim;
        int k_start_col = params.kv_head * params.head_dim;

        for (int q_local = 0; q_local < q_count; ++q_local)
        {
            int q_row = q_start + q_local;

            for (int kv = 0; kv < k_count; ++kv)
            {
                int k_row = k_start + kv;

                float dot = compute_scaled_dot(
                    params.Q, params.K,
                    q_row, k_row,
                    q_start_col, k_start_col,
                    params.head_dim,
                    params.q_blocks_per_row,
                    params.kv_blocks_per_row);

                scores[q_local * k_count + kv] = dot * params.attention_scale;
            }
        }
    }

    // ============================================================================
    // Integer-Domain Variant
    // ============================================================================

    int32_t q16_dot_product_int32(
        const Q16_1Block *Q,
        const Q16_1Block *K,
        int q_row,
        int k_row,
        int head,
        int kv_head,
        int head_dim,
        int q_blocks_per_row,
        int kv_blocks_per_row)
    {
        constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;

        int q_start_col = head * head_dim;
        int k_start_col = kv_head * head_dim;

        int32_t dot = 0;

        for (int d = 0; d < head_dim; ++d)
        {
            int q_col = q_start_col + d;
            int q_block_idx = q_col / BLOCK_SIZE;
            int q_elem_idx = q_col % BLOCK_SIZE;
            const Q16_1Block &q_block = Q[q_row * q_blocks_per_row + q_block_idx];
            int16_t q_val = q_block.qs[q_elem_idx];

            int k_col = k_start_col + d;
            int k_block_idx = k_col / BLOCK_SIZE;
            int k_elem_idx = k_col % BLOCK_SIZE;
            const Q16_1Block &k_block = K[k_row * kv_blocks_per_row + k_block_idx];
            int16_t k_val = k_block.qs[k_elem_idx];

            dot += static_cast<int32_t>(q_val) * static_cast<int32_t>(k_val);
        }

        return dot;
    }

} // namespace llaminar2::kernels::q16_1::microkernels
