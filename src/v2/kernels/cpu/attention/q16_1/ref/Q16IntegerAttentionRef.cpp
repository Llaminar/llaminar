/**
 * @file Q16IntegerAttentionRef.cpp
 * @brief TRUE integer-only Q16 attention reference implementation (v2)
 *
 * This implementation fulfills the promise of "pure integer" attention that v1 failed to deliver.
 * The key insight is matching block size to head_dim so we get 1 scale per head.
 *
 * @see Q16IntegerAttentionRef.h for design rationale
 * @see docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md for full project plan
 */

#include "Q16IntegerAttentionRef.h"
#include "microkernels/Exp2FixedSoftmax.h"
#include "microkernels/Q16DotProduct.h"
#include "microkernels/PVAccumulate.h"
#include "microkernels/WoProjection.h"
#include "microkernels/OnlineSoftmax.h"
#include "utils/Assertions.h"
#include "utils/Logger.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace llaminar2::kernels::q16_1
{

    // ============================================================================
    // Internal Constants
    // ============================================================================

    namespace
    {
        // FA2 tile sizes
        constexpr int TILE_BR = 4;  // Query tile size
        constexpr int TILE_BC = 32; // KV tile size

        // INT16 weight max for softmax output (VNNI-friendly)
        constexpr int16_t WEIGHT_MAX = 32767;

        // Mask value for causal attention
        constexpr int32_t MASKED_SCORE = std::numeric_limits<int32_t>::min();

    } // namespace

    // ============================================================================
    // Block Access Helpers (Template for Variable Block Sizes)
    // ============================================================================

    namespace
    {
        /**
         * @brief Get INT16 value from variable-size block at specified position.
         *
         * @tparam BlockType One of Q16_1Block_64, Q16_1Block_128
         */
        template <typename BlockType>
        inline int16_t get_block_value(
            const BlockType *blocks,
            int row,
            int col,
            int blocks_per_row)
        {
            constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;
            int block_idx = col / BLOCK_SIZE;
            int elem_idx = col % BLOCK_SIZE;
            const BlockType &block = blocks[row * blocks_per_row + block_idx];
            return block.qs[elem_idx];
        }

        /**
         * @brief Get scale factor from variable-size block.
         */
        template <typename BlockType>
        inline float get_block_scale(
            const BlockType *blocks,
            int row,
            int col,
            int blocks_per_row)
        {
            constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;
            int block_idx = col / BLOCK_SIZE;
            const BlockType &block = blocks[row * blocks_per_row + block_idx];
            return block.d;
        }

        /**
         * @brief Get pointer to INT16 data in block (for VNNI loads).
         */
        template <typename BlockType>
        inline const int16_t *get_block_data(
            const BlockType *blocks,
            int row,
            int block_idx,
            int blocks_per_row)
        {
            return blocks[row * blocks_per_row + block_idx].qs;
        }

    } // namespace

    // ============================================================================
    // INTEGER Q×K^T Dot Product - Delegated to Q16DotProduct Microkernel
    // ============================================================================

    // NOTE: The actual implementations are in microkernels/Q16DotProduct.h/.cpp
    // The inline helpers below are retained only for internal dispatch compatibility.

    namespace
    {
        /**
         * @brief Compute Q×K^T for all KV positions (Flash Decode).
         *
         * Delegates to the Q16DotProduct microkernel.
         */
        template <typename BlockType>
        void integer_qk_gemv(
            const BlockType *Q_blocks,
            const BlockType *K_blocks,
            int32_t *scores,
            int q_row,
            int k_count,
            int head_dim,
            int blocks_per_row)
        {
            // Get Q pointer for the specified row
            const BlockType *Q_row = Q_blocks + q_row * blocks_per_row;

            // Delegate to microkernel
            microkernels::q16_qk_gemv<BlockType>(
                Q_row, K_blocks, scores,
                k_count, head_dim, blocks_per_row);
        }

    } // namespace

    // ============================================================================
    // INTEGER P×V Accumulation - Delegated to PVAccumulate Microkernel
    // ============================================================================

    // NOTE: The actual implementations are in microkernels/PVAccumulate.h/.cpp
    // The inline helper below is retained only for internal dispatch compatibility.

    namespace
    {
        /**
         * @brief Accumulate P×V in pure INT32.
         *
         * Delegates to the PVAccumulate microkernel.
         */
        template <typename BlockType>
        void integer_pv_accumulate_impl(
            const int16_t *weights,
            const BlockType *V_blocks,
            int32_t *context,
            int kv_len,
            int head_dim,
            int blocks_per_row)
        {
            // Delegate to microkernel
            microkernels::q16_pv_accumulate<BlockType>(
                weights, V_blocks, context,
                kv_len, head_dim, blocks_per_row);
        }

    } // namespace

    // ============================================================================
    // Flash Decode Implementation (seq_len_q = 1) - Online Softmax
    // ============================================================================

    /**
     * @brief TRUE FlashDecode with online softmax - block-at-a-time processing.
     *
     * This is the memory-efficient FlashDecode algorithm:
     * - Processes KV cache in blocks (default 32 positions)
     * - Maintains online softmax state (running max, sum, context)
     * - Never materializes full [kv_len] scores array
     * - Rescales previous context when running max changes
     *
     * Algorithm:
     *   m = -inf, l = 0, O = 0
     *   for each KV block:
     *     scores = Q × K[block]^T
     *     m_new = max(m, max(scores))
     *     if m_new > m:
     *       correction = exp(m - m_new)
     *       l *= correction
     *       O *= correction
     *     for k in block:
     *       w = exp(scores[k] - m_new)
     *       l += w
     *       O += w * V[k]
     *     m = m_new
     *   output = O / l
     */
    bool q16_integer_attention_decode(const Q16IntegerAttentionParams &params)
    {
        // Validate
        if (!q16_validate_integer_params(params))
        {
            return false;
        }

        LLAMINAR_ASSERT(params.is_decode(), "Decode path requires seq_len_q=1");

        const int num_heads = params.num_heads;
        const int kv_len = params.kv_len;
        const int head_dim = params.head_dim;
        const int blocks_per_row = params.q_blocks_per_row();

        // Compute cache-aware KV block size
        const int KV_BLOCK_SIZE = microkernels::compute_flash_decode_kv_block_size(head_dim);

        LOG_DEBUG("FlashDecode: KV_BLOCK_SIZE=" << KV_BLOCK_SIZE
                                                << " (scratch=" << (KV_BLOCK_SIZE * 8) << " bytes)");

        // Per-head scratch buffers (sized to computed block size)
        std::vector<int32_t> scores_scratch(KV_BLOCK_SIZE);
        std::vector<int32_t> weights_scratch(KV_BLOCK_SIZE);
        std::vector<int32_t> context(head_dim);

        // For snapshots: store scores (not weights) so we can recompute with final max
        std::vector<int32_t> all_scores_raw;
        if (params.snapshot_weights || params.snapshot_scores)
        {
            all_scores_raw.resize(kv_len, MASKED_SCORE);
        }

        // Process each head
        for (int h = 0; h < num_heads; ++h)
        {
            int kv_h = params.get_kv_head(h);

            // Get scales for this head
            float qk_scale = params.get_qk_scale(h, kv_h);
            float pv_scale = params.get_pv_scale(kv_h);

            // Initialize online softmax state
            microkernels::OnlineSoftmaxState state;
            state.frac_bits = 11;
            state.lut_value_bits = 30;

            // Zero context
            std::memset(context.data(), 0, head_dim * sizeof(int32_t));

            // ================================================================
            // FlashDecode: Process KV cache in blocks
            // ================================================================

            switch (params.block_size)
            {
            case Q16BlockSize::BLOCK_64:
            {
                auto Q = reinterpret_cast<const Q16_1Block_64 *>(params.Q);
                auto K = reinterpret_cast<const Q16_1Block_64 *>(params.K);
                auto V = reinterpret_cast<const Q16_1Block_64 *>(params.V);

                const Q16_1Block_64 *Q_head = Q + h * blocks_per_row;
                const Q16_1Block_64 *K_head = K + kv_h * kv_len * blocks_per_row;
                const Q16_1Block_64 *V_head = V + kv_h * kv_len * blocks_per_row;

                // Process in blocks
                for (int kv_start = 0; kv_start < kv_len; kv_start += KV_BLOCK_SIZE)
                {
                    int block_size = std::min(KV_BLOCK_SIZE, kv_len - kv_start);

                    microkernels::flash_decode_process_kv_block<Q16_1Block_64>(
                        Q_head, K_head, V_head,
                        context.data(), state,
                        scores_scratch.data(), weights_scratch.data(),
                        kv_start, block_size,
                        head_dim, blocks_per_row, qk_scale);

                    // Capture raw scores (before softmax) for snapshot
                    // We'll recompute weights at the end using final max
                    if (params.snapshot_weights || params.snapshot_scores)
                    {
                        for (int i = 0; i < block_size; ++i)
                        {
                            all_scores_raw[kv_start + i] = scores_scratch[i];
                        }
                    }
                }
                break;
            }
            case Q16BlockSize::BLOCK_128:
            {
                auto Q = reinterpret_cast<const Q16_1Block_128 *>(params.Q);
                auto K = reinterpret_cast<const Q16_1Block_128 *>(params.K);
                auto V = reinterpret_cast<const Q16_1Block_128 *>(params.V);

                const Q16_1Block_128 *Q_head = Q + h * blocks_per_row;
                const Q16_1Block_128 *K_head = K + kv_h * kv_len * blocks_per_row;
                const Q16_1Block_128 *V_head = V + kv_h * kv_len * blocks_per_row;

                for (int kv_start = 0; kv_start < kv_len; kv_start += KV_BLOCK_SIZE)
                {
                    int block_size = std::min(KV_BLOCK_SIZE, kv_len - kv_start);

                    microkernels::flash_decode_process_kv_block<Q16_1Block_128>(
                        Q_head, K_head, V_head,
                        context.data(), state,
                        scores_scratch.data(), weights_scratch.data(),
                        kv_start, block_size,
                        head_dim, blocks_per_row, qk_scale);

                    // Capture raw scores for snapshot
                    if (params.snapshot_weights || params.snapshot_scores)
                    {
                        for (int i = 0; i < block_size; ++i)
                        {
                            all_scores_raw[kv_start + i] = scores_scratch[i];
                        }
                    }
                }
                break;
            }
            default:
                LOG_ERROR("Unsupported block size: " << static_cast<int>(params.block_size));
                return false;
            }

            // ================================================================
            // Finalize: Normalize context by sum of weights
            // ================================================================

            // Context is currently: Σ (w_unnorm * V) where w_unnorm are in LUT precision
            // Final context = context / l (the running sum)

            // Snapshot: pre-softmax scores (if requested)
            // We stored raw INT32 scores, just need to convert to float
            if (params.snapshot_scores)
            {
                for (int k = 0; k < kv_len; ++k)
                {
                    // Raw score is already in INT32 format, scale to FP32
                    params.snapshot_scores[h * kv_len + k] =
                        static_cast<float>(all_scores_raw[k]) * qk_scale;
                }
            }

            // Snapshot: post-softmax weights (if requested)
            // Recompute weights from stored scores using final max (state.m)
            // This ensures all weights are relative to the same max
            if (params.snapshot_weights)
            {
                // Ensure exp2 LUT is initialized
                microkernels::ensure_exp2_lut_initialized(state.lut_value_bits);
                const uint32_t *lut = microkernels::get_exp2_lut_data();

                // Compute beta = alpha * log2(e) for exp2 conversion
                constexpr double LOG2E = 1.4426950408889634073599246810018921;
                double beta = static_cast<double>(qk_scale) * LOG2E;
                int beta_scale_bits = 24;
                int64_t M = static_cast<int64_t>(
                    std::llround(beta * static_cast<double>(1ULL << beta_scale_bits)));
                int shift_for_t = beta_scale_bits - state.frac_bits;
                uint32_t one = static_cast<uint32_t>(1U << state.lut_value_bits);

                // Recompute all weights using final max and accumulate sum
                int64_t weight_sum = 0;
                std::vector<int64_t> weights_recomputed(kv_len);

                for (int k = 0; k < kv_len; ++k)
                {
                    if (all_scores_raw[k] == MASKED_SCORE)
                    {
                        weights_recomputed[k] = 0;
                        continue;
                    }

                    int32_t delta = state.m - all_scores_raw[k];

                    if (delta <= 0)
                    {
                        // This is the max position
                        weights_recomputed[k] = one;
                        weight_sum += one;
                        continue;
                    }

                    // t_fixed = delta * M in Q(frac_bits) format
                    int64_t prod = static_cast<int64_t>(delta) * M;
                    int64_t t_fixed = (shift_for_t >= 0)
                                          ? (prod >> shift_for_t)
                                          : (prod << (-shift_for_t));

                    // Decompose: t = ip + frac
                    int64_t ip = t_fixed >> state.frac_bits;
                    int frac_idx = static_cast<int>(t_fixed & ((1 << state.frac_bits) - 1));

                    // Underflow check
                    if (ip >= 31)
                    {
                        weights_recomputed[k] = 0;
                        continue;
                    }

                    // 2^(-t) = lut[frac] >> ip
                    uint32_t w = lut[frac_idx] >> static_cast<int>(ip);
                    weights_recomputed[k] = static_cast<int64_t>(w);
                    weight_sum += w;
                }

                // Normalize to [0, 1]
                for (int k = 0; k < kv_len; ++k)
                {
                    float w_norm = (weight_sum > 0)
                                       ? static_cast<float>(weights_recomputed[k]) / static_cast<float>(weight_sum)
                                       : 0.0f;
                    params.snapshot_weights[h * kv_len + k] = w_norm;
                }
            }

            // Snapshot: attention context (if requested)
            // Context needs to be divided by l and scaled
            if (params.snapshot_context)
            {
                float context_scale = (state.l > 0)
                                          ? pv_scale / static_cast<float>(state.l >> (state.lut_value_bits - 15))
                                          : 0.0f;
                for (int d = 0; d < head_dim; ++d)
                {
                    params.snapshot_context[h * head_dim + d] =
                        static_cast<float>(context[d]) * context_scale;
                }
            }

            // ================================================================
            // Step 4: Wo projection → Q16_1 output
            // ================================================================

            // TODO(v2): Implement Wo projection with VPDPWSSD
            // For now, this is a placeholder
            // The Wo projection should:
            // 1. Take INT32 context [head_dim]
            // 2. Multiply by packed Wo weights
            // 3. Produce Q16_1 output [d_model]

            // ================================================================
            // Step 5: Residual add (integer)
            // ================================================================

            // TODO(v2): Implement integer residual add
            // Should handle scale alignment between attention output and residual
        }

        return true;
    }

    // ============================================================================
    // FA2 Prefill Implementation (seq_len_q > 1)
    // ============================================================================

    /**
     * @brief TRUE FA2 Prefill with online softmax - tiled block-at-a-time processing.
     *
     * This is the FlashAttention-2 algorithm adapted for multi-query prefill:
     * - Processes queries in tiles of Br rows (default 4)
     * - Processes KV cache in tiles of Bc columns (default 32)
     * - Maintains per-row online softmax state (m, l per query row)
     * - Rescales previous context when running max changes
     *
     * Algorithm (for each query tile):
     *   m[:] = -inf, l[:] = 0, O[:, :] = 0
     *   for each KV tile:
     *     S = Q_tile × K_tile^T                    [Br × Bc]
     *     for each row r:
     *       m_new[r] = max(m[r], max(S[r, :]))
     *       if m_new[r] > m[r]:
     *         correction = exp(m[r] - m_new[r])
     *         l[r] *= correction
     *         O[r, :] *= correction
     *       for k in tile:
     *         w = exp(S[r, k] - m_new[r])
     *         l[r] += w
     *         O[r, :] += w * V[k, :]
     *       m[r] = m_new[r]
     *   output[r, :] = O[r, :] / l[r]
     */
    bool q16_integer_attention_prefill(const Q16IntegerAttentionParams &params)
    {
        // Validate
        if (!q16_validate_integer_params(params))
        {
            return false;
        }

        LLAMINAR_ASSERT(!params.is_decode(), "Prefill path requires seq_len_q>1");

        const int seq_len_q = params.seq_len_q;
        const int kv_len = params.kv_len;
        const int num_heads = params.num_heads;
        const int head_dim = params.head_dim;
        const int blocks_per_row = params.q_blocks_per_row();

        // Compute cache-aware tile sizes based on CPU cache hierarchy
        const auto tile_cfg = microkernels::compute_fa2_tile_config(head_dim, kv_len);
        const int TILE_BR = tile_cfg.Br;
        const int TILE_BC = tile_cfg.Bc;

        LOG_DEBUG("FA2 Prefill: Br=" << TILE_BR << " Bc=" << TILE_BC
                                     << " (scratch=" << tile_cfg.scratch_bytes() << " bytes"
                                     << ", context=" << tile_cfg.context_bytes() << " bytes)");

        // Per-tile scratch buffers (sized to computed tile dimensions)
        std::vector<int32_t> scores_scratch(TILE_BR * TILE_BC);
        std::vector<int32_t> weights_scratch(TILE_BR * TILE_BC);

        // Per-row context accumulators (for current Q tile)
        std::vector<int32_t> context_tile(TILE_BR * head_dim);

        // For snapshots: store raw scores so we can recompute weights with final max
        std::vector<int32_t> all_scores_raw;
        if (params.snapshot_weights || params.snapshot_scores)
        {
            all_scores_raw.resize(seq_len_q * kv_len, MASKED_SCORE);
        }

        // Process each head
        for (int h = 0; h < num_heads; ++h)
        {
            int kv_h = params.get_kv_head(h);

            // Get scales for this head
            float qk_scale = params.get_qk_scale(h, kv_h);
            float pv_scale = params.get_pv_scale(kv_h);

            // Process query sequence in tiles of TILE_BR
            for (int q_tile_start = 0; q_tile_start < seq_len_q; q_tile_start += TILE_BR)
            {
                int Br = std::min(TILE_BR, seq_len_q - q_tile_start);

                // Initialize batch softmax state for this Q tile
                microkernels::OnlineSoftmaxStateBatch state;
                state.init(Br);

                // Zero context tile
                std::memset(context_tile.data(), 0, TILE_BR * head_dim * sizeof(int32_t));

                // ================================================================
                // FA2: Process KV cache in tiles of TILE_BC
                // ================================================================

                switch (params.block_size)
                {
                case Q16BlockSize::BLOCK_64:
                {
                    auto Q = reinterpret_cast<const Q16_1Block_64 *>(params.Q);
                    auto K = reinterpret_cast<const Q16_1Block_64 *>(params.K);
                    auto V = reinterpret_cast<const Q16_1Block_64 *>(params.V);

                    // Q pointer for this head and Q tile
                    const Q16_1Block_64 *Q_tile = Q + h * seq_len_q * blocks_per_row +
                                                  q_tile_start * blocks_per_row;

                    // K/V pointers for this KV head
                    const Q16_1Block_64 *K_head = K + kv_h * kv_len * blocks_per_row;
                    const Q16_1Block_64 *V_head = V + kv_h * kv_len * blocks_per_row;

                    // Process KV in tiles
                    for (int kv_tile_start = 0; kv_tile_start < kv_len; kv_tile_start += TILE_BC)
                    {
                        int Bc = std::min(TILE_BC, kv_len - kv_tile_start);

                        microkernels::fa2_prefill_process_kv_tile<Q16_1Block_64>(
                            Q_tile, K_head, V_head,
                            context_tile.data(), state,
                            scores_scratch.data(), weights_scratch.data(),
                            kv_tile_start, Br, Bc,
                            head_dim, blocks_per_row,
                            blocks_per_row, // q_stride
                            blocks_per_row, // k_stride
                            head_dim,       // context_stride
                            qk_scale,
                            true,          // causal
                            q_tile_start); // q_offset for causal mask

                        // Capture raw scores (with causal mask applied) for snapshot
                        // Note: microkernel uses r * Bc stride, not r * TILE_BC
                        if (params.snapshot_weights || params.snapshot_scores)
                        {
                            for (int r = 0; r < Br; ++r)
                            {
                                for (int c = 0; c < Bc; ++c)
                                {
                                    int q_idx = q_tile_start + r;
                                    int k_idx = kv_tile_start + c;
                                    all_scores_raw[q_idx * kv_len + k_idx] =
                                        scores_scratch[r * Bc + c];
                                }
                            }
                        }
                    }
                    break;
                }
                case Q16BlockSize::BLOCK_128:
                {
                    auto Q = reinterpret_cast<const Q16_1Block_128 *>(params.Q);
                    auto K = reinterpret_cast<const Q16_1Block_128 *>(params.K);
                    auto V = reinterpret_cast<const Q16_1Block_128 *>(params.V);

                    const Q16_1Block_128 *Q_tile = Q + h * seq_len_q * blocks_per_row +
                                                   q_tile_start * blocks_per_row;
                    const Q16_1Block_128 *K_head = K + kv_h * kv_len * blocks_per_row;
                    const Q16_1Block_128 *V_head = V + kv_h * kv_len * blocks_per_row;

                    for (int kv_tile_start = 0; kv_tile_start < kv_len; kv_tile_start += TILE_BC)
                    {
                        int Bc = std::min(TILE_BC, kv_len - kv_tile_start);

                        microkernels::fa2_prefill_process_kv_tile<Q16_1Block_128>(
                            Q_tile, K_head, V_head,
                            context_tile.data(), state,
                            scores_scratch.data(), weights_scratch.data(),
                            kv_tile_start, Br, Bc,
                            head_dim, blocks_per_row,
                            blocks_per_row, blocks_per_row, head_dim,
                            qk_scale, true, q_tile_start);

                        // Capture raw scores for snapshot
                        // Note: microkernel uses r * Bc stride, not r * TILE_BC
                        if (params.snapshot_weights || params.snapshot_scores)
                        {
                            for (int r = 0; r < Br; ++r)
                            {
                                for (int c = 0; c < Bc; ++c)
                                {
                                    int q_idx = q_tile_start + r;
                                    int k_idx = kv_tile_start + c;
                                    all_scores_raw[q_idx * kv_len + k_idx] =
                                        scores_scratch[r * Bc + c];
                                }
                            }
                        }
                    }
                    break;
                }
                default:
                    LOG_ERROR("Unsupported block size: " << static_cast<int>(params.block_size));
                    return false;
                }

                // ================================================================
                // Finalize: Store normalized context for this Q tile
                // ================================================================

                for (int r = 0; r < Br; ++r)
                {
                    int q_idx = q_tile_start + r;

                    // Snapshot: pre-softmax scores (raw INT32)
                    if (params.snapshot_scores)
                    {
                        for (int k = 0; k < kv_len; ++k)
                        {
                            int32_t score_raw = all_scores_raw[q_idx * kv_len + k];
                            // Convert to FP32. Masked scores stay as large negative
                            if (score_raw == MASKED_SCORE)
                            {
                                params.snapshot_scores[(h * seq_len_q + q_idx) * kv_len + k] = -1e9f;
                            }
                            else
                            {
                                params.snapshot_scores[(h * seq_len_q + q_idx) * kv_len + k] =
                                    static_cast<float>(score_raw) * qk_scale;
                            }
                        }
                    }

                    // Snapshot: post-softmax weights
                    // Recompute weights from scores using final max (state.m[r]) for this row
                    if (params.snapshot_weights)
                    {
                        // Ensure exp2 LUT is initialized
                        microkernels::ensure_exp2_lut_initialized(state.lut_value_bits);
                        const uint32_t *lut = microkernels::get_exp2_lut_data();

                        // Compute beta = alpha * log2(e)
                        constexpr double LOG2E = 1.4426950408889634073599246810018921;
                        double beta = static_cast<double>(qk_scale) * LOG2E;
                        int beta_scale_bits = 24;
                        int64_t M = static_cast<int64_t>(
                            std::llround(beta * static_cast<double>(1ULL << beta_scale_bits)));
                        int shift_for_t = beta_scale_bits - state.frac_bits;
                        uint32_t one = static_cast<uint32_t>(1U << state.lut_value_bits);

                        // Recompute all weights using final max and accumulate sum
                        int64_t weight_sum = 0;
                        std::vector<int64_t> weights_recomputed(kv_len);

                        for (int k = 0; k < kv_len; ++k)
                        {
                            int32_t score_raw = all_scores_raw[q_idx * kv_len + k];
                            if (score_raw == MASKED_SCORE)
                            {
                                weights_recomputed[k] = 0;
                                continue;
                            }

                            int32_t delta = state.m[r] - score_raw;

                            if (delta <= 0)
                            {
                                // This is the max position
                                weights_recomputed[k] = one;
                                weight_sum += one;
                                continue;
                            }

                            // t_fixed = delta * M in Q(frac_bits) format
                            int64_t prod = static_cast<int64_t>(delta) * M;
                            int64_t t_fixed = (shift_for_t >= 0)
                                                  ? (prod >> shift_for_t)
                                                  : (prod << (-shift_for_t));

                            // Decompose: t = ip + frac
                            int64_t ip = t_fixed >> state.frac_bits;
                            int frac_idx = static_cast<int>(t_fixed & ((1 << state.frac_bits) - 1));

                            // Underflow check
                            if (ip >= 31)
                            {
                                weights_recomputed[k] = 0;
                                continue;
                            }

                            // 2^(-t) = lut[frac] >> ip
                            uint32_t w = lut[frac_idx] >> static_cast<int>(ip);
                            weights_recomputed[k] = static_cast<int64_t>(w);
                            weight_sum += w;
                        }

                        // Normalize to [0, 1]
                        for (int k = 0; k < kv_len; ++k)
                        {
                            float w_norm = (weight_sum > 0)
                                               ? static_cast<float>(weights_recomputed[k]) / static_cast<float>(weight_sum)
                                               : 0.0f;
                            params.snapshot_weights[(h * seq_len_q + q_idx) * kv_len + k] = w_norm;
                        }
                    }

                    // Snapshot: attention context
                    if (params.snapshot_context)
                    {
                        int64_t l_val = state.l[r];
                        int shift_amt = state.lut_value_bits - 15;
                        int64_t l_shifted = l_val >> shift_amt;
                        float context_scale = (l_shifted > 0)
                                                  ? pv_scale / static_cast<float>(l_shifted)
                                                  : 0.0f;
                        for (int d = 0; d < head_dim; ++d)
                        {
                            params.snapshot_context[(h * seq_len_q + q_idx) * head_dim + d] =
                                static_cast<float>(context_tile[r * head_dim + d]) * context_scale;
                        }
                    }
                }

                // ================================================================
                // Step 4: Wo projection → Q16_1 output (TODO)
                // ================================================================

                // TODO(v2): Implement Wo projection with VPDPWSSD

                // ================================================================
                // Step 5: Residual add (integer) (TODO)
                // ================================================================

                // TODO(v2): Implement integer residual add
            }
        }

        return true;
    }

    // ============================================================================
    // Validation
    // ============================================================================

    bool q16_validate_integer_params(const Q16IntegerAttentionParams &params)
    {
        // Required pointers
        if (!params.Q || !params.K || !params.V)
        {
            LOG_ERROR("Q16IntegerAttention: Q, K, V pointers required");
            return false;
        }

        // Dimensions
        if (params.seq_len_q <= 0 || params.kv_len <= 0)
        {
            LOG_ERROR("Q16IntegerAttention: Invalid sequence lengths");
            return false;
        }

        if (params.num_heads <= 0 || params.num_kv_heads <= 0)
        {
            LOG_ERROR("Q16IntegerAttention: Invalid head counts");
            return false;
        }

        if (params.head_dim <= 0)
        {
            LOG_ERROR("Q16IntegerAttention: Invalid head dimension");
            return false;
        }

        // GQA validation
        if (params.num_heads % params.num_kv_heads != 0)
        {
            LOG_ERROR("Q16IntegerAttention: num_heads must be divisible by num_kv_heads");
            return false;
        }

        // Block size validation
        int bs = static_cast<int>(params.block_size);
        if (bs != 32 && bs != 64 && bs != 128 && bs != 192)
        {
            LOG_ERROR("Q16IntegerAttention: Invalid block size: " << bs);
            return false;
        }

        // Warn if head_dim doesn't align with block size
        if (!params.is_head_aligned())
        {
            LOG_WARN("Q16IntegerAttention: head_dim=" << params.head_dim
                                                      << " not aligned with any optimal block size. "
                                                      << "Consider using block_size=" << static_cast<int>(optimal_q16_block_size(params.head_dim))
                                                      << " for best integer performance.");
        }

        return true;
    }

    // ============================================================================
    // Public Microkernel Dispatch (for testing)
    // ============================================================================

    void q16_integer_qk_dotproduct(
        const void *Q,
        const void *K,
        int32_t *scores,
        int k_count,
        int head_dim,
        Q16BlockSize block_size)
    {
        // Delegate to microkernel dispatch
        microkernels::q16_qk_gemv_dispatch(
            Q, K, scores,
            k_count, head_dim, block_size);
    }

    void q16_integer_pv_accumulate(
        const int16_t *weights,
        const void *V,
        int32_t *context,
        int kv_len,
        int head_dim,
        Q16BlockSize block_size)
    {
        // Delegate to microkernel dispatch
        microkernels::q16_pv_accumulate_dispatch(
            weights, V, context,
            kv_len, head_dim, block_size);
    }

} // namespace llaminar2::kernels::q16_1
