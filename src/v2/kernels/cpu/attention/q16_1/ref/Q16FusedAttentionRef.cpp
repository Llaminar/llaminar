/**
 * @file Q16FusedAttentionRef.cpp
 * @brief Scalar reference implementation of Q16_1 fully fused attention (PURE INTEGER)
 *
 * This file implements the FULL fused attention kernel with two execution paths:
 *
 * FLASH DECODE (seq_len_q=1):
 * - Single query against full KV cache
 * - Parallel over KV positions, no tiling
 * - Single-pass Exp2FixedSoftmax (no online state tracking)
 * - Steps: Q×K^T (INT32) → Softmax (INT16) → P×V (INT32) → Wo (Q16_1) → Residual (Q16_1)
 * - Optimized for latency in token generation
 *
 * FA2 PREFILL (seq_len_q>1):
 * - Batched queries with per-query processing
 * - Per-query Exp2FixedSoftmax (integer domain)
 * - Steps: [Per-query: Q×K^T → Softmax → P×V] → Wo → Residual
 * - Optimized for throughput in prompt processing
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * PURE INTEGER-DOMAIN PIPELINE (NO FP32 INTERMEDIATE!)
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * The entire attention-to-residual pipeline uses integer operations:
 *
 * 1. Q×K^T → INT32 scores
 *    - Int8RequantRef: Q16→INT8 requant for stable alpha
 *    - INT8×INT8 → INT32 dot products
 *
 * 2. Exp2FixedSoftmax → INT16 weights
 *    - Fixed-point exp2 approximation
 *    - Outputs INT16 weights [0, 32767]
 *
 * 3. P×V → INT32 accumulators (NOT FP32!)
 *    - INT16 weights × INT16 V values → INT32 accumulators
 *    - Tracks weight_sum and v_scale_product for deferred scaling
 *
 * 4. Wo Projection (VPDPWSSD) → Q16_1 output
 *    - INT32→INT16 context requantization (keeps 16 bits!)
 *    - VPDPWSSD: INT16 context × INT16 weights → INT32
 *    - INT32→Q16_1 output requantization
 *
 * 5. Native Q16_1 Residual Add
 *    - Uses simd::q16_1_add_q16_1() for Q16_1 + Q16_1
 *    - No FP32 conversion anywhere in the pipeline!
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * @see Q16FusedAttentionRef.h for API documentation
 * @see PROJECT_Q16_INTEGER_ATTENTION.md for design details
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include "Q16FusedAttentionRef.h"
#include "microkernels/Q16DotProductRef.h"
#include "microkernels/Exp2FixedSoftmaxRef.h"
#include "microkernels/Int8RequantRef.h"
#include "microkernels/WoProjectionVNNIRef.h"
#include "tensors/SIMDHelpers.h"
#include "utils/Logger.h"
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cmath>

namespace llaminar2::kernels::q16_1
{

    // Import microkernel namespace for convenience
    using namespace microkernels;

    // ============================================================================
    // Full Fused Kernel Validation
    // ============================================================================

    bool q16_validate_full_params(const Q16FusedAttentionWoResidualParams &params)
    {
        // Validate attention inputs
        if (params.Q == nullptr)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: Q tensor is null");
            return false;
        }
        if (params.K == nullptr)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: K tensor is null");
            return false;
        }
        if (params.V == nullptr)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: V tensor is null");
            return false;
        }

        // Validate Wo weights
        if (params.Wo_packed == nullptr)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: Wo_packed weights are null");
            return false;
        }

        // Validate residual
        if (params.residual_in == nullptr)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: residual_in is null");
            return false;
        }
        if (params.residual_out == nullptr)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: residual_out is null");
            return false;
        }

        // Validate dimensions
        if (params.seq_len_q <= 0)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: seq_len_q must be > 0");
            return false;
        }
        if (params.kv_len <= 0)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: kv_len must be > 0");
            return false;
        }
        if (params.num_heads <= 0)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: num_heads must be > 0");
            return false;
        }
        if (params.num_kv_heads <= 0)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: num_kv_heads must be > 0");
            return false;
        }
        if (params.head_dim <= 0)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: head_dim must be > 0");
            return false;
        }
        if (params.d_model <= 0)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: d_model must be > 0");
            return false;
        }
        if (params.num_heads % params.num_kv_heads != 0)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: num_heads must be divisible by num_kv_heads");
            return false;
        }

        return true;
    }

    // ============================================================================
    // Flash Decode: Pure Integer Attention Core Helper (Single Query)
    // ============================================================================

    /**
     * @brief Process attention for a single head in Flash Decode mode (PURE INTEGER)
     *
     * INTEGER-DOMAIN PIPELINE (NO FP32 INTERMEDIATE!):
     * 1. Q×K^T → INT32 scores (via Int8Requant)
     * 2. Exp2FixedSoftmax → INT16 weights
     * 3. P×V → INT32 accumulators (NOT FP32!)
     *
     * The INT32 accumulators and scale info are returned for Wo projection.
     *
     * @param params Full kernel parameters
     * @param head Query head index
     * @param int32_accum Output INT32 accumulator for this head [head_dim]
     * @param weight_sum Output sum of INT16 weights
     * @param v_scale_product Output product of V block scales
     */
    static void flash_decode_attention_head_integer(
        const Q16FusedAttentionWoResidualParams &params,
        int head,
        int32_t *int32_accum,
        int32_t *weight_sum,
        float *v_scale_product)
    {
        const int head_dim = params.head_dim;
        const int kv_len = params.kv_len;
        const int kv_head = params.get_kv_head(head);
        const int q_blocks_per_row = params.q_blocks_per_row();
        const int kv_blocks_per_row = params.kv_blocks_per_row();
        const float attention_scale = params.get_scale(); // 1/sqrt(head_dim)
        const bool causal = params.causal;
        const int position_offset = params.position_offset;

        constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;
        constexpr int q_row = 0; // Decode: single query

        // Zero the accumulator
        std::memset(int32_accum, 0, head_dim * sizeof(int32_t));
        *weight_sum = 0;
        *v_scale_product = 1.0f;

        // Determine effective KV length (causal masking)
        int kv_end = kv_len;
        if (causal)
        {
            kv_end = std::min(kv_len, position_offset + 1);
        }

        if (kv_end <= 0)
        {
            return;
        }

        // ════════════════════════════════════════════════════════════════════════
        // STEP 1: Q×K^T → INT32 scores (via Int8Requant for stable alpha)
        // ════════════════════════════════════════════════════════════════════════

        std::vector<int32_t> int32_scores(kv_end);
        float alpha = 0.0f;

        {
            microkernels::Int8RequantParams requant_params{};
            requant_params.Q = params.Q;
            requant_params.K = params.K;
            requant_params.q_row = q_row;
            requant_params.head = head;
            requant_params.kv_head = kv_head;
            requant_params.head_dim = head_dim;
            requant_params.kv_end = kv_end;
            requant_params.q_blocks_per_row = q_blocks_per_row;
            requant_params.kv_blocks_per_row = kv_blocks_per_row;
            requant_params.attention_scale = attention_scale;

            microkernels::compute_int8_requant_logits(requant_params, int32_scores.data(), &alpha);
        }

        // ════════════════════════════════════════════════════════════════════════
        // STEP 2: Exp2FixedSoftmax → INT16 weights [0, 32767]
        // ════════════════════════════════════════════════════════════════════════

        std::vector<int16_t> int16_weights(kv_end);

        microkernels::exp2_fixed_softmax_row(
            int32_scores.data(),
            int16_weights.data(),
            kv_end,
            alpha,
            weight_sum);

        // ════════════════════════════════════════════════════════════════════════
        // STEP 3: P×V → INT32 accumulators (PURE INTEGER, NO FP32!)
        // INT16 weights × INT16 V values → INT32 accumulators
        // ════════════════════════════════════════════════════════════════════════

        // Track V scale for later requantization
        float v_scale_sum = 0.0f;
        int v_scale_count = 0;

        const int v_start_col = kv_head * head_dim;

        for (int kv_pos = 0; kv_pos < kv_end; ++kv_pos)
        {
            int16_t w = int16_weights[kv_pos];
            if (w == 0)
                continue;

            // Accumulate weighted V values (all in integer domain)
            for (int d = 0; d < head_dim; ++d)
            {
                int v_col = v_start_col + d;
                int v_block_idx = v_col / BLOCK_SIZE;
                int v_elem_idx = v_col % BLOCK_SIZE;

                const Q16_1Block &v_block = params.V[kv_pos * kv_blocks_per_row + v_block_idx];
                int16_t v_val = v_block.qs[v_elem_idx];

                // INT16 × INT16 → INT32 (matches VNNI VPDPWSSD)
                int32_accum[d] += static_cast<int32_t>(w) * static_cast<int32_t>(v_val);

                // Track V scale (sample representative blocks)
                if (d == 0)
                {
                    v_scale_sum += v_block.d;
                    v_scale_count++;
                }
            }
        }

        // Compute average V scale
        if (v_scale_count > 0)
        {
            *v_scale_product = v_scale_sum / static_cast<float>(v_scale_count);
        }
    }

    // ============================================================================
    // FA2 Prefill: Pure Integer Attention Core Helper (One Query Row)
    // ============================================================================

    /**
     * @brief Process attention for a single query row in FA2 Prefill mode (PURE INTEGER)
     */
    static void fa2_prefill_attention_row_integer(
        const Q16FusedAttentionWoResidualParams &params,
        int q_row,
        int head,
        int32_t *int32_accum,
        int32_t *weight_sum,
        float *v_scale_product)
    {
        const int head_dim = params.head_dim;
        const int kv_len = params.kv_len;
        const int kv_head = params.get_kv_head(head);
        const int q_blocks_per_row = params.q_blocks_per_row();
        const int kv_blocks_per_row = params.kv_blocks_per_row();
        const float attention_scale = params.get_scale();
        const bool causal = params.causal;

        constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;

        // Zero the accumulator
        std::memset(int32_accum, 0, head_dim * sizeof(int32_t));
        *weight_sum = 0;
        *v_scale_product = 1.0f;

        // Determine effective KV length (causal masking)
        int kv_end = kv_len;
        if (causal)
        {
            kv_end = std::min(kv_len, q_row + 1);
        }

        if (kv_end <= 0)
        {
            return;
        }

        // ════════════════════════════════════════════════════════════════════════
        // STEP 1: Q×K^T → INT32 scores (via Int8Requant)
        // ════════════════════════════════════════════════════════════════════════

        std::vector<int32_t> int32_scores(kv_end);
        float alpha = 0.0f;

        {
            microkernels::Int8RequantParams requant_params{};
            requant_params.Q = params.Q;
            requant_params.K = params.K;
            requant_params.q_row = q_row;
            requant_params.head = head;
            requant_params.kv_head = kv_head;
            requant_params.head_dim = head_dim;
            requant_params.kv_end = kv_end;
            requant_params.q_blocks_per_row = q_blocks_per_row;
            requant_params.kv_blocks_per_row = kv_blocks_per_row;
            requant_params.attention_scale = attention_scale;

            microkernels::compute_int8_requant_logits(requant_params, int32_scores.data(), &alpha);
        }

        // ════════════════════════════════════════════════════════════════════════
        // STEP 2: Exp2FixedSoftmax → INT16 weights [0, 32767]
        // ════════════════════════════════════════════════════════════════════════

        std::vector<int16_t> int16_weights(kv_end);

        microkernels::exp2_fixed_softmax_row(
            int32_scores.data(),
            int16_weights.data(),
            kv_end,
            alpha,
            weight_sum);

        // ════════════════════════════════════════════════════════════════════════
        // STEP 3: P×V → INT32 accumulators (PURE INTEGER, NO FP32!)
        // ════════════════════════════════════════════════════════════════════════

        float v_scale_sum = 0.0f;
        int v_scale_count = 0;

        const int v_start_col = kv_head * head_dim;

        for (int kv_pos = 0; kv_pos < kv_end; ++kv_pos)
        {
            int16_t w = int16_weights[kv_pos];
            if (w == 0)
                continue;

            for (int d = 0; d < head_dim; ++d)
            {
                int v_col = v_start_col + d;
                int v_block_idx = v_col / BLOCK_SIZE;
                int v_elem_idx = v_col % BLOCK_SIZE;

                const Q16_1Block &v_block = params.V[kv_pos * kv_blocks_per_row + v_block_idx];
                int16_t v_val = v_block.qs[v_elem_idx];

                // INT16 × INT16 → INT32
                int32_accum[d] += static_cast<int32_t>(w) * static_cast<int32_t>(v_val);

                if (d == 0)
                {
                    v_scale_sum += v_block.d;
                    v_scale_count++;
                }
            }
        }

        if (v_scale_count > 0)
        {
            *v_scale_product = v_scale_sum / static_cast<float>(v_scale_count);
        }
    }

    // ============================================================================
    // Full Fused Kernel - Flash Decode Path (PURE INTEGER)
    // ============================================================================

    bool q16_fused_attention_wo_residual_decode(const Q16FusedAttentionWoResidualParams &params)
    {
        // Validate parameters
        if (!q16_validate_full_params(params))
        {
            return false;
        }

        if (!params.is_decode())
        {
            LOG_WARN("Q16FusedAttentionWoResidual: decode called with seq_len_q > 1, "
                     "redirecting to FA2 prefill");
            return q16_fused_attention_wo_residual_prefill(params);
        }

        LOG_DEBUG("Q16FusedAttentionWoResidual FLASH DECODE (PURE INTEGER): heads=" << params.num_heads
                                                                                    << "/" << params.num_kv_heads
                                                                                    << ", head_dim=" << params.head_dim
                                                                                    << ", d_model=" << params.d_model
                                                                                    << ", kv_len=" << params.kv_len);

        const int num_heads = params.num_heads;
        const int head_dim = params.head_dim;
        const int d_model = params.d_model;
        const int input_dim = num_heads * head_dim;
        const int output_blocks = (d_model + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;

        // ════════════════════════════════════════════════════════════════════════
        // STEPS 1-3: Attention core → INT32 context (all heads concatenated)
        // ════════════════════════════════════════════════════════════════════════

        std::vector<int32_t> int32_context(input_dim, 0);
        int32_t total_weight_sum = 0;
        float total_v_scale = 0.0f;
        int heads_processed = 0;

        for (int h = 0; h < num_heads; ++h)
        {
            int32_t *head_accum = int32_context.data() + h * head_dim;
            int32_t head_weight_sum = 0;
            float head_v_scale = 1.0f;

            flash_decode_attention_head_integer(
                params, h, head_accum, &head_weight_sum, &head_v_scale);

            if (head_weight_sum > 0)
            {
                total_weight_sum += head_weight_sum;
                total_v_scale += head_v_scale;
                heads_processed++;
            }
        }

        // Average the scale info across heads
        if (heads_processed > 0)
        {
            total_v_scale /= heads_processed;
            // Use per-head average weight sum for normalization
            total_weight_sum = total_weight_sum / heads_processed;
        }
        if (total_weight_sum == 0)
            total_weight_sum = 1;

        // ════════════════════════════════════════════════════════════════════════
        // SNAPSHOT: Capture attention context (INT32 → FP32 for debugging)
        // ════════════════════════════════════════════════════════════════════════

        if (params.context_snapshot)
        {
            // Convert INT32 context to FP32 for snapshot
            // The INT32 values need to be scaled by weight_sum and v_scale to get FP32 equivalents
            float *ctx_out = params.context_snapshot;
            const float inv_weight_sum = 1.0f / static_cast<float>(total_weight_sum);

            for (int d = 0; d < input_dim; ++d)
            {
                // INT32 accumulator / weight_sum * v_scale → approximate FP32 context value
                float val = static_cast<float>(int32_context[d]) * inv_weight_sum * total_v_scale;
                ctx_out[d] = val;
            }

            LOG_DEBUG("[Q16FusedAttention DECODE] Captured context snapshot: "
                      << input_dim << " elements, weight_sum=" << total_weight_sum
                      << ", v_scale=" << total_v_scale);
        }

        // ════════════════════════════════════════════════════════════════════════
        // STEP 4: Wo Projection (VPDPWSSD) → Q16_1 output (ALL INTEGER)
        // ════════════════════════════════════════════════════════════════════════

        // Allocate Q16_1 projection output
        std::vector<Q16_1Block> projection_q16(output_blocks);

        // Create IntegerContext for Wo projection
        IntegerContext context;
        context.int32_data = int32_context.data();
        context.weight_sum = total_weight_sum;
        context.v_scale_product = total_v_scale;
        context.count = input_dim;

        // Create output structure
        Q16_1Projection projection_out;
        projection_out.blocks = projection_q16.data();
        projection_out.count = d_model;

        // Set up Wo projection params
        WoProjectionVNNIParams wo_params;
        wo_params.Wo_packed = params.Wo_packed;
        wo_params.input_dim = input_dim;
        wo_params.d_model = d_model;
        wo_params.bias = nullptr;

        // Execute VPDPWSSD projection → INT32 → requant → Q16_1 output
        wo_projection_vpdpwssd_to_q16_1_gemv(wo_params, context, projection_out);

        // ════════════════════════════════════════════════════════════════════════
        // STEP 5: Native Q16_1 Residual Add (ALL INTEGER: Q16_1 + Q16_1 → Q16_1)
        // ════════════════════════════════════════════════════════════════════════

        // Copy residual_in to residual_out if different pointers
        if (params.residual_in != params.residual_out)
        {
            std::memcpy(params.residual_out, params.residual_in,
                        output_blocks * sizeof(Q16_1Block));
        }

        // Q16_1 + Q16_1 → Q16_1 (in-place on residual_out) - NO FP32!
        simd::q16_1_add_q16_1(
            params.residual_out,   // Residual (accumulates result in-place)
            projection_out.blocks, // Projection to add
            params.residual_out,   // Output (same as residual for in-place)
            d_model);

        return true;
    }

    // ============================================================================
    // Full Fused Kernel - FA2 Prefill Path (PURE INTEGER)
    // ============================================================================

    bool q16_fused_attention_wo_residual_prefill(const Q16FusedAttentionWoResidualParams &params)
    {
        // Validate parameters
        if (!q16_validate_full_params(params))
        {
            return false;
        }

        if (params.is_decode())
        {
            LOG_WARN("Q16FusedAttentionWoResidual: prefill called with seq_len_q == 1, "
                     "redirecting to Flash Decode");
            return q16_fused_attention_wo_residual_decode(params);
        }

        LOG_DEBUG("Q16FusedAttentionWoResidual FA2 PREFILL (PURE INTEGER): seq_len=" << params.seq_len_q
                                                                                     << ", heads=" << params.num_heads
                                                                                     << "/" << params.num_kv_heads
                                                                                     << ", head_dim=" << params.head_dim
                                                                                     << ", d_model=" << params.d_model
                                                                                     << ", kv_len=" << params.kv_len);

        const int seq_len_q = params.seq_len_q;
        const int num_heads = params.num_heads;
        const int head_dim = params.head_dim;
        const int d_model = params.d_model;
        const int input_dim = num_heads * head_dim;
        const int output_blocks = (d_model + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;
        const int residual_blocks_per_row = params.residual_blocks_per_row();

        // Process each query position
        for (int q = 0; q < seq_len_q; ++q)
        {
            // ════════════════════════════════════════════════════════════════════
            // STEPS 1-3: Attention core → INT32 context
            // ════════════════════════════════════════════════════════════════════

            std::vector<int32_t> int32_context(input_dim, 0);
            int32_t total_weight_sum = 0;
            float total_v_scale = 0.0f;
            int heads_processed = 0;

            for (int h = 0; h < num_heads; ++h)
            {
                int32_t *head_accum = int32_context.data() + h * head_dim;
                int32_t head_weight_sum = 0;
                float head_v_scale = 1.0f;

                fa2_prefill_attention_row_integer(
                    params, q, h, head_accum, &head_weight_sum, &head_v_scale);

                if (head_weight_sum > 0)
                {
                    total_weight_sum += head_weight_sum;
                    total_v_scale += head_v_scale;
                    heads_processed++;
                }
            }

            if (heads_processed > 0)
            {
                total_v_scale /= heads_processed;
                total_weight_sum = total_weight_sum / heads_processed;
            }
            if (total_weight_sum == 0)
                total_weight_sum = 1;

            // ════════════════════════════════════════════════════════════════════
            // SNAPSHOT: Capture attention context for this query position
            // ════════════════════════════════════════════════════════════════════

            if (params.context_snapshot)
            {
                // Convert INT32 context to FP32 for snapshot
                // Output layout: [seq_len_q × input_dim] where input_dim = num_heads × head_dim
                float *ctx_row = params.context_snapshot + q * input_dim;
                const float inv_weight_sum = 1.0f / static_cast<float>(total_weight_sum);

                for (int d = 0; d < input_dim; ++d)
                {
                    float val = static_cast<float>(int32_context[d]) * inv_weight_sum * total_v_scale;
                    ctx_row[d] = val;
                }

                if (q == 0)
                {
                    LOG_DEBUG("[Q16FusedAttention PREFILL] Capturing context snapshot for "
                              << seq_len_q << " positions, input_dim=" << input_dim);
                }
            }

            // ════════════════════════════════════════════════════════════════════
            // STEP 4: Wo Projection → Q16_1 output (ALL INTEGER)
            // ════════════════════════════════════════════════════════════════════

            std::vector<Q16_1Block> projection_q16(output_blocks);

            IntegerContext context;
            context.int32_data = int32_context.data();
            context.weight_sum = total_weight_sum;
            context.v_scale_product = total_v_scale;
            context.count = input_dim;

            Q16_1Projection projection_out;
            projection_out.blocks = projection_q16.data();
            projection_out.count = d_model;

            WoProjectionVNNIParams wo_params;
            wo_params.Wo_packed = params.Wo_packed;
            wo_params.input_dim = input_dim;
            wo_params.d_model = d_model;
            wo_params.bias = nullptr;

            wo_projection_vpdpwssd_to_q16_1_gemv(wo_params, context, projection_out);

            // ════════════════════════════════════════════════════════════════════
            // STEP 5: Native Q16_1 Residual Add (ALL INTEGER)
            // ════════════════════════════════════════════════════════════════════

            Q16_1Block *residual_out_row = params.residual_out + q * residual_blocks_per_row;
            const Q16_1Block *residual_in_row = params.residual_in + q * residual_blocks_per_row;

            // Copy input to output if different
            if (residual_in_row != residual_out_row)
            {
                std::memcpy(residual_out_row, residual_in_row,
                            output_blocks * sizeof(Q16_1Block));
            }

            // Q16_1 + Q16_1 → Q16_1 - NO FP32!
            simd::q16_1_add_q16_1(
                residual_out_row,
                projection_out.blocks,
                residual_out_row,
                d_model);
        }

        return true;
    }

} // namespace llaminar2::kernels::q16_1
