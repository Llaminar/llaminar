/**
 * @file Q16IntegerAttentionRef.h
 * @brief TRUE integer-only Q16 attention reference implementation (v2)
 *
 * DESIGN RATIONALE
 * ================
 * This is a complete rewrite of Q16 attention after discovering that v1's
 * "PURE INTEGER" implementation actually used FP32 everywhere (std::exp for
 * softmax, FP32 accumulators, FP64 scale tracking). Only residual add was integer.
 *
 * Key insight: The 32-element block size inherited from Q8_1 was a mistake.
 * - INT16 has 256× more precision than INT8, can handle larger blocks
 * - Common head_dim values (64, 128, 192) don't align with 32-block
 * - Per-block scales forced FP32 fallbacks to combine scales across blocks
 *
 * Solution: Model-aware block sizes that match head_dim:
 * - 64-block: Small models (Qwen2.5-0.5B, GPT-2)         → 1 block/head
 * - 128-block: Standard models (Qwen3, Llama, Mistral)  → 1 block/head
 * - 192-block: MLA Q/K (DeepSeek V3, Kimi K2)           → 1 block/head
 *
 * With 1 block per head, we get a SINGLE scale factor per head - no normalization
 * or FP32 fallbacks needed! The entire attention computation stays in INT16/INT32.
 *
 * INTEGER PIPELINE
 * ================
 * Flash Decode (seq_len_q=1):
 *   Q×K^T  → INT32 scores via VPDPWSSD (single scale: s_Q × s_K)
 *   Softmax → Exp2FixedSoftmax LUT → INT16 weights [0, 32767]
 *   P×V    → INT32 accumulators (single scale: s_P × s_V)
 *   Wo     → VPDPWSSD projection → Q16 output
 *   +res   → Integer residual add with scale alignment
 *
 * FA2 Prefill (seq_len_q>1):
 *   Same pipeline, but tiled with online softmax state.
 *   Block size Br=4 (queries), Bc=32 (KV positions).
 *
 * SCALE HANDLING
 * ==============
 * Input normalization (done in FusedGemmStage / KV cache write):
 *   Q_norm = Q / head_scale[h], stored with scale = head_scale[h]
 *   K_norm = K / head_scale[h], stored with scale = head_scale[h]
 *   V_norm = V / head_scale[h], stored with scale = head_scale[h]
 *
 * This ensures all heads within a tensor have the SAME effective scale,
 * enabling pure integer arithmetic without per-element scale lookups.
 *
 * FALLBACK PATH
 * =============
 * For models where head_dim doesn't match any block size (rare), we fall
 * back to per-head scale normalization which adds minimal FP32 overhead
 * at quantization boundaries, not during the attention computation.
 *
 * @see docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include "tensors/BlockStructures.h"

// Forward declaration - Wo projection will be added in a later phase
namespace llaminar2::kernels::q16_1
{
    struct QuantisedPackedWeights; // Placeholder for Wo projection weights
}

namespace llaminar2::kernels::q16_1
{

    // ============================================================================
    // Block Size Configuration (use canonical types from BlockStructures.h)
    // ============================================================================

    // Re-export Q16BlockSize and related utilities from BlockStructures.h
    // for convenience within this namespace
    using llaminar2::optimal_q16_block_size;
    using llaminar2::Q16_1Block;
    using llaminar2::Q16_1Block_128;
    using llaminar2::Q16_1Block_64;
    using llaminar2::Q16BlockSize;
    using llaminar2::Q16BlockType;
    using llaminar2::Q16BlockType_t;

    /**
     * @brief Check if head_dim perfectly aligns with a supported block size.
     *
     * When aligned, we get exactly 1 block per head = single scale = no FP32 fallback.
     * Note: MLA architectures should use separate NOPE (128) + ROPE (64) tensors.
     */
    constexpr bool is_head_aligned(int head_dim)
    {
        return head_dim == 64 || head_dim == 128;
    }
    /**
     * @brief Calculate blocks per row for given dimension and block size.
     */
    constexpr int blocks_per_row(int dim, Q16BlockSize block_size)
    {
        int bs = static_cast<int>(block_size);
        return (dim + bs - 1) / bs;
    }

    // ============================================================================
    // Kernel Parameters
    // ============================================================================

    /**
     * @brief Parameters for TRUE integer Q16 attention kernel.
     *
     * Key differences from deprecated v1:
     * 1. block_size field for model-aware block selection
     * 2. head_scales arrays for pre-normalized tensors
     * 3. No FP32 intermediate buffers (all computation stays INT16/INT32)
     */
    struct Q16IntegerAttentionParams
    {
        // === Input tensors (Q16_1 quantized, pre-normalized) ===
        const void *Q = nullptr; ///< Query: [seq_len_q, num_heads, head_dim] as Q16_1 blocks
        const void *K = nullptr; ///< Key: [kv_len, num_kv_heads, head_dim] as Q16_1 blocks
        const void *V = nullptr; ///< Value: [kv_len, num_kv_heads, head_dim] as Q16_1 blocks

        // === Per-head scale factors (from normalization) ===
        const float *q_head_scales = nullptr;  ///< [num_heads] scale applied during Q normalization
        const float *kv_head_scales = nullptr; ///< [num_kv_heads] scale applied during K/V normalization

        // === Output projection weights ===
        const QuantisedPackedWeights *Wo_packed = nullptr; ///< Packed Wo weights for VPDPWSSD

        // === Residual connection ===
        const void *residual_in = nullptr; ///< Input residual [seq_len_q, d_model] as Q16_1
        void *residual_out = nullptr;      ///< Output [seq_len_q, d_model] as Q16_1

        // === Dimensions ===
        int seq_len_q = 0;    ///< Query sequence length (1 for decode, >1 for prefill)
        int kv_len = 0;       ///< KV cache length
        int num_heads = 0;    ///< Number of query/attention heads
        int num_kv_heads = 0; ///< Number of KV heads (for GQA)
        int head_dim = 0;     ///< Dimension per head
        int d_model = 0;      ///< Model dimension (num_heads × head_dim)

        // === Block configuration ===
        Q16BlockSize block_size = Q16BlockSize::BLOCK_128; ///< Block size for this model

        // === Optional: Snapshot buffers for debugging ===
        float *snapshot_scores = nullptr;    ///< [seq_len_q, num_heads, kv_len] pre-softmax
        float *snapshot_weights = nullptr;   ///< [seq_len_q, num_heads, kv_len] post-softmax
        float *snapshot_context = nullptr;   ///< [seq_len_q, num_heads, head_dim] attention output
        float *snapshot_projected = nullptr; ///< [seq_len_q, d_model] after Wo projection

        // === Helper methods ===

        /**
         * @brief Get combined QK scale for attention scores.
         *
         * For normalized tensors with aligned block size:
         *   score_scale = q_head_scales[h] × kv_head_scales[kv_h] / sqrt(head_dim)
         *
         * This is applied ONCE to the INT32 accumulator, not per-element.
         */
        float get_qk_scale(int q_head, int kv_head) const
        {
            float s_q = q_head_scales ? q_head_scales[q_head] : 1.0f;
            float s_k = kv_head_scales ? kv_head_scales[kv_head] : 1.0f;
            return s_q * s_k / std::sqrt(static_cast<float>(head_dim));
        }

        /**
         * @brief Get PV scale for context accumulation.
         *
         * The context accumulator holds: Σ w_scaled * V_int16
         * where w_scaled = w_raw >> (lut_value_bits - 15) ≈ w_raw / 2^15
         *
         * After normalizing by l (scaled):
         *   result_int = context / (l >> 15)
         *              = Σ (w_raw/l) * V_int16
         *
         * To convert to FP32:
         *   result_fp32 = result_int * s_v
         *
         * So pv_scale is just the V tensor's scale.
         */
        float get_pv_scale(int kv_head) const
        {
            float s_v = kv_head_scales ? kv_head_scales[kv_head] : 1.0f;
            return s_v;
        }

        bool is_decode() const { return seq_len_q == 1; }

        int get_kv_head(int q_head) const
        {
            return (num_kv_heads == num_heads) ? q_head : (q_head / (num_heads / num_kv_heads));
        }

        int q_blocks_per_row() const
        {
            return blocks_per_row(head_dim, block_size);
        }

        int kv_blocks_per_row() const
        {
            return blocks_per_row(head_dim, block_size);
        }

        int residual_blocks_per_row() const
        {
            return blocks_per_row(d_model, block_size);
        }

        bool is_head_aligned() const
        {
            return q16_1::is_head_aligned(head_dim);
        }
    };

    // ============================================================================
    // TRUE Integer Attention API
    // ============================================================================

    /**
     * @brief Execute TRUE integer attention (Flash Decode path).
     *
     * Integer pipeline:
     *   1. Q×K^T via VPDPWSSD → INT32 scores
     *   2. Exp2FixedSoftmax → INT16 weights
     *   3. P×V via VPDPWSSD → INT32 context
     *   4. Wo projection → Q16_1 output
     *   5. Integer residual add
     *
     * NO FP32 operations in the hot path (only at quantization boundaries).
     *
     * @param params Kernel parameters (seq_len_q must be 1)
     * @return true on success
     */
    bool q16_integer_attention_decode(const Q16IntegerAttentionParams &params);

    /**
     * @brief Execute TRUE integer attention (FA2 Prefill path).
     *
     * Same integer pipeline as decode, but tiled:
     *   - Block size Br=4 (queries), Bc=32 (KV positions)
     *   - Online softmax state tracking in INT32
     *
     * @param params Kernel parameters (seq_len_q > 1)
     * @return true on success
     */
    bool q16_integer_attention_prefill(const Q16IntegerAttentionParams &params);

    /**
     * @brief Auto-dispatch based on seq_len_q.
     */
    inline bool q16_integer_attention_reference(const Q16IntegerAttentionParams &params)
    {
        if (params.is_decode())
        {
            return q16_integer_attention_decode(params);
        }
        else
        {
            return q16_integer_attention_prefill(params);
        }
    }

    // ============================================================================
    // Validation
    // ============================================================================

    /**
     * @brief Validate kernel parameters.
     */
    bool q16_validate_integer_params(const Q16IntegerAttentionParams &params);

    // ============================================================================
    // Microkernel Dispatch (for testing)
    // ============================================================================

    /**
     * @brief Compute Q×K^T dot products in pure INT32.
     *
     * Uses VPDPWSSD for INT16×INT16→INT32 accumulation.
     * Output is INT32 scores (not scaled to FP32).
     *
     * @param Q Query block(s)
     * @param K Key block(s)
     * @param scores Output INT32 scores [k_count]
     * @param k_count Number of KV positions
     * @param head_dim Head dimension
     * @param block_size Block size being used
     */
    void q16_integer_qk_dotproduct(
        const void *Q,
        const void *K,
        int32_t *scores,
        int k_count,
        int head_dim,
        Q16BlockSize block_size);

    /**
     * @brief Compute P×V accumulation in pure INT32.
     *
     * @param weights INT16 softmax weights [kv_len]
     * @param V Value blocks
     * @param context Output INT32 context [head_dim]
     * @param kv_len Number of KV positions
     * @param head_dim Head dimension
     * @param block_size Block size being used
     */
    void q16_integer_pv_accumulate(
        const int16_t *weights,
        const void *V,
        int32_t *context,
        int kv_len,
        int head_dim,
        Q16BlockSize block_size);

} // namespace llaminar2::kernels::q16_1
