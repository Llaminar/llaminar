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

// Forward declaration for VNNI packed weights (in gemm_v4 namespace)
namespace llaminar2::gemm_v4
{
    struct QuantisedPackedWeights;
}

namespace llaminar2::kernels::q16_1
{
    // Import VNNI packed weights type for use in this namespace
    using llaminar2::gemm_v4::QuantisedPackedWeights;

    // ============================================================================
    // Block Size Configuration (use canonical types from BlockStructures.h)
    // ============================================================================

    // Re-export Q16BlockSize and related utilities from BlockStructures.h
    // for convenience within this namespace
    using llaminar2::optimal_q16_block_size;
    using llaminar2::Q16_1Block;
    using llaminar2::Q16_1Block_128;
    using llaminar2::Q16_1Block_64;
    using llaminar2::Q16BlockMutablePtr;
    using llaminar2::Q16BlockPtr;
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
     * 4. TYPE-SAFE block pointers via Q16BlockPtr (no more const void*)
     *
     * NOTE: Q16BlockPtr and Q16BlockMutablePtr are imported from BlockStructures.h
     *       via 'using' statements above.
     */
    struct Q16IntegerAttentionParams
    {
        // === Input tensors (Q16_1 quantized, pre-normalized) ===
        // TYPE-SAFE: Uses Q16BlockPtr instead of const void*
        Q16BlockPtr Q; ///< Query: [seq_len_q, num_heads, head_dim] as Q16_1 blocks
        Q16BlockPtr K; ///< Key: [kv_len, num_kv_heads, head_dim] as Q16_1 blocks
        Q16BlockPtr V; ///< Value: [kv_len, num_kv_heads, head_dim] as Q16_1 blocks

        // === Per-head scale factors (from normalization) ===
        const float *q_head_scales = nullptr;  ///< [num_heads] scale applied during Q normalization
        const float *kv_head_scales = nullptr; ///< [num_kv_heads] scale applied during K/V normalization

        // === Per-position K scale factors (HybridQ16 K precision fix) ===
        // When K uses dynamic per-head scale from RoPE (instead of uniform kv_head_scale),
        // these per-position scales must be provided for correct Q×K^T computation.
        // Shape: [kv_len * num_kv_heads] (position-major: k_position_scales[pos * num_kv_heads + kv_h])
        // When nullptr, K scales come from kv_head_scales (uniform per-head).
        //
        // NOTE: k_position_scales is DEPRECATED. Use read_k_scales_from_blocks=true instead.
        // The k_position_scales buffer is shared across layers and gets overwritten.
        const float *k_position_scales = nullptr;

        // === Read K scales directly from K block headers ===
        // When true, the attention kernel reads K scales from K cache block .d fields
        // instead of from k_position_scales buffer. This avoids cross-layer contamination
        // since KV cache is per-layer.
        // This is the CORRECT mode for HybridQ16 K precision fix.
        bool read_k_scales_from_blocks = false;

        // === Output projection weights ===
        const QuantisedPackedWeights *Wo_packed = nullptr; ///< Packed Wo weights for VPDPWSSD

        // === Output buffer (Wo projection output) ===
        Q16BlockMutablePtr output; ///< [seq_len_q, d_model] as Q16_1 blocks (Wo projection output)

        // === Residual connection (uses 32-element blocks for compatibility) ===
        const Q16_1Block *residual_in = nullptr; ///< Input residual [seq_len_q, d_model] as Q16_1Block
        Q16_1Block *residual_out = nullptr;      ///< Output [seq_len_q, d_model] as Q16_1Block

        // === Dimensions ===
        int seq_len_q = 0;    ///< Query sequence length (1 for decode, >1 for prefill)
        int kv_len = 0;       ///< KV cache length (number of valid positions)
        int num_heads = 0;    ///< Number of query/attention heads
        int num_kv_heads = 0; ///< Number of KV heads (for GQA)
        int head_dim = 0;     ///< Dimension per head
        int d_model = 0;      ///< Model dimension (num_heads × head_dim)

        // === KV Cache Layout ===
        // For HEAD_MAJOR sparse cache: stride between heads = max_seq_len (sparse allocation)
        // For dense/transposed data: stride between heads = kv_len (packed)
        // If 0, defaults to kv_len (dense layout assumption)
        int kv_head_stride = 0; ///< Stride in positions between consecutive KV heads

        // === Block configuration ===
        // Can be explicitly set, or will be derived from Q.block_size
        Q16BlockSize block_size = Q16BlockSize::BLOCK_64;

        // === Optional: Snapshot buffers for debugging ===
        float *snapshot_scores = nullptr;       ///< [seq_len_q, num_heads, kv_len] pre-softmax
        float *snapshot_weights = nullptr;      ///< [seq_len_q, num_heads, kv_len] post-softmax
        float *snapshot_context = nullptr;      ///< [seq_len_q, num_heads, head_dim] attention output
        float *snapshot_projected = nullptr;    ///< [seq_len_q, d_model] after Wo projection
        float *snapshot_wo_output = nullptr;    ///< [seq_len_q, d_model] Wo output before residual
        float *snapshot_residual_out = nullptr; ///< [seq_len_q, d_model] after residual add

        // === Helper methods ===

        /**
         * @brief Get effective stride between KV heads (in positions).
         *
         * For HEAD_MAJOR sparse cache: returns kv_head_stride (= max_seq_len)
         * For dense/transposed data: returns kv_len (packed)
         *
         * This handles the difference between:
         * - Sparse cache: head_1 at offset max_seq_len from head_0
         * - Dense data: head_1 at offset kv_len from head_0
         */
        int effective_kv_head_stride() const
        {
            return (kv_head_stride > 0) ? kv_head_stride : kv_len;
        }

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
         * @brief Get K scale for specific position and KV head.
         *
         * Uses k_position_scales if available (dynamic-scale from RoPE),
         * otherwise falls back to kv_head_scales (from block header).
         *
         * @param kv_pos KV cache position index
         * @param kv_head KV head index
         * @return K scale factor for this position/head
         */
        float get_k_scale(int kv_pos, int kv_head) const
        {
            if (k_position_scales)
            {
                return k_position_scales[kv_pos * num_kv_heads + kv_head];
            }
            return kv_head_scales ? kv_head_scales[kv_head] : 1.0f;
        }

        /**
         * @brief Check if per-position K scales are available.
         *
         * When true, attention kernel should use get_k_scale() for each K position
         * instead of uniform qk_scale.
         *
         * This returns true if EITHER:
         * 1. k_position_scales buffer is provided (legacy, cross-layer contamination risk), OR
         * 2. read_k_scales_from_blocks is true (correct, reads from K cache blocks)
         */
        bool has_per_position_k_scales() const
        {
            return k_position_scales != nullptr || read_k_scales_from_blocks;
        }

        /**
         * @brief Get Q scale for base alpha computation (Option C: Pass Scale to Softmax).
         *
         * When using per-position K scales, we split qk_scale into:
         *   - base_alpha = q_scale / sqrt(head_dim) * log2(e) (computed once)
         *   - per_position_alpha = base_alpha * k_scale[pos] (per-position)
         *
         * @param q_head Query head index
         * @return Q scale factor (without K scale or sqrt(head_dim) normalization)
         */
        float get_q_scale(int q_head) const
        {
            return q_head_scales ? q_head_scales[q_head] : 1.0f;
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

        /// Get effective block size - uses member if set non-default, otherwise derives from Q
        Q16BlockSize effective_block_size() const
        {
            // If Q has valid data, use its block_size; otherwise use the member
            return Q.data ? Q.block_size : block_size;
        }

        int q_blocks_per_row() const
        {
            return blocks_per_row(head_dim, effective_block_size());
        }

        int kv_blocks_per_row() const
        {
            return blocks_per_row(head_dim, effective_block_size());
        }

        int residual_blocks_per_row() const
        {
            // Residual always uses 32-element blocks
            return blocks_per_row(d_model, Q16BlockSize::BLOCK_32);
        }

        bool is_head_aligned() const
        {
            return q16_1::is_head_aligned(head_dim);
        }

        /// Validate that Q, K, V all have the same block size
        bool validate_block_sizes() const
        {
            if (Q.empty() || K.empty() || V.empty())
                return false;
            return Q.block_size == K.block_size && K.block_size == V.block_size;
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
