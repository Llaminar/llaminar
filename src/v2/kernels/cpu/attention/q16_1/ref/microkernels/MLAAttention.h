/**
 * @file MLAAttention.h
 * @brief Multi-head Latent Attention (MLA) microkernels for Q16 attention
 *
 * OVERVIEW
 * ========
 * MLA (Multi-head Latent Attention) is used by DeepSeek V3 and Kimi K2 models.
 * It splits Q/K into two components with fundamentally different distributions:
 *
 *   - NOPE (No Position Embedding): 128 dimensions, context-dependent
 *   - ROPE (Rotary Position Embedding): 64 dimensions, position-encoded
 *
 * The attention score is computed as:
 *   score = Q_nope · K_nope^T  +  Q_rope · K_rope^T
 *
 * ACCUMULATOR OVERFLOW ANALYSIS
 * =============================
 * AVX-512 VNNI (VPDPWSSD) performs INT16×INT16→INT32 accumulation. With
 * 192 dimensions (128 NOPE + 64 ROPE), we must ensure no overflow:
 *
 *   INT32_MAX = 2,147,483,647 ≈ 2.1B
 *   Max safe per-element: sqrt(INT32_MAX / 192) ≈ 3344
 *
 * SAFE VALUE RANGE: ±3344 per int16 element guarantees no overflow.
 *
 * CRITICAL CONTRACT: This safety guarantee requires FIXED-SCALE quantization:
 *   - Q16_1 must use: d = kv_cache_scale / 32767.0f (FIXED)
 *   - NOT adaptive: d = max_abs / 32767.0f (DANGEROUS - always saturates INT16!)
 *
 * With fixed scale (kv_cache_scale=8.0), typical activations map to safe INT16:
 *   FP32 ±8.0 → INT16 ±32767 (full range, risky for >96 dims)
 *   FP32 ±4.0 → INT16 ±16383 (safe for 192 dims)
 *   FP32 ±1.0 → INT16 ±4096  (very safe, typical case)
 *
 * See PROJECT_Q16_INTEGER_ATTENTION_V2.md "VNNI OVERFLOW PREVENTION CONTRACT"
 * for the three-party contract between kv_cache_scale, fixed-scale quantization,
 * and per-head normalization.
 *
 * SUB-BLOCK ACCUMULATION (CURRENT IMPLEMENTATION)
 * ===============================================
 * As a defense-in-depth measure, we process NOPE and ROPE separately:
 *   1. Accumulate NOPE (128 dims) into int32 partial
 *   2. Accumulate ROPE (64 dims) into int32 partial
 *   3. Sum partials (2 int32 values → 1 int32)
 *
 * This bounds worst-case per-component to 128 × val² instead of 192 × val².
 *
 * DESIGN RATIONALE
 * ================
 * Why separate tensors instead of a single 192-block?
 *
 * 1. **Different value distributions**: NOPE and ROPE have fundamentally
 *    different activation patterns. A single scale compromises both.
 *
 * 2. **Per-component quantization**: Each component gets its own optimal
 *    scale factor, preserving precision where it matters.
 *
 * 3. **Memory layout efficiency**: Separate tensors allow contiguous access
 *    patterns for each component during dot product.
 *
 * BLOCK SIZE MAPPING
 * ==================
 * - Q/K NOPE: Q16_1Block_128 (1 block per head, 128 elements)
 * - Q/K ROPE: Q16_1Block_64 (1 block per head, 64 elements)
 * - V values: Q16_1Block_128 (standard, no NOPE/ROPE split)
 *
 * @see docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md "MLA Architecture" section
 */
#pragma once

#include <cstdint>
#include "tensors/BlockStructures.h"

namespace llaminar2::kernels::q16_1::microkernels
{

    // Import block types from BlockStructures.h
    using llaminar2::Q16_1Block_128;
    using llaminar2::Q16_1Block_64;

    // ============================================================================
    // MLA Configuration
    // ============================================================================

    /**
     * @brief Configuration constants for MLA architectures.
     *
     * These match DeepSeek V3 and Kimi K2 specifications.
     */
    struct MLAConfig
    {
        /// NOPE (No Position Embedding) dimension - context-dependent features
        static constexpr int NOPE_DIM = 128;

        /// ROPE (Rotary Position Embedding) dimension - positional features
        static constexpr int ROPE_DIM = 64;

        /// Total Q/K head dimension (NOPE + ROPE)
        static constexpr int TOTAL_QK_DIM = NOPE_DIM + ROPE_DIM; // 192

        /// V dimension (same as NOPE, no split)
        static constexpr int V_DIM = 128;

        /// Max safe int16 value for combined 192-dim accumulation without INT32 overflow
        /// floor(sqrt(INT32_MAX / 192)) = floor(sqrt(11,184,810)) = 3344
        static constexpr int16_t MAX_SAFE_VALUE = 3344;

        /// Max safe int16 value for NOPE-only (128-dim) accumulation
        /// floor(sqrt(INT32_MAX / 128)) = floor(sqrt(16,777,215)) = 4095
        static constexpr int16_t MAX_SAFE_VALUE_NOPE = 4095;

        /// Max safe int16 value for ROPE-only (64-dim) accumulation
        /// floor(sqrt(INT32_MAX / 64)) = floor(sqrt(33,554,431)) = 5792
        static constexpr int16_t MAX_SAFE_VALUE_ROPE = 5792;

        /// Blocks per head for NOPE component (128 / 128 = 1)
        static constexpr int NOPE_BLOCKS_PER_HEAD = 1;

        /// Blocks per head for ROPE component (64 / 64 = 1)
        static constexpr int ROPE_BLOCKS_PER_HEAD = 1;
    };

    // ============================================================================
    // MLA Attention Parameters
    // ============================================================================

    /**
     * @brief Parameters for MLA attention computation.
     *
     * Holds pointers to the split NOPE/ROPE tensors for Q and K.
     * V is not split - it uses standard Q16_1Block_128.
     */
    struct MLAAttentionParams
    {
        // ========== Query tensors (single position for decode, tile for prefill) ==========

        /// Query NOPE component [num_queries, n_heads, NOPE_BLOCKS_PER_HEAD]
        const Q16_1Block_128 *Q_nope = nullptr;

        /// Query ROPE component [num_queries, n_heads, ROPE_BLOCKS_PER_HEAD]
        const Q16_1Block_64 *Q_rope = nullptr;

        // ========== Key cache tensors ==========

        /// Key NOPE component [kv_len, n_kv_heads, NOPE_BLOCKS_PER_HEAD]
        const Q16_1Block_128 *K_nope = nullptr;

        /// Key ROPE component [kv_len, n_kv_heads, ROPE_BLOCKS_PER_HEAD]
        const Q16_1Block_64 *K_rope = nullptr;

        // ========== Value cache (not split) ==========

        /// Value cache [kv_len, n_kv_heads, blocks_per_head]
        const Q16_1Block_128 *V = nullptr;

        // ========== Dimensions ==========

        /// Number of query positions (1 for decode, tile size for prefill)
        int num_queries = 1;

        /// Number of KV cache positions
        int kv_len = 0;

        /// Number of attention heads (Q heads)
        int n_heads = 1;

        /// Number of KV heads (for GQA: n_kv_heads <= n_heads)
        int n_kv_heads = 1;

        // ========== Strides (in blocks) ==========

        /// Stride between Q positions (blocks) - default: n_heads * blocks_per_head
        int q_nope_stride = 0;
        int q_rope_stride = 0;

        /// Stride between K positions (blocks) - default: n_kv_heads * blocks_per_head
        int k_nope_stride = 0;
        int k_rope_stride = 0;

        /// Stride between V positions (blocks)
        int v_stride = 0;

        /**
         * @brief Initialize strides with default values based on dimensions.
         *
         * Call this after setting n_heads and n_kv_heads if using default layout.
         */
        void initDefaultStrides()
        {
            // Q layout: [num_queries, n_heads, 1 block per head]
            q_nope_stride = n_heads * MLAConfig::NOPE_BLOCKS_PER_HEAD;
            q_rope_stride = n_heads * MLAConfig::ROPE_BLOCKS_PER_HEAD;

            // K layout: [kv_len, n_kv_heads, 1 block per head]
            k_nope_stride = n_kv_heads * MLAConfig::NOPE_BLOCKS_PER_HEAD;
            k_rope_stride = n_kv_heads * MLAConfig::ROPE_BLOCKS_PER_HEAD;

            // V layout: [kv_len, n_kv_heads, 1 block per head]
            v_stride = n_kv_heads * MLAConfig::NOPE_BLOCKS_PER_HEAD; // V uses 128-dim blocks
        }
    };

    // ============================================================================
    // Flash Decode: Single Query MLA GEMV
    // ============================================================================

    /**
     * @brief Compute single MLA Q×K dot product for one query-key pair.
     *
     * score = Q_nope · K_nope + Q_rope · K_rope
     *
     * Pure INT32 accumulation - no FP32 anywhere in computation.
     *
     * @param Q_nope Query NOPE block (128 elements)
     * @param Q_rope Query ROPE block (64 elements)
     * @param K_nope Key NOPE block (128 elements)
     * @param K_rope Key ROPE block (64 elements)
     * @return INT32 combined dot product score
     */
    int32_t q16_dot_single_mla(
        const Q16_1Block_128 *Q_nope,
        const Q16_1Block_64 *Q_rope,
        const Q16_1Block_128 *K_nope,
        const Q16_1Block_64 *K_rope);

    /**
     * @brief Compute MLA Q×K^T for all KV positions (GEMV pattern).
     *
     * Flash Decode path for MLA: single query against all cached keys.
     * Computes: scores[k] = Q_nope · K_nope[k] + Q_rope · K_rope[k]
     *
     * @param Q_nope Query NOPE blocks [NOPE_BLOCKS_PER_HEAD]
     * @param Q_rope Query ROPE blocks [ROPE_BLOCKS_PER_HEAD]
     * @param K_nope Key NOPE cache [kv_len, NOPE_BLOCKS_PER_HEAD]
     * @param K_rope Key ROPE cache [kv_len, ROPE_BLOCKS_PER_HEAD]
     * @param scores Output INT32 scores [kv_len]
     * @param kv_len Number of KV cache positions
     * @param k_nope_stride Stride between K_nope positions (blocks)
     * @param k_rope_stride Stride between K_rope positions (blocks)
     */
    void q16_qk_gemv_mla(
        const Q16_1Block_128 *Q_nope,
        const Q16_1Block_64 *Q_rope,
        const Q16_1Block_128 *K_nope,
        const Q16_1Block_64 *K_rope,
        int32_t *scores,
        int kv_len,
        int k_nope_stride = MLAConfig::NOPE_BLOCKS_PER_HEAD,
        int k_rope_stride = MLAConfig::ROPE_BLOCKS_PER_HEAD);

    /**
     * @brief Overload using MLAAttentionParams for a single head.
     *
     * @param params MLA attention parameters
     * @param scores Output INT32 scores [kv_len]
     * @param head_idx Which attention head to process
     */
    void q16_qk_gemv_mla(
        const MLAAttentionParams &params,
        int32_t *scores,
        int head_idx);

    // ============================================================================
    // FA2 Prefill: Multi-Query MLA GEMM Tile
    // ============================================================================

    /**
     * @brief Compute MLA Q×K^T for a tile of queries and keys (GEMM pattern).
     *
     * FA2 Prefill path for MLA: multiple queries against a tile of keys.
     * Produces a score tile [Br, Bc] for online softmax.
     *
     * scores[q, k] = Q_nope[q] · K_nope[k] + Q_rope[q] · K_rope[k]
     *
     * @param Q_nope Query NOPE tile [Br, NOPE_BLOCKS_PER_HEAD]
     * @param Q_rope Query ROPE tile [Br, ROPE_BLOCKS_PER_HEAD]
     * @param K_nope Key NOPE tile [Bc, NOPE_BLOCKS_PER_HEAD]
     * @param K_rope Key ROPE tile [Bc, ROPE_BLOCKS_PER_HEAD]
     * @param scores Output score tile [Br, Bc] (row-major)
     * @param Br Number of queries in tile
     * @param Bc Number of keys in tile
     * @param q_nope_stride Stride between Q_nope rows (blocks)
     * @param q_rope_stride Stride between Q_rope rows (blocks)
     * @param k_nope_stride Stride between K_nope rows (blocks)
     * @param k_rope_stride Stride between K_rope rows (blocks)
     */
    void q16_qk_gemm_tile_mla(
        const Q16_1Block_128 *Q_nope,
        const Q16_1Block_64 *Q_rope,
        const Q16_1Block_128 *K_nope,
        const Q16_1Block_64 *K_rope,
        int32_t *scores,
        int Br,
        int Bc,
        int q_nope_stride = MLAConfig::NOPE_BLOCKS_PER_HEAD,
        int q_rope_stride = MLAConfig::ROPE_BLOCKS_PER_HEAD,
        int k_nope_stride = MLAConfig::NOPE_BLOCKS_PER_HEAD,
        int k_rope_stride = MLAConfig::ROPE_BLOCKS_PER_HEAD);

    // ============================================================================
    // Scale Factor Computation for MLA
    // ============================================================================

    /**
     * @brief Compute combined scale factor for MLA attention scores.
     *
     * MLA has two separate scale factors (NOPE and ROPE) that must be combined
     * for the final attention computation:
     *
     *   score_fp = score_int * (q_nope.d * k_nope.d + q_rope.d * k_rope.d)
     *
     * NOTE: This is an approximation. The exact combination depends on the
     * relative magnitudes of NOPE vs ROPE contributions per position.
     *
     * For high precision, consider tracking NOPE and ROPE contributions
     * separately and combining after softmax.
     *
     * @param q_nope_scale Query NOPE scale factor
     * @param k_nope_scale Key NOPE scale factor
     * @param q_rope_scale Query ROPE scale factor
     * @param k_rope_scale Key ROPE scale factor
     * @return Combined scale factor for attention scores
     */
    inline float mla_combine_scales(
        float q_nope_scale,
        float k_nope_scale,
        float q_rope_scale,
        float k_rope_scale)
    {
        // The integer score is: int_score = nope_dot + rope_dot
        // Where: nope_dot = Σ q_nope_int[i] * k_nope_int[i]
        //        rope_dot = Σ q_rope_int[i] * k_rope_int[i]
        //
        // The FP score would be:
        //   fp_score = q_nope_scale * k_nope_scale * nope_dot
        //            + q_rope_scale * k_rope_scale * rope_dot
        //
        // We can't factor this cleanly, but if NOPE and ROPE scales are similar,
        // a weighted average works well. For simplicity, we use the max:
        float nope_combined = q_nope_scale * k_nope_scale;
        float rope_combined = q_rope_scale * k_rope_scale;

        // Use the larger scale to avoid underrepresenting any component
        return (nope_combined > rope_combined) ? nope_combined : rope_combined;
    }

    /**
     * @brief Compute per-position scale factors for precise MLA attention.
     *
     * When NOPE and ROPE have significantly different scales, this function
     * computes the exact combined scale for each position.
     *
     * @param Q_nope Query NOPE block
     * @param Q_rope Query ROPE block
     * @param K_nope Key NOPE blocks [kv_len]
     * @param K_rope Key ROPE blocks [kv_len]
     * @param int_scores INT32 scores from q16_qk_gemv_mla [kv_len]
     * @param fp_scores Output FP32 scaled scores [kv_len]
     * @param kv_len Number of KV positions
     * @param k_nope_stride Stride between K_nope positions
     * @param k_rope_stride Stride between K_rope positions
     */
    void mla_apply_scales(
        const Q16_1Block_128 *Q_nope,
        const Q16_1Block_64 *Q_rope,
        const Q16_1Block_128 *K_nope,
        const Q16_1Block_64 *K_rope,
        const int32_t *int_scores,
        float *fp_scores,
        int kv_len,
        int k_nope_stride = MLAConfig::NOPE_BLOCKS_PER_HEAD,
        int k_rope_stride = MLAConfig::ROPE_BLOCKS_PER_HEAD);

} // namespace llaminar2::kernels::q16_1::microkernels
