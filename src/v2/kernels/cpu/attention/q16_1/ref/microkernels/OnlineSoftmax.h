/**
 * @file OnlineSoftmax.h
 * @brief Online softmax microkernel for FlashDecode/FlashAttention
 *
 * ALGORITHM (Online Softmax)
 * ==========================
 * Standard softmax requires two passes: find max, then compute exp.
 * Online softmax computes incrementally in a single streaming pass:
 *
 *   Initialize: m = -∞, l = 0, O = 0
 *
 *   For each block of KV positions:
 *     1. Compute scores for block: s[k] = Q · K[k]
 *     2. Find local max: m_block = max(s[k])
 *     3. Update running max: m_new = max(m, m_block)
 *     4. Rescale previous state if max changed:
 *        correction = exp(m - m_new)   // In integer: 2^(-(m_new - m) * scale)
 *        l = l * correction
 *        O = O * correction
 *     5. Accumulate this block:
 *        for k in block:
 *          w = exp(s[k] - m_new)       // In integer: exp2 LUT
 *          l += w
 *          O += w * V[k]
 *     6. m = m_new
 *
 *   Finalize: output = O / l
 *
 * WHY THIS MATTERS
 * ================
 * - Memory efficiency: Never stores full [kv_len] scores array
 * - Cache friendly: Streams through KV cache in blocks
 * - Numerically stable: Running max prevents overflow
 * - JIT-friendly: Clean block-at-a-time pattern
 *
 * INTEGER DOMAIN
 * ==============
 * - m (max score) stored as INT32
 * - l (sum of weights) stored as INT64 for precision
 * - O (context) stored as INT32[head_dim]
 * - Rescaling uses exp2 LUT + integer shifts
 *
 * @see docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md
 * @see Exp2Core.h for shared exp2 LUT primitives
 */
#pragma once

#include "Exp2Core.h" // Shared exp2 LUT primitives and AdaptiveAlphaConfig

#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>
#include "utils/CPUFeatures.h"

namespace llaminar2::kernels::q16_1::microkernels
{

    // ============================================================================
    // FA2 Prefill Tile Size Configuration
    // ============================================================================

    /**
     * @brief Cache-aware tile size configuration for FA2 Prefill.
     *
     * FA2 (FlashAttention-2) processes attention in tiles to optimize cache usage:
     *   - Q tile: [Br × head_dim] query rows processed together
     *   - KV tile: [Bc × head_dim] KV positions processed together
     *   - Score tile: [Br × Bc] attention scores (must fit in L1)
     *   - Context tile: [Br × head_dim] accumulated context (should fit in L1/L2)
     *
     * Memory Layout per tile (Q16_1 format):
     *   - Q tile:       Br × (head_dim / block_size) × sizeof(Q16_1Block)
     *   - K tile:       Bc × (head_dim / block_size) × sizeof(Q16_1Block)
     *   - V tile:       Bc × (head_dim / block_size) × sizeof(Q16_1Block)
     *   - Score tile:   Br × Bc × 4 bytes (INT32)
     *   - Weights tile: Br × Bc × 4 bytes (INT32)
     *   - Context tile: Br × head_dim × 4 bytes (INT32)
     *
     * Cache Targets:
     *   - L1: Score tile + Weights tile + current Q row should fit
     *   - L2: Context tile + K/V tiles being streamed
     */
    struct FA2TileConfig
    {
        int Br;       ///< Query tile rows (typically 2-8)
        int Bc;       ///< KV tile columns (typically 16-64)
        int head_dim; ///< Head dimension for memory calculations

        /// Compute memory footprint of score + weights scratch [Br × Bc × 2 × 4 bytes]
        size_t scratch_bytes() const
        {
            return static_cast<size_t>(Br) * Bc * 2 * sizeof(int32_t);
        }

        /// Compute memory footprint of context tile [Br × head_dim × 4 bytes]
        size_t context_bytes() const
        {
            return static_cast<size_t>(Br) * head_dim * sizeof(int32_t);
        }

        /// Compute memory footprint of K or V tile [Bc × blocks_per_row × block_size_bytes]
        /// For Q16_1Block_64: 66 bytes, Q16_1Block_128: 130 bytes
        size_t kv_tile_bytes(int block_size) const
        {
            int blocks_per_row = (head_dim + block_size - 1) / block_size;
            size_t bytes_per_block = (block_size == 64) ? 130 : 258; // Q16_1Block_64 or Q16_1Block_128
            return static_cast<size_t>(Bc) * blocks_per_row * bytes_per_block;
        }
    };

    /**
     * @brief Compute optimal FA2 tile sizes based on detected CPU cache hierarchy.
     *
     * Algorithm:
     * 1. Target L1 for score/weights scratch: Br × Bc × 8 bytes should be ≤ 50% L1
     * 2. Target L2 for context tile: Br × head_dim × 4 bytes should be ≤ 25% L2
     * 3. Bc should be multiple of 4 for SIMD efficiency
     * 4. Br should be 2-8 for register pressure balance
     *
     * Derivation for typical 32KB L1:
     *   - 50% L1 = 16KB for scratch
     *   - Br=4, Bc=32: 4×32×8 = 1024 bytes ✓ (well under 16KB)
     *   - Could support Br=4, Bc=64: 4×64×8 = 2048 bytes ✓
     *
     * Derivation for typical 1MB L2:
     *   - 25% L2 = 256KB for context
     *   - Br=4, head_dim=128: 4×128×4 = 2048 bytes ✓
     *
     * @param head_dim Attention head dimension (e.g., 64, 128)
     * @param kv_len Total KV sequence length (for prefetch tuning)
     * @return FA2TileConfig with optimal Br and Bc values
     */
    inline FA2TileConfig compute_fa2_tile_config(int head_dim, int kv_len = 0)
    {
        const auto &cache = cache_info();

        FA2TileConfig cfg;
        cfg.head_dim = head_dim;

        // ===== Step 1: Determine Bc (KV tile width) =====
        // Target: scratch buffers [Br × Bc × 8 bytes] fit in 50% of L1
        // Start with conservative Br=4, solve for Bc
        const size_t l1_target = cache.l1_size / 2; // 50% of L1
        const int bytes_per_scratch_elem = 8;       // INT32 scores + INT32 weights
        const int initial_Br = 4;

        // Bc_max from L1 constraint: Br × Bc × 8 ≤ l1_target
        int Bc_from_l1 = static_cast<int>(l1_target / (initial_Br * bytes_per_scratch_elem));

        // Round down to multiple of 8 for SIMD alignment
        Bc_from_l1 = (Bc_from_l1 / 8) * 8;

        // Clamp Bc to reasonable range [8, 128]
        // Larger Bc = fewer tiles = less overhead, but more memory pressure
        cfg.Bc = std::clamp(Bc_from_l1, 8, 128);

        // ===== Step 2: Determine Br (query tile height) =====
        // Target: context tile [Br × head_dim × 4 bytes] fits in 25% of L2
        const size_t l2_target = cache.l2_size / 4; // 25% of L2
        const int bytes_per_context_elem = sizeof(int32_t);

        // Br_max from L2 constraint: Br × head_dim × 4 ≤ l2_target
        int Br_from_l2 = static_cast<int>(l2_target / (head_dim * bytes_per_context_elem));

        // Clamp Br to reasonable range [2, 8]
        // Larger Br = more parallel queries, but higher register pressure
        cfg.Br = std::clamp(Br_from_l2, 2, 8);

        // ===== Step 3: Adjust Bc based on final Br =====
        // Recompute Bc with actual Br
        int Bc_adjusted = static_cast<int>(l1_target / (cfg.Br * bytes_per_scratch_elem));
        Bc_adjusted = (Bc_adjusted / 8) * 8;
        cfg.Bc = std::clamp(Bc_adjusted, 8, 128);

        // ===== Step 4: KV length-based tuning =====
        // For short sequences, smaller Bc reduces overhead
        // For long sequences, larger Bc amortizes loop overhead
        if (kv_len > 0)
        {
            if (kv_len < 64)
            {
                cfg.Bc = std::min(cfg.Bc, 16); // Small KV → small tiles
            }
            else if (kv_len > 1024)
            {
                cfg.Bc = std::max(cfg.Bc, 32); // Large KV → larger tiles
            }
        }

        return cfg;
    }

    /**
     * @brief Get default FA2 tile configuration (compile-time constants).
     *
     * These are conservative defaults that work well across most CPUs:
     *   - Br=4: Good balance of parallelism vs register pressure
     *   - Bc=32: Fits in L1 with room for Q, good SIMD vectorization
     */
    inline constexpr FA2TileConfig default_fa2_tile_config(int head_dim)
    {
        return FA2TileConfig{4, 32, head_dim};
    }

    /// Legacy constants for backwards compatibility
    /// @deprecated Use compute_fa2_tile_config() or default_fa2_tile_config() instead
    constexpr int FA2_TILE_BR = 8; // Max Br value from compute_fa2_tile_config()
    constexpr int FA2_TILE_BC = 32;

    // Note: AdaptiveAlphaConfig is now provided by Exp2Core.h for all
    // softmax implementations to share.

    // ============================================================================
    // Online Softmax State
    // ============================================================================

    /**
     * @brief Running state for online softmax computation.
     *
     * Maintains the incremental softmax state across KV blocks.
     * All values are in integer domain for JIT-friendly computation.
     *
     * RUNNING AVERAGE DESIGN (v2):
     * ============================
     * Instead of accumulating raw sums that grow with sequence length,
     * we maintain context as a NORMALIZED running average:
     *
     *   context = Σ(w_i * v_i) / Σ(w_i)
     *
     * This keeps context values bounded regardless of sequence length.
     *
     * Incremental update for each block:
     *   l_new = l_old + block_sum
     *   context_new = (context_old * l_old + block_contribution) / l_new
     *
     * The key insight: softmax weights sum to 1, so the final weighted
     * average of V values stays bounded by the V value range.
     */
    struct OnlineSoftmaxState
    {
        /// Running maximum score (INT32) - in SCALED domain when adaptive scaling
        int32_t m = std::numeric_limits<int32_t>::min();

        /// Running sum of unnormalized weights (INT64 for precision)
        /// This is Σ exp2_lut(s[k] - m) before final normalization
        int64_t l = 0;

        /// Sum of weights already incorporated into context (for running average)
        /// Using double to avoid precision loss during rescaling at very long sequences.
        /// The merge formula needs: l_old / l_new which is exact with double.
        double l_processed = 0.0;

        /// Number of unmasked positions processed so far
        int count = 0;

        /// Configuration: fractional bits for exp2 computation
        int frac_bits = 11;

        /// Configuration: LUT value bits for exp2
        int lut_value_bits = 30;

        /// Adaptive alpha configuration (computed once at init)
        AdaptiveAlphaConfig alpha_config;

        // ===== Per-position K scale support (HybridQ16 K precision fix) =====

        /// Base alpha for per-position mode: q_scale / sqrt(head_dim) * log2(e)
        /// Final alpha = base_alpha_fp32 * k_scale[pos]
        float base_alpha_fp32 = 0.0f;

        /// Head dimension (needed for per-position scale computation)
        int head_dim_for_scale = 0;

        /// Flag indicating per-position K scale mode is active
        bool per_position_k_mode = false;

        /// Reset to initial state
        void reset()
        {
            m = std::numeric_limits<int32_t>::min();
            l = 0;
            l_processed = 0.0;
            count = 0;
            // Note: alpha_config, base_alpha_fp32, per_position_k_mode preserved across reset
        }

        /// Initialize with alpha for adaptive scaling (uniform K scale mode)
        void init(float alpha, int frac = 11, int lut_bits = 30)
        {
            reset();
            frac_bits = frac;
            lut_value_bits = lut_bits;
            alpha_config = AdaptiveAlphaConfig::compute(alpha);
            per_position_k_mode = false;
            base_alpha_fp32 = 0.0f;
            head_dim_for_scale = 0;
        }

        /**
         * @brief Initialize for per-position K scale mode (Option C: Pass Scale to Softmax).
         *
         * In this mode, we separate the QK scale into:
         *   - base_alpha = q_scale / sqrt(head_dim) * log2(e) (computed once)
         *   - per_position_alpha = base_alpha * k_scale[pos] (per-position)
         *
         * The softmax processes integer scores and applies per-position K scale
         * during the exp2 computation.
         *
         * @param q_scale Q head scale factor
         * @param head_dim Head dimension (for 1/sqrt normalization)
         * @param frac Fractional bits for exp2 computation
         * @param lut_bits LUT value precision bits
         */
        void init_per_position(float q_scale, int head_dim, int frac = 11, int lut_bits = 30)
        {
            reset();
            frac_bits = frac;
            lut_value_bits = lut_bits;
            per_position_k_mode = true;
            head_dim_for_scale = head_dim;

            // Pre-compute base alpha (without K scale or LOG2E)
            // LOG2E is applied inside AdaptiveAlphaConfig/exp2_compute_weight
            // So base_alpha = q_scale / sqrt(head_dim), NOT including LOG2E
            float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim));
            base_alpha_fp32 = q_scale * inv_sqrt_d; // Note: NO LOG2E here!

            // Initialize alpha_config with a representative scale for rescaling
            // Use q_scale only; actual per-position alpha is computed on-the-fly
            float representative_alpha = q_scale * inv_sqrt_d;
            alpha_config = AdaptiveAlphaConfig::compute(representative_alpha);
        }

        /// Check if any positions have been processed
        bool empty() const { return count == 0; }
    };

    // ============================================================================
    // V2: Deferred Normalization Online Softmax State
    // ============================================================================

    /**
     * @brief VNNI-safe weight configuration for deferred normalization.
     *
     * With 10-bit scaled weights (weight_shift=20), we can safely accumulate
     * up to 60 positions in INT32 before dumping to INT64.
     *
     * Proof:
     *   max_weight_scaled = 2^30 >> 20 = 1024 (10 bits)
     *   max_V = 32767 (INT16)
     *   max_per_product = 1024 × 32767 ≈ 2^25
     *   max_per_chunk = 60 × 2^25 ≈ 2×10^9 < 2.15×10^9 (INT32_MAX) ✓
     */
    struct VNNIChunkConfig
    {
        static constexpr int WEIGHT_SHIFT = 20; ///< 30-bit LUT → 10-bit VNNI-safe
        static constexpr int MAX_WEIGHT = 1023; ///< After shift
        static constexpr int CHUNK_SIZE = 60;   ///< Positions per INT32 accumulation
        static constexpr int MAX_V = 32767;     ///< INT16 max
    };

    /**
     * @brief Running state for deferred-normalization online softmax (V2).
     *
     * KEY DIFFERENCE FROM V1:
     * =======================
     * V1 normalized context after EVERY block (O(N × head_dim) FP divisions).
     * V2 accumulates unnormalized sums in INT64, divides ONCE at finalization.
     *
     * INVARIANT:
     *   context_accum[d] = Σ(w_scaled[k] × V[k][d]) for all k processed
     *   sum_w_scaled     = Σ(w_scaled[k]) for all k processed
     *
     * At finalization: output[d] = context_accum[d] / sum_w_scaled
     *
     * This eliminates ~725K FP divisions for a 596-token prefill.
     */
    struct OnlineSoftmaxStateV2
    {
        // === Max tracking (same as V1) ===
        int32_t m = std::numeric_limits<int32_t>::min(); ///< Running max score

        // === Weight sums (BOTH tracked for correctness) ===
        int64_t sum_w_unscaled = 0; ///< Sum before weight_shift (for debugging/verification)
        int64_t sum_w_scaled = 0;   ///< Sum after weight_shift (for final division)

        // === Position tracking ===
        int count = 0; ///< Unmasked positions processed

        // === Configuration ===
        int frac_bits = 11;                               ///< Fractional bits for exp2 computation
        int lut_value_bits = 30;                          ///< Total bits in LUT output
        int weight_shift = VNNIChunkConfig::WEIGHT_SHIFT; ///< Shift for VNNI-safe weights
        int chunk_size = VNNIChunkConfig::CHUNK_SIZE;     ///< P×V accumulation chunk

        // === Alpha configuration ===
        AdaptiveAlphaConfig alpha_config;
        float base_alpha_fp32 = 0.0f;     ///< For per-position K scale mode
        int head_dim_for_scale = 0;       ///< Head dimension for scale computation
        bool per_position_k_mode = false; ///< Whether using per-position K scales

        // REMOVED: double l_processed (no longer needed!)

        /// Reset to initial state (preserves configuration)
        void reset()
        {
            m = std::numeric_limits<int32_t>::min();
            sum_w_unscaled = 0;
            sum_w_scaled = 0;
            count = 0;
        }

        /// Initialize with alpha for adaptive scaling (uniform K scale mode)
        void init(float alpha, int frac = 11, int lut_bits = 30)
        {
            reset();
            frac_bits = frac;
            lut_value_bits = lut_bits;
            weight_shift = VNNIChunkConfig::WEIGHT_SHIFT;
            chunk_size = VNNIChunkConfig::CHUNK_SIZE;
            alpha_config = AdaptiveAlphaConfig::compute(alpha);
            per_position_k_mode = false;
            base_alpha_fp32 = 0.0f;
            head_dim_for_scale = 0;
        }

        /// Initialize for per-position K scale mode
        void init_per_position(float q_scale, int head_dim, int frac = 11, int lut_bits = 30)
        {
            reset();
            frac_bits = frac;
            lut_value_bits = lut_bits;
            weight_shift = VNNIChunkConfig::WEIGHT_SHIFT;
            chunk_size = VNNIChunkConfig::CHUNK_SIZE;
            per_position_k_mode = true;
            head_dim_for_scale = head_dim;

            float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim));
            base_alpha_fp32 = q_scale * inv_sqrt_d;

            float representative_alpha = q_scale * inv_sqrt_d;
            alpha_config = AdaptiveAlphaConfig::compute(representative_alpha);
        }

        /// Check if any positions have been processed
        bool empty() const { return count == 0; }
    };

    // ============================================================================
    // V2: 128-bit Rescale Helper
    // ============================================================================

    /**
     * @brief Rescale INT64 value using 128-bit intermediate.
     *
     * Computes: (value × scale_num) >> scale_shift
     *
     * Uses __int128 to prevent overflow when value is large (e.g., 2^50)
     * and scale_num is up to 2^30.
     *
     * @param value INT64 value to rescale
     * @param scale_num Rescale numerator (from exp2 LUT)
     * @param scale_shift Rescale denominator as power of 2
     * @return Rescaled INT64 value
     */
    inline int64_t rescale_int64_v2(__int128 value, int32_t scale_num, int scale_shift)
    {
        __int128 product = value * static_cast<__int128>(scale_num);
        return static_cast<int64_t>(product >> scale_shift);
    }

    // ============================================================================
    // Block-wise Online Softmax Operations
    // ============================================================================

    /**
     * @brief Find maximum score in a block (for online max update).
     *
     * @param scores INT32 scores for this block [block_size]
     * @param block_size Number of scores in block
     * @param mask_value Value indicating masked position (INT32_MIN)
     * @return Maximum unmasked score, or INT32_MIN if all masked
     */
    int32_t online_softmax_find_block_max(
        const int32_t *scores,
        int block_size,
        int32_t mask_value = std::numeric_limits<int32_t>::min());

    /**
     * @brief Compute exp2 rescaling factor for max update.
     *
     * When running max changes from m_old to m_new, previous state must be
     * rescaled by exp(m_old - m_new) = 2^((m_old - m_new) * beta).
     *
     * Returns integer representation: (scale_num, scale_shift) such that
     *   rescale_factor ≈ scale_num >> scale_shift
     *
     * @param m_old Previous running max (INT32)
     * @param m_new New running max (INT32)
     * @param alpha Score scaling factor (qk_scale)
     * @param scale_num Output: numerator for rescaling
     * @param scale_shift Output: right-shift for rescaling
     * @param config Exp2 configuration
     */
    void online_softmax_compute_rescale(
        int32_t m_old,
        int32_t m_new,
        float alpha,
        int32_t &scale_num,
        int &scale_shift,
        int frac_bits = 11,
        int lut_value_bits = 30);

    /**
     * @brief Compute rescale factor with adaptive alpha (internal).
     */
    void online_softmax_compute_rescale_adaptive(
        int32_t m_old,
        int32_t m_new,
        const AdaptiveAlphaConfig &alpha_config,
        int32_t &scale_num,
        int &scale_shift,
        int frac_bits = 11,
        int lut_value_bits = 30);

    /**
     * @brief Compute unnormalized exp2 weights for a block of scores.
     *
     * For each score s[k], computes w[k] = exp2_lut(s[k] - m) where m is
     * the running max. Returns sum of weights for this block.
     *
     * @param scores INT32 scores [block_size]
     * @param weights Output INT32 unnormalized weights [block_size]
     * @param block_size Number of scores
     * @param running_max Current running max (state.m)
     * @param alpha Score scaling factor
     * @param frac_bits Fractional bits for exp2
     * @param lut_value_bits LUT value precision
     * @return Sum of weights for this block (INT64)
     */
    int64_t online_softmax_compute_block_weights(
        const int32_t *scores,
        int32_t *weights,
        int block_size,
        int32_t running_max,
        float alpha,
        int frac_bits = 11,
        int lut_value_bits = 30);

    /**
     * @brief Compute block weights with adaptive alpha (internal).
     */
    int64_t online_softmax_compute_block_weights_adaptive(
        const int32_t *scores,
        int32_t *weights,
        int block_size,
        int32_t running_max,
        const AdaptiveAlphaConfig &alpha_config,
        int frac_bits = 11,
        int lut_value_bits = 30);

    /**
     * @brief Update online softmax state with a new block of scores.
     *
     * This is the main online softmax update function:
     * 1. Find block max
     * 2. Update running max
     * 3. Compute rescale factor if max changed
     * 4. Compute weights for this block
     * 5. Update running sum
     *
     * @param state Online softmax state (modified in place)
     * @param scores INT32 scores for this block [block_size]
     * @param weights Output INT32 unnormalized weights [block_size]
     * @param block_size Number of scores in block
     * @param alpha Score scaling factor
     * @param scale_num Output: rescale numerator (1.0 if no rescale needed)
     * @param scale_shift Output: rescale shift (0 if no rescale needed)
     * @return true if rescaling is needed (max changed)
     */
    bool online_softmax_update_block(
        OnlineSoftmaxState &state,
        const int32_t *scores,
        int32_t *weights,
        int block_size,
        float alpha,
        int32_t &scale_num,
        int &scale_shift);

    /**
     * @brief Finalize weights: normalize to INT16 range [0, 32767].
     *
     * After processing all blocks, unnormalized weights in [block_weights]
     * are scaled by weight_max / state.l to produce final INT16 weights.
     *
     * @param state Final online softmax state
     * @param block_weights Unnormalized INT32 weights for all blocks
     * @param final_weights Output INT16 normalized weights
     * @param total_len Total number of KV positions
     * @param weight_max Maximum output weight (32767)
     * @return Sum of final weights (should be close to weight_max * count)
     */
    int32_t online_softmax_finalize_weights(
        const OnlineSoftmaxState &state,
        const int32_t *block_weights,
        int16_t *final_weights,
        int total_len,
        int16_t weight_max = 32767);

    // ============================================================================
    // FlashDecode-Specific Operations
    // ============================================================================

    /**
     * @brief Process one KV block in FlashDecode with online softmax.
     *
     * High-level function that combines:
     * 1. Q×K dot products for block
     * 2. Online softmax state update
     * 3. Context rescaling if needed
     * 4. P×V accumulation for block
     *
     * @tparam BlockType Q16_1Block_64 or Q16_1Block_128
     * @param Q Query blocks [blocks_per_row]
     * @param K Key blocks for this block [block_size, blocks_per_row]
     * @param V Value blocks for this block [block_size, blocks_per_row]
     * @param context Running INT32 context [head_dim] (modified in place)
     * @param state Online softmax state (modified in place)
     * @param scores_scratch INT32 scratch for scores [block_size]
     * @param weights_scratch INT32 scratch for unnorm weights [block_size]
     * @param kv_block_start Starting KV position for this block
     * @param kv_block_size Number of positions in this block
     * @param head_dim Head dimension
     * @param blocks_per_row Blocks per head
     * @param alpha QK scale factor
     */
    template <typename BlockType>
    void flash_decode_process_kv_block(
        const BlockType *Q,
        const BlockType *K,
        const BlockType *V,
        int32_t *context,
        OnlineSoftmaxState &state,
        int32_t *scores_scratch,
        int32_t *weights_scratch,
        int kv_block_start,
        int kv_block_size,
        int head_dim,
        int blocks_per_row,
        float alpha);

    /**
     * @brief Process one KV block in FlashDecode with per-position K scales.
     *
     * Option C implementation: Pass per-position K scales to softmax.
     * The dot product remains pure integer, and per-position K scale is applied
     * during the exp2 computation in softmax.
     *
     * @tparam BlockType Q16_1Block_64 or Q16_1Block_128
     * @param Q Query blocks [blocks_per_row]
     * @param K Key blocks for this block [block_size, blocks_per_row]
     * @param V Value blocks for this block [block_size, blocks_per_row]
     * @param context Running INT32 context [head_dim] (modified in place)
     * @param state Online softmax state (modified in place, must use init_per_position())
     * @param scores_scratch INT32 scratch for scores [block_size]
     * @param weights_scratch INT32 scratch for unnorm weights [block_size]
     * @param kv_block_start Starting KV position for this block
     * @param kv_block_size Number of positions in this block
     * @param head_dim Head dimension
     * @param blocks_per_row Blocks per head
     * @param k_scales Per-position K scale factors [kv_block_size] for this block
     */
    template <typename BlockType>
    void flash_decode_process_kv_block_with_k_scales(
        const BlockType *Q,
        const BlockType *K,
        const BlockType *V,
        int32_t *context,
        OnlineSoftmaxState &state,
        int32_t *scores_scratch,
        int32_t *weights_scratch,
        int kv_block_start,
        int kv_block_size,
        int head_dim,
        int blocks_per_row,
        const float *k_scales);

    // ============================================================================
    // V2: Deferred Normalization Block Processing (PURE INTEGER)
    // ============================================================================

    /**
     * @brief Process one KV block in FlashDecode with deferred normalization (V2).
     *
     * KEY DIFFERENCE FROM V1:
     * - context_accum is INT64 (unnormalized accumulator)
     * - NO division happens in this function
     * - Uses VNNI-safe chunking (60 positions per INT32 accumulation)
     *
     * @tparam BlockType Q16_1Block_64 or Q16_1Block_128
     * @param Q Query blocks [blocks_per_row]
     * @param K Key blocks for this block [block_size, blocks_per_row]
     * @param V Value blocks for this block [block_size, blocks_per_row]
     * @param context_accum Running INT64 unnormalized context [head_dim] (modified in place)
     * @param state V2 online softmax state (modified in place)
     * @param scores_scratch INT32 scratch for scores [block_size]
     * @param weights_scratch INT32 scratch for unnorm weights [block_size]
     * @param kv_block_start Starting KV position for this block
     * @param kv_block_size Number of positions in this block
     * @param head_dim Head dimension
     * @param blocks_per_row Blocks per head
     * @param alpha QK scale factor
     */
    template <typename BlockType>
    void flash_decode_process_kv_block_v2(
        const BlockType *Q,
        const BlockType *K,
        const BlockType *V,
        int64_t *context_accum,
        OnlineSoftmaxStateV2 &state,
        int32_t *scores_scratch,
        int32_t *weights_scratch,
        int kv_block_start,
        int kv_block_size,
        int head_dim,
        int blocks_per_row,
        float alpha);

    /**
     * @brief Finalize FlashDecode context with single division pass (V2).
     *
     * Divides the accumulated unnormalized context by the total weight sum
     * to produce the final normalized INT32 context.
     *
     * @param context_accum Unnormalized INT64 context [head_dim]
     * @param context_out Output INT32 context [head_dim]
     * @param state V2 online softmax state (read-only)
     * @param head_dim Head dimension
     */
    void flash_decode_finalize_v2(
        const int64_t *context_accum,
        int32_t *context_out,
        const OnlineSoftmaxStateV2 &state,
        int head_dim);

    // ============================================================================
    // FA2 Prefill: Batch Online Softmax State
    // ============================================================================

    /**
     * @brief Per-row online softmax state for FA2 Prefill.
     *
     * Unlike FlashDecode (single query), FA2 Prefill processes Br query rows
     * simultaneously. Each row maintains independent softmax state.
     *
     * This structure holds state for up to FA2_TILE_BR rows.
     */
    struct OnlineSoftmaxStateBatch
    {
        /// Running maximum scores [Br] - one per query row
        int32_t m[FA2_TILE_BR];

        /// Running sum of unnormalized weights [Br] - one per query row
        int64_t l[FA2_TILE_BR];

        /// Sum of weights already incorporated into context [Br] (for running average)
        /// Using double to avoid precision loss during rescaling at very long sequences.
        double l_processed[FA2_TILE_BR];

        /// Number of unmasked positions processed [Br]
        int count[FA2_TILE_BR];

        /// Actual number of rows in this batch (may be < FA2_TILE_BR at end)
        int num_rows = 0;

        /// Configuration: fractional bits for exp2 computation
        int frac_bits = 11;

        /// Configuration: LUT value bits for exp2
        int lut_value_bits = 30;

        /// Adaptive alpha configuration (computed once at init, shared by all rows)
        AdaptiveAlphaConfig alpha_config;

        /// Initialize for Br rows with adaptive alpha
        void init(int br, float alpha, int frac = 11, int lut_bits = 30)
        {
            num_rows = br;
            frac_bits = frac;
            lut_value_bits = lut_bits;
            base_alpha_fp32 = 0.0f; // Not using per-position mode
            alpha_config = AdaptiveAlphaConfig::compute(alpha);
            for (int r = 0; r < FA2_TILE_BR; ++r)
            {
                m[r] = std::numeric_limits<int32_t>::min();
                l[r] = 0;
                l_processed[r] = 0.0;
                count[r] = 0;
            }
        }

        /**
         * @brief Initialize for per-position K scale mode (Option C).
         *
         * In this mode, each K position has its own scale factor, so we can't
         * pre-compute a single alpha. Instead we store base_alpha = q_scale / sqrt(head_dim)
         * and compute per-position alpha on-the-fly: alpha[k] = base_alpha * k_scale[k]
         *
         * @param br Number of query rows in this batch
         * @param q_scale Q head scale factor
         * @param head_dim Head dimension (for 1/sqrt normalization)
         * @param frac Fractional bits for exp2 computation
         * @param lut_bits LUT value precision bits
         */
        void init_per_position(int br, float q_scale, int head_dim, int frac = 11, int lut_bits = 30)
        {
            num_rows = br;
            frac_bits = frac;
            lut_value_bits = lut_bits;

            // Pre-compute base alpha (without K scale)
            float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim));
            base_alpha_fp32 = q_scale * inv_sqrt_d;

            // Initialize alpha_config with representative scale for rescaling
            alpha_config = AdaptiveAlphaConfig::compute(base_alpha_fp32);

            for (int r = 0; r < FA2_TILE_BR; ++r)
            {
                m[r] = std::numeric_limits<int32_t>::min();
                l[r] = 0;
                l_processed[r] = 0.0;
                count[r] = 0;
            }
        }

        /// Legacy init without alpha (for backwards compatibility)
        void init(int br, int frac = 11, int lut_bits = 30)
        {
            init(br, 0.0f, frac, lut_bits);
        }

        /// Check if row r has any positions processed
        bool empty(int r) const { return count[r] == 0; }

        /// Base alpha for per-position K scale mode (q_scale / sqrt(head_dim))
        /// Zero when not in per-position mode
        float base_alpha_fp32 = 0.0f;
    };

    // ============================================================================
    // FA2 Prefill Operations
    // ============================================================================

    /**
     * @brief Process one KV tile in FA2 Prefill with online softmax.
     *
     * FA2 Prefill version of flash_decode_process_kv_block.
     * Processes [Br × Bc] tile instead of [1 × Bc] block.
     *
     * Algorithm for each query row r:
     *   1. Compute scores[r, :] = Q[r] × K[kv_tile]^T  (GEMM row)
     *   2. Update online softmax state for row r
     *   3. Rescale context[r] if max changed
     *   4. Accumulate P[r] × V[kv_tile] into context[r]
     *
     * @tparam BlockType Q16_1Block_64 or Q16_1Block_128
     * @param Q Query tile blocks [Br, blocks_per_row]
     * @param K Key tile blocks [Bc, blocks_per_row]
     * @param V Value tile blocks [Bc, blocks_per_row]
     * @param context Running INT32 context tile [Br, head_dim] (modified)
     * @param state Batch online softmax state [Br] (modified)
     * @param scores_scratch INT32 scratch for score tile [Br × Bc]
     * @param weights_scratch INT32 scratch for unnorm weights [Br × Bc]
     * @param kv_tile_start Starting KV position for this tile
     * @param Br Number of query rows in tile (may be < FA2_TILE_BR)
     * @param Bc Number of KV positions in tile (may be < FA2_TILE_BC)
     * @param head_dim Head dimension
     * @param blocks_per_row Blocks per head
     * @param q_stride Stride between Q rows in blocks
     * @param k_stride Stride between K rows in blocks (usually blocks_per_row)
     * @param context_stride Stride between context rows (usually head_dim)
     * @param alpha QK scale factor
     * @param causal Whether to apply causal masking
     * @param q_offset Query position offset (for causal mask calculation)
     */
    template <typename BlockType>
    void fa2_prefill_process_kv_tile(
        const BlockType *Q,
        const BlockType *K,
        const BlockType *V,
        int32_t *context,
        OnlineSoftmaxStateBatch &state,
        int32_t *scores_scratch,
        int32_t *weights_scratch,
        int kv_tile_start,
        int Br,
        int Bc,
        int head_dim,
        int blocks_per_row,
        int q_stride,
        int k_stride,
        int context_stride,
        float alpha,
        bool causal = false,
        int q_offset = 0);

    /**
     * @brief Process one KV tile in FA2 Prefill with per-position K scales.
     *
     * Same as fa2_prefill_process_kv_tile but with per-position K scale support.
     * Each KV position has its own K scale, allowing for dynamic quantization
     * where different positions may have different scale factors.
     *
     * The base_alpha_fp32 is computed as: q_scale / sqrt(head_dim)
     * For each KV position k, the effective alpha is: base_alpha_fp32 * k_scales[k]
     *
     * This is critical for HybridQ16 pipeline where K cache blocks have per-position
     * scales stored in their .d header field, and using a uniform qk_scale causes
     * numerical divergence.
     *
     * @tparam BlockType Q16_1Block_64 or Q16_1Block_128
     * @param Q Query tile blocks [Br, blocks_per_row]
     * @param K Key tile blocks [Bc, blocks_per_row]
     * @param V Value tile blocks [Bc, blocks_per_row]
     * @param context Running INT32 context tile [Br, head_dim] (modified)
     * @param state Batch online softmax state [Br] (modified)
     * @param scores_scratch INT32 scratch for score tile [Br × Bc]
     * @param weights_scratch INT32 scratch for unnorm weights [Br × Bc]
     * @param kv_tile_start Starting KV position for this tile
     * @param Br Number of query rows in tile (may be < FA2_TILE_BR)
     * @param Bc Number of KV positions in tile (may be < FA2_TILE_BC)
     * @param head_dim Head dimension
     * @param blocks_per_row Blocks per head
     * @param q_stride Stride between Q rows in blocks
     * @param k_stride Stride between K rows in blocks (usually blocks_per_row)
     * @param context_stride Stride between context rows (usually head_dim)
     * @param base_alpha_fp32 Base alpha = q_scale / sqrt(head_dim) (WITHOUT k_scale)
     * @param k_scales Per-position K scales [Bc] for this tile (starting at kv_tile_start)
     * @param causal Whether to apply causal masking
     * @param q_offset Query position offset (for causal mask calculation)
     */
    template <typename BlockType>
    void fa2_prefill_process_kv_tile_with_k_scales(
        const BlockType *Q,
        const BlockType *K,
        const BlockType *V,
        int32_t *context,
        OnlineSoftmaxStateBatch &state,
        int32_t *scores_scratch,
        int32_t *weights_scratch,
        int kv_tile_start,
        int Br,
        int Bc,
        int head_dim,
        int blocks_per_row,
        int q_stride,
        int k_stride,
        int context_stride,
        float base_alpha_fp32,
        const float *k_scales,
        bool causal = false,
        int q_offset = 0);

    // ============================================================================
    // FlashDecode KV Block Size Configuration
    // ============================================================================

    /**
     * @brief Compute optimal KV block size for FlashDecode based on cache hierarchy.
     *
     * FlashDecode processes a single query row against the KV cache in blocks.
     * The block size affects:
     *   - Score scratch: [1 × Bc × 4 bytes] for INT32 scores
     *   - Weights scratch: [1 × Bc × 4 bytes] for INT32 weights
     *   - KV access pattern: Bc consecutive KV positions per iteration
     *
     * Cache Targets:
     *   - L1: Score + weight scratch (2 × Bc × 4 bytes) should fit with headroom
     *   - Vectorization: Bc should be multiple of 8 for AVX2/AVX-512
     *   - Prefetch: Bc should allow effective prefetch of next KV block
     *
     * @param head_dim Attention head dimension
     * @return Optimal KV block size (multiple of 8, clamped to [16, 128])
     */
    inline int compute_flash_decode_kv_block_size(int head_dim)
    {
        const auto &cache = cache_info();

        // Target: scratch buffers [2 × Bc × 4 bytes] fit in 25% of L1
        // Leave room for K/V data and context accumulator
        const size_t l1_target = cache.l1_size / 4;
        const int bytes_per_elem = 2 * sizeof(int32_t); // scores + weights

        int Bc = static_cast<int>(l1_target / bytes_per_elem);

        // Round down to multiple of 8 for SIMD alignment
        Bc = (Bc / 8) * 8;

        // Clamp to reasonable range [16, 128]
        return std::clamp(Bc, 16, 128);
    }

    /// Default KV block size for FlashDecode (compile-time constant)
    /// 32 is a good balance: small enough for cache, large enough for SIMD
    /// @deprecated Use compute_flash_decode_kv_block_size() for runtime optimization
    constexpr int FLASH_DECODE_KV_BLOCK_SIZE = 32;

} // namespace llaminar2::kernels::q16_1::microkernels
