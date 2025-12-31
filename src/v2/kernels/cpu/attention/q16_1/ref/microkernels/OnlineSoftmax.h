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
 */
#pragma once

#include <cstdint>
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

    // ============================================================================
    // Online Softmax State
    // ============================================================================

    /**
     * @brief Running state for online softmax computation.
     *
     * Maintains the incremental softmax state across KV blocks.
     * All values are in integer domain for JIT-friendly computation.
     */
    struct OnlineSoftmaxState
    {
        /// Running maximum score (INT32)
        int32_t m = std::numeric_limits<int32_t>::min();

        /// Running sum of unnormalized weights (INT64 for precision)
        /// This is Σ exp2_lut(s[k] - m) before final normalization
        int64_t l = 0;

        /// Number of unmasked positions processed so far
        int count = 0;

        /// Configuration: fractional bits for exp2 computation
        int frac_bits = 11;

        /// Configuration: LUT value bits for exp2
        int lut_value_bits = 30;

        /// Reset to initial state
        void reset()
        {
            m = std::numeric_limits<int32_t>::min();
            l = 0;
            count = 0;
        }

        /// Check if any positions have been processed
        bool empty() const { return count == 0; }
    };

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

        /// Number of unmasked positions processed [Br]
        int count[FA2_TILE_BR];

        /// Actual number of rows in this batch (may be < FA2_TILE_BR at end)
        int num_rows = 0;

        /// Configuration: fractional bits for exp2 computation
        int frac_bits = 11;

        /// Configuration: LUT value bits for exp2
        int lut_value_bits = 30;

        /// Initialize for Br rows
        void init(int br, int frac = 11, int lut_bits = 30)
        {
            num_rows = br;
            frac_bits = frac;
            lut_value_bits = lut_bits;
            for (int r = 0; r < FA2_TILE_BR; ++r)
            {
                m[r] = std::numeric_limits<int32_t>::min();
                l[r] = 0;
                count[r] = 0;
            }
        }

        /// Check if row r has any positions processed
        bool empty(int r) const { return count[r] == 0; }
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
