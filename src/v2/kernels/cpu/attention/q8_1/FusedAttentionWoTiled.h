/**
 * @file FusedAttentionWoTiled.h
 * @brief Cache-blocked tiled implementation of fused attention + Wo projection.
 *
 * This implementation transforms the O(N) cache misses per query in the reference
 * implementation to O(N/KV_TILE) by:
 * 1. Loading K/V tiles into L2 cache
 * 2. Processing multiple query positions against each K/V tile
 * 3. Amortizing K/V memory bandwidth across Q_TILE queries
 *
 * The tile sizes are computed dynamically based on detected L2/L3 cache sizes
 * using CPUFeatures.h, ensuring optimal performance across different CPUs.
 *
 * @author David Sanftenberg
 * @date December 2025
 */
#pragma once

#include "FusedAttentionWoRef.h"
#include "../../../../utils/CPUFeatures.h"

#include <cstdint>
#include <cmath>
#include <limits>

namespace llaminar::v2::kernels
{

    /**
     * @brief Configuration for cache-optimal tiling.
     *
     * Computed at runtime based on detected CPU cache hierarchy.
     * These values determine how much data we process per tile to
     * maximize cache reuse and minimize memory bandwidth.
     */
    struct TileConfig
    {
        int kv_tile;      ///< K/V positions per tile (fits in L2)
        int q_tile;       ///< Query positions per tile (for K/V reuse)
        int wo_tile;      ///< Wo output dimensions per tile (optional L3 blocking)
        uint32_t l2_size; ///< Detected L2 cache size in bytes
        uint32_t l3_size; ///< Detected L3 cache size in bytes

        /**
         * @brief Check if tiling is beneficial for given dimensions.
         *
         * For very short sequences, the overhead of tiling may exceed benefits.
         *
         * @param kv_seq_len Total KV sequence length
         * @return true if tiling is beneficial
         */
        bool should_tile(int kv_seq_len) const
        {
            // Tiling beneficial when KV sequence exceeds one tile
            return kv_seq_len > kv_tile;
        }
    };

    /**
     * @brief Compute optimal tile sizes based on detected cache hierarchy.
     *
     * Uses CPUFeatures.h to detect L2/L3 cache sizes and computes tile
     * dimensions that maximize cache reuse while avoiding thrashing.
     *
     * @param head_dim Head dimension (64 or 128 typically)
     * @param d_model Output model dimension
     * @return TileConfig with optimal tile sizes
     *
     * Memory budget allocation:
     * - L2 (per core): 50% for K/V tile, 25% for context buffers, 25% for Q/scratch
     * - L3 (shared): Wo weight slices, prefetch buffers
     */
    TileConfig compute_tile_config(int head_dim, int d_model);

    /**
     * @brief Cache-blocked tiled implementation of fused attention + Wo.
     *
     * Algorithm (FlashAttention-style with Wo fusion):
     *
     * ```
     * for q_start in range(0, seq_len, Q_TILE):
     *     Initialize per-query softmax states and context buffers
     *
     *     for kv_start in range(0, kv_len, KV_TILE):
     *         // K/V tile now in L2 cache
     *         for each query m in [q_start, q_start + Q_TILE):
     *             for each kv position n in [kv_start, kv_start + KV_TILE):
     *                 if n > m (causal): skip
     *                 score = Q[m] · K[n]
     *                 update softmax state for m
     *                 context[m] += weight * V[n]
     *
     *     // Finalize Q tile
     *     for each query m in [q_start, q_start + Q_TILE):
     *         context[m] /= sum_exp[m]
     *         output[m] += context[m] × Wo  (all heads)
     * ```
     *
     * Cache behavior:
     * - K/V rows loaded once per KV_TILE, reused across Q_TILE queries
     * - Q rows loaded once per query (no reuse needed)
     * - Context buffers in L2 for duration of Q tile processing
     * - Wo projection streams through output (minimal cache footprint)
     */
    class FusedAttentionWoTiled
    {
    public:
        /**
         * @brief Execute tiled fused attention + Wo projection.
         *
         * Falls back to reference implementation if tiling is not beneficial
         * (e.g., very short sequences where tile overhead exceeds savings).
         *
         * @param params Kernel parameters (same as FusedAttentionWoRef)
         * @return true on success, false on error
         */
        static bool execute(const FusedAttentionWoParams &params);

        /**
         * @brief Execute with explicit tile configuration.
         *
         * Useful for testing specific tile sizes or benchmarking.
         *
         * @param params Kernel parameters
         * @param config Tile configuration to use
         * @return true on success, false on error
         */
        static bool execute(const FusedAttentionWoParams &params, const TileConfig &config);

        /**
         * @brief Get the tile configuration that would be used for given params.
         *
         * Useful for debugging and performance analysis.
         *
         * @param params Kernel parameters
         * @return TileConfig that would be used
         */
        static TileConfig get_tile_config(const FusedAttentionWoParams &params);

    private:
        /**
         * @brief Process a batch item with tiling.
         */
        static void process_batch_item_tiled(
            const FusedAttentionWoParams &params,
            int batch_idx,
            const TileConfig &config);

        /**
         * @brief Process a single attention head with tiling.
         *
         * This is the core tiled algorithm that processes one head's attention
         * computation with K/V cache blocking.
         *
         * @param params Kernel parameters
         * @param batch_idx Batch index
         * @param head_idx Head index
         * @param q_start Start of Q tile
         * @param q_end End of Q tile (exclusive)
         * @param kv_len Total KV sequence length
         * @param pos_offset Position offset for causal masking
         * @param config Tile configuration
         * @param context_buffers Pre-allocated context buffers [Q_TILE][head_dim]
         * @param softmax_states Pre-allocated softmax states [Q_TILE]
         */
        static void process_head_tiled(
            const FusedAttentionWoParams &params,
            int batch_idx,
            int head_idx,
            int q_start,
            int q_end,
            int kv_len,
            int pos_offset,
            const TileConfig &config,
            float *context_buffers,
            struct OnlineSoftmaxStateTiled *softmax_states);

        /**
         * @brief Prefetch K/V tile into cache.
         *
         * Uses software prefetch instructions to bring K/V data into L2
         * before it's needed, hiding memory latency.
         *
         * @param K_base Base pointer to K tensor for this batch
         * @param V_base Base pointer to V tensor for this batch
         * @param kv_head KV head index
         * @param kv_start Start of KV tile
         * @param kv_end End of KV tile (exclusive)
         * @param num_kv_heads Total number of KV heads
         * @param num_blocks Blocks per head (head_dim / 32)
         */
        static void prefetch_kv_tile(
            const Q8_1Block *K_base,
            const Q8_1Block *V_base,
            int kv_head,
            int kv_start,
            int kv_end,
            int num_kv_heads,
            int num_blocks);
    };

    /**
     * @brief Softmax state for tiled execution (per query position in tile).
     *
     * Same as OnlineSoftmaxState but grouped for tile processing.
     */
    struct OnlineSoftmaxStateTiled
    {
        float max_score;  ///< Running maximum score
        float sum_exp;    ///< Running sum of exp(score - max)
        bool initialized; ///< Has first score been seen?

        OnlineSoftmaxStateTiled()
            : max_score(-std::numeric_limits<float>::infinity()),
              sum_exp(0.0f),
              initialized(false) {}
    };

} // namespace llaminar::v2::kernels
