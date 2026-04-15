/**
 * @file StageModelMetadata.h
 * @brief Lightweight per-stage model metadata for PP stages
 *
 * StageModelMetadata replaces per-stage ModelContext for dimension queries
 * in Pipeline Parallel mode. It contains only the hyperparameters needed
 * by graph builders and stage runners, without any association to a
 * ModelLoader, WeightManager, or GGUF file handle.
 *
 * Created from PoolModelMetadata + LayerRange during PP stage initialization.
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#pragma once

#include "ISharedWeightPool.h"
#include "WeightManagerConfig.h"
#include <string>

namespace llaminar2
{

    /**
     * @brief Per-stage model metadata for PP pipeline stages
     *
     * Unlike ModelContext (which owns a ModelLoader, WeightManager, and parses GGUF),
     * StageModelMetadata is a plain data struct with factory methods. It is created
     * from a SharedWeightPool's PoolModelMetadata with a specific LayerRange.
     *
     * Usage:
     * @code
     * auto pool_meta = pool->metadata();
     * auto stage_meta = StageModelMetadata::fromPool(pool_meta, {0, 12, true, false});
     *
     * // Use for graph building
     * int local_layers = stage_meta.localLayerCount();    // 12
     * int d_model = stage_meta.d_model;                   // 896
     * bool has_embed = stage_meta.has_embedding;           // true
     * @endcode
     */
    struct StageModelMetadata
    {
        // =========================================================================
        // Architecture
        // =========================================================================

        std::string architecture; ///< e.g., "qwen2", "llama", "qwen3"
        std::string model_path;   ///< Path to the GGUF model file

        // =========================================================================
        // Full Model Dimensions
        // =========================================================================

        int total_layers = 0;   ///< Total layers in the full model
        int d_model = 0;        ///< Hidden dimension (embedding length)
        int n_heads = 0;        ///< Number of query attention heads
        int n_kv_heads = 0;     ///< Number of KV heads (GQA)
        int head_dim = 0;       ///< Dimension per head
        int d_ff = 0;           ///< FFN intermediate dimension
        int vocab_size = 0;     ///< Vocabulary size
        int context_length = 0; ///< Maximum context length

        // GDN-specific dimensions
        int gdn_n_k_heads = 0;
        int gdn_n_v_heads = 0;
        int gdn_d_state = 0;

        // =========================================================================
        // PP Stage Assignment
        // =========================================================================

        int first_layer = 0;        ///< First layer index (inclusive)
        int last_layer = 0;         ///< Last layer index (exclusive)
        bool has_embedding = false; ///< This stage owns the embedding table
        bool has_lm_head = false;   ///< This stage owns the output norm + LM head

        // =========================================================================
        // Derived Accessors
        // =========================================================================

        /** @brief Number of layers in THIS stage */
        int localLayerCount() const { return last_layer - first_layer; }

        /** @brief Whether this has GDN dimensions */
        bool hasGDN() const { return gdn_n_k_heads > 0 && gdn_n_v_heads > 0; }

        /** @brief Get the layer range struct */
        LayerRange layerRange() const
        {
            return {first_layer, last_layer, has_embedding, has_lm_head};
        }

        // =========================================================================
        // Factory Methods
        // =========================================================================

        /**
         * @brief Create stage metadata from pool metadata + layer range
         *
         * @param pool_meta Global model metadata from SharedWeightPool
         * @param range Layer range assignment for this PP stage
         * @return StageModelMetadata with per-stage configuration
         */
        static StageModelMetadata fromPool(
            const PoolModelMetadata &pool_meta,
            const LayerRange &range)
        {
            StageModelMetadata meta;
            meta.architecture = pool_meta.architecture;
            meta.model_path = pool_meta.model_path;
            meta.total_layers = pool_meta.n_layers;
            meta.d_model = pool_meta.d_model;
            meta.n_heads = pool_meta.n_heads;
            meta.n_kv_heads = pool_meta.n_kv_heads;
            meta.head_dim = pool_meta.head_dim;
            meta.d_ff = pool_meta.d_ff;
            meta.vocab_size = pool_meta.vocab_size;
            meta.context_length = pool_meta.context_length;
            meta.gdn_n_k_heads = pool_meta.gdn_n_k_heads;
            meta.gdn_n_v_heads = pool_meta.gdn_n_v_heads;
            meta.gdn_d_state = pool_meta.gdn_d_state;
            meta.first_layer = range.first;
            meta.last_layer = range.last;
            meta.has_embedding = range.has_embedding;
            meta.has_lm_head = range.has_lm_head;
            return meta;
        }
    };

} // namespace llaminar2
