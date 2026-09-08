/**
 * @file WeightManagerConfig.h
 * @brief Immutable configuration for WeightManager construction
 *
 * Replaces multi-phase setter initialization with a single config struct.
 * All configuration that was previously set via setModelDimensions(),
 * setGDNDimensions(), setTensorParallelConfig(), setLayerRange(), etc.
 * is now provided upfront at WeightManager construction time.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "WeightTypes.h"
#include "../config/TensorParallelConfig.h"
#include "../execution/local_execution/graph/GraphSchema.h"
#include "../execution/config/RuntimeConfig.h"
#include <memory>
#include <optional>

namespace llaminar2
{

    /**
     * @brief Model head and dimension information for weight slicing
     *
     * Consolidates what was previously set via three separate APIs:
     * - setModelDimensions(n_heads, n_kv_heads, head_dim)
     * - setGDNDimensions(n_k_heads, n_v_heads, d_state)
     * - WeightShardingConfig::getDimensionType() (pattern-based)
     *
     * All three sources are now in one place so they can't disagree.
     */
    struct ModelDimensions
    {
        // Standard attention dimensions
        int n_heads = 0;    ///< Number of query attention heads
        int n_kv_heads = 0; ///< Number of KV attention heads (GQA)
        int head_dim = 0;   ///< Dimension per attention head

        // GDN (Gated Delta Net) dimensions — zero means no GDN
        int gdn_n_k_heads = 0; ///< Number of GDN key heads (ssm.group_count)
        int gdn_n_v_heads = 0; ///< Number of GDN value heads (ssm.time_step_rank)
        int gdn_d_state = 0;   ///< GDN state dimension per head (ssm.state_size)

        /**
         * @brief Check if GDN dimensions are configured
         */
        bool hasGDN() const { return gdn_n_k_heads > 0 && gdn_n_v_heads > 0 && gdn_d_state > 0; }

        /**
         * @brief Check if standard attention dimensions are valid
         */
        bool isValid() const { return n_heads > 0 && head_dim > 0; }

        /**
         * @brief Compute expected FusedQKV total rows for standard attention
         * Layout: [Q(n_heads*hd) | K(n_kv_heads*hd) | V(n_kv_heads*hd)]
         */
        size_t expectedFusedQKVRows() const
        {
            return static_cast<size_t>(n_heads) * head_dim + 2 * static_cast<size_t>(n_kv_heads) * head_dim;
        }

        /**
         * @brief Compute expected FusedQKV total rows for GDN layout
         * Layout: [Q(n_k*d) | K(n_k*d) | V(n_v*d)]
         */
        size_t expectedGDNFusedQKVRows() const
        {
            return 2 * static_cast<size_t>(gdn_n_k_heads) * gdn_d_state +
                   static_cast<size_t>(gdn_n_v_heads) * gdn_d_state;
        }
    };

    /**
     * @brief Layer range for Pipeline Parallelism
     *
     * When set, restricts which layer weights are loaded.
     * Weights outside this range return nullptr from getWeightForDevice().
     */
    struct LayerRange
    {
        int first = 0;             ///< First layer index (inclusive)
        int last = -1;             ///< Last layer index (exclusive, -1 = all remaining)
        bool has_embedding = true; ///< True if this PP stage loads token embedding
        bool has_lm_head = true;   ///< True if this PP stage loads output norm + LM head
    };

    /**
     * @brief Immutable whole-expert assignment used while slicing MoE weights.
     *
     * This is deliberately the same policy coordinate consumed by graph
     * construction. Weight loading must never invent an independent contiguous
     * rank rule, because that would let physical expert bytes disagree with the
     * graph's router ownership mask.
     */
    struct RoutedExpertWeightAssignment
    {
        RoutedExpertOwnerOrder owner_order = RoutedExpertOwnerOrder::Ordinal;
        int participant_index = 0;
        int participant_count = 1;

        [[nodiscard]] bool valid() const noexcept
        {
            return participant_count > 0 &&
                   participant_index >= 0 &&
                   participant_index < participant_count;
        }
    };

    // WeightPreprocessor is defined in WeightTypes.h

    /**
     * @brief Immutable configuration for WeightManager
     *
     * Replaces multi-phase setter initialization. All needed configuration
     * is provided upfront. Callers construct this struct once, then pass it
     * to the WeightManager constructor.
     *
     * Example:
     * @code
     * WeightManagerConfig config;
     * config.sharding = SchemaFactoryRegistry::getWeightShardingConfig(arch);
     * config.dimensions.n_heads = 32;
     * config.dimensions.n_kv_heads = 8;
     * config.dimensions.head_dim = 128;
     * config.strategy = WeightDistributionStrategy::SHARDED;
     * auto wm = std::make_shared<WeightManager>(loader, mpi_ctx, config);
     * @endcode
     */
    struct WeightManagerConfig
    {
        /// Model-specific weight sharding patterns (from SchemaFactory)
        WeightShardingConfig sharding;

        /// Model head/dimension information (for FusedQKV slicing)
        ModelDimensions dimensions;

        /// Tensor parallel configuration for LOCAL TP device-aware slicing (optional)
        std::shared_ptr<TensorParallelConfig> tp_config;

        /// Distribution strategy
        WeightDistributionStrategy strategy = WeightDistributionStrategy::REPLICATED;

        /// Weight loading precision
        WeightPrecision precision = WeightPrecision::NATIVE;

        /// Pipeline parallelism layer range (optional)
        std::optional<LayerRange> layer_range;

        /// Exact static routed-expert owner coordinates for physical slicing.
        RoutedExpertWeightAssignment routed_expert_assignment;

        /// Optional per-weight transform before GEMM packing
        WeightPreprocessor preprocessor;

        /**
         * @brief Check if sharding config has been set (non-empty patterns)
         */
        bool hasShardingConfig() const { return !sharding.patterns.empty() || !sharding.exact_matches.empty(); }

        /**
         * @brief Check if model dimensions are valid for FusedQKV slicing
         */
        bool hasModelDimensions() const { return dimensions.isValid(); }

        /**
         * @brief Check if GDN dimensions are configured
         */
        bool hasGDNDimensions() const { return dimensions.hasGDN(); }

        /**
         * @brief Check if tensor parallel config is set
         */
        bool hasTensorParallelConfig() const { return tp_config != nullptr; }

        /**
         * @brief Check if a layer range is configured
         */
        bool hasLayerRange() const { return layer_range.has_value(); }
    };

} // namespace llaminar2
