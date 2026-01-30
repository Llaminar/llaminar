/**
 * @file IPartialWeightLoader.h
 * @brief Interface for loading partial model weights based on execution plan
 *
 * Part of Phase 3: Pipeline Parallelism Integration
 *
 * In pipeline parallelism, each rank only needs weights for its assigned layers.
 * This interface enables memory-efficient partial weight loading, reducing
 * memory usage by ~50% for 2-stage PP.
 *
 * Usage:
 *   auto loader = createPartialWeightLoader(model_config);
 *   auto loaded_names = loader->loadWeightsForPlan(weight_manager, plan, model_path);
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../execution/mpi_orchestration/RankExecutionPlan.h"
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class WeightManager;
    struct ModelConfig;

    /**
     * @brief Information about weights that will be loaded
     */
    struct PartialWeightInfo
    {
        std::vector<std::string> weight_names; ///< List of weight tensor names
        size_t estimated_bytes = 0;            ///< Estimated total memory in bytes
        int layer_count = 0;                   ///< Number of layers
        bool has_embedding = false;            ///< Whether embedding is included
        bool has_lm_head = false;              ///< Whether LM head is included
    };

    /**
     * @brief Interface for partial weight loading based on execution plan
     *
     * The partial weight loader analyzes the execution plan to determine
     * exactly which weights need to be loaded for this rank's pipeline stage.
     *
     * Weight naming follows the GGUF convention:
     * - Embedding: "token_embd.weight"
     * - Per-layer: "blk.{layer}.{component}.weight"
     * - Output norm: "output_norm.weight"
     * - LM head: "output.weight"
     */
    class IPartialWeightLoader
    {
    public:
        virtual ~IPartialWeightLoader() = default;

        // =====================================================================
        // Weight Loading
        // =====================================================================

        /**
         * @brief Load only weights needed for the execution plan
         *
         * Loads weights into the weight manager for the layer range specified
         * in the execution plan. This includes:
         * - Embedding (if plan.has_embedding)
         * - Layer weights for [plan.first_layer, plan.last_layer]
         * - Output norm and LM head (if plan.has_lm_head)
         *
         * @param weight_manager The weight manager to load weights into
         * @param plan The execution plan specifying which layers this rank owns
         * @param model_path Path to the model file
         * @return List of weight names that were loaded
         */
        virtual std::vector<std::string> loadWeightsForPlan(
            WeightManager &weight_manager,
            const RankExecutionPlan &plan,
            const std::string &model_path) = 0;

        // =====================================================================
        // Weight Analysis
        // =====================================================================

        /**
         * @brief Get list of weight names needed for a layer range
         *
         * Returns the full list of GGUF weight tensor names required for
         * the specified layer range, optionally including embedding and LM head.
         *
         * @param first_layer First layer index (0-based)
         * @param last_layer Last layer index (inclusive)
         * @param include_embedding Whether to include token embedding weight
         * @param include_lm_head Whether to include output norm and LM head weights
         * @return Vector of weight tensor names
         */
        virtual std::vector<std::string> weightsForLayerRange(
            int first_layer,
            int last_layer,
            bool include_embedding,
            bool include_lm_head) const = 0;

        /**
         * @brief Get weight info for an execution plan
         *
         * Returns detailed information about what weights would be loaded
         * for the given plan, without actually loading them.
         *
         * @param plan The execution plan to analyze
         * @return PartialWeightInfo with weight names and estimates
         */
        virtual PartialWeightInfo getWeightInfoForPlan(
            const RankExecutionPlan &plan) const = 0;

        /**
         * @brief Estimate memory needed for partial weights
         *
         * Provides a rough estimate of memory required to load the weights
         * specified in the execution plan. Useful for capacity planning.
         *
         * @param plan The execution plan to estimate
         * @param model_path Path to model for metadata lookup (optional)
         * @return Estimated bytes needed
         */
        virtual size_t estimateMemoryForPlan(
            const RankExecutionPlan &plan,
            const std::string &model_path = "") const = 0;

        // =====================================================================
        // Weight Categories
        // =====================================================================

        /**
         * @brief Get attention weight names for a layer
         *
         * Returns: attn_q, attn_k, attn_v, attn_output, attn_norm
         *
         * @param layer_idx Layer index
         * @return Vector of attention weight names
         */
        virtual std::vector<std::string> attentionWeightsForLayer(int layer_idx) const = 0;

        /**
         * @brief Get FFN weight names for a layer
         *
         * Returns: ffn_gate, ffn_up, ffn_down, ffn_norm
         *
         * @param layer_idx Layer index
         * @return Vector of FFN weight names
         */
        virtual std::vector<std::string> ffnWeightsForLayer(int layer_idx) const = 0;

        /**
         * @brief Get all weight names for a single layer
         *
         * Combined attention + FFN weights for a layer.
         *
         * @param layer_idx Layer index
         * @return Vector of all weight names for the layer
         */
        virtual std::vector<std::string> allWeightsForLayer(int layer_idx) const = 0;
    };

    // =========================================================================
    // Factory
    // =========================================================================

    /**
     * @brief Create a partial weight loader
     *
     * @param model_config Model configuration (for layer count, etc.)
     * @return Unique pointer to loader instance
     */
    std::unique_ptr<IPartialWeightLoader> createPartialWeightLoader(
        const ModelConfig *model_config = nullptr);

} // namespace llaminar2
