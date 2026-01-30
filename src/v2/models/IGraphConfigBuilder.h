/**
 * @file IGraphConfigBuilder.h
 * @brief Interface for building graph configuration from execution plans
 *
 * IGraphConfigBuilder bridges the gap between cluster orchestration (Layer 2)
 * and graph building (Layer 3). It takes a RankExecutionPlan and produces
 * model-specific graph configuration.
 *
 * Key Design:
 * - Interface enables mocking for unit tests
 * - Factory function returns model-specific implementation
 * - Works with any model architecture (Qwen2, Llama, etc.)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../execution/mpi_orchestration/RankExecutionPlan.h"
#include "../execution/local_execution/device/LayerDevicePlacement.h"
#include "../execution/mpi_orchestration/IExecutionPlanBuilder.h" // For ModelConfig
#include "../loaders/IWeightManager.h"                            // For IWeightManager interface
#include <memory>

namespace llaminar2
{

    // Forward declarations - model-specific configs
    struct Qwen2GraphConfig;
    class IKVCache;

    /**
     * @brief Result of building graph configuration
     *
     * Contains the built configuration along with any derived information
     * that may be needed during graph building.
     */
    struct GraphConfigBuildResult
    {
        /// Successfully built configuration
        bool success = false;

        /// Error message if failed
        std::string error;

        /// Layer device placement (owned by result)
        std::shared_ptr<LayerDevicePlacement> placement;

        /// Model configuration extracted/validated
        struct ModelInfo
        {
            int n_layers = 0;
            int n_heads = 0;
            int n_kv_heads = 0;
            int d_model = 0;
            int d_ff = 0;
            int vocab_size = 0;
            int head_dim = 0;
        };
        ModelInfo model_info;

        /// Execution info for this rank
        struct ExecutionInfo
        {
            int first_layer = 0;
            int last_layer = -1;
            bool has_embedding = true;
            bool has_lm_head = true;
            int local_heads = 0;
            int local_kv_heads = 0;
            int shard_index = 0;
            int total_shards = 1;
        };
        ExecutionInfo execution_info;
    };

    /**
     * @brief Interface for building graph configuration from execution plan
     *
     * Model-specific implementations translate RankExecutionPlan into
     * the configuration structure needed by the model's graph builder.
     */
    class IGraphConfigBuilder
    {
    public:
        virtual ~IGraphConfigBuilder() = default;

        /**
         * @brief Build graph configuration from execution plan
         *
         * Takes the rank-specific execution plan and model configuration,
         * producing a fully populated graph configuration.
         *
         * @param plan Execution plan for this rank
         * @param model_config Model architecture parameters
         * @param weight_manager Weight manager for accessing model weights
         * @return Build result with configuration or error
         */
        virtual GraphConfigBuildResult buildConfig(
            const RankExecutionPlan &plan,
            const ModelConfig &model_config,
            IWeightManager &weight_manager) = 0;

        /**
         * @brief Build Qwen2-specific graph configuration
         *
         * Convenience method that returns the Qwen2GraphConfig directly.
         * Only valid for Qwen2-based models.
         *
         * @param plan Execution plan for this rank
         * @param model_config Model architecture parameters
         * @param weight_manager Weight manager for accessing model weights
         * @param[out] config Output configuration to populate
         * @return true if successful
         */
        virtual bool buildQwen2Config(
            const RankExecutionPlan &plan,
            const ModelConfig &model_config,
            IWeightManager &weight_manager,
            Qwen2GraphConfig &config) = 0;
    };

    // =========================================================================
    // Factory Functions
    // =========================================================================

    /**
     * @brief Create Qwen2-specific graph config builder
     * @return Unique pointer to builder instance
     */
    std::unique_ptr<IGraphConfigBuilder> createQwen2GraphConfigBuilder();

    /**
     * @brief Create graph config builder for a model type
     *
     * Factory that returns the appropriate builder based on model type.
     *
     * @param model_type Model type string (e.g., "qwen2", "llama")
     * @return Unique pointer to builder instance, or nullptr if unsupported
     */
    std::unique_ptr<IGraphConfigBuilder> createGraphConfigBuilder(
        const std::string &model_type);

} // namespace llaminar2
