/**
 * @file Qwen2GraphConfigBuilder.h
 * @brief Qwen2-specific graph configuration builder
 *
 * Translates RankExecutionPlan into Qwen2GraphConfig for graph building.
 * Handles all Qwen2-specific configuration including:
 * - Head distribution for TP
 * - FFN dimension sharding
 * - Layer ranges for PP
 * - Activation precision settings
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../IGraphConfigBuilder.h"
#include "../../execution/local_execution/device/LayerDevicePlacement.h"
#include <memory>

namespace llaminar2
{

    // Forward declaration to avoid include chain that triggers
    // duplicate CollectiveBackendType definition.
    // Full definition included in Qwen2GraphConfigBuilder.cpp.
    struct Qwen2GraphConfig;

    /**
     * @brief Qwen2-specific implementation of IGraphConfigBuilder
     *
     * Creates Qwen2GraphConfig from RankExecutionPlan, properly configuring:
     * - Tensor parallelism (head sharding, FFN sharding)
     * - Pipeline parallelism (layer ranges)
     * - Device placement
     * - Precision settings
     */
    class Qwen2GraphConfigBuilder : public IGraphConfigBuilder
    {
    public:
        Qwen2GraphConfigBuilder() = default;
        ~Qwen2GraphConfigBuilder() override = default;

        /**
         * @brief Build generic graph configuration
         */
        GraphConfigBuildResult buildConfig(
            const RankExecutionPlan &plan,
            const ModelConfig &model_config,
            IWeightManager &weight_manager) override;

        /**
         * @brief Build Qwen2-specific configuration
         */
        bool buildQwen2Config(
            const RankExecutionPlan &plan,
            const ModelConfig &model_config,
            IWeightManager &weight_manager,
            Qwen2GraphConfig &config) override;

    private:
        /**
         * @brief Create appropriate LayerDevicePlacement from plan
         */
        std::unique_ptr<LayerDevicePlacement> createPlacement(
            const RankExecutionPlan &plan,
            const ModelConfig &model_config);

        /**
         * @brief Configure attention TP parameters
         */
        void configureAttentionTP(
            Qwen2GraphConfig &config,
            const RankExecutionPlan &plan,
            const ModelConfig &model_config,
            const LayerDevicePlacement &placement);

        /**
         * @brief Configure FFN TP parameters
         */
        void configureFFNTP(
            Qwen2GraphConfig &config,
            const RankExecutionPlan &plan,
            const ModelConfig &model_config);

        /**
         * @brief Configure LM head TP parameters
         */
        void configureLMHeadTP(
            Qwen2GraphConfig &config,
            const RankExecutionPlan &plan,
            const ModelConfig &model_config);

        /**
         * @brief Configure pipeline parallelism parameters
         */
        void configurePipelineParallel(
            Qwen2GraphConfig &config,
            const RankExecutionPlan &plan,
            const ModelConfig &model_config);

        /**
         * @brief Compute head start index for this shard
         */
        int computeHeadStart(
            int total_heads,
            int shard_index,
            int total_shards,
            const std::vector<float> &weights);

        /**
         * @brief Compute local head count for this shard
         */
        int computeLocalHeads(
            int total_heads,
            int shard_index,
            int total_shards,
            const std::vector<float> &weights);
    };

} // namespace llaminar2
