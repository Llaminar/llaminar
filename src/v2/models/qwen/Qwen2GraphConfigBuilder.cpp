/**
 * @file Qwen2GraphConfigBuilder.cpp
 * @brief Implementation of Qwen2GraphConfigBuilder
 * @author David Sanftenberg
 * @date January 2026
 */

#include "Qwen2GraphConfigBuilder.h"
#include "Qwen2Graph.h" // For Qwen2GraphConfig definition
#include "../../loaders/WeightManager.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace llaminar2
{

    // =========================================================================
    // Factory Functions
    // =========================================================================

    std::unique_ptr<IGraphConfigBuilder> createQwen2GraphConfigBuilder()
    {
        return std::make_unique<Qwen2GraphConfigBuilder>();
    }

    std::unique_ptr<IGraphConfigBuilder> createGraphConfigBuilder(const std::string &model_type)
    {
        // Currently only Qwen2 is supported
        if (model_type == "qwen2" || model_type == "Qwen2" ||
            model_type == "qwen3" || model_type == "Qwen3")
        {
            return createQwen2GraphConfigBuilder();
        }
        return nullptr;
    }

    // =========================================================================
    // Qwen2GraphConfigBuilder Implementation
    // =========================================================================

    GraphConfigBuildResult Qwen2GraphConfigBuilder::buildConfig(
        const RankExecutionPlan &plan,
        const ModelConfig &model_config,
        IWeightManager & /*weight_manager*/)
    {
        GraphConfigBuildResult result;

        // Create placement strategy
        auto placement = createPlacement(plan, model_config);
        if (!placement)
        {
            result.success = false;
            result.error = "Failed to create layer device placement";
            return result;
        }

        // Extract model info (mapping ModelConfig field names to GraphConfigBuildResult names)
        result.model_info.n_layers = model_config.n_layers;
        result.model_info.n_heads = model_config.n_heads;
        result.model_info.n_kv_heads = model_config.n_kv_heads;
        result.model_info.d_model = model_config.hidden_size;    // ModelConfig uses hidden_size
        result.model_info.d_ff = model_config.intermediate_size; // ModelConfig uses intermediate_size
        result.model_info.vocab_size = model_config.vocab_size;
        result.model_info.head_dim = model_config.head_dim;

        // Extract execution info from plan
        result.execution_info.first_layer = plan.first_layer;
        result.execution_info.last_layer = plan.last_layer;
        result.execution_info.has_embedding = plan.has_embedding;
        result.execution_info.has_lm_head = plan.has_lm_head;
        result.execution_info.shard_index = plan.weight_shard.shard_index;
        result.execution_info.total_shards = plan.weight_shard.total_shards;

        // Calculate local heads using placement
        result.execution_info.local_heads =
            placement->headsForDevice(plan.primary_device);
        result.execution_info.local_kv_heads =
            placement->kvHeadsForDevice(plan.primary_device);

        result.placement = std::move(placement);
        result.success = true;
        return result;
    }

    bool Qwen2GraphConfigBuilder::buildQwen2Config(
        const RankExecutionPlan &plan,
        const ModelConfig &model_config,
        IWeightManager & /*weight_manager*/,
        Qwen2GraphConfig &config)
    {
        // Create placement strategy
        auto placement = createPlacement(plan, model_config);
        if (!placement)
        {
            return false;
        }

        // Basic model architecture
        // Note: ModelConfig uses hidden_size/intermediate_size,
        // Qwen2GraphConfig uses d_model/d_ff
        config.n_layers = model_config.n_layers;
        config.d_model = model_config.hidden_size;
        config.n_heads = model_config.n_heads;
        config.n_kv_heads = model_config.n_kv_heads;
        config.head_dim = model_config.head_dim;
        config.d_ff = model_config.intermediate_size;
        config.vocab_size = model_config.vocab_size;

        // Precision settings: use defaults from Qwen2GraphConfig
        // (ModelConfig doesn't carry rms_norm_eps/rope_theta)
        // config.rms_norm_eps = 1e-6f;  // default
        // config.rope_theta = 10000.0f; // default

        // Device settings
        config.default_device = plan.primary_device.toLocalDeviceId();

        // Configure TP for attention
        configureAttentionTP(config, plan, model_config, *placement);

        // Configure TP for FFN
        configureFFNTP(config, plan, model_config);

        // Configure TP for LM head
        configureLMHeadTP(config, plan, model_config);

        // Configure pipeline parallelism
        configurePipelineParallel(config, plan, model_config);

        return true;
    }

    std::unique_ptr<LayerDevicePlacement> Qwen2GraphConfigBuilder::createPlacement(
        const RankExecutionPlan &plan,
        const ModelConfig &model_config)
    {
        return LayerDevicePlacement::fromExecutionPlan(
            plan,
            model_config.n_layers,
            model_config.n_heads,
            model_config.n_kv_heads);
    }

    void Qwen2GraphConfigBuilder::configureAttentionTP(
        Qwen2GraphConfig &config,
        const RankExecutionPlan &plan,
        const ModelConfig &model_config,
        const LayerDevicePlacement &placement)
    {
        // Get shard info
        int shard_index = plan.weight_shard.shard_index;
        int total_shards = plan.weight_shard.total_shards;

        if (total_shards <= 1)
        {
            // No TP - use full heads
            config.local_n_heads = model_config.n_heads;
            config.local_n_kv_heads = model_config.n_kv_heads;
            config.head_start = 0;
            config.qkv_column_parallel = false;
            return;
        }

        // Get weights for proportional distribution
        std::vector<float> weights;
        if (plan.usesLocalTP())
        {
            weights = plan.local_tp_weights;
        }

        // Calculate local heads from placement
        config.local_n_heads = placement.headsForDevice(plan.primary_device);
        config.local_n_kv_heads = placement.kvHeadsForDevice(plan.primary_device);
        config.head_start = computeHeadStart(
            model_config.n_heads, shard_index, total_shards, weights);
        config.qkv_column_parallel = true;
    }

    void Qwen2GraphConfigBuilder::configureFFNTP(
        Qwen2GraphConfig &config,
        const RankExecutionPlan &plan,
        const ModelConfig &model_config)
    {
        int total_shards = plan.weight_shard.total_shards;

        if (total_shards <= 1)
        {
            // No TP
            config.d_ff_local = model_config.intermediate_size;
            config.ffn_column_parallel = false;
            return;
        }

        // Proportional FFN distribution
        if (plan.usesLocalTP() && !plan.local_tp_weights.empty())
        {
            float my_weight = plan.local_tp_weights[plan.weight_shard.shard_index];
            config.d_ff_local = static_cast<int>(
                std::round(model_config.intermediate_size * my_weight));
        }
        else
        {
            // Equal distribution
            config.d_ff_local = model_config.intermediate_size / total_shards;
        }

        // Round to alignment (typically 128 for SIMD)
        constexpr int FFN_ALIGNMENT = 128;
        config.d_ff_local = ((config.d_ff_local + FFN_ALIGNMENT - 1) / FFN_ALIGNMENT) * FFN_ALIGNMENT;

        config.ffn_column_parallel = true;
    }

    void Qwen2GraphConfigBuilder::configureLMHeadTP(
        Qwen2GraphConfig &config,
        const RankExecutionPlan &plan,
        const ModelConfig &model_config)
    {
        int total_shards = plan.weight_shard.total_shards;

        if (total_shards <= 1)
        {
            // No TP
            config.vocab_local = model_config.vocab_size;
            config.lm_head_column_parallel = false;
            return;
        }

        // Proportional vocab distribution
        if (plan.usesLocalTP() && !plan.local_tp_weights.empty())
        {
            float my_weight = plan.local_tp_weights[plan.weight_shard.shard_index];
            config.vocab_local = static_cast<int>(
                std::round(model_config.vocab_size * my_weight));
        }
        else
        {
            // Equal distribution
            config.vocab_local = model_config.vocab_size / total_shards;
        }

        config.lm_head_column_parallel = true;
    }

    void Qwen2GraphConfigBuilder::configurePipelineParallel(
        Qwen2GraphConfig & /*config*/,
        const RankExecutionPlan & /*plan*/,
        const ModelConfig & /*model_config*/)
    {
        // Pipeline parallelism doesn't directly modify Qwen2GraphConfig fields.
        // The layer building decisions (which layers, embedding, lm_head) are
        // determined by LayerDevicePlacement::shouldBuildLayer(),
        // shouldBuildEmbedding(), and shouldBuildLMHead().
        //
        // The RankExecutionPlan.has_embedding and has_lm_head flags are used
        // when creating the LayerDevicePlacement, not when configuring the
        // graph config.
        //
        // If future PP features need config changes (e.g., async pipeline
        // execution settings), they would be configured here.
    }

    int Qwen2GraphConfigBuilder::computeHeadStart(
        int total_heads,
        int shard_index,
        int total_shards,
        const std::vector<float> &weights)
    {
        if (total_shards <= 1 || shard_index == 0)
        {
            return 0;
        }

        if (!weights.empty() && static_cast<int>(weights.size()) >= total_shards)
        {
            // Proportional distribution
            int start = 0;
            for (int i = 0; i < shard_index; ++i)
            {
                start += computeLocalHeads(total_heads, i, total_shards, weights);
            }
            return start;
        }

        // Equal distribution
        int heads_per_shard = total_heads / total_shards;
        return shard_index * heads_per_shard;
    }

    int Qwen2GraphConfigBuilder::computeLocalHeads(
        int total_heads,
        int shard_index,
        int total_shards,
        const std::vector<float> &weights)
    {
        if (total_shards <= 1)
        {
            return total_heads;
        }

        if (!weights.empty() && static_cast<int>(weights.size()) >= total_shards)
        {
            // Proportional distribution based on weights
            float total_weight = std::accumulate(weights.begin(),
                                                 weights.begin() + total_shards, 0.0f);

            if (shard_index == total_shards - 1)
            {
                // Last shard gets remainder
                int used = 0;
                for (int i = 0; i < total_shards - 1; ++i)
                {
                    used += static_cast<int>(
                        std::round(total_heads * weights[i] / total_weight));
                }
                return total_heads - used;
            }

            return static_cast<int>(
                std::round(total_heads * weights[shard_index] / total_weight));
        }

        // Equal distribution
        if (shard_index == total_shards - 1)
        {
            // Last shard gets remainder
            int used = (total_heads / total_shards) * (total_shards - 1);
            return total_heads - used;
        }

        return total_heads / total_shards;
    }

} // namespace llaminar2
