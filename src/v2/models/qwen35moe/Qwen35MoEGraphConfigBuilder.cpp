/**
 * @file Qwen35MoEGraphConfigBuilder.cpp
 * @brief Implementation of Qwen35MoEGraphConfigBuilder (stub)
 */

#include "Qwen35MoEGraphConfigBuilder.h"
#include "../GraphTypes.h"
#include "../../interfaces/IModelContext.h"
#include "../../loaders/IModelLoader.h"
#include "../../utils/Logger.h"

namespace llaminar2
{

    bool Qwen35MoEGraphConfigBuilder::populateFromModelContext(
        IModelContext &ctx,
        GraphConfig &config)
    {
        // Base Qwen3.5 fields (GDN, hybrid layer types, etc.)
        if (!Qwen35GraphConfigBuilder::populateFromModelContext(ctx, config))
        {
            return false;
        }

        auto loader = ctx.loader();
        if (!loader)
        {
            LOG_ERROR("[Qwen35MoEGraphConfigBuilder] No loader available");
            return false;
        }

        const std::string arch = ctx.architecture();

        // =====================================================================
        // MoE metadata
        // =====================================================================
        config.moe.num_experts = loader->getInt(arch + ".expert_count", 0);
        config.moe.top_k = loader->getInt(arch + ".expert_used_count", 0);
        int shared_count = loader->getInt(arch + ".expert_shared_count", 0);
        config.moe.has_shared_expert = (shared_count > 0);
        config.moe.norm_topk_prob = true; // Qwen3.5 MoE normalizes routing weights
        config.moe.shared_expert_gate = config.moe.has_shared_expert; // sigmoid gate on shared expert

        // Expert FFN intermediate dimension — read from GGUF or infer from weight shapes
        config.moe.intermediate_size = loader->getInt(arch + ".expert_feed_forward_length", 0);
        if (config.moe.intermediate_size == 0)
        {
            // Fallback: will be inferred from expert weight tensor shapes at graph build time
            config.moe.intermediate_size = loader->getInt(arch + ".feed_forward_length", 0);
        }
        config.moe.shared_intermediate_size = config.moe.intermediate_size; // same for Qwen3.5

        LOG_DEBUG("[Qwen35MoEGraphConfigBuilder] MoE config:"
                  << " num_experts=" << config.moe.num_experts
                  << " top_k=" << config.moe.top_k
                  << " has_shared_expert=" << config.moe.has_shared_expert);

        return true;
    }

    ModelWeights Qwen35MoEGraphConfigBuilder::buildWeights(WeightAccessor get_weight)
    {
        // Start with base Qwen3.5 weight building (handles GDN + FA attention weights)
        ModelWeights weights = Qwen35GraphConfigBuilder::buildWeights(get_weight);

        // Wrap the base layer accessor to also populate MoE fields
        auto base_accessor = weights.get_layer_weights;
        weights.get_layer_weights = [get_weight, base_accessor](int layer_idx) -> LayerWeights
        {
            // Get base weights (attention + dense FFN + GDN)
            LayerWeights layer = base_accessor(layer_idx);

            // Populate MoE weight fields for layers that have them
            std::string prefix = "blk." + std::to_string(layer_idx) + ".";

            auto gate_inp = get_weight(prefix + "ffn_gate_inp.weight");
            if (gate_inp)
            {
                // This is a MoE layer — populate expert weights
                layer.moe_gate = gate_inp.get();
                layer.moe_gate_exps = get_weight(prefix + "ffn_gate_exps.weight").get();
                layer.moe_up_exps = get_weight(prefix + "ffn_up_exps.weight").get();
                layer.moe_down_exps = get_weight(prefix + "ffn_down_exps.weight").get();

                // Shared expert weights
                auto shexp_gate = get_weight(prefix + "ffn_gate_shexp.weight");
                auto shexp_up = get_weight(prefix + "ffn_up_shexp.weight");
                auto shexp_down = get_weight(prefix + "ffn_down_shexp.weight");
                layer.shared_expert_gate = shexp_gate ? shexp_gate.get() : nullptr;
                layer.shared_expert_up = shexp_up ? shexp_up.get() : nullptr;
                layer.shared_expert_down = shexp_down ? shexp_down.get() : nullptr;

                // Shared expert sigmoid gate
                auto gate_inp_shexp = get_weight(prefix + "ffn_gate_inp_shexp.weight");
                layer.shared_expert_gate_inp = gate_inp_shexp ? gate_inp_shexp.get() : nullptr;
            }

            return layer;
        };

        return weights;
    }

} // namespace llaminar2
