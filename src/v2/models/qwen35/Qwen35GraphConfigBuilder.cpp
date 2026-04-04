/**
 * @file Qwen35GraphConfigBuilder.cpp
 * @brief Implementation of Qwen35GraphConfigBuilder
 */

#include "Qwen35GraphConfigBuilder.h"
#include "../GraphTypes.h"
#include "../../interfaces/IModelContext.h"
#include "../../loaders/IModelLoader.h"
#include "../../utils/Logger.h"

namespace llaminar2
{

    bool Qwen35GraphConfigBuilder::populateFromModelContext(
        IModelContext &ctx,
        GraphConfig &config)
    {
        // Base Qwen2 fields (n_layers, d_model, n_heads, etc.)
        if (!Qwen2GraphConfigBuilder::populateFromModelContext(ctx, config))
        {
            return false;
        }

        auto loader = ctx.loader();
        if (!loader)
        {
            LOG_ERROR("[Qwen35GraphConfigBuilder] No loader available");
            return false;
        }

        const std::string arch = ctx.architecture();

        // =====================================================================
        // GDN / SSM metadata
        // =====================================================================
        config.gdn_conv_kernel_size = loader->getInt(arch + ".ssm.conv_kernel", 0);
        config.gdn_state_size = loader->getInt(arch + ".ssm.state_size", 0);
        config.gdn_inner_size = loader->getInt(arch + ".ssm.inner_size", 0);
        config.gdn_group_count = loader->getInt(arch + ".ssm.group_count", 0);
        config.gdn_time_step_rank = loader->getInt(arch + ".ssm.time_step_rank", 0);

        LOG_DEBUG("[Qwen35GraphConfigBuilder] GDN config:"
                  << " conv_kernel=" << config.gdn_conv_kernel_size
                  << " state_size=" << config.gdn_state_size
                  << " inner_size=" << config.gdn_inner_size
                  << " group_count=" << config.gdn_group_count
                  << " time_step_rank=" << config.gdn_time_step_rank);

        // =====================================================================
        // Hybrid layer types (full_attention_interval)
        // =====================================================================
        config.full_attention_interval = loader->getInt(
            arch + ".full_attention_interval", 0);

        if (config.full_attention_interval > 0)
        {
            const int n_layers = config.n_layers;
            config.layer_types.resize(n_layers);

            int fa_count = 0;
            int gdn_count = 0;
            for (int i = 0; i < n_layers; ++i)
            {
                // Pattern: every Nth layer is full attention (1-indexed check)
                // Layer indices 3, 7, 11, ... are FA when interval=4
                if ((i + 1) % config.full_attention_interval == 0)
                {
                    config.layer_types[i] = "full_attention";
                    ++fa_count;
                }
                else
                {
                    config.layer_types[i] = "gdn";
                    ++gdn_count;
                }
            }

            LOG_INFO("[Qwen35GraphConfigBuilder] Hybrid architecture:"
                     << " " << gdn_count << " GDN layers + "
                     << fa_count << " FA layers"
                     << " (interval=" << config.full_attention_interval << ")");
        }

        // =====================================================================
        // Attention output gate (always present in Qwen3.5)
        // =====================================================================
        config.has_attention_output_gate = true;

        // =====================================================================
        // Partial RoPE for FA layers
        // =====================================================================
        // Qwen3.5 uses partial RoPE: rope.dimension_count / head_dim
        const int rope_dim_count = loader->getInt(arch + ".rope.dimension_count", 0);
        if (rope_dim_count > 0 && config.head_dim > 0)
        {
            config.partial_rotary_factor =
                static_cast<float>(rope_dim_count) / static_cast<float>(config.head_dim);
            LOG_DEBUG("[Qwen35GraphConfigBuilder] Partial RoPE: "
                      << rope_dim_count << "/" << config.head_dim
                      << " = " << config.partial_rotary_factor);
        }

        // =====================================================================
        // Per-layer head dimensions (FA: head_dim from GGUF, GDN: state_size)
        // =====================================================================
        if (!config.layer_types.empty() && config.gdn_state_size > 0)
        {
            const int n_layers = config.n_layers;
            config.layer_head_dim.resize(n_layers);
            for (int i = 0; i < n_layers; ++i)
            {
                if (config.layer_types[i] == "full_attention")
                {
                    config.layer_head_dim[i] = config.head_dim; // FA head_dim (e.g. 256)
                }
                else
                {
                    config.layer_head_dim[i] = config.gdn_state_size; // GDN state dim (e.g. 128)
                }
            }
        }

        return true;
    }

    ModelWeights Qwen35GraphConfigBuilder::buildWeights(WeightAccessor get_weight)
    {
        ModelWeights weights;

        // Global weights
        auto embedding = get_weight("token_embd.weight");
        auto final_norm = get_weight("output_norm.weight");
        auto lm_head = get_weight("output.weight");

        // Tied embeddings: if output.weight is missing, reuse token_embd.weight
        if (!lm_head && embedding)
        {
            LOG_INFO("[Qwen35GraphConfigBuilder] output.weight not found, using tied embeddings");
            lm_head = embedding;
        }

        weights.embedding_table = embedding.get();
        weights.final_norm = final_norm.get();
        weights.lm_head = lm_head.get();

        // Per-layer weight accessor — dispatches based on which weights exist
        // (GDN layers have attn_qkv, FA layers have attn_q/k/v)
        weights.get_layer_weights = [get_weight](int layer_idx) -> LayerWeights
        {
            LayerWeights layer;
            std::string prefix = "blk." + std::to_string(layer_idx) + ".";

            // ==============================================================
            // Shared weights (present on ALL layers)
            // ==============================================================
            layer.attn_norm = get_weight(prefix + "attn_norm.weight").get();
            layer.ffn_norm = get_weight(prefix + "post_attention_norm.weight").get();
            layer.gate_proj = get_weight(prefix + "ffn_gate.weight").get();
            layer.up_proj = get_weight(prefix + "ffn_up.weight").get();
            layer.down_proj = get_weight(prefix + "ffn_down.weight").get();

            // Attention output gate (present on both GDN and FA layers)
            auto gate = get_weight(prefix + "attn_gate.weight");
            layer.attn_gate = gate ? gate.get() : nullptr;

            // ==============================================================
            // Detect layer type by probing which weights exist
            // ==============================================================
            auto attn_qkv = get_weight(prefix + "attn_qkv.weight");

            if (attn_qkv)
            {
                // GDN layer — load GDN-specific weights
                layer.attn_qkv = attn_qkv.get();

                auto ssm_alpha = get_weight(prefix + "ssm_alpha.weight");
                auto ssm_beta = get_weight(prefix + "ssm_beta.weight");
                auto ssm_conv1d = get_weight(prefix + "ssm_conv1d.weight");
                auto ssm_dt_bias = get_weight(prefix + "ssm_dt.bias");
                auto ssm_a = get_weight(prefix + "ssm_a");
                auto ssm_norm = get_weight(prefix + "ssm_norm.weight");
                auto ssm_out = get_weight(prefix + "ssm_out.weight");

                layer.ssm_alpha = ssm_alpha ? ssm_alpha.get() : nullptr;
                layer.ssm_beta = ssm_beta ? ssm_beta.get() : nullptr;
                layer.ssm_conv1d = ssm_conv1d ? ssm_conv1d.get() : nullptr;
                layer.ssm_dt_bias = ssm_dt_bias ? ssm_dt_bias.get() : nullptr;
                layer.ssm_a = ssm_a ? ssm_a.get() : nullptr;
                layer.ssm_norm = ssm_norm ? ssm_norm.get() : nullptr;
                layer.ssm_out = ssm_out ? ssm_out.get() : nullptr;
            }
            else
            {
                // Full attention layer — load FA-specific weights
                layer.wq = get_weight(prefix + "attn_q.weight").get();
                layer.wk = get_weight(prefix + "attn_k.weight").get();
                layer.wv = get_weight(prefix + "attn_v.weight").get();
                layer.wo = get_weight(prefix + "attn_output.weight").get();

                // QK norms (Qwen3.5 FA layers have per-head norms)
                auto q_norm = get_weight(prefix + "attn_q_norm.weight");
                auto k_norm = get_weight(prefix + "attn_k_norm.weight");
                layer.q_norm = q_norm ? q_norm.get() : nullptr;
                layer.k_norm = k_norm ? k_norm.get() : nullptr;
            }

            return layer;
        };

        return weights;
    }

} // namespace llaminar2
