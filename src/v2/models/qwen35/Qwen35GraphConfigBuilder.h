/**
 * @file Qwen35GraphConfigBuilder.h
 * @brief Qwen3.5-specific graph configuration builder
 *
 * Extends QwenStandardGraphConfigBuilder with:
 * - GDN-specific GGUF metadata parsing (ssm.*, full_attention_interval)
 * - Heterogeneous layer type assignment (GDN vs FA)
 * - GDN weight loading (attn_qkv, attn_gate, ssm_*)
 * - Per-layer head dimensions (FA=256, GDN=128)
 * - Partial RoPE factor for FA layers
 *
 * Prompt and sampling defaults are resolved by ModelGenerationPolicy using
 * the loaded revision's metadata, separately from this shared tensor layout.
 */

#pragma once

#include "../qwen/QwenStandardGraphConfigBuilder.h"

#include <string>

namespace llaminar2
{

    /**
     * @brief Qwen3.5-specific implementation of IGraphConfigBuilder
     *
     * Inherits all TP/PP/device configuration from QwenStandardGraphConfigBuilder.
     * Overrides only the methods that differ for Qwen3.5:
     * - populateFromModelContext: reads GDN/hybrid metadata
     * - buildWeights: handles per-layer GDN vs FA weights
     */
    class Qwen35GraphConfigBuilder : public QwenStandardGraphConfigBuilder
    {
    public:
        Qwen35GraphConfigBuilder() = default;
        ~Qwen35GraphConfigBuilder() override = default;

        /**
         * @brief Populate config from IModelContext with Qwen3.5-specific fields
         *
         * In addition to standard fields (n_layers, d_model, etc.), reads:
         * - ssm.conv_kernel → gdn.conv_kernel_size
         * - ssm.state_size → gdn.state_size
         * - ssm.inner_size → gdn.inner_size
         * - ssm.group_count → gdn.group_count
         * - ssm.time_step_rank → gdn.time_step_rank
         * - full_attention_interval → generates layer_types
         * - partial_rotary_factor from rope.dimension_count
         * - has_attention_output_gate = true
         */
        bool populateFromModelContext(
            IModelContext &ctx,
            GraphConfig &config) override;

        /**
         * @brief Build weights with Qwen3.5 per-layer dispatch
         *
         * GDN layers load: attn_qkv, attn_gate, ssm_alpha, ssm_beta,
         *                  ssm_conv1d, ssm_dt.bias, ssm_a, ssm_norm, ssm_out
         * FA layers load:  attn_q, attn_k, attn_v, attn_output,
         *                  attn_q_norm, attn_k_norm, attn_gate
         * Both load:       attn_norm, post_attention_norm (as ffn_norm), ffn_gate/up/down
         */
        ModelWeights buildWeights(WeightAccessor get_weight) override;

    };

} // namespace llaminar2
