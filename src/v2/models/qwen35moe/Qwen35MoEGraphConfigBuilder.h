/**
 * @file Qwen35MoEGraphConfigBuilder.h
 * @brief Qwen3.5 MoE-specific graph configuration builder
 *
 * Extends Qwen35GraphConfigBuilder with:
 * - MoE-specific GGUF metadata parsing (expert_count, expert_used_count)
 * - Expert weight loading
 * - Shared expert + router configuration
 */

#pragma once

#include "../qwen35/Qwen35GraphConfigBuilder.h"

namespace llaminar2
{

    /**
     * @brief Qwen3.5 MoE-specific implementation of IGraphConfigBuilder
     *
     * Inherits all GDN/FA configuration from Qwen35GraphConfigBuilder.
     * Overrides only the methods that differ for MoE:
     * - populateFromModelContext: reads expert count metadata
     * - buildWeights: handles MoE expert + shared expert + router weights
     */
    class Qwen35MoEGraphConfigBuilder : public Qwen35GraphConfigBuilder
    {
    public:
        Qwen35MoEGraphConfigBuilder() = default;
        ~Qwen35MoEGraphConfigBuilder() override = default;

        /**
         * @brief Populate config from IModelContext with MoE-specific fields
         *
         * In addition to Qwen3.5 fields, reads:
         * - expert_count → moe.n_experts
         * - expert_used_count → moe.n_experts_used
         * - expert_shared_count → moe.n_shared_experts
         */
        bool populateFromModelContext(
            IModelContext &ctx,
            GraphConfig &config) override;

        /**
         * @brief Build weights with MoE per-layer dispatch
         *
         * MoE layers load: ffn_gate_exps, ffn_up_exps, ffn_down_exps,
         *                  ffn_gate_inp (router), ffn_gate_shexp, ffn_up_shexp,
         *                  ffn_down_shexp, ffn_gate_inp_shexp (shared expert gate)
         */
        ModelWeights buildWeights(WeightAccessor get_weight) override;
    };

} // namespace llaminar2
