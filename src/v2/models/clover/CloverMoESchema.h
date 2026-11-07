/**
 * @file CloverMoESchema.h
 * @brief Declarative Clover autoregressive stage, sharding and snapshot contracts.
 *
 * Reuse the identical causal hybrid GDN/full-attention and normalized softmax
 * MoE operations, not a renamed Qwen checkpoint. CloverMoEConfig validates the
 * exact conditional attention/shared-expert tensor inventory before admission.
 * The artifact's own chat template remains authoritative; diffusion operands
 * and learned nextn heads are not part of this schema.
 */
#pragma once
#include "../qwen35moe/Qwen35MoESchema.h"

namespace llaminar2
{
    /** @brief First-class causal Clover schema sharing qualified native operators. */
    class CloverMoESchemaFactory final : public Qwen35MoESchemaFactory
    {
    public:
        /** @return The exact GGUF architecture, independent of its Qwen ancestry. */
        std::string architectureName() const override { return "clover_moe"; }

        /** @return Shared causal stage/layout declarations with Clover source identity. */
        GraphSchema createSchema() const override
        {
            auto schema = Qwen35MoESchemaFactory::createSchema();
            schema.name = architectureName();
            return schema;
        }

        /**
         * @brief Require the routed expert triplet and router on every Clover block.
         * @param name Canonical GGUF weight name or layer suffix.
         * @return Conditional presence only for roles validated by CloverMoEConfig.
         */
        bool isWeightOptional(const std::string &name) const override
        {
            for (const auto suffix : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_inp.weight"})
                if (name == suffix || name.ends_with(std::string(".") + suffix)) return false;
            return Qwen35MoESchemaFactory::isWeightOptional(name);
        }
    };
}
