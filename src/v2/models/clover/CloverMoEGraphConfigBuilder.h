/**
 * @file CloverMoEGraphConfigBuilder.h
 * @brief Bind Clover's validated causal source to the shared hybrid MoE graph.
 *
 * Architecture admission is separate from execution placement. Existing GDN,
 * gated full-attention and expert weight roles retain their original ownership
 * and prepared representation; no diffusion workspace or controller is created.
 */
#pragma once
#include "../qwen35moe/Qwen35MoEGraphConfigBuilder.h"

namespace llaminar2
{
    /** @brief Clover-specific source admission with reusable causal weight binding. */
    class CloverMoEGraphConfigBuilder final : public Qwen35MoEGraphConfigBuilder
    {
    public:
        /**
         * @brief Authenticate the full source before projecting participant-local geometry.
         * @param ctx Model context retaining the original complete GGUF directory.
         * @param config Existing execution policy, populated with Clover architecture fields.
         * @return True on complete initialization.
         * @throws std::invalid_argument For a missing directory or unsupported source geometry.
         */
        bool populateFromModelContext(IModelContext &ctx, GraphConfig &config) override;
    };
}
