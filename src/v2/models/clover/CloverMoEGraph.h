/**
 * @file CloverMoEGraph.h
 * @brief Participant-local causal Clover graph using the qualified hybrid MoE machinery.
 *
 * Clover's AR equations and converted tensor layout match the shared graph:
 * tiled-head GDN, gated partial-RoPE attention, normalized top-k SwiGLU experts,
 * and an optional sigmoid-gated shared expert. Reusing that graph retains its
 * native capture, transfer, prefix-state and expert-placement authorities.
 * This class does not implement block diffusion or self-speculative drafting.
 */
#pragma once
#include "../qwen35moe/Qwen35MoEGraph.h"

namespace llaminar2
{
    /** @brief Declare Clover identity without copying attention or MoE orchestration. */
    class CloverMoEGraph final : public Qwen35MoEGraph
    {
    public:
        /**
         * @brief Construct one participant from admitted architecture and placement.
         * @param config Clover geometry and the participant's frozen execution policy.
         * @param mpi Existing communicator owner; no additional coordination is created.
         */
        CloverMoEGraph(const GraphConfig &config, std::shared_ptr<IMPIContext> mpi)
            : Qwen35MoEGraph(config, std::move(mpi)) {}

        /** @return Exact source architecture for graph identity and diagnostics. */
        std::string architectureName() const override { return "clover_moe"; }

        /** @return The complete inherited native schema, including resolved projection placement. */
        GraphSchema getSchema() const override
        {
            auto schema = Qwen35MoEGraph::getSchema();
            schema.name = architectureName();
            return schema;
        }
    };
}
