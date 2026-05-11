/**
 * @file MoEExpertParallelPlanner.h
 * @brief Static residency planner for same-layer MoE expert-parallel overlays.
 */

#pragma once

#include "MoEExpertParallelPlan.h"

#include <cstddef>
#include <string>
#include <vector>

namespace llaminar2
{

    class DecodeExpertHistogram;

    struct MoEExpertModelMetadata
    {
        int num_layers = 0;
        int num_experts = 0;
        int d_model = 0;
        int routed_intermediate_size = 0;
        int shared_intermediate_size = 0;
        bool has_shared_expert = false;
        std::string routed_quant_type = "F32";
        std::string shared_quant_type = "F32";
    };

    struct MoEExpertLayerTierMask
    {
        int layer = -1;
        int tier_index = -1;
        std::vector<int> expert_ids;
    };

    struct MoEExpertParallelPlannerOptions
    {
        const DecodeExpertHistogram *decode_histogram = nullptr;
        std::vector<ExpertLayerPlacement> explicit_placements;
        std::vector<MoEExpertLayerTierMask> explicit_masks;
    };

    struct MoEExpertParallelPlannerInput
    {
        MoEExpertParallelPlan plan;
        MoEExpertModelMetadata metadata;
        MoEExpertParallelPlannerOptions options;
    };

    struct MoEExpertTierMemoryEstimate
    {
        int tier_index = -1;
        std::string tier_name;
        std::string domain;
        size_t routed_expert_count = 0;
        size_t routed_expert_bytes = 0;
    };

    struct MoEExpertDomainMemoryEstimate
    {
        std::string domain;
        size_t shared_expert_bytes = 0;
        size_t routed_expert_bytes = 0;

        size_t totalBytes() const
        {
            return shared_expert_bytes + routed_expert_bytes;
        }
    };

    struct MoEExpertParallelMemoryEstimate
    {
        std::string shared_expert_domain;
        size_t routed_expert_bytes_per_expert = 0;
        size_t shared_expert_bytes_per_layer = 0;
        size_t total_shared_expert_bytes = 0;
        size_t total_routed_expert_bytes = 0;
        std::vector<MoEExpertTierMemoryEstimate> tiers;
        std::vector<MoEExpertDomainMemoryEstimate> domains;
    };

    struct MoEExpertParallelPlannerResult
    {
        MoEExpertParallelPlan planned_plan;
        MoEExpertParallelMemoryEstimate memory;
    };

    class MoEExpertParallelPlanner
    {
    public:
        static MoEExpertParallelPlannerResult plan(const MoEExpertParallelPlannerInput &input);

        static MoEExpertParallelPlannerResult plan(
            const MoEExpertParallelPlan &plan,
            const MoEExpertModelMetadata &metadata,
            const MoEExpertParallelPlannerOptions &options = {});

        static size_t estimateRoutedExpertBytesPerExpert(const MoEExpertModelMetadata &metadata);
        static size_t estimateSharedExpertBytesPerLayer(const MoEExpertModelMetadata &metadata);
        static size_t estimateTotalSharedExpertBytes(const MoEExpertModelMetadata &metadata);
    };

} // namespace llaminar2
