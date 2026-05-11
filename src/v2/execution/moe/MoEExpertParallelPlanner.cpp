#include "execution/moe/MoEExpertParallelPlanner.h"

#include "execution/moe/DecodeExpertHistogram.h"
#include "planning/WeightMemoryEstimator.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace llaminar2
{
namespace
{

    size_t ceilBytes(long double bytes)
    {
        if (bytes <= 0.0L)
            return 0;
        return static_cast<size_t>(std::ceil(bytes));
    }

    void validateMetadata(const MoEExpertModelMetadata &metadata)
    {
        if (metadata.num_layers <= 0)
            throw std::invalid_argument("MoE expert planner metadata num_layers must be > 0");
        if (metadata.num_experts <= 0)
            throw std::invalid_argument("MoE expert planner metadata num_experts must be > 0");
        if (metadata.d_model <= 0)
            throw std::invalid_argument("MoE expert planner metadata d_model must be > 0");
        if (metadata.routed_intermediate_size <= 0)
            throw std::invalid_argument("MoE expert planner metadata routed_intermediate_size must be > 0");
        if (metadata.has_shared_expert && metadata.shared_intermediate_size <= 0)
            throw std::invalid_argument("MoE expert planner metadata shared_intermediate_size must be > 0 when has_shared_expert is true");
    }

    std::string formatValidationErrors(const MoEExpertParallelValidationResult &validation)
    {
        std::ostringstream message;
        message << "Invalid MoE expert parallel plan for planner:";
        for (const auto &error : validation.errors)
        {
            message << "\n - " << error;
        }
        return message.str();
    }

    int fallbackTierIndex(const MoEExpertParallelPlan &plan)
    {
        for (size_t tier_idx = 0; tier_idx < plan.routed_tiers.size(); ++tier_idx)
        {
            if (plan.routed_tiers[tier_idx].fallback)
                return static_cast<int>(tier_idx);
        }
        throw std::invalid_argument("MoE expert parallel planner requires exactly one fallback routed tier");
    }

    std::vector<int> sortedNonFallbackTierIndices(const MoEExpertParallelPlan &plan)
    {
        std::vector<int> tier_indices;
        tier_indices.reserve(plan.routed_tiers.size());
        for (size_t tier_idx = 0; tier_idx < plan.routed_tiers.size(); ++tier_idx)
        {
            if (!plan.routed_tiers[tier_idx].fallback)
                tier_indices.push_back(static_cast<int>(tier_idx));
        }

        std::sort(tier_indices.begin(), tier_indices.end(), [&](int lhs, int rhs) {
            const auto &left = plan.routed_tiers[lhs];
            const auto &right = plan.routed_tiers[rhs];
            if (left.priority != right.priority)
                return left.priority < right.priority;
            return lhs < rhs;
        });
        return tier_indices;
    }

    std::vector<int> byIdExpertOrder(int num_experts)
    {
        std::vector<int> expert_order(static_cast<size_t>(num_experts));
        std::iota(expert_order.begin(), expert_order.end(), 0);
        return expert_order;
    }

    bool histogramLayerHasCounts(const DecodeExpertHistogram &histogram, int layer, int num_experts)
    {
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            if (histogram.activationCount(layer, expert_id) != 0)
                return true;
        }
        return false;
    }

    std::vector<int> histogramExpertOrder(const DecodeExpertHistogram &histogram, int layer, int num_experts)
    {
        std::vector<std::pair<int, uint64_t>> expert_counts;
        expert_counts.reserve(static_cast<size_t>(num_experts));
        for (int expert_id = 0; expert_id < num_experts; ++expert_id)
        {
            expert_counts.emplace_back(expert_id, histogram.activationCount(layer, expert_id));
        }

        std::sort(expert_counts.begin(), expert_counts.end(), [](const auto &lhs, const auto &rhs) {
            if (lhs.second != rhs.second)
                return lhs.second > rhs.second;
            return lhs.first < rhs.first;
        });

        std::vector<int> expert_order;
        expert_order.reserve(expert_counts.size());
        for (const auto &[expert_id, _] : expert_counts)
        {
            expert_order.push_back(expert_id);
        }
        return expert_order;
    }

    size_t tierCapacityPerLayer(
        const ExpertRoutedTier &tier,
        int num_experts,
        size_t routed_expert_bytes_per_expert)
    {
        size_t capacity = static_cast<size_t>(num_experts);
        if (tier.max_experts_per_layer > 0)
        {
            capacity = std::min(capacity, static_cast<size_t>(tier.max_experts_per_layer));
        }
        if (tier.memory_budget_bytes > 0)
        {
            const size_t memory_capacity = routed_expert_bytes_per_expert == 0
                                               ? 0
                                               : tier.memory_budget_bytes / routed_expert_bytes_per_expert;
            capacity = std::min(capacity, memory_capacity);
        }
        return capacity;
    }

    ExpertLayerPlacement buildPlacementFromExpertOrder(
        const MoEExpertParallelPlan &plan,
        const MoEExpertModelMetadata &metadata,
        int layer,
        const std::vector<int> &expert_order,
        size_t routed_expert_bytes_per_expert)
    {
        const int fallback_tier = fallbackTierIndex(plan);
        ExpertLayerPlacement placement;
        placement.layer = layer;
        placement.routed_expert_tier.assign(static_cast<size_t>(metadata.num_experts), fallback_tier);

        size_t next_expert = 0;
        const auto tier_indices = sortedNonFallbackTierIndices(plan);
        for (const int tier_idx : tier_indices)
        {
            const auto &tier = plan.routed_tiers[tier_idx];
            const size_t capacity = tierCapacityPerLayer(tier, metadata.num_experts, routed_expert_bytes_per_expert);
            size_t assigned = 0;
            while (assigned < capacity && next_expert < expert_order.size())
            {
                const int expert_id = expert_order[next_expert++];
                placement.routed_expert_tier[static_cast<size_t>(expert_id)] = tier_idx;
                ++assigned;
            }
        }

        return placement;
    }

    std::vector<ExpertLayerPlacement> convertExplicitMasksToPlacements(
        const std::vector<MoEExpertLayerTierMask> &explicit_masks,
        const MoEExpertModelMetadata &metadata,
        const MoEExpertParallelPlan &plan)
    {
        std::vector<ExpertLayerPlacement> placements;
        placements.reserve(static_cast<size_t>(metadata.num_layers));
        for (int layer = 0; layer < metadata.num_layers; ++layer)
        {
            ExpertLayerPlacement placement;
            placement.layer = layer;
            placement.routed_expert_tier.assign(static_cast<size_t>(metadata.num_experts), -1);
            placements.push_back(std::move(placement));
        }

        for (const auto &mask : explicit_masks)
        {
            if (mask.layer < 0 || mask.layer >= metadata.num_layers)
                throw std::invalid_argument("Explicit MoE expert tier mask references layer outside model metadata range: " + std::to_string(mask.layer));
            if (mask.tier_index < 0 || mask.tier_index >= static_cast<int>(plan.routed_tiers.size()))
                throw std::invalid_argument("Explicit MoE expert tier mask references unknown tier index: " + std::to_string(mask.tier_index));

            auto &placement = placements[static_cast<size_t>(mask.layer)];
            for (const int expert_id : mask.expert_ids)
            {
                if (expert_id < 0 || expert_id >= metadata.num_experts)
                    throw std::invalid_argument("Explicit MoE expert tier mask references expert outside model metadata range: " + std::to_string(expert_id));
                auto &assigned_tier = placement.routed_expert_tier[static_cast<size_t>(expert_id)];
                if (assigned_tier != -1)
                {
                    throw std::invalid_argument("Explicit MoE expert tier masks assign layer " + std::to_string(mask.layer) +
                                                " expert " + std::to_string(expert_id) + " more than once");
                }
                assigned_tier = mask.tier_index;
            }
        }

        return placements;
    }

    std::vector<ExpertLayerPlacement> explicitPlacements(
        const MoEExpertParallelPlan &plan,
        const MoEExpertModelMetadata &metadata,
        const MoEExpertParallelPlannerOptions &options)
    {
        if (!options.explicit_placements.empty())
            return options.explicit_placements;
        if (!options.explicit_masks.empty())
            return convertExplicitMasksToPlacements(options.explicit_masks, metadata, plan);
        if (!plan.placements.empty())
            return plan.placements;
        throw std::invalid_argument("ExplicitMasks MoE expert residency policy requires explicit placements or masks");
    }

    std::vector<ExpertLayerPlacement> staticByIdPlacements(
        const MoEExpertParallelPlan &plan,
        const MoEExpertModelMetadata &metadata,
        size_t routed_expert_bytes_per_expert)
    {
        std::vector<ExpertLayerPlacement> placements;
        placements.reserve(static_cast<size_t>(metadata.num_layers));
        const auto expert_order = byIdExpertOrder(metadata.num_experts);
        for (int layer = 0; layer < metadata.num_layers; ++layer)
        {
            placements.push_back(buildPlacementFromExpertOrder(plan, metadata, layer, expert_order, routed_expert_bytes_per_expert));
        }
        return placements;
    }

    std::vector<ExpertLayerPlacement> histogramTieredCachePlacements(
        const MoEExpertParallelPlan &plan,
        const MoEExpertModelMetadata &metadata,
        const MoEExpertParallelPlannerOptions &options,
        size_t routed_expert_bytes_per_expert)
    {
        if (!options.decode_histogram)
            return staticByIdPlacements(plan, metadata, routed_expert_bytes_per_expert);

        const auto &histogram = *options.decode_histogram;
        if (histogram.config().num_layers < metadata.num_layers || histogram.config().num_experts < metadata.num_experts)
        {
            throw std::invalid_argument("DecodeExpertHistogram shape is smaller than MoE expert planner model metadata");
        }

        std::vector<ExpertLayerPlacement> placements;
        placements.reserve(static_cast<size_t>(metadata.num_layers));
        const auto by_id_order = byIdExpertOrder(metadata.num_experts);
        for (int layer = 0; layer < metadata.num_layers; ++layer)
        {
            const auto expert_order = histogramLayerHasCounts(histogram, layer, metadata.num_experts)
                                          ? histogramExpertOrder(histogram, layer, metadata.num_experts)
                                          : by_id_order;
            placements.push_back(buildPlacementFromExpertOrder(plan, metadata, layer, expert_order, routed_expert_bytes_per_expert));
        }
        return placements;
    }

    MoEExpertDomainMemoryEstimate &domainEstimate(
        MoEExpertParallelMemoryEstimate &estimate,
        const std::string &domain)
    {
        auto found = std::find_if(estimate.domains.begin(), estimate.domains.end(), [&](const auto &entry) {
            return entry.domain == domain;
        });
        if (found != estimate.domains.end())
            return *found;

        MoEExpertDomainMemoryEstimate entry;
        entry.domain = domain;
        estimate.domains.push_back(std::move(entry));
        return estimate.domains.back();
    }

    MoEExpertParallelMemoryEstimate estimateMemory(
        const MoEExpertParallelPlan &planned_plan,
        const MoEExpertModelMetadata &metadata,
        size_t routed_expert_bytes_per_expert)
    {
        MoEExpertParallelMemoryEstimate estimate;
        estimate.shared_expert_domain = planned_plan.shared_expert_domain;
        estimate.routed_expert_bytes_per_expert = routed_expert_bytes_per_expert;
        estimate.shared_expert_bytes_per_layer = MoEExpertParallelPlanner::estimateSharedExpertBytesPerLayer(metadata);
        estimate.total_shared_expert_bytes = MoEExpertParallelPlanner::estimateTotalSharedExpertBytes(metadata);
        estimate.tiers.reserve(planned_plan.routed_tiers.size());

        if (!planned_plan.shared_expert_domain.empty())
        {
            auto &shared_domain = domainEstimate(estimate, planned_plan.shared_expert_domain);
            shared_domain.shared_expert_bytes = estimate.total_shared_expert_bytes;
        }

        for (size_t tier_idx = 0; tier_idx < planned_plan.routed_tiers.size(); ++tier_idx)
        {
            const auto &tier = planned_plan.routed_tiers[tier_idx];
            MoEExpertTierMemoryEstimate tier_estimate;
            tier_estimate.tier_index = static_cast<int>(tier_idx);
            tier_estimate.tier_name = tier.name;
            tier_estimate.domain = tier.domain;
            estimate.tiers.push_back(std::move(tier_estimate));
            (void)domainEstimate(estimate, tier.domain);
        }

        for (const auto &placement : planned_plan.placements)
        {
            for (const int tier_idx : placement.routed_expert_tier)
            {
                if (tier_idx < 0 || tier_idx >= static_cast<int>(estimate.tiers.size()))
                    continue;
                auto &tier_estimate = estimate.tiers[static_cast<size_t>(tier_idx)];
                ++tier_estimate.routed_expert_count;
                tier_estimate.routed_expert_bytes += routed_expert_bytes_per_expert;
                estimate.total_routed_expert_bytes += routed_expert_bytes_per_expert;
            }
        }

        for (const auto &tier_estimate : estimate.tiers)
        {
            auto &domain = domainEstimate(estimate, tier_estimate.domain);
            domain.routed_expert_bytes += tier_estimate.routed_expert_bytes;
        }

        return estimate;
    }

} // namespace

    MoEExpertParallelPlannerResult MoEExpertParallelPlanner::plan(const MoEExpertParallelPlannerInput &input)
    {
        return plan(input.plan, input.metadata, input.options);
    }

    MoEExpertParallelPlannerResult MoEExpertParallelPlanner::plan(
        const MoEExpertParallelPlan &base_plan,
        const MoEExpertModelMetadata &metadata,
        const MoEExpertParallelPlannerOptions &options)
    {
        if (!base_plan.enabled)
        {
            MoEExpertParallelPlannerResult result;
            result.planned_plan = base_plan;
            return result;
        }

        validateMetadata(metadata);

        const auto base_validation = validateMoEExpertParallelPlan(base_plan);
        if (!base_validation.ok())
            throw std::invalid_argument(formatValidationErrors(base_validation));

        MoEExpertParallelPlan planned_plan = base_plan;
        const size_t routed_expert_bytes_per_expert = estimateRoutedExpertBytesPerExpert(metadata);

        switch (base_plan.residency_policy)
        {
        case ExpertResidencyPolicy::StaticById:
            planned_plan.placements = staticByIdPlacements(base_plan, metadata, routed_expert_bytes_per_expert);
            break;
        case ExpertResidencyPolicy::ExplicitMasks:
            planned_plan.placements = explicitPlacements(base_plan, metadata, options);
            break;
        case ExpertResidencyPolicy::HistogramTieredCache:
            planned_plan.placements = histogramTieredCachePlacements(base_plan, metadata, options, routed_expert_bytes_per_expert);
            break;
        case ExpertResidencyPolicy::Disabled:
            if (base_plan.placements.empty())
                throw std::invalid_argument("MoE expert parallel planner cannot plan an enabled overlay with Disabled residency policy and no placements");
            planned_plan.placements = base_plan.placements;
            break;
        }

        const MoEExpertParallelValidationOptions validation_options{
            .layer_count = metadata.num_layers,
            .routed_expert_count = metadata.num_experts,
        };
        const auto planned_validation = validateMoEExpertParallelPlan(planned_plan, validation_options);
        if (!planned_validation.ok())
            throw std::invalid_argument(formatValidationErrors(planned_validation));

        MoEExpertParallelPlannerResult result;
        result.planned_plan = std::move(planned_plan);
        result.memory = estimateMemory(result.planned_plan, metadata, routed_expert_bytes_per_expert);
        return result;
    }

    size_t MoEExpertParallelPlanner::estimateRoutedExpertBytesPerExpert(const MoEExpertModelMetadata &metadata)
    {
        validateMetadata(metadata);
        const long double elements = 3.0L * static_cast<long double>(metadata.d_model) *
                                     static_cast<long double>(metadata.routed_intermediate_size);
        const long double bytes_per_weight = WeightMemoryEstimator::getNativeBytesPerWeight(metadata.routed_quant_type);
        return ceilBytes(elements * bytes_per_weight);
    }

    size_t MoEExpertParallelPlanner::estimateSharedExpertBytesPerLayer(const MoEExpertModelMetadata &metadata)
    {
        validateMetadata(metadata);
        if (!metadata.has_shared_expert)
            return 0;
        const long double elements = 3.0L * static_cast<long double>(metadata.d_model) *
                                     static_cast<long double>(metadata.shared_intermediate_size);
        const long double bytes_per_weight = WeightMemoryEstimator::getNativeBytesPerWeight(metadata.shared_quant_type);
        return ceilBytes(elements * bytes_per_weight);
    }

    size_t MoEExpertParallelPlanner::estimateTotalSharedExpertBytes(const MoEExpertModelMetadata &metadata)
    {
        validateMetadata(metadata);
        return estimateSharedExpertBytesPerLayer(metadata) * static_cast<size_t>(metadata.num_layers);
    }

} // namespace llaminar2
