/**
 * @file MoEExpertParallelPlan.h
 * @brief Value types for same-layer MoE expert-parallel placement plans.
 *
 * This is a configuration contract only. TieredExpertOverlay describes
 * multiple role domains contributing to the same MoE layer and must not be
 * lowered as sequential pipeline-parallel stage ownership.
 */

#pragma once

#include "config/ExecutionDomainDefinition.h"

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llaminar2
{

    /**
     * @brief High-level MoE expert execution policy.
     *
     * SingleDomainExpertSharded models one domain covering routed experts by
     * expert id. TieredExpertOverlay models ordered same-layer expert tiers
     * whose partial outputs are reduced back to the continuation domain.
     */
    enum class MoEExpertExecutionKind
    {
        SingleDomainExpertSharded,
        TieredExpertOverlay,
    };

    enum class ExpertDomainKind
    {
        SingleDevice,
        LocalTP,
        NodeLocalTP,
    };

    enum class ExpertPlacementRole
    {
        SharedExpert,
        RoutedExpertTier,
    };

    enum class ExpertResidencyPolicy
    {
        Disabled,
        StaticById,
        HistogramTieredCache,
        ExplicitMasks,
    };

    /**
     * @brief Domain-internal expert compute strategy.
     *
     * ExpertIdSharded is the configuration-level representation of the narrow
     * TPMode::ExpertParallel-style expert-id split inside one TP context.
     * TensorParallelExperts means each selected expert's GEMMs are sharded
     * across a multi-participant domain-scoped TP context.
     */
    enum class ExpertDomainComputeKind
    {
        ReplicatedExperts,
        ExpertIdSharded,
        TensorParallelExperts,
    };

    struct ExpertComputeDomain
    {
        // Migration note: ExpertComputeDomain is a compatibility wrapper for
        // ExecutionDomainDefinition plus MoE-specific placement references.
        // New domain fields belong on ExecutionDomainDefinition; continuation,
        // shared expert, and routed tier ownership must remain placements over
        // domains rather than domain-type semantics.
        std::string name;
        ExpertDomainKind kind = ExpertDomainKind::SingleDevice;
        CollectiveBackendType backend = CollectiveBackendType::AUTO;
        std::vector<GlobalDeviceAddress> participants;
        std::vector<int> world_ranks;
        int owner_rank = -1;
        ExpertDomainComputeKind compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
        std::vector<float> weights;

        ExecutionDomainDefinition toExecutionDomainDefinition() const
        {
            ExecutionDomainDefinition domain;
            domain.name = name;
            domain.participants = participants;
            domain.weights = weights;
            domain.backend = backend;
            domain.owner_rank = owner_rank >= 0 ? std::optional<int>(owner_rank) : std::nullopt;
            domain.ranks = world_ranks;

            switch (kind)
            {
            case ExpertDomainKind::SingleDevice:
                domain.scope = ExecutionDomainScope::SINGLE;
                break;
            case ExpertDomainKind::LocalTP:
                domain.scope = ExecutionDomainScope::LOCAL;
                break;
            case ExpertDomainKind::NodeLocalTP:
                domain.scope = ExecutionDomainScope::NODE_LOCAL;
                break;
            }

            switch (compute_kind)
            {
            case ExpertDomainComputeKind::ReplicatedExperts:
                domain.compute_kind = ExecutionDomainComputeKind::REPLICATED_EXPERTS;
                break;
            case ExpertDomainComputeKind::ExpertIdSharded:
                domain.compute_kind = ExecutionDomainComputeKind::EXPERT_ID_SHARDED;
                break;
            case ExpertDomainComputeKind::TensorParallelExperts:
                domain.compute_kind = ExecutionDomainComputeKind::TENSOR_PARALLEL_EXPERTS;
                break;
            }

            return domain;
        }

        static ExpertComputeDomain fromExecutionDomainDefinition(const ExecutionDomainDefinition &domain)
        {
            ExpertComputeDomain result;
            result.name = domain.name;
            result.backend = domain.backend;
            result.participants = domain.participants;
            result.world_ranks = domain.ranks;
            result.owner_rank = domain.owner_rank.value_or(-1);
            result.weights = domain.weights;

            switch (domain.scope)
            {
            case ExecutionDomainScope::SINGLE:
                result.kind = ExpertDomainKind::SingleDevice;
                break;
            case ExecutionDomainScope::LOCAL:
                result.kind = ExpertDomainKind::LocalTP;
                break;
            case ExecutionDomainScope::NODE_LOCAL:
                result.kind = ExpertDomainKind::NodeLocalTP;
                break;
            case ExecutionDomainScope::AUTO:
                result.kind = domain.participants.size() > 1
                                  ? ExpertDomainKind::LocalTP
                                  : ExpertDomainKind::SingleDevice;
                break;
            case ExecutionDomainScope::GLOBAL:
                throw std::invalid_argument("MoE expert overlay domains do not support scope=global; use node_local or local");
            }

            switch (domain.compute_kind)
            {
            case ExecutionDomainComputeKind::UNSPECIFIED:
            case ExecutionDomainComputeKind::REPLICATED_EXPERTS:
                result.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
                break;
            case ExecutionDomainComputeKind::EXPERT_ID_SHARDED:
                result.compute_kind = ExpertDomainComputeKind::ExpertIdSharded;
                break;
            case ExecutionDomainComputeKind::TENSOR_PARALLEL_EXPERTS:
                result.compute_kind = ExpertDomainComputeKind::TensorParallelExperts;
                break;
            }

            return result;
        }

        bool isDomainScopedTPKind() const
        {
            return kind == ExpertDomainKind::LocalTP || kind == ExpertDomainKind::NodeLocalTP;
        }

        bool hasMultipleParticipants() const
        {
            return participants.size() > 1;
        }

        bool supportsDomainScopedTensorParallelExperts() const
        {
            return isDomainScopedTPKind() && hasMultipleParticipants();
        }

        bool supportsExpertIdSharding() const
        {
            return isDomainScopedTPKind() && hasMultipleParticipants();
        }
    };

    struct ExpertRoutedTier
    {
        std::string name;
        std::string domain;
        int priority = 0;
        int max_experts_per_layer = 0;
        size_t memory_budget_bytes = 0;
        bool fallback = false;
    };

    struct ExpertLayerPlacement
    {
        int layer = -1;

        /// Dense primary ownership: index = routed expert id, value = routed tier index.
        /// This representation assigns each routed expert to exactly one tier.
        std::vector<int> routed_expert_tier;
    };

    struct MoEExpertParallelPlan
    {
        bool enabled = false;
        MoEExpertExecutionKind execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
        std::string continuation_domain;
        std::string base_model_domain;
        std::string shared_expert_domain;
        ExpertResidencyPolicy residency_policy = ExpertResidencyPolicy::Disabled;
        std::vector<ExpertComputeDomain> domains;
        std::vector<ExpertRoutedTier> routed_tiers;
        std::vector<ExpertLayerPlacement> placements;

        bool isTieredOverlay() const
        {
            return enabled && execution_kind == MoEExpertExecutionKind::TieredExpertOverlay;
        }

        std::string effectiveBaseModelDomain() const
        {
            return base_model_domain.empty() ? continuation_domain : base_model_domain;
        }
    };

    struct MoEExpertParallelValidationOptions
    {
        /// When > 0 and placements are provided, require one placement per layer [0, layer_count).
        int layer_count = 0;

        /// When > 0 and placements are provided, each placement must cover exactly this many experts.
        int routed_expert_count = 0;
    };

    struct MoEExpertParallelValidationResult
    {
        std::vector<std::string> errors;

        bool ok() const
        {
            return errors.empty();
        }

        explicit operator bool() const
        {
            return ok();
        }
    };

    inline const char *toString(MoEExpertExecutionKind kind)
    {
        switch (kind)
        {
        case MoEExpertExecutionKind::SingleDomainExpertSharded:
            return "SingleDomainExpertSharded";
        case MoEExpertExecutionKind::TieredExpertOverlay:
            return "TieredExpertOverlay";
        }
        return "Unknown";
    }

    inline const char *toString(ExpertDomainKind kind)
    {
        switch (kind)
        {
        case ExpertDomainKind::SingleDevice:
            return "SingleDevice";
        case ExpertDomainKind::LocalTP:
            return "LocalTP";
        case ExpertDomainKind::NodeLocalTP:
            return "NodeLocalTP";
        }
        return "Unknown";
    }

    inline const char *toString(ExpertDomainComputeKind kind)
    {
        switch (kind)
        {
        case ExpertDomainComputeKind::ReplicatedExperts:
            return "ReplicatedExperts";
        case ExpertDomainComputeKind::ExpertIdSharded:
            return "ExpertIdSharded";
        case ExpertDomainComputeKind::TensorParallelExperts:
            return "TensorParallelExperts";
        }
        return "Unknown";
    }

    inline MoEExpertParallelValidationResult validateMoEExpertParallelPlan(
        const MoEExpertParallelPlan &plan,
        const MoEExpertParallelValidationOptions &options = {})
    {
        MoEExpertParallelValidationResult result;
        auto addError = [&](const std::string &message) {
            result.errors.push_back(message);
        };

        if (!plan.enabled)
            return result;

        std::unordered_map<std::string, const ExpertComputeDomain *> domains_by_name;
        for (const auto &domain : plan.domains)
        {
            const auto canonical_domain = domain.toExecutionDomainDefinition();
            for (const auto &error : canonical_domain.validate())
            {
                addError("expert compute domain '" + domain.name + "': " + error);
            }

            if (domain.name.empty())
            {
                addError("expert compute domain name must not be empty");
                continue;
            }
            auto inserted = domains_by_name.emplace(domain.name, &domain).second;
            if (!inserted)
            {
                addError("duplicate expert compute domain name: " + domain.name);
            }

            if (domain.participants.empty())
            {
                addError("expert compute domain '" + domain.name + "' must declare at least one participant");
            }

            if (!domain.world_ranks.empty())
            {
                if (domain.world_ranks.size() != domain.participants.size())
                {
                    addError("expert compute domain '" + domain.name + "' declares " +
                             std::to_string(domain.world_ranks.size()) +
                             " world ranks but " + std::to_string(domain.participants.size()) +
                             " participants");
                }

                std::unordered_set<int> seen_ranks;
                for (int rank : domain.world_ranks)
                {
                    if (rank < 0)
                    {
                        addError("expert compute domain '" + domain.name + "' has a negative world rank");
                    }
                    else if (!seen_ranks.insert(rank).second)
                    {
                        addError("expert compute domain '" + domain.name + "' has duplicate world rank " +
                                 std::to_string(rank));
                    }
                }

                if (domain.owner_rank >= 0 && seen_ranks.find(domain.owner_rank) == seen_ranks.end())
                {
                    addError("expert compute domain '" + domain.name + "' owner rank " +
                             std::to_string(domain.owner_rank) + " is not in its world rank list");
                }
            }

            if (domain.owner_rank < -1)
            {
                addError("expert compute domain '" + domain.name + "' has invalid owner rank");
            }

            if (domain.kind == ExpertDomainKind::SingleDevice && domain.participants.size() > 1)
            {
                addError("expert compute domain '" + domain.name + "' is SingleDevice but declares multiple participants");
            }

            if (domain.compute_kind == ExpertDomainComputeKind::ExpertIdSharded && !domain.supportsExpertIdSharding())
            {
                addError("expert compute domain '" + domain.name + "' uses ExpertIdSharded, which maps to TPMode::ExpertParallel and requires a multi-participant TP domain");
            }

            if (domain.compute_kind == ExpertDomainComputeKind::TensorParallelExperts && !domain.supportsDomainScopedTensorParallelExperts())
            {
                addError("expert compute domain '" + domain.name + "' uses TensorParallelExperts but is not a multi-participant domain-scoped TP domain");
            }
        }

        auto requireDomain = [&](const std::string &field_name, const std::string &domain_name) {
            if (domain_name.empty())
            {
                addError(field_name + " domain must not be empty");
                return;
            }
            if (domains_by_name.find(domain_name) == domains_by_name.end())
            {
                addError(field_name + " domain references unknown expert compute domain: " + domain_name);
            }
        };

        requireDomain("continuation", plan.continuation_domain);
        if (!plan.base_model_domain.empty())
        {
            requireDomain("base/non-expert model", plan.base_model_domain);
        }
        requireDomain("shared expert", plan.shared_expert_domain);

        if (plan.routed_tiers.empty())
        {
            addError("enabled MoE expert parallel plan must declare at least one routed tier");
        }

        std::unordered_map<std::string, size_t> tiers_by_name;
        int fallback_count = 0;
        for (size_t tier_idx = 0; tier_idx < plan.routed_tiers.size(); ++tier_idx)
        {
            const auto &tier = plan.routed_tiers[tier_idx];
            if (tier.name.empty())
            {
                addError("routed tier name must not be empty");
            }
            else
            {
                auto inserted = tiers_by_name.emplace(tier.name, tier_idx).second;
                if (!inserted)
                {
                    addError("duplicate routed tier name: " + tier.name);
                }
            }

            if (tier.domain.empty())
            {
                addError("routed tier '" + tier.name + "' domain must not be empty");
            }
            else if (domains_by_name.find(tier.domain) == domains_by_name.end())
            {
                addError("routed tier '" + tier.name + "' references unknown expert compute domain: " + tier.domain);
            }

            if (tier.fallback)
                ++fallback_count;
        }

        if (!plan.routed_tiers.empty() && fallback_count != 1)
        {
            addError("enabled MoE expert parallel plan with routed tiers must declare exactly one fallback tier");
        }

        if (!plan.placements.empty())
        {
            std::unordered_set<int> covered_layers;
            for (const auto &placement : plan.placements)
            {
                if (placement.layer < 0)
                {
                    addError("expert layer placement has invalid negative layer index");
                    continue;
                }

                if (options.layer_count > 0 && placement.layer >= options.layer_count)
                {
                    addError("expert layer placement references layer outside validation range: " + std::to_string(placement.layer));
                }

                if (!covered_layers.insert(placement.layer).second)
                {
                    addError("duplicate expert layer placement for layer: " + std::to_string(placement.layer));
                }

                if (options.routed_expert_count > 0 &&
                    static_cast<int>(placement.routed_expert_tier.size()) != options.routed_expert_count)
                {
                    addError("expert layer placement for layer " + std::to_string(placement.layer) +
                             " does not cover every routed expert");
                }

                if (placement.routed_expert_tier.empty())
                {
                    addError("expert layer placement for layer " + std::to_string(placement.layer) +
                             " must assign at least one routed expert");
                }

                for (size_t expert_id = 0; expert_id < placement.routed_expert_tier.size(); ++expert_id)
                {
                    const int tier_idx = placement.routed_expert_tier[expert_id];
                    if (tier_idx < 0)
                    {
                        addError("expert layer placement for layer " + std::to_string(placement.layer) +
                                 " leaves routed expert " + std::to_string(expert_id) + " without a tier");
                    }
                    else if (tier_idx >= static_cast<int>(plan.routed_tiers.size()))
                    {
                        addError("expert layer placement for layer " + std::to_string(placement.layer) +
                                 " routes expert " + std::to_string(expert_id) + " to unknown tier index " +
                                 std::to_string(tier_idx));
                    }
                }
            }

            if (options.layer_count > 0)
            {
                for (int layer = 0; layer < options.layer_count; ++layer)
                {
                    if (covered_layers.find(layer) == covered_layers.end())
                    {
                        addError("missing expert layer placement for layer: " + std::to_string(layer));
                    }
                }
            }
        }

        if (plan.execution_kind == MoEExpertExecutionKind::SingleDomainExpertSharded)
        {
            std::unordered_set<std::string> routed_domains;
            for (const auto &tier : plan.routed_tiers)
            {
                if (!tier.domain.empty())
                    routed_domains.insert(tier.domain);
            }

            if (routed_domains.size() > 1)
            {
                addError("SingleDomainExpertSharded plans must route experts through one compute domain");
            }
        }

        return result;
    }

    inline bool isValidMoEExpertParallelPlan(
        const MoEExpertParallelPlan &plan,
        const MoEExpertParallelValidationOptions &options = {})
    {
        return validateMoEExpertParallelPlan(plan, options).ok();
    }

    inline void validateMoEExpertParallelPlanOrThrow(
        const MoEExpertParallelPlan &plan,
        const MoEExpertParallelValidationOptions &options = {})
    {
        auto result = validateMoEExpertParallelPlan(plan, options);
        if (result.ok())
            return;

        std::ostringstream message;
        message << "Invalid MoE expert parallel plan:";
        for (const auto &error : result.errors)
        {
            message << "\n - " << error;
        }
        throw std::invalid_argument(message.str());
    }

} // namespace llaminar2
