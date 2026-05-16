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

#include <algorithm>
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

    enum class MoEContinuationActivationLayout
    {
        ReplicatedHidden,
        RootOnlyHidden,
        ShardedHiddenRequiresGather,
    };

    enum class ExpertResidencyPolicy
    {
        Disabled,
        StaticById,
        HistogramTieredCache,
        ExplicitMasks,
        RoutedTierRebalanced,
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

    struct MoEContinuationDomainSpec
    {
        std::string domain;
        int logical_root_participant = 0;
        bool dense_tp_enabled = false;
        MoEContinuationActivationLayout hidden_layout = MoEContinuationActivationLayout::ReplicatedHidden;
        bool shared_expert_uses_dense_tp = true;
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
        MoEContinuationDomainSpec continuation_domain_spec;
        ExpertResidencyPolicy residency_policy = ExpertResidencyPolicy::Disabled;

        /// Generic dense execution domains for continuation/base/shared model flow.
        /// These may use SINGLE, LOCAL, NODE_LOCAL, or GLOBAL scope and are
        /// validated separately from routed whole-expert ownership domains.
        std::vector<ExecutionDomainDefinition> dense_domains;

        /// Routed expert ownership domains. These remain whole-expert domains
        /// in the graph-native reset; TensorParallelExperts is rejected by
        /// validation unless explicitly allowed by legacy compatibility code.
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

        /// Legacy compatibility only: graph-native routed tiers reject true
        /// tensor-sharded expert GEMMs by default and use whole-expert owners.
        bool allow_routed_tensor_parallel_experts = false;
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

    inline const char *toString(MoEContinuationActivationLayout layout)
    {
        switch (layout)
        {
        case MoEContinuationActivationLayout::ReplicatedHidden:
            return "ReplicatedHidden";
        case MoEContinuationActivationLayout::RootOnlyHidden:
            return "RootOnlyHidden";
        case MoEContinuationActivationLayout::ShardedHiddenRequiresGather:
            return "ShardedHiddenRequiresGather";
        }
        return "Unknown";
    }

    inline const char *toString(ExpertResidencyPolicy policy)
    {
        switch (policy)
        {
        case ExpertResidencyPolicy::Disabled:
            return "Disabled";
        case ExpertResidencyPolicy::StaticById:
            return "StaticById";
        case ExpertResidencyPolicy::HistogramTieredCache:
            return "HistogramTieredCache";
        case ExpertResidencyPolicy::ExplicitMasks:
            return "ExplicitMasks";
        case ExpertResidencyPolicy::RoutedTierRebalanced:
            return "RoutedTierRebalanced";
        }
        return "Unknown";
    }

    inline std::string renderMoEExpertParallelPlanExplanation(
        const MoEExpertParallelPlan &plan,
        int model_routed_expert_count = 0)
    {
        auto memoryText = [](size_t bytes)
        {
            if (bytes == 0)
                return std::string("auto/model-aware");
            return std::to_string(bytes) + " bytes";
        };

        auto appendDevices = [](std::ostringstream &out,
                                const std::vector<GlobalDeviceAddress> &participants)
        {
            out << "[";
            for (size_t index = 0; index < participants.size(); ++index)
            {
                if (index > 0)
                    out << ",";
                out << participants[index].toShortString();
            }
            out << "]";
        };

        auto appendStringList = [](std::ostringstream &out,
                                   const std::vector<std::string> &values)
        {
            out << "[";
            for (size_t index = 0; index < values.size(); ++index)
            {
                if (index > 0)
                    out << ",";
                out << values[index];
            }
            out << "]";
        };

        auto findExpertDomain = [&](const std::string &name) -> const ExpertComputeDomain *
        {
            for (const auto &domain : plan.domains)
            {
                if (domain.name == name)
                    return &domain;
            }
            return nullptr;
        };

        auto addUnique = [](std::vector<std::string> &values, const std::string &value)
        {
            if (std::find(values.begin(), values.end(), value) == values.end())
                values.push_back(value);
        };

        std::ostringstream out;
        out << "  moe_expert_overlay:\n";
        out << "    enabled: " << (plan.enabled ? "true" : "false") << "\n";
        if (!plan.enabled)
        {
            out << "    execution: disabled\n";
            return out.str();
        }

        out << "    execution: graph-native " << toString(plan.execution_kind)
            << " (whole-expert routed ownership, no shadow LocalTP runtime)\n";
        out << "    residency_policy: " << toString(plan.residency_policy) << "\n";
        out << "    continuation_domain: " << plan.continuation_domain
            << " root_participant=" << plan.continuation_domain_spec.logical_root_participant
            << " dense_tp=" << (plan.continuation_domain_spec.dense_tp_enabled ? "true" : "false")
            << " hidden_layout=" << toString(plan.continuation_domain_spec.hidden_layout) << "\n";
        out << "    base_model_domain: " << plan.effectiveBaseModelDomain() << "\n";
        out << "    shared_expert_domain: " << plan.shared_expert_domain << "\n";

        if (!plan.dense_domains.empty())
        {
            out << "    dense_domains:\n";
            for (const auto &domain : plan.dense_domains)
            {
                out << "      - " << domain.toString() << "\n";
            }
        }

        if (!plan.domains.empty())
        {
            out << "    routed_domains:\n";
            for (const auto &domain : plan.domains)
            {
                out << "      - " << domain.name << " devices=";
                appendDevices(out, domain.participants);
                out << " kind=" << toString(domain.kind)
                    << " compute=" << toString(domain.compute_kind)
                    << " backend=" << collectiveBackendTypeToString(domain.backend);
                if (domain.owner_rank >= 0)
                    out << " owner=" << domain.owner_rank;
                if (!domain.world_ranks.empty())
                {
                    out << " ranks=[";
                    for (size_t index = 0; index < domain.world_ranks.size(); ++index)
                    {
                        if (index > 0)
                            out << ",";
                        out << domain.world_ranks[index];
                    }
                    out << "]";
                }
                out << "\n";
            }
        }

        size_t total_non_fallback_capacity = 0;
        bool total_capacity_model_dependent = false;
        int fallback_count = 0;
        bool has_cpu_fallback = false;
        std::vector<std::string> fallback_domains;

        if (!plan.routed_tiers.empty())
        {
            out << "    routed_tiers:\n";
            for (const auto &tier : plan.routed_tiers)
            {
                out << "      - " << tier.name
                    << " domain=" << tier.domain
                    << " priority=" << tier.priority
                    << " capacity=";

                if (tier.max_experts_per_layer > 0)
                    out << tier.max_experts_per_layer << " experts/layer";
                else
                    out << "model-dependent";

                out << " memory=" << memoryText(tier.memory_budget_bytes)
                    << " fallback=" << (tier.fallback ? "true" : "false") << "\n";

                if (tier.fallback)
                {
                    ++fallback_count;
                    addUnique(fallback_domains, tier.domain);
                    if (const auto *domain = findExpertDomain(tier.domain))
                    {
                        has_cpu_fallback = has_cpu_fallback ||
                                           std::any_of(domain->participants.begin(),
                                                       domain->participants.end(),
                                                       [](const GlobalDeviceAddress &device)
                                                       {
                                                           return device.isCPU();
                                                       });
                    }
                    continue;
                }

                if (model_routed_expert_count > 0)
                {
                    size_t tier_capacity = static_cast<size_t>(model_routed_expert_count);
                    if (tier.max_experts_per_layer > 0)
                    {
                        tier_capacity = std::min(
                            tier_capacity,
                            static_cast<size_t>(tier.max_experts_per_layer));
                    }
                    if (tier.memory_budget_bytes > 0)
                        total_capacity_model_dependent = true;
                    total_non_fallback_capacity += tier_capacity;
                }
                else if (tier.max_experts_per_layer > 0 && tier.memory_budget_bytes == 0)
                {
                    total_non_fallback_capacity += static_cast<size_t>(tier.max_experts_per_layer);
                }
                else
                {
                    total_capacity_model_dependent = true;
                }
            }
        }

        out << "    total_non_fallback_capacity: ";
        if (total_capacity_model_dependent && model_routed_expert_count <= 0)
            out << "model-dependent";
        else
            out << total_non_fallback_capacity << " experts/layer";
        if (total_capacity_model_dependent)
            out << " (memory/model-aware resolution may reduce usable capacity)";
        out << "\n";

        out << "    fallback: count=" << fallback_count << " domains=";
        appendStringList(out, fallback_domains);
        out << " cpu_fallback=" << (has_cpu_fallback ? "true" : "false") << "\n";

        if (model_routed_expert_count > 0)
        {
            if (fallback_count > 0)
            {
                out << "    coverage: fallback present; uncovered routed experts can use fallback domains for "
                    << model_routed_expert_count << " model experts\n";
            }
            else if (total_capacity_model_dependent)
            {
                out << "    coverage: non-fallback capacity upper bound "
                    << total_non_fallback_capacity << "/" << model_routed_expert_count
                    << "; final coverage is validated during model-aware resolution\n";
            }
            else
            {
                out << "    coverage: "
                    << (total_non_fallback_capacity >= static_cast<size_t>(model_routed_expert_count)
                            ? "all routed experts covered"
                            : "incomplete without fallback")
                    << " (" << total_non_fallback_capacity << "/"
                    << model_routed_expert_count << ")\n";
            }
        }
        else
        {
            out << "    coverage: model expert count is not known at config parse time; "
                << "coverage is validated during model-aware resolution\n";
        }

        out << "    rebalance_hint: ";
        switch (plan.residency_policy)
        {
        case ExpertResidencyPolicy::RoutedTierRebalanced:
            out << "uses histogram-aware routed tier rebalancing at safe step boundaries";
            break;
        case ExpertResidencyPolicy::HistogramTieredCache:
            out << "uses histogram tier ordering when model-aware planning has histogram data";
            break;
        case ExpertResidencyPolicy::StaticById:
            out << "uses deterministic expert-id ordering across routed tiers";
            break;
        case ExpertResidencyPolicy::ExplicitMasks:
            out << "uses explicit masks/placements supplied by configuration or caller";
            break;
        case ExpertResidencyPolicy::Disabled:
            out << "requires precomputed placements for enabled overlays";
            break;
        }
        out << "\n";

        return out.str();
    }

    inline bool isAllowedMoEContinuationDenseScope(ExecutionDomainScope scope)
    {
        switch (scope)
        {
        case ExecutionDomainScope::SINGLE:
        case ExecutionDomainScope::LOCAL:
        case ExecutionDomainScope::NODE_LOCAL:
        case ExecutionDomainScope::GLOBAL:
            return true;
        case ExecutionDomainScope::AUTO:
            return false;
        }
        return false;
    }

    inline MoEExpertParallelValidationResult validateMoEContinuationDomainSpec(
        const MoEContinuationDomainSpec &spec,
        const ExecutionDomainDefinition &domain,
        const std::string &role_name = "continuation")
    {
        MoEExpertParallelValidationResult result;
        auto addError = [&](const std::string &message)
        {
            result.errors.push_back(message);
        };

        if (spec.domain.empty())
            addError(role_name + " domain spec must not be empty");
        else if (spec.domain != domain.name)
            addError(role_name + " domain spec references '" + spec.domain +
                     "' but was validated against domain '" + domain.name + "'");

        for (const auto &error : domain.validate())
            addError(role_name + " dense domain '" + domain.name + "': " + error);

        if (!isAllowedMoEContinuationDenseScope(domain.scope))
        {
            addError(role_name + " dense domain '" + domain.name +
                     "' must use scope=single, local, node_local, or global; got scope=" +
                     executionDomainScopeToString(domain.scope));
        }

        if (spec.logical_root_participant < 0 ||
            spec.logical_root_participant >= static_cast<int>(domain.participants.size()))
        {
            addError(role_name + " domain '" + domain.name +
                     "' logical_root_participant " + std::to_string(spec.logical_root_participant) +
                     " is outside participant range [0, " +
                     std::to_string(domain.participants.size()) + ")");
        }

        if (spec.dense_tp_enabled && domain.participants.size() <= 1)
        {
            addError(role_name + " domain '" + domain.name +
                     "' enables dense TP but declares fewer than two participants");
        }

        return result;
    }

    inline MoEExpertParallelValidationResult validateMoEExpertParallelPlan(
        const MoEExpertParallelPlan &plan,
        const MoEExpertParallelValidationOptions &options = {})
    {
        MoEExpertParallelValidationResult result;
        auto addError = [&](const std::string &message)
        {
            result.errors.push_back(message);
        };

        if (!plan.enabled)
            return result;

        std::unordered_map<std::string, const ExecutionDomainDefinition *> execution_domains_by_name;
        for (const auto &domain : plan.dense_domains)
        {
            for (const auto &error : domain.validate())
            {
                addError("dense execution domain '" + domain.name + "': " + error);
            }

            if (domain.name.empty())
            {
                addError("dense execution domain name must not be empty");
                continue;
            }

            execution_domains_by_name.emplace(domain.name, &domain);
        }

        std::vector<ExecutionDomainDefinition> canonical_expert_domains;
        canonical_expert_domains.reserve(plan.domains.size());

        std::unordered_map<std::string, const ExpertComputeDomain *> domains_by_name;
        for (const auto &domain : plan.domains)
        {
            canonical_expert_domains.push_back(domain.toExecutionDomainDefinition());
            const auto &canonical_domain = canonical_expert_domains.back();
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

            execution_domains_by_name.emplace(domain.name, &canonical_domain);

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

        auto requireDenseDomain = [&](const std::string &field_name, const std::string &domain_name)
        {
            if (domain_name.empty())
            {
                addError(field_name + " domain must not be empty");
                return;
            }
            if (execution_domains_by_name.find(domain_name) == execution_domains_by_name.end())
            {
                addError(field_name + " domain references unknown execution domain: " + domain_name);
            }
        };

        requireDenseDomain("continuation", plan.continuation_domain);
        if (!plan.base_model_domain.empty())
        {
            requireDenseDomain("base/non-expert model", plan.base_model_domain);
        }
        requireDenseDomain("shared expert", plan.shared_expert_domain);

        auto validateDenseRole = [&](const std::string &role_name,
                                     const MoEContinuationDomainSpec &spec)
        {
            auto it = execution_domains_by_name.find(spec.domain);
            if (it == execution_domains_by_name.end() || !it->second)
                return;

            auto role_result = validateMoEContinuationDomainSpec(spec, *it->second, role_name);
            for (const auto &error : role_result.errors)
                addError(error);
        };

        MoEContinuationDomainSpec continuation_spec = plan.continuation_domain_spec;
        if (continuation_spec.domain.empty())
            continuation_spec.domain = plan.continuation_domain;
        validateDenseRole("continuation", continuation_spec);

        MoEContinuationDomainSpec base_spec;
        base_spec.domain = plan.effectiveBaseModelDomain();
        validateDenseRole("base/non-expert model", base_spec);

        MoEContinuationDomainSpec shared_spec;
        shared_spec.domain = plan.shared_expert_domain;
        shared_spec.dense_tp_enabled = continuation_spec.dense_tp_enabled &&
                                       continuation_spec.shared_expert_uses_dense_tp &&
                                       shared_spec.domain == continuation_spec.domain;
        validateDenseRole("shared expert", shared_spec);

        if (plan.routed_tiers.empty())
        {
            addError("enabled MoE expert parallel plan must declare at least one routed tier");
        }

        std::unordered_map<std::string, size_t> tiers_by_name;
        std::unordered_set<std::string> routed_domain_names;
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
                addError("routed tier '" + tier.name + "' references unknown routed expert compute domain: " + tier.domain);
            }
            else
            {
                routed_domain_names.insert(tier.domain);
            }

            if (tier.fallback)
                ++fallback_count;
        }

        if (!options.allow_routed_tensor_parallel_experts)
        {
            for (const auto &domain_name : routed_domain_names)
            {
                auto domain_it = domains_by_name.find(domain_name);
                if (domain_it == domains_by_name.end() || !domain_it->second)
                    continue;
                const auto &domain = *domain_it->second;
                if (domain.compute_kind == ExpertDomainComputeKind::TensorParallelExperts)
                {
                    addError("routed expert domain '" + domain.name +
                             "' uses TensorParallelExperts, which is unsupported for graph-native whole-expert routed tiers; "
                             "graph-native MoE overlay has no shadow LocalTP runtime. Use whole-expert ownership or "
                             "enable an explicit future TensorParallelExperts path with its own reduction stage");
                }
            }
        }

        if (fallback_count > 1)
        {
            addError("enabled MoE expert parallel plan with routed tiers must declare at most one fallback tier");
        }

        if (fallback_count == 0 && options.routed_expert_count > 0 && !plan.routed_tiers.empty())
        {
            bool all_non_fallback_capacity_known = true;
            size_t non_fallback_capacity = 0;
            for (const auto &tier : plan.routed_tiers)
            {
                if (tier.fallback)
                    continue;
                if (tier.max_experts_per_layer <= 0)
                {
                    all_non_fallback_capacity_known = false;
                    break;
                }
                non_fallback_capacity += static_cast<size_t>(tier.max_experts_per_layer);
            }

            if (all_non_fallback_capacity_known &&
                non_fallback_capacity < static_cast<size_t>(options.routed_expert_count))
            {
                addError("graph-native whole-expert overlay has no fallback tier and non-fallback routed tier capacity covers only " +
                         std::to_string(non_fallback_capacity) + " of " +
                         std::to_string(options.routed_expert_count) +
                         " routed experts; increase routed tier capacity or configure one fallback tier");
            }
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

                std::vector<int> tier_assignment_counts(plan.routed_tiers.size(), 0);
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
                    else
                    {
                        ++tier_assignment_counts[static_cast<size_t>(tier_idx)];
                    }
                }

                for (size_t tier_idx = 0; tier_idx < plan.routed_tiers.size(); ++tier_idx)
                {
                    const auto &tier = plan.routed_tiers[tier_idx];
                    if (tier.max_experts_per_layer > 0 &&
                        tier_assignment_counts[tier_idx] > tier.max_experts_per_layer)
                    {
                        addError("expert layer placement for layer " + std::to_string(placement.layer) +
                                 " assigns " + std::to_string(tier_assignment_counts[tier_idx]) +
                                 " routed experts to tier '" + tier.name +
                                 "' but max_experts_per_layer is " +
                                 std::to_string(tier.max_experts_per_layer));
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
