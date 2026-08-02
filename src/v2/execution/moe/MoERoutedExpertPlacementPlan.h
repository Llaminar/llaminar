/**
 * @file MoERoutedExpertPlacementPlan.h
 * @brief Declarative same-layer placement plans for routed MoE experts.
 *
 * Routed-expert placement is independent from both dense tensor parallelism
 * and the compute distribution used inside a routed domain.  A tiered overlay
 * may therefore contain whole-expert-apportioned, replicated, or
 * tensor-sharded domains without changing its topology name.
 */

#pragma once

#include "config/ExecutionDomainDefinition.h"
#include "execution/config/RuntimeConfig.h"

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

    /** @brief Topology of routed-expert placement domains. */
    enum class RoutedExpertPlacementTopology
    {
        SingleDomain,
        TieredOverlay,
    };

    /** @brief Model role assigned to an execution domain. */
    enum class RoutedExpertPlacementRole
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

    /** @brief Policy controlling which complete routed experts are resident. */
    enum class RoutedExpertResidencyPolicy
    {
        Disabled,
        StaticById,
        HistogramTieredCache,
        ExplicitMasks,
        RoutedTierRebalanced,
    };

    /**
     * @brief One hardware domain that owns routed-expert compute.
     *
     * `routed_compute_policy` describes whether full experts are replicated,
     * apportioned by expert id, or tensor-sharded. `routed_phase_policy`
     * describes whether prefill and decode schedule those physical residents
     * differently. `routed_assignment_policy` applies only to row scheduling
     * among eligible complete residents. `scope` describes participant
     * topology and carries no compute semantics.
     */
    struct RoutedExpertDomain
    {
        std::string name;
        ExecutionDomainScope scope = ExecutionDomainScope::SINGLE;
        CollectiveBackendType backend = CollectiveBackendType::AUTO;
        std::vector<GlobalDeviceAddress> participants;
        std::vector<int> world_ranks;
        int owner_rank = -1;
        RoutedExpertComputePolicy routed_compute_policy =
            RoutedExpertComputePolicy::Apportioned;
        RoutedExpertPhasePolicy routed_phase_policy =
            RoutedExpertPhasePolicy::Uniform;
        RoutedExpertAssignmentPolicy routed_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
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

            domain.scope = scope;
            domain.routed_compute_policy = routed_compute_policy;
            domain.routed_phase_policy = routed_phase_policy;
            domain.routed_assignment_policy = routed_assignment_policy;

            return domain;
        }

        static RoutedExpertDomain fromExecutionDomainDefinition(
            const ExecutionDomainDefinition &domain)
        {
            RoutedExpertDomain result;
            result.name = domain.name;
            result.backend = domain.backend;
            result.participants = domain.participants;
            result.world_ranks = domain.ranks;
            result.owner_rank = domain.owner_rank.value_or(-1);
            result.weights = domain.weights;
            result.scope = domain.scope == ExecutionDomainScope::AUTO
                               ? (domain.participants.size() > 1
                                      ? ExecutionDomainScope::LOCAL
                                      : ExecutionDomainScope::SINGLE)
                               : domain.scope;
            result.routed_compute_policy =
                domain.routed_compute_policy == RoutedExpertComputePolicy::Unspecified
                    ? RoutedExpertComputePolicy::Apportioned
                    : domain.routed_compute_policy;
            result.routed_phase_policy =
                domain.routed_phase_policy == RoutedExpertPhasePolicy::Unspecified
                    ? RoutedExpertPhasePolicy::Uniform
                    : domain.routed_phase_policy;
            result.routed_assignment_policy =
                domain.routed_assignment_policy == RoutedExpertAssignmentPolicy::Unspecified
                    ? RoutedExpertAssignmentPolicy::StaticOwner
                    : domain.routed_assignment_policy;

            return result;
        }

        bool isCollectiveDomain() const
        {
            return scope == ExecutionDomainScope::LOCAL ||
                   scope == ExecutionDomainScope::NODE_LOCAL ||
                   scope == ExecutionDomainScope::GLOBAL;
        }

        bool hasMultipleParticipants() const
        {
            return participants.size() > 1;
        }

        bool supportsRoutedExpertTensorSharding() const
        {
            return isCollectiveDomain() && hasMultipleParticipants();
        }

        bool supportsWholeExpertApportionment() const
        {
            return !participants.empty();
        }

        bool supportsLeastLoadedResidentAssignment() const
        {
            const bool apportioned_execution =
                routed_compute_policy == RoutedExpertComputePolicy::Apportioned ||
                (routed_compute_policy == RoutedExpertComputePolicy::Replicated &&
                 routed_phase_policy ==
                     RoutedExpertPhasePolicy::
                         PrefillApportionedDecodeReplicated);
            return apportioned_execution &&
                   isCollectiveDomain() &&
                   hasMultipleParticipants();
        }

    };

    struct RoutedExpertTier
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
        bool dense_decode_replicated = false;
        bool dense_decode_mirrored_embedding = false;
        DenseParallelPolicy dense_policy = DenseParallelPolicy::Replicated;
        MoEContinuationActivationLayout hidden_layout = MoEContinuationActivationLayout::ReplicatedHidden;
        bool shared_expert_uses_dense_tp = true;

        DenseParallelPolicy effectiveDensePolicy() const
        {
            return denseParallelPolicyFromFlags(
                dense_tp_enabled,
                dense_decode_replicated,
                dense_decode_mirrored_embedding);
        }

        void setDensePolicy(DenseParallelPolicy policy)
        {
            dense_policy = policy;
            dense_tp_enabled = denseParallelPolicyEnablesTP(policy);
            dense_decode_replicated = denseParallelPolicyReplicatesDecode(policy);
            dense_decode_mirrored_embedding =
                denseParallelPolicyMirrorsDecodeEmbedding(policy) &&
                !dense_decode_replicated;
        }

        void refreshDensePolicyFromFlags()
        {
            dense_policy = effectiveDensePolicy();
        }
    };

    struct RoutedExpertLayerPlacement
    {
        int layer = -1;

        /// Dense primary ownership: index = routed expert id, value = routed tier index.
        /// This representation assigns each routed expert to exactly one tier.
        std::vector<int> routed_expert_tier;
    };

    struct MoERoutedExpertPlacementPlan
    {
        bool enabled = false;
        RoutedExpertPlacementTopology topology =
            RoutedExpertPlacementTopology::TieredOverlay;
        std::string continuation_domain;
        std::string base_model_domain;
        std::string shared_expert_domain;
        MoEContinuationDomainSpec continuation_domain_spec;
        RoutedExpertResidencyPolicy residency_policy =
            RoutedExpertResidencyPolicy::Disabled;

        /// Generic dense execution domains for continuation/base/shared model flow.
        /// These may use SINGLE, LOCAL, NODE_LOCAL, or GLOBAL scope and are
        /// validated separately from routed whole-expert ownership domains.
        std::vector<ExecutionDomainDefinition> dense_domains;

        std::vector<RoutedExpertDomain> domains;
        std::vector<RoutedExpertTier> routed_tiers;
        std::vector<RoutedExpertLayerPlacement> placements;

        bool isTieredOverlay() const
        {
            return enabled && topology == RoutedExpertPlacementTopology::TieredOverlay;
        }

        std::string effectiveBaseModelDomain() const
        {
            return base_model_domain.empty() ? continuation_domain : base_model_domain;
        }
    };

    struct MoERoutedExpertPlacementValidationOptions
    {
        /// When > 0 and placements are provided, require one placement per layer [0, layer_count).
        int layer_count = 0;

        /// When > 0 and placements are provided, each placement must cover exactly this many experts.
        int routed_expert_count = 0;

    };

    struct MoERoutedExpertPlacementValidationResult
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

    inline const char *toString(RoutedExpertPlacementTopology topology)
    {
        switch (topology)
        {
        case RoutedExpertPlacementTopology::SingleDomain:
            return "single-domain";
        case RoutedExpertPlacementTopology::TieredOverlay:
            return "tiered-overlay";
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

    inline const char *toString(RoutedExpertResidencyPolicy policy)
    {
        switch (policy)
        {
        case RoutedExpertResidencyPolicy::Disabled:
            return "Disabled";
        case RoutedExpertResidencyPolicy::StaticById:
            return "StaticById";
        case RoutedExpertResidencyPolicy::HistogramTieredCache:
            return "HistogramTieredCache";
        case RoutedExpertResidencyPolicy::ExplicitMasks:
            return "ExplicitMasks";
        case RoutedExpertResidencyPolicy::RoutedTierRebalanced:
            return "RoutedTierRebalanced";
        }
        return "Unknown";
    }

    inline std::string renderMoERoutedExpertPlacementPlanExplanation(
        const MoERoutedExpertPlacementPlan &plan,
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

        auto findRoutedExpertDomain = [&](const std::string &name) -> const RoutedExpertDomain *
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
        out << "  moe_routed_expert_placement:\n";
        out << "    enabled: " << (plan.enabled ? "true" : "false") << "\n";
        if (!plan.enabled)
        {
            out << "    execution: disabled\n";
            return out.str();
        }

        out << "    topology: graph-native " << toString(plan.topology) << "\n";
        out << "    residency_policy: " << toString(plan.residency_policy) << "\n";
        out << "    continuation_domain: " << plan.continuation_domain
            << " root_participant=" << plan.continuation_domain_spec.logical_root_participant
            << " dense_policy=" << denseParallelPolicyToString(plan.continuation_domain_spec.effectiveDensePolicy())
            << " dense_tp=" << (plan.continuation_domain_spec.dense_tp_enabled ? "true" : "false")
            << " dense_decode_replicated=" << (plan.continuation_domain_spec.dense_decode_replicated ? "true" : "false")
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
                out << " scope=" << executionDomainScopeToString(domain.scope)
                    << " routed_compute="
                    << routedExpertComputePolicyToString(domain.routed_compute_policy)
                    << " routed_phase="
                    << routedExpertPhasePolicyToString(domain.routed_phase_policy)
                    << " routed_assignment="
                    << routedExpertAssignmentPolicyToString(domain.routed_assignment_policy)
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
                    if (const auto *domain = findRoutedExpertDomain(tier.domain))
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
        case RoutedExpertResidencyPolicy::RoutedTierRebalanced:
            out << "uses histogram-aware routed tier rebalancing at safe step boundaries";
            break;
        case RoutedExpertResidencyPolicy::HistogramTieredCache:
            out << "uses histogram tier ordering when model-aware planning has histogram data";
            break;
        case RoutedExpertResidencyPolicy::StaticById:
            out << "uses deterministic expert-id ordering across routed tiers";
            break;
        case RoutedExpertResidencyPolicy::ExplicitMasks:
            out << "uses explicit masks/placements supplied by configuration or caller";
            break;
        case RoutedExpertResidencyPolicy::Disabled:
            out << "requires precomputed placements for enabled routed-expert plans";
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

    inline MoERoutedExpertPlacementValidationResult validateMoEContinuationDomainSpec(
        const MoEContinuationDomainSpec &spec,
        const ExecutionDomainDefinition &domain,
        const std::string &role_name = "continuation")
    {
        MoERoutedExpertPlacementValidationResult result;
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

    inline MoERoutedExpertPlacementValidationResult validateMoERoutedExpertPlacementPlan(
        const MoERoutedExpertPlacementPlan &plan,
        const MoERoutedExpertPlacementValidationOptions &options = {})
    {
        MoERoutedExpertPlacementValidationResult result;
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

        std::unordered_map<std::string, const RoutedExpertDomain *> domains_by_name;
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

            if (domain.scope == ExecutionDomainScope::SINGLE &&
                domain.participants.size() > 1)
            {
                addError("routed expert domain '" + domain.name +
                         "' has scope=single but declares multiple participants");
            }

            if (domain.routed_compute_policy == RoutedExpertComputePolicy::Apportioned && !domain.supportsWholeExpertApportionment())
            {
                addError("routed expert domain '" + domain.name +
                         "' uses routed_compute=apportioned but declares no participants");
            }

            const bool apportioned_prefill_over_replicated_weights =
                domain.routed_compute_policy ==
                    RoutedExpertComputePolicy::Replicated &&
                domain.routed_phase_policy ==
                    RoutedExpertPhasePolicy::
                        PrefillApportionedDecodeReplicated;

            if (domain.routed_phase_policy ==
                    RoutedExpertPhasePolicy::
                        PrefillApportionedDecodeReplicated &&
                domain.routed_compute_policy !=
                    RoutedExpertComputePolicy::Replicated)
            {
                addError(
                    "routed expert domain '" + domain.name +
                    "' uses routed_phase=prefill-apportioned-decode-replicated "
                    "but routed_compute is not replicated");
            }

            if (domain.routed_phase_policy ==
                    RoutedExpertPhasePolicy::
                        PrefillApportionedDecodeReplicated &&
                !domain.supportsLeastLoadedResidentAssignment())
            {
                addError(
                    "routed expert domain '" + domain.name +
                    "' uses routed_phase=prefill-apportioned-decode-replicated "
                    "but is not a multi-participant collective domain");
            }

            if (domain.routed_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident &&
                domain.routed_compute_policy != RoutedExpertComputePolicy::Apportioned &&
                !apportioned_prefill_over_replicated_weights)
            {
                addError("routed expert domain '" + domain.name +
                         "' uses routed_assignment=least-loaded-resident but does not use routed_compute=apportioned");
            }

            if (domain.routed_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident &&
                !domain.supportsLeastLoadedResidentAssignment())
            {
                addError("routed expert domain '" + domain.name +
                         "' uses routed_assignment=least-loaded-resident but is not a multi-participant collective domain");
            }

            if (domain.routed_compute_policy == RoutedExpertComputePolicy::TensorSharded && !domain.supportsRoutedExpertTensorSharding())
            {
                addError("routed expert domain '" + domain.name +
                         "' uses routed_compute=tensor-sharded but is not a multi-participant collective domain");
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
            addError("enabled routed-expert placement plan must declare at least one routed tier");
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
                addError("routed tier '" + tier.name +
                         "' references unknown routed expert domain: " + tier.domain);
            }
            else
            {
                routed_domain_names.insert(tier.domain);
            }

            if (tier.fallback)
                ++fallback_count;
        }

        if (fallback_count > 1)
        {
            addError("enabled routed-expert placement plan must declare at most one fallback tier");
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
                addError("graph-native whole-expert routed placement has no fallback tier and non-fallback routed tier capacity covers only " +
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

        if (plan.topology == RoutedExpertPlacementTopology::SingleDomain)
        {
            std::unordered_set<std::string> routed_domains;
            for (const auto &tier : plan.routed_tiers)
            {
                if (!tier.domain.empty())
                    routed_domains.insert(tier.domain);
            }

            if (routed_domains.size() > 1)
            {
                addError("SingleDomain routed-expert plans must use one routed compute domain");
            }
        }

        return result;
    }

    inline bool isValidMoERoutedExpertPlacementPlan(
        const MoERoutedExpertPlacementPlan &plan,
        const MoERoutedExpertPlacementValidationOptions &options = {})
    {
        return validateMoERoutedExpertPlacementPlan(plan, options).ok();
    }

    inline void validateMoERoutedExpertPlacementPlanOrThrow(
        const MoERoutedExpertPlacementPlan &plan,
        const MoERoutedExpertPlacementValidationOptions &options = {})
    {
        auto result = validateMoERoutedExpertPlacementPlan(plan, options);
        if (result.ok())
            return;

        std::ostringstream message;
        message << "Invalid routed-expert placement plan:";
        for (const auto &error : result.errors)
        {
            message << "\n - " << error;
        }
        throw std::invalid_argument(message.str());
    }

} // namespace llaminar2
