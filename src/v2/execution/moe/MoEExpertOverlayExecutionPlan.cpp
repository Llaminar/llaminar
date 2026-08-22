/**
 * @file MoEExpertOverlayExecutionPlan.cpp
 * @brief Resolves portable MoE overlay topology into rank-owned runtime work.
 *
 * This translation unit is the authority that turns model-level domain intent
 * into concrete devices and MPI ownership. Inventory binding deliberately
 * preserves the distinction between rank-local TP (`owner_rank`) and
 * cross-rank NodeTP (`world_ranks`); downstream typed configuration uses
 * that distinction to select the correct collective and graph shape.
 */

#include "MoEExpertOverlayExecutionPlan.h"
#include "execution/mpi_orchestration/DeviceInventory.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace llaminar2
{
    namespace
    {
        template <typename T>
        void addUnique(std::vector<T> &values, const T &value)
        {
            if (std::find(values.begin(), values.end(), value) == values.end())
                values.push_back(value);
        }

        template <typename T>
        bool contains(const std::vector<T> &values, const T &value)
        {
            return std::find(values.begin(), values.end(), value) != values.end();
        }

        std::string formatExecutionPlanErrors(const std::vector<std::string> &errors)
        {
            std::ostringstream message;
            message << "Invalid MoE expert overlay execution plan:";
            for (const auto &error : errors)
                message << "\n - " << error;
            return message.str();
        }

        std::vector<int> deterministicParticipantRanks(const RoutedExpertDomain &domain)
        {
            if (!domain.world_ranks.empty())
                return domain.world_ranks;

            if (domain.scope == ExecutionDomainScope::NODE_LOCAL)
            {
                std::vector<int> ranks;
                ranks.reserve(domain.participants.size());
                for (size_t index = 0; index < domain.participants.size(); ++index)
                    ranks.push_back(static_cast<int>(index));
                return ranks;
            }

            if (domain.owner_rank >= 0)
                return std::vector<int>(domain.participants.size(), domain.owner_rank);

            if (domain.scope == ExecutionDomainScope::SINGLE)
                return std::vector<int>(domain.participants.size(), 0);

            return std::vector<int>(domain.participants.size(), -1);
        }

        int inferredOwnerRank(const RoutedExpertDomain &domain, const std::vector<int> &participant_ranks)
        {
            if (domain.owner_rank >= 0)
                return domain.owner_rank;
            if (!participant_ranks.empty())
                return participant_ranks.front();
            return -1;
        }

        int inferMinimumWorldSize(const MoERoutedExpertPlacementPlan &plan, int current_world_rank)
        {
            int world_size = std::max(1, current_world_rank + 1);
            for (const auto &domain : plan.domains)
            {
                if (domain.owner_rank >= 0)
                    world_size = std::max(world_size, domain.owner_rank + 1);

                if (!domain.world_ranks.empty())
                {
                    for (int rank : domain.world_ranks)
                        world_size = std::max(world_size, rank + 1);
                }
                else if (domain.scope == ExecutionDomainScope::NODE_LOCAL)
                {
                    world_size = std::max(world_size, static_cast<int>(domain.participants.size()));
                }
            }
            return world_size;
        }

        std::unordered_map<std::string, const RoutedExpertDomain *> sourceDomainsByName(
            const MoERoutedExpertPlacementPlan &plan)
        {
            std::unordered_map<std::string, const RoutedExpertDomain *> by_name;
            for (const auto &domain : plan.domains)
                by_name.emplace(domain.name, &domain);
            return by_name;
        }

        const RoutedExpertDomain *findSourceDomain(
            const std::unordered_map<std::string, const RoutedExpertDomain *> &by_name,
            const std::string &name)
        {
            auto it = by_name.find(name);
            if (it == by_name.end())
                return nullptr;
            return it->second;
        }

        int domainOwnerRankOrInvalid(
            const std::unordered_map<std::string, const RoutedExpertDomain *> &by_name,
            const std::string &name)
        {
            const auto *domain = findSourceDomain(by_name, name);
            if (!domain)
                return -1;
            return inferredOwnerRank(*domain, deterministicParticipantRanks(*domain));
        }

        std::vector<std::string> validateExecutionTopology(
            const MoERoutedExpertPlacementPlan &plan,
            int current_world_rank,
            int world_size)
        {
            std::vector<std::string> errors;
            const auto validation = validateMoERoutedExpertPlacementPlan(plan);
            for (const auto &error : validation.errors)
                errors.push_back(error);

            if (current_world_rank < 0 || current_world_rank >= world_size)
            {
                errors.push_back("current rank " + std::to_string(current_world_rank) +
                                 " is outside overlay MPI world size " + std::to_string(world_size));
            }

            const auto by_name = sourceDomainsByName(plan);
            for (const auto &domain : plan.domains)
            {
                const auto participant_ranks = deterministicParticipantRanks(domain);
                if (contains(participant_ranks, -1))
                {
                    errors.push_back("domain '" + domain.name +
                                     "' has ambiguous rank ownership; set owner=<rank> or ranks=<rank-list>");
                    continue;
                }

                if (domain.owner_rank >= world_size)
                {
                    errors.push_back("domain '" + domain.name + "' owner rank " +
                                     std::to_string(domain.owner_rank) +
                                     " is outside overlay MPI world size " + std::to_string(world_size));
                }

                std::set<int> distinct_ranks;
                for (size_t index = 0; index < participant_ranks.size(); ++index)
                {
                    const int rank = participant_ranks[index];
                    distinct_ranks.insert(rank);
                    if (rank < 0 || rank >= world_size)
                    {
                        errors.push_back("domain '" + domain.name + "' participant " +
                                         std::to_string(index) + " maps to rank " +
                                         std::to_string(rank) +
                                         " outside overlay MPI world size " + std::to_string(world_size));
                    }
                }

                if (domain.scope == ExecutionDomainScope::RANK_LOCAL && distinct_ranks.size() > 1)
                {
                    errors.push_back("domain '" + domain.name +
                                     "' is LocalTP but maps participants to multiple ranks; use NodeTP for cross-rank domains");
                }
            }

            const int continuation_owner = domainOwnerRankOrInvalid(by_name, plan.continuation_domain);
            if (continuation_owner < 0)
            {
                errors.push_back("continuation domain '" + plan.continuation_domain +
                                 "' does not resolve to a deterministic owner rank");
            }

            const int shared_owner = domainOwnerRankOrInvalid(by_name, plan.shared_expert_domain);
            if (shared_owner < 0)
            {
                errors.push_back("shared expert domain '" + plan.shared_expert_domain +
                                 "' does not resolve to a deterministic owner rank");
            }
            else if (continuation_owner >= 0 && shared_owner != continuation_owner)
            {
                errors.push_back("shared expert domain '" + plan.shared_expert_domain +
                                 "' owner rank " + std::to_string(shared_owner) +
                                 " does not match continuation root rank " +
                                 std::to_string(continuation_owner));
            }

            const std::string base_domain_name = plan.effectiveBaseModelDomain();
            const int base_owner = domainOwnerRankOrInvalid(by_name, base_domain_name);
            if (base_owner < 0)
            {
                errors.push_back("base/non-expert model domain '" + base_domain_name +
                                 "' does not resolve to a deterministic owner rank");
            }
            else if (continuation_owner >= 0 && base_owner != continuation_owner)
            {
                errors.push_back("base/non-expert model domain '" + base_domain_name +
                                 "' owner rank " + std::to_string(base_owner) +
                                 " does not match continuation root rank " +
                                 std::to_string(continuation_owner));
            }

            return errors;
        }

        std::set<std::string> fallbackDomainNames(const MoERoutedExpertPlacementPlan &plan)
        {
            std::set<std::string> domains;
            for (const auto &tier : plan.routed_tiers)
            {
                if (tier.fallback)
                    domains.insert(tier.domain);
            }
            return domains;
        }

        std::set<std::string> routedDomainNames(const MoERoutedExpertPlacementPlan &plan)
        {
            std::set<std::string> domains;
            for (const auto &tier : plan.routed_tiers)
                domains.insert(tier.domain);
            return domains;
        }

        struct DomainRoleDescriptor
        {
            const RoutedExpertDomain *source = nullptr;
            const MoEOverlayRuntimeDomain *runtime = nullptr;
            std::vector<int> participant_ranks;
            int owner_rank = -1;
        };

        std::vector<DomainRoleDescriptor> buildDomainRoleDescriptors(
            const MoERoutedExpertPlacementPlan &source_plan,
            const std::vector<MoEOverlayRuntimeDomain> &runtime_domains)
        {
            std::unordered_map<std::string, const MoEOverlayRuntimeDomain *> runtime_by_name;
            for (const auto &domain : runtime_domains)
                runtime_by_name.emplace(domain.name, &domain);

            std::vector<DomainRoleDescriptor> descriptors;
            descriptors.reserve(source_plan.domains.size());
            for (const auto &domain : source_plan.domains)
            {
                DomainRoleDescriptor descriptor;
                descriptor.source = &domain;
                auto runtime_it = runtime_by_name.find(domain.name);
                if (runtime_it != runtime_by_name.end())
                    descriptor.runtime = runtime_it->second;
                descriptor.participant_ranks = deterministicParticipantRanks(domain);
                descriptor.owner_rank = inferredOwnerRank(domain, descriptor.participant_ranks);
                descriptors.push_back(std::move(descriptor));
            }
            return descriptors;
        }

        const DomainRoleDescriptor *descriptorForDomain(
            const std::vector<DomainRoleDescriptor> &descriptors,
            const std::string &name)
        {
            auto it = std::find_if(descriptors.begin(), descriptors.end(),
                                   [&](const auto &descriptor)
                                   {
                                       return descriptor.source && descriptor.source->name == name;
                                   });
            if (it == descriptors.end())
                return nullptr;
            return &*it;
        }

        bool rankParticipatesInDomain(const DomainRoleDescriptor &descriptor, int rank)
        {
            return contains(descriptor.participant_ranks, rank);
        }

        bool rankHasDeviceType(
            const DomainRoleDescriptor &descriptor,
            int rank,
            bool want_gpu)
        {
            if (!descriptor.source)
                return false;
            for (size_t index = 0; index < descriptor.source->participants.size(); ++index)
            {
                if (index >= descriptor.participant_ranks.size() || descriptor.participant_ranks[index] != rank)
                    continue;
                if (want_gpu && descriptor.source->participants[index].isGPU())
                    return true;
                if (!want_gpu && descriptor.source->participants[index].isCPU())
                    return true;
            }
            return false;
        }

        void addDomainDevicesForRank(
            OverlayRankPlan &rank_plan,
            const DomainRoleDescriptor &descriptor,
            int rank)
        {
            if (!descriptor.source)
                return;
            for (size_t index = 0; index < descriptor.source->participants.size(); ++index)
            {
                if (index >= descriptor.participant_ranks.size() || descriptor.participant_ranks[index] != rank)
                    continue;
                addUnique(rank_plan.local_devices, descriptor.source->participants[index].toLocalDeviceId());
            }
        }

        bool isCpuFallbackDomain(
            const DomainRoleDescriptor &descriptor,
            const std::set<std::string> &fallback_domains)
        {
            return descriptor.source &&
                   fallback_domains.find(descriptor.source->name) != fallback_domains.end() &&
                   std::any_of(descriptor.source->participants.begin(), descriptor.source->participants.end(),
                               [](const auto &participant)
                               {
                                   return participant.isCPU();
                               });
        }

        bool isLocalAcceleratorParticipantDomain(
            const DomainRoleDescriptor &descriptor,
            int rank,
            int continuation_root_rank)
        {
            if (!descriptor.source || !rankHasDeviceType(descriptor, rank, true))
                return false;

            if (descriptor.source->scope == ExecutionDomainScope::RANK_LOCAL &&
                descriptor.source->routed_compute_policy ==
                    RoutedExpertComputePolicy::TensorSharded)
            {
                return true;
            }

            return descriptor.source->scope == ExecutionDomainScope::SINGLE &&
                   descriptor.owner_rank == continuation_root_rank &&
                   rank == continuation_root_rank;
        }

        bool isExpertDomain(
            const DomainRoleDescriptor &descriptor,
            const std::set<std::string> &routed_domains,
            const std::string &shared_domain)
        {
            return descriptor.source &&
                   (descriptor.source->name == shared_domain ||
                    routed_domains.find(descriptor.source->name) != routed_domains.end());
        }

        OverlayRankRole primaryRoleFor(const OverlayRankPlan &rank_plan)
        {
            if (rank_plan.hasRole(OverlayRankRole::ContinuationRoot))
                return OverlayRankRole::ContinuationRoot;
            if (rank_plan.hasRole(
                    OverlayRankRole::ContinuationParticipant))
                return OverlayRankRole::ContinuationParticipant;
            if (rank_plan.hasRole(OverlayRankRole::LocalAcceleratorParticipant))
                return OverlayRankRole::LocalAcceleratorParticipant;
            if (rank_plan.hasRole(OverlayRankRole::CpuFallbackParticipant))
                return OverlayRankRole::CpuFallbackParticipant;
            if (rank_plan.hasRole(OverlayRankRole::RemoteExpertParticipant))
                return OverlayRankRole::RemoteExpertParticipant;
            return OverlayRankRole::RelayOnly;
        }

        OverlayRankExecutionKind executionKindFor(
            const OverlayRankPlan &rank_plan)
        {
            const bool is_authority =
                rank_plan.hasRole(OverlayRankRole::ContinuationRoot);
            const bool is_peer =
                rank_plan.hasRole(OverlayRankRole::ContinuationParticipant);
            if (is_authority && is_peer)
            {
                throw std::invalid_argument(
                    "ExpertOverlay rank cannot be both continuation authority and continuation peer");
            }
            if (is_authority)
                return OverlayRankExecutionKind::ContinuationAuthority;
            if (is_peer)
                return OverlayRankExecutionKind::ContinuationPeer;

            const bool owns_expert_endpoint =
                rank_plan.hasRole(
                    OverlayRankRole::LocalAcceleratorParticipant) ||
                rank_plan.hasRole(
                    OverlayRankRole::CpuFallbackParticipant) ||
                rank_plan.hasRole(
                    OverlayRankRole::RemoteExpertParticipant);
            return owns_expert_endpoint
                       ? OverlayRankExecutionKind::ExpertOnlyFollower
                       : OverlayRankExecutionKind::RelayOnly;
        }

        void appendRoles(std::ostringstream &out, const std::vector<OverlayRankRole> &roles)
        {
            if (roles.empty())
            {
                out << "<none>";
                return;
            }
            for (size_t index = 0; index < roles.size(); ++index)
            {
                if (index != 0)
                    out << ",";
                out << toString(roles[index]);
            }
        }

        template <typename T, typename Formatter>
        void appendList(std::ostringstream &out, const std::vector<T> &values, Formatter formatter)
        {
            if (values.empty())
            {
                out << "<none>";
                return;
            }
            for (size_t index = 0; index < values.size(); ++index)
            {
                if (index != 0)
                    out << ",";
                out << formatter(values[index]);
            }
        }

        bool wildcardLocalHostname(const std::string &hostname)
        {
            return hostname.empty() || hostname == "localhost";
        }

        bool rankMatchesParticipantHost(
            const RankInventory &rank,
            const GlobalDeviceAddress &participant)
        {
            return wildcardLocalHostname(participant.hostname) ||
                   rank.hostname == participant.hostname;
        }

        std::vector<int> uniqueSortedRanks(std::vector<int> ranks)
        {
            std::sort(ranks.begin(), ranks.end());
            ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
            return ranks;
        }

        struct ResolvedParticipantBinding
        {
            int world_rank = -1;
            GlobalDeviceAddress address;
        };

        struct GpuBindingCandidate
        {
            const RankInventory *rank = nullptr;
            const DeviceInfo *device = nullptr;
        };

        ResolvedParticipantBinding resolveGpuParticipantBinding(
            const GlobalDeviceAddress &participant,
            const ClusterInventory &inventory,
            const std::string &domain_name,
            std::optional<int> pinned_rank)
        {
            std::vector<GpuBindingCandidate> visible;
            std::vector<GpuBindingCandidate> local;
            for (const auto &rank : inventory.ranks)
            {
                if (pinned_rank.has_value() && rank.rank != *pinned_rank)
                    continue;
                if (!rankMatchesParticipantHost(rank, participant))
                    continue;
                for (const auto &gpu : rank.gpus)
                {
                    if (gpu.type != participant.device_type ||
                        gpu.local_device_id != participant.device_ordinal)
                    {
                        continue;
                    }
                    if (participant.hasValidNuma() &&
                        gpu.numa_node >= 0 &&
                        gpu.numa_node != participant.numa_node)
                    {
                        continue;
                    }
                    visible.push_back(GpuBindingCandidate{&rank, &gpu});
                    if (gpu.numa_node >= 0 &&
                        rank.local_rank == gpu.numa_node)
                    {
                        local.push_back(GpuBindingCandidate{&rank, &gpu});
                    }
                }
            }

            auto uniqueByRank = [](std::vector<GpuBindingCandidate> values)
            {
                std::sort(
                    values.begin(),
                    values.end(),
                    [](const auto &left, const auto &right)
                    {
                        return left.rank->rank < right.rank->rank;
                    });
                values.erase(
                    std::unique(
                        values.begin(),
                        values.end(),
                        [](const auto &left, const auto &right)
                        {
                            return left.rank->rank == right.rank->rank;
                        }),
                    values.end());
                return values;
            };
            visible = uniqueByRank(std::move(visible));
            local = uniqueByRank(std::move(local));
            const auto &candidates =
                (pinned_rank.has_value() || local.empty()) ? visible : local;
            if (candidates.empty())
            {
                std::ostringstream error;
                error << "MoE overlay domain '" << domain_name
                      << "' requires " << participant.toShortString();
                if (pinned_rank.has_value())
                    error << " on explicitly pinned MPI rank " << *pinned_rank;
                error << ", but the discovered cluster inventory does not expose that device";
                throw std::invalid_argument(error.str());
            }
            if (candidates.size() != 1u)
            {
                std::ostringstream error;
                error << "MoE overlay domain '" << domain_name
                      << "' device " << participant.toShortString()
                      << " is visible from multiple equally local MPI ranks [";
                for (size_t index = 0; index < candidates.size(); ++index)
                {
                    if (index != 0)
                        error << ",";
                    error << candidates[index].rank->rank;
                }
                error << "]; specify an explicit NUMA node or world-rank binding";
                throw std::invalid_argument(error.str());
            }

            const auto &selected = candidates.front();
            GlobalDeviceAddress address = participant;
            if (wildcardLocalHostname(address.hostname) &&
                !selected.rank->hostname.empty())
            {
                address.hostname = selected.rank->hostname;
            }
            if (!address.hasValidNuma() && selected.device->numa_node >= 0)
                address.numa_node = selected.device->numa_node;
            return ResolvedParticipantBinding{
                .world_rank = selected.rank->rank,
                .address = std::move(address),
            };
        }

        ResolvedParticipantBinding resolveCpuParticipantBinding(
            const GlobalDeviceAddress &participant,
            const ClusterInventory &inventory,
            const std::string &domain_name,
            size_t participant_index,
            std::optional<int> pinned_rank)
        {
            std::vector<const RankInventory *> candidates;
            for (const auto &rank : inventory.ranks)
            {
                if (pinned_rank.has_value() && rank.rank != *pinned_rank)
                    continue;
                if (!rankMatchesParticipantHost(rank, participant))
                    continue;
                if (participant.hasValidNuma() &&
                    rank.local_rank != participant.numa_node)
                {
                    continue;
                }
                candidates.push_back(&rank);
            }

            if (candidates.empty())
            {
                std::ostringstream error;
                error << "MoE overlay domain '" << domain_name
                      << "' requires CPU participant "
                      << participant.toShortString();
                if (pinned_rank.has_value())
                    error << " on explicitly pinned MPI rank " << *pinned_rank;
                error << ", but no discovered rank owns that CPU/NUMA address";
                throw std::invalid_argument(error.str());
            }

            const RankInventory *selected = nullptr;
            if (pinned_rank.has_value() || candidates.size() == 1u)
            {
                selected = candidates.front();
            }
            else if (!participant.hasValidNuma())
            {
                std::set<int> node_ids;
                for (const auto *candidate : candidates)
                    node_ids.insert(candidate->node_id);
                if (node_ids.size() > 1u)
                {
                    throw std::invalid_argument(
                        "MoE overlay domain '" + domain_name +
                        "' uses an unqualified CPU participant across multiple hosts; specify hostname and NUMA node");
                }
                std::sort(
                    candidates.begin(),
                    candidates.end(),
                    [](const auto *left, const auto *right)
                    {
                        if (left->local_rank != right->local_rank)
                            return left->local_rank < right->local_rank;
                        return left->rank < right->rank;
                    });
                if (participant_index < candidates.size())
                    selected = candidates[participant_index];
            }

            if (!selected)
            {
                throw std::invalid_argument(
                    "MoE overlay domain '" + domain_name +
                    "' CPU participant " + participant.toShortString() +
                    " is ambiguous across MPI ranks; specify a NUMA node");
            }

            GlobalDeviceAddress address = participant;
            if (wildcardLocalHostname(address.hostname) &&
                !selected->hostname.empty())
            {
                address.hostname = selected->hostname;
            }
            if (!address.hasValidNuma())
                address.numa_node = selected->local_rank;
            return ResolvedParticipantBinding{
                .world_rank = selected->rank,
                .address = std::move(address),
            };
        }

        ResolvedParticipantBinding resolveParticipantBinding(
            const GlobalDeviceAddress &participant,
            const ClusterInventory &inventory,
            const std::string &domain_name,
            size_t participant_index,
            std::optional<int> pinned_rank = std::nullopt)
        {
            if (participant.isGPU())
            {
                return resolveGpuParticipantBinding(
                    participant, inventory, domain_name, pinned_rank);
            }
            if (participant.isCPU())
            {
                return resolveCpuParticipantBinding(
                    participant,
                    inventory,
                    domain_name,
                    participant_index,
                    pinned_rank);
            }
            throw std::invalid_argument(
                "MoE overlay domain '" + domain_name +
                "' uses an unsupported participant type for automatic rank binding");
        }
    } // namespace

    const char *toString(OverlayRankRole role)
    {
        switch (role)
        {
        case OverlayRankRole::ContinuationRoot:
            return "ContinuationRoot";
        case OverlayRankRole::ContinuationParticipant:
            return "ContinuationParticipant";
        case OverlayRankRole::LocalAcceleratorParticipant:
            return "LocalAcceleratorParticipant";
        case OverlayRankRole::CpuFallbackParticipant:
            return "CpuFallbackParticipant";
        case OverlayRankRole::RemoteExpertParticipant:
            return "RemoteExpertParticipant";
        case OverlayRankRole::RelayOnly:
            return "RelayOnly";
        }
        return "Unknown";
    }

    const char *toString(OverlayRankExecutionKind kind)
    {
        switch (kind)
        {
        case OverlayRankExecutionKind::ContinuationAuthority:
            return "ContinuationAuthority";
        case OverlayRankExecutionKind::ContinuationPeer:
            return "ContinuationPeer";
        case OverlayRankExecutionKind::ExpertOnlyFollower:
            return "ExpertOnlyFollower";
        case OverlayRankExecutionKind::RelayOnly:
            return "RelayOnly";
        }
        return "Unknown";
    }

    bool OverlayRankPlan::hasRole(OverlayRankRole candidate) const
    {
        return std::find(roles.begin(), roles.end(), candidate) != roles.end();
    }

    bool OverlayRankPlan::ownsDomain(const std::string &domain_name) const
    {
        return std::find(owned_domains.begin(), owned_domains.end(), domain_name) != owned_domains.end();
    }

    bool OverlayRankPlan::hasLocalDevice(DeviceId device) const
    {
        return std::find(local_devices.begin(), local_devices.end(), device) != local_devices.end();
    }

    bool OverlayRankPlan::ownsContinuationGraph() const
    {
        return execution_kind ==
                   OverlayRankExecutionKind::ContinuationAuthority ||
               execution_kind ==
                   OverlayRankExecutionKind::ContinuationPeer;
    }

    bool OverlayRankPlan::ownsTransactionAuthority() const
    {
        return execution_kind ==
               OverlayRankExecutionKind::ContinuationAuthority;
    }

    bool OverlayRankPlan::usesExpertTransactionFollower() const
    {
        return execution_kind ==
               OverlayRankExecutionKind::ExpertOnlyFollower;
    }

    const OverlayRankPlan *MoEExpertOverlayExecutionPlan::rankPlanFor(int world_rank) const
    {
        auto it = std::find_if(rank_plans.begin(), rank_plans.end(),
                               [&](const auto &rank_plan)
                               {
                                   return rank_plan.world_rank == world_rank;
                               });
        if (it == rank_plans.end())
            return nullptr;
        return &*it;
    }

    MoEExpertOverlayExecutionPlan buildMoEExpertOverlayExecutionPlan(
        const MoEExpertOverlayRuntimePlan &runtime_plan,
        int requested_world_size)
    {
        const auto &source_plan = runtime_plan.sourcePlan();
        const int world_size = requested_world_size > 0
                                   ? requested_world_size
                                   : inferMinimumWorldSize(source_plan, runtime_plan.currentWorldRank());

        const auto errors = validateExecutionTopology(source_plan, runtime_plan.currentWorldRank(), world_size);
        if (!errors.empty())
            throw std::invalid_argument(formatExecutionPlanErrors(errors));

        MoEExpertOverlayExecutionPlan result;
        result.world_size = world_size;
        result.continuation_domain = source_plan.continuation_domain;
        result.base_model_domain = source_plan.effectiveBaseModelDomain();
        result.shared_expert_domain = source_plan.shared_expert_domain;
        result.domains = runtime_plan.domains();

        const auto descriptors = buildDomainRoleDescriptors(source_plan, runtime_plan.domains());
        const auto *continuation = descriptorForDomain(descriptors, source_plan.continuation_domain);
        if (!continuation)
            throw std::invalid_argument("MoEExpertOverlayExecutionPlan could not find continuation domain '" +
                                        source_plan.continuation_domain + "'");
        const auto *base_model = descriptorForDomain(descriptors, result.base_model_domain);
        if (!base_model)
            throw std::invalid_argument("MoEExpertOverlayExecutionPlan could not find base/non-expert model domain '" +
                                        result.base_model_domain + "'");
        result.continuation_root_rank = continuation->owner_rank;

        const bool node_local_dense_continuation =
            base_model->source &&
            base_model->source->scope == ExecutionDomainScope::NODE_LOCAL;
        if (node_local_dense_continuation &&
            source_plan.continuation_domain_spec.effectiveDensePolicy() !=
                DenseParallelPolicy::TensorParallel)
        {
            throw std::invalid_argument(
                "NodeLocal ExpertOverlay continuation domain '" +
                result.base_model_domain +
                "' requires dense_policy=tensor_parallel");
        }
        if (node_local_dense_continuation &&
            !source_plan.continuation_domain_spec.domain.empty() &&
            source_plan.continuation_domain_spec.domain !=
                result.base_model_domain)
        {
            throw std::invalid_argument(
                "NodeLocal ExpertOverlay continuation policy names domain '" +
                source_plan.continuation_domain_spec.domain +
                "' but the base continuation domain is '" +
                result.base_model_domain + "'");
        }

        result.rank_plans.reserve(static_cast<size_t>(world_size));
        for (int rank = 0; rank < world_size; ++rank)
        {
            OverlayRankPlan rank_plan;
            rank_plan.world_rank = rank;
            result.rank_plans.push_back(std::move(rank_plan));
        }

        const auto fallback_domains = fallbackDomainNames(source_plan);
        const auto routed_domains = routedDomainNames(source_plan);

        for (auto &rank_plan : result.rank_plans)
        {
            const bool command_root =
                rank_plan.world_rank == result.continuation_root_rank;
            const bool distributed_dense_participant =
                node_local_dense_continuation &&
                rankParticipatesInDomain(
                    *base_model,
                    rank_plan.world_rank);
            if (command_root)
            {
                addUnique(rank_plan.roles, OverlayRankRole::ContinuationRoot);
            }
            else if (distributed_dense_participant)
            {
                addUnique(
                    rank_plan.roles,
                    OverlayRankRole::ContinuationParticipant);
            }

            if (command_root || distributed_dense_participant)
            {
                addUnique(rank_plan.owned_domains, source_plan.continuation_domain);
                addUnique(rank_plan.owned_domains, result.base_model_domain);
                addUnique(rank_plan.root_weight_domains, result.base_model_domain);
                addUnique(rank_plan.shared_expert_weight_domains, source_plan.shared_expert_domain);
                addDomainDevicesForRank(rank_plan, *continuation, rank_plan.world_rank);
                addDomainDevicesForRank(rank_plan, *base_model, rank_plan.world_rank);
            }

            for (const auto &descriptor : descriptors)
            {
                if (!descriptor.source || !rankParticipatesInDomain(descriptor, rank_plan.world_rank))
                    continue;

                addDomainDevicesForRank(rank_plan, descriptor, rank_plan.world_rank);

                if (isCpuFallbackDomain(descriptor, fallback_domains) &&
                    rankHasDeviceType(descriptor, rank_plan.world_rank, false))
                {
                    addUnique(rank_plan.roles, OverlayRankRole::CpuFallbackParticipant);
                    addUnique(rank_plan.owned_domains, descriptor.source->name);
                    addUnique(rank_plan.cpu_fallback_expert_domains, descriptor.source->name);
                    if (!command_root && !distributed_dense_participant)
                        addUnique(rank_plan.worker_fallback_expert_domains, descriptor.source->name);
                    continue;
                }

                if (isLocalAcceleratorParticipantDomain(
                        descriptor, rank_plan.world_rank, result.continuation_root_rank))
                {
                    addUnique(rank_plan.roles, OverlayRankRole::LocalAcceleratorParticipant);
                    addUnique(rank_plan.owned_domains, descriptor.source->name);
                    addUnique(rank_plan.accelerator_routed_expert_domains, descriptor.source->name);
                    continue;
                }

                if (descriptor.source->name != source_plan.continuation_domain &&
                    isExpertDomain(descriptor, routed_domains, source_plan.shared_expert_domain))
                {
                    addUnique(rank_plan.roles, OverlayRankRole::RemoteExpertParticipant);
                    addUnique(rank_plan.owned_domains, descriptor.source->name);
                    if (rankHasDeviceType(descriptor, rank_plan.world_rank, true))
                        addUnique(rank_plan.accelerator_routed_expert_domains, descriptor.source->name);
                    else if (rankHasDeviceType(descriptor, rank_plan.world_rank, false))
                        addUnique(rank_plan.cpu_fallback_expert_domains, descriptor.source->name);
                }
            }

            if (rank_plan.roles.empty())
                addUnique(rank_plan.roles, OverlayRankRole::RelayOnly);

            rank_plan.role = primaryRoleFor(rank_plan);
            rank_plan.execution_kind = executionKindFor(rank_plan);
            rank_plan.loads_tokenizer =
                rank_plan.hasRole(OverlayRankRole::ContinuationRoot);
            rank_plan.loads_worker_tokenizer_state = !rank_plan.ownsContinuationGraph() &&
                                                     rank_plan.hasRole(OverlayRankRole::CpuFallbackParticipant);
            rank_plan.loads_full_model_metadata = !rank_plan.hasRole(OverlayRankRole::RelayOnly);
            rank_plan.loads_root_weights = rank_plan.ownsContinuationGraph();
            rank_plan.loads_shared_expert_weights = !rank_plan.shared_expert_weight_domains.empty();
            rank_plan.loads_accelerator_routed_experts = !rank_plan.accelerator_routed_expert_domains.empty();
            rank_plan.loads_cpu_fallback_experts = !rank_plan.cpu_fallback_expert_domains.empty();
            rank_plan.loads_worker_fallback_experts = !rank_plan.worker_fallback_expert_domains.empty();
            rank_plan.loads_expert_weights = rank_plan.loads_shared_expert_weights ||
                                             rank_plan.loads_accelerator_routed_experts ||
                                             rank_plan.loads_cpu_fallback_experts ||
                                             rank_plan.loads_worker_fallback_experts;
        }

        const auto *current = result.rankPlanFor(runtime_plan.currentWorldRank());
        if (!current)
        {
            throw std::invalid_argument("MoEExpertOverlayExecutionPlan current rank " +
                                        std::to_string(runtime_plan.currentWorldRank()) +
                                        " is outside planned rank list");
        }
        result.current_rank = *current;
        return result;
    }

    MoEExpertOverlayExecutionPlan resolveMoEExpertOverlayExecutionPlan(
        std::shared_ptr<const MoERoutedExpertPlacementPlan> plan,
        int current_world_rank)
    {
        return resolveMoEExpertOverlayExecutionPlan(
            std::move(plan),
            MoEExpertOverlayExecutionPlanResolverOptions{
                .current_world_rank = current_world_rank,
                .world_size = 0,
            });
    }

    MoEExpertOverlayExecutionPlan resolveMoEExpertOverlayExecutionPlan(
        std::shared_ptr<const MoERoutedExpertPlacementPlan> plan,
        const MoEExpertOverlayExecutionPlanResolverOptions &options)
    {
        if (!plan || !plan->usesExpertOverlayAuthority())
            throw std::invalid_argument("MoEExpertOverlayExecutionPlan requires an enabled tiered overlay plan");

        const int world_size = options.world_size > 0
                                   ? options.world_size
                                   : inferMinimumWorldSize(*plan, options.current_world_rank);

        const auto errors = validateExecutionTopology(*plan, options.current_world_rank, world_size);
        if (!errors.empty())
            throw std::invalid_argument(formatExecutionPlanErrors(errors));

        auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(
            std::move(plan),
            MoEExpertOverlayRuntimeResolverOptions{
                .current_world_rank = options.current_world_rank,
                .validate_mvp_root_reachability = false,
            });
        if (!runtime_plan)
            throw std::invalid_argument("MoEExpertOverlayExecutionPlan requires an enabled tiered overlay plan");
        return buildMoEExpertOverlayExecutionPlan(*runtime_plan, world_size);
    }

    std::shared_ptr<MoERoutedExpertPlacementPlan>
    bindMoEExpertOverlayPlanToClusterInventory(
        const MoERoutedExpertPlacementPlan &plan,
        const ClusterInventory &inventory)
    {
        if (!plan.usesExpertOverlayAuthority())
        {
            throw std::invalid_argument(
                "Automatic MoE overlay rank binding requires an enabled tiered overlay plan");
        }
        if (inventory.ranks.empty() || inventory.world_size <= 0)
        {
            throw std::invalid_argument(
                "Automatic MoE overlay rank binding requires a non-empty cluster inventory");
        }

        auto bound = std::make_shared<MoERoutedExpertPlacementPlan>(plan);
        for (auto &domain : bound->domains)
        {
            if (domain.participants.empty())
            {
                throw std::invalid_argument(
                    "MoE overlay domain '" + domain.name +
                    "' has no participants to bind");
            }

            std::vector<std::optional<int>> requested_ranks(
                domain.participants.size(), std::nullopt);
            if (!domain.world_ranks.empty())
            {
                if (domain.world_ranks.size() != domain.participants.size())
                {
                    throw std::invalid_argument(
                        "MoE overlay domain '" + domain.name +
                        "' has " + std::to_string(domain.world_ranks.size()) +
                        " explicit rank bindings for " +
                        std::to_string(domain.participants.size()) +
                        " participants");
                }
                for (size_t index = 0; index < domain.world_ranks.size(); ++index)
                    requested_ranks[index] = domain.world_ranks[index];
            }
            else if (domain.owner_rank >= 0)
            {
                std::fill(
                    requested_ranks.begin(),
                    requested_ranks.end(),
                    std::optional<int>(domain.owner_rank));
            }

            std::vector<int> resolved_ranks;
            resolved_ranks.reserve(domain.participants.size());
            for (size_t participant_index = 0;
                 participant_index < domain.participants.size();
                 ++participant_index)
            {
                auto binding = resolveParticipantBinding(
                    domain.participants[participant_index],
                    inventory,
                    domain.name,
                    participant_index,
                    requested_ranks[participant_index]);
                domain.participants[participant_index] =
                    std::move(binding.address);
                resolved_ranks.push_back(binding.world_rank);
            }

            const auto distinct = uniqueSortedRanks(resolved_ranks);
            if (domain.scope == ExecutionDomainScope::AUTO)
            {
                /*
                 * AUTO expresses physical intent without baking the host's
                 * current NUMA/rank placement into the CLI. Resolve it only
                 * after inventory binding: one participant is SINGLE, several
                 * devices on one rank are LocalTP, and participants spanning
                 * ranks are NodeTP. This remains deterministic because every
                 * rank consumes the same gathered inventory.
                 */
                if (domain.participants.size() == 1u)
                    domain.scope = ExecutionDomainScope::SINGLE;
                else if (distinct.size() == 1u)
                    domain.scope = ExecutionDomainScope::RANK_LOCAL;
                else
                    domain.scope = ExecutionDomainScope::NODE_LOCAL;
            }

            if (domain.scope == ExecutionDomainScope::RANK_LOCAL)
            {
                if (distinct.size() != 1u)
                {
                    throw std::invalid_argument(
                        "MoE overlay LocalTP domain '" + domain.name +
                        "' resolves across MPI ranks; declare NodeTP for cross-rank participants");
                }

                if (domain.owner_rank >= 0 && domain.owner_rank != distinct.front())
                {
                    throw std::invalid_argument(
                        "MoE overlay LocalTP domain '" + domain.name +
                        "' explicitly names owner rank " +
                        std::to_string(domain.owner_rank) +
                        " but its participants resolve to rank " +
                        std::to_string(distinct.front()));
                }

                /*
                 * LocalTP is one rank owning several devices. Its single owner
                 * is the complete rank authority, so it needs no parallel
                 * participant-rank map.
                 */
                domain.owner_rank = distinct.front();
                domain.world_ranks.clear();
            }
            else
            {
                /*
                 * SINGLE and NODE_LOCAL retain participant-level bindings.
                 * Repeated NodeTP ranks are valid when one MPI process owns
                 * several devices; collective code derives distinct MPI group
                 * membership separately from this physical participant map.
                 */
                domain.world_ranks = std::move(resolved_ranks);
                if (domain.owner_rank < 0)
                    domain.owner_rank = domain.world_ranks.front();
            }
        }

        /*
         * Dense-domain declarations are another view of the same named
         * hardware pools. Keep their participant addresses and rank bindings
         * identical to the routed-domain view so downstream configuration
         * consumers cannot observe two topologies for one logical domain.
         */
        for (auto &dense_domain : bound->dense_domains)
        {
            auto resolved = std::find_if(
                bound->domains.begin(),
                bound->domains.end(),
                [&](const auto &candidate)
                {
                    return candidate.name == dense_domain.name;
                });
            if (resolved != bound->domains.end())
                dense_domain = resolved->toExecutionDomainDefinition();
        }
        return bound;
    }

    std::string MoEExpertOverlayExecutionPlan::diagnostics() const
    {
        std::ostringstream out;
        out << "MoE expert overlay execution plan: current_rank=" << current_rank.world_rank
            << " world_size=" << world_size
            << " continuation_domain=" << continuation_domain
            << " continuation_root_rank=" << continuation_root_rank
            << " base_model_domain=" << base_model_domain
            << " shared_expert_domain=" << shared_expert_domain;

        for (const auto &rank_plan : rank_plans)
        {
            out << "\n  rank[" << rank_plan.world_rank << "]: primary_role="
                << toString(rank_plan.role)
                << " roles=";
            appendRoles(out, rank_plan.roles);
            out << " owned_domains=";
            appendList(out, rank_plan.owned_domains,
                       [](const std::string &value) -> std::string
                       {
                           return value;
                       });
            out << " root_weight_domains=";
            appendList(out, rank_plan.root_weight_domains,
                       [](const std::string &value) -> std::string
                       {
                           return value;
                       });
            out << " shared_expert_domains=";
            appendList(out, rank_plan.shared_expert_weight_domains,
                       [](const std::string &value) -> std::string
                       {
                           return value;
                       });
            out << " accelerator_routed_domains=";
            appendList(out, rank_plan.accelerator_routed_expert_domains,
                       [](const std::string &value) -> std::string
                       {
                           return value;
                       });
            out << " cpu_fallback_domains=";
            appendList(out, rank_plan.cpu_fallback_expert_domains,
                       [](const std::string &value) -> std::string
                       {
                           return value;
                       });
            out << " worker_fallback_domains=";
            appendList(out, rank_plan.worker_fallback_expert_domains,
                       [](const std::string &value) -> std::string
                       {
                           return value;
                       });
            out << " local_devices=";
            appendList(out, rank_plan.local_devices,
                       [](const DeviceId &device) -> std::string
                       {
                           return device.to_string();
                       });
            out << " execution_kind=" << toString(rank_plan.execution_kind)
                << " owns_continuation_graph=" << (rank_plan.ownsContinuationGraph() ? "true" : "false")
                << " loads_tokenizer=" << (rank_plan.loads_tokenizer ? "true" : "false")
                << " loads_worker_tokenizer_state=" << (rank_plan.loads_worker_tokenizer_state ? "true" : "false")
                << " loads_full_model_metadata=" << (rank_plan.loads_full_model_metadata ? "true" : "false")
                << " loads_root_weights=" << (rank_plan.loads_root_weights ? "true" : "false")
                << " loads_shared_expert_weights=" << (rank_plan.loads_shared_expert_weights ? "true" : "false")
                << " loads_accelerator_routed_experts=" << (rank_plan.loads_accelerator_routed_experts ? "true" : "false")
                << " loads_cpu_fallback_experts=" << (rank_plan.loads_cpu_fallback_experts ? "true" : "false")
                << " loads_worker_fallback_experts=" << (rank_plan.loads_worker_fallback_experts ? "true" : "false")
                << " loads_expert_weights=" << (rank_plan.loads_expert_weights ? "true" : "false");
        }
        return out.str();
    }

} // namespace llaminar2
