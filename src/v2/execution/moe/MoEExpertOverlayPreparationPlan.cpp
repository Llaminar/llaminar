#include "MoEExpertOverlayPreparationPlan.h"

#include "MoEExpertOverlayExecutionPlan.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace llaminar2
{
    namespace
    {
        const char *residencyPolicyName(ExpertResidencyPolicy policy)
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

        MoEExpertOverlayDomainPreparationStats &statsFor(
            MoEExpertOverlayPreparationDiagnostics &diagnostics,
            const std::string &domain_name,
            DeviceId device,
            int participant_index,
            int participant_world_rank,
            bool participant_world_rank_known,
            int owner_world_rank,
            WeightResidencyCategory residency_category,
            ExpertResidencyPolicy residency_policy)
        {
            auto it = std::find_if(diagnostics.domains.begin(), diagnostics.domains.end(),
                                   [&](const auto &stats)
                                   {
                                       return stats.domain_name == domain_name &&
                                              stats.device == device &&
                                              stats.participant_index == participant_index &&
                                              stats.participant_world_rank == participant_world_rank &&
                                              stats.residency_category == residency_category;
                                   });
            if (it != diagnostics.domains.end())
                return *it;

            MoEExpertOverlayDomainPreparationStats stats;
            stats.domain_name = domain_name;
            stats.device = device;
            stats.participant_index = participant_index;
            stats.participant_world_rank = participant_world_rank;
            stats.participant_world_rank_known = participant_world_rank_known;
            stats.owner_world_rank = owner_world_rank;
            stats.residency_category = residency_category;
            stats.residency_policy = residency_policy;
            stats.accelerator = device.is_gpu();
            diagnostics.domains.push_back(std::move(stats));
            return diagnostics.domains.back();
        }

        struct PreparationParticipant
        {
            DeviceId device = DeviceId::invalid();
            int participant_index = -1;
            int world_rank = -1;
            bool world_rank_known = false;
            int owner_rank = -1;
        };

        int effectiveParticipantRank(const MoEOverlayRuntimeDomain &domain, const MoEOverlayDomainParticipant &participant)
        {
            if (participant.world_rank_known)
                return participant.world_rank;
            if (domain.kind == ExpertDomainKind::LocalTP && domain.owner_rank >= 0)
                return domain.owner_rank;
            if (domain.participants.size() == 1 && domain.owner_rank >= 0)
                return domain.owner_rank;
            return participant.world_rank;
        }

        bool effectiveParticipantRankKnown(const MoEOverlayRuntimeDomain &domain, const MoEOverlayDomainParticipant &participant)
        {
            return participant.world_rank_known ||
                   (domain.kind == ExpertDomainKind::LocalTP && domain.owner_rank >= 0) ||
                   (domain.participants.size() == 1 && domain.owner_rank >= 0);
        }

        std::vector<PreparationParticipant> preparationParticipantsFor(const MoEOverlayRuntimeDomain &domain)
        {
            std::vector<PreparationParticipant> participants;
            for (const auto &participant : domain.participants)
            {
                if (!participant.local_device.is_valid())
                    continue;

                PreparationParticipant prepared;
                prepared.device = participant.local_device;
                prepared.participant_index = participant.participant_index;
                prepared.world_rank = effectiveParticipantRank(domain, participant);
                prepared.world_rank_known = effectiveParticipantRankKnown(domain, participant);
                prepared.owner_rank = domain.owner_rank;
                participants.push_back(prepared);
            }

            if (participants.empty() && domain.primary_device.is_valid())
            {
                PreparationParticipant prepared;
                prepared.device = domain.primary_device;
                prepared.participant_index = 0;
                prepared.world_rank = domain.primary_world_rank;
                prepared.world_rank_known = domain.primary_world_rank_known;
                prepared.owner_rank = domain.owner_rank;
                participants.push_back(prepared);
            }

            return participants;
        }

        WeightResidencyCategory routedResidencyCategoryFor(const ExpertRoutedTier &tier, DeviceId device)
        {
            if (tier.fallback || device.is_cpu())
                return WeightResidencyCategory::CpuFallbackExpert;
            return WeightResidencyCategory::AcceleratorRoutedExpert;
        }

        bool requestBelongsToRank(
            const MoEExpertOverlayPreparationRequest &request,
            const OverlayRankPlan &rank_plan)
        {
            if (!rank_plan.ownsDomain(request.domain_name))
                return false;

            if (request.participant_world_rank_known)
                return request.participant_world_rank == rank_plan.world_rank;

            return rank_plan.hasLocalDevice(request.device);
        }

        WeightResidencyCategory categoryForFilteredRank(
            const MoEExpertOverlayPreparationRequest &request,
            const OverlayRankPlan &rank_plan)
        {
            if (request.fallback && request.device.is_cpu() && !rank_plan.builds_root_graph)
                return WeightResidencyCategory::WorkerFallbackExpert;
            return request.residency_category;
        }

        MoEExpertOverlayDomainPreparationStats &recordRequestStats(
            MoEExpertOverlayPreparationDiagnostics &diagnostics,
            const MoEExpertOverlayPreparationRequest &request)
        {
            auto &stats = statsFor(
                diagnostics,
                request.domain_name,
                request.device,
                request.participant_index,
                request.participant_world_rank,
                request.participant_world_rank_known,
                request.owner_world_rank,
                request.residency_category,
                request.residency_policy);
            stats.fallback = stats.fallback || request.fallback;
            stats.memory_budget_bytes = std::max(stats.memory_budget_bytes, request.memory_budget_bytes);
            ++stats.planned_engine_count;
            return stats;
        }

        void sortDiagnostics(MoEExpertOverlayPreparationDiagnostics &diagnostics)
        {
            std::sort(diagnostics.domains.begin(), diagnostics.domains.end(),
                      [](const auto &lhs, const auto &rhs)
                      {
                          if (lhs.domain_name != rhs.domain_name)
                              return lhs.domain_name < rhs.domain_name;
                          if (lhs.device != rhs.device)
                              return lhs.device < rhs.device;
                          if (lhs.participant_world_rank != rhs.participant_world_rank)
                              return lhs.participant_world_rank < rhs.participant_world_rank;
                          if (lhs.participant_index != rhs.participant_index)
                              return lhs.participant_index < rhs.participant_index;
                          return static_cast<int>(lhs.residency_category) < static_cast<int>(rhs.residency_category);
                      });
        }
    } // namespace

    const MoEExpertOverlayDomainPreparationStats *MoEExpertOverlayPreparationDiagnostics::domainStats(
        const std::string &domain_name,
        DeviceId device) const
    {
        auto it = std::find_if(domains.begin(), domains.end(),
                               [&](const auto &stats)
                               {
                                   return stats.domain_name == domain_name && stats.device == device;
                               });
        return it == domains.end() ? nullptr : &(*it);
    }

    const MoEExpertOverlayDomainPreparationStats *MoEExpertOverlayPreparationDiagnostics::domainStats(
        const std::string &domain_name,
        DeviceId device,
        int participant_world_rank,
        int participant_index) const
    {
        auto it = std::find_if(domains.begin(), domains.end(),
                               [&](const auto &stats)
                               {
                                   return stats.domain_name == domain_name &&
                                          stats.device == device &&
                                          stats.participant_world_rank == participant_world_rank &&
                                          stats.participant_index == participant_index;
                               });
        return it == domains.end() ? nullptr : &(*it);
    }

    std::string MoEExpertOverlayPreparationDiagnostics::render() const
    {
        std::ostringstream out;
        out << "MoE expert overlay preparation diagnostics:";
        if (domains.empty())
        {
            out << " no domains";
            return out.str();
        }

        size_t root_bytes = 0;
        size_t shared_bytes = 0;
        size_t accelerator_bytes = 0;
        size_t fallback_bytes = 0;
        size_t worker_bytes = 0;
        for (const auto &stats : domains)
        {
            switch (stats.residency_category)
            {
            case WeightResidencyCategory::RootNonExpert:
                root_bytes += stats.estimated_routed_bytes;
                break;
            case WeightResidencyCategory::SharedExpert:
                shared_bytes += stats.estimated_routed_bytes;
                break;
            case WeightResidencyCategory::AcceleratorRoutedExpert:
                accelerator_bytes += stats.estimated_routed_bytes;
                break;
            case WeightResidencyCategory::CpuFallbackExpert:
                fallback_bytes += stats.estimated_routed_bytes;
                break;
            case WeightResidencyCategory::WorkerFallbackExpert:
                worker_bytes += stats.estimated_routed_bytes;
                break;
            case WeightResidencyCategory::Unspecified:
                break;
            }
        }

        out << " memory_by_role{root=" << root_bytes
            << ", shared=" << shared_bytes
            << ", routed_tier=" << accelerator_bytes
            << ", fallback=" << fallback_bytes
            << ", worker=" << worker_bytes
            << "}";

        for (const auto &stats : domains)
        {
            out << "\n  domain " << stats.domain_name
                << " device=" << stats.device.to_string()
                << " participant=" << stats.participant_index
                << " rank=";
            if (stats.participant_world_rank_known)
                out << stats.participant_world_rank;
            else
                out << "unknown";
            out << " owner_rank=" << stats.owner_world_rank
                << " category=" << toString(stats.residency_category)
                << " residency=" << residencyPolicyName(stats.residency_policy)
                << " accelerator=" << (stats.accelerator ? "true" : "false")
                << " fallback=" << (stats.fallback ? "true" : "false")
                << " assigned_experts=" << stats.assigned_routed_experts
                << " planned_engines=" << stats.planned_engine_count
                << " estimated_routed_bytes=" << stats.estimated_routed_bytes
                << " memory_budget_bytes=" << stats.memory_budget_bytes;
        }
        return out.str();
    }

    MoEExpertOverlayPreparationPlan MoEExpertOverlayPreparationPlan::build(
        const MoEExpertOverlayRuntimePlan &runtime_plan,
        size_t routed_expert_bytes_per_expert)
    {
        const auto &source = runtime_plan.sourcePlan();
        if (!source.isTieredOverlay())
            return {};

        MoEExpertOverlayPreparationPlan result;
        constexpr ExpertGemmRegistry::WeightRole kRoles[] = {
            ExpertGemmRegistry::WeightRole::GATE,
            ExpertGemmRegistry::WeightRole::UP,
            ExpertGemmRegistry::WeightRole::DOWN,
        };

        for (size_t tier_index = 0; tier_index < source.routed_tiers.size(); ++tier_index)
        {
            const auto &tier = source.routed_tiers[tier_index];
            const auto &domain = runtime_plan.domainForTier(tier_index);
            for (const auto &participant : preparationParticipantsFor(domain))
            {
                const auto category = routedResidencyCategoryFor(tier, participant.device);
                auto &stats = statsFor(
                    result.diagnostics_,
                    tier.domain,
                    participant.device,
                    participant.participant_index,
                    participant.world_rank,
                    participant.world_rank_known,
                    participant.owner_rank,
                    category,
                    source.residency_policy);
                stats.fallback = stats.fallback || tier.fallback;
                stats.memory_budget_bytes = std::max(stats.memory_budget_bytes, tier.memory_budget_bytes);
            }
        }

        std::set<std::tuple<std::string, DeviceId, int, int, WeightResidencyCategory, int, int>> counted_experts;
        for (const auto &placement : source.placements)
        {
            for (size_t expert_index = 0; expert_index < placement.routed_expert_tier.size(); ++expert_index)
            {
                const int tier_index = placement.routed_expert_tier[expert_index];
                if (tier_index < 0 || tier_index >= static_cast<int>(source.routed_tiers.size()))
                {
                    std::ostringstream message;
                    message << "MoE expert overlay preparation plan has invalid tier "
                            << tier_index << " for layer " << placement.layer
                            << " expert " << expert_index;
                    throw std::runtime_error(message.str());
                }

                const auto &tier = source.routed_tiers[static_cast<size_t>(tier_index)];
                const auto &domain = runtime_plan.domainForTier(static_cast<size_t>(tier_index));
                for (const auto &participant : preparationParticipantsFor(domain))
                {
                    const auto category = routedResidencyCategoryFor(tier, participant.device);
                    auto &stats = statsFor(
                        result.diagnostics_,
                        tier.domain,
                        participant.device,
                        participant.participant_index,
                        participant.world_rank,
                        participant.world_rank_known,
                        participant.owner_rank,
                        category,
                        source.residency_policy);
                    const auto expert_key = std::make_tuple(
                        tier.domain,
                        participant.device,
                        participant.participant_index,
                        participant.world_rank,
                        category,
                        placement.layer,
                        static_cast<int>(expert_index));
                    if (counted_experts.insert(expert_key).second)
                    {
                        ++stats.assigned_routed_experts;
                        stats.estimated_routed_bytes += routed_expert_bytes_per_expert;
                    }
                    stats.planned_engine_count += 3;
                    stats.fallback = stats.fallback || tier.fallback;
                    stats.memory_budget_bytes = std::max(stats.memory_budget_bytes, tier.memory_budget_bytes);

                    for (const auto role : kRoles)
                    {
                        MoEExpertOverlayPreparationRequest request;
                        request.layer = placement.layer;
                        request.expert_id = static_cast<int>(expert_index);
                        request.role = role;
                        request.tier_index = tier_index;
                        request.tier_name = tier.name;
                        request.domain_name = tier.domain;
                        request.device = participant.device;
                        request.participant_index = participant.participant_index;
                        request.participant_world_rank = participant.world_rank;
                        request.participant_world_rank_known = participant.world_rank_known;
                        request.owner_world_rank = participant.owner_rank;
                        request.residency_category = category;
                        request.residency_policy = source.residency_policy;
                        request.estimated_routed_bytes = routed_expert_bytes_per_expert;
                        request.memory_budget_bytes = tier.memory_budget_bytes;
                        request.fallback = tier.fallback;
                        result.requests_.push_back(std::move(request));
                    }
                }
            }
        }

        sortDiagnostics(result.diagnostics_);
        return result;
    }

    bool MoEExpertOverlayPreparationPlan::hasRequestsForDevice(DeviceId device) const
    {
        return std::any_of(requests_.begin(), requests_.end(),
                           [&](const auto &request)
                           { return request.device == device; });
    }

    bool MoEExpertOverlayPreparationPlan::hasAcceleratorRequests() const
    {
        return std::any_of(requests_.begin(), requests_.end(),
                           [](const auto &request)
                           { return request.device.is_gpu(); });
    }

    bool MoEExpertOverlayPreparationPlan::hasCpuRoutedAssignments() const
    {
        return std::any_of(requests_.begin(), requests_.end(),
                           [](const auto &request)
                           { return request.device.is_cpu(); });
    }

    std::vector<DeviceId> MoEExpertOverlayPreparationPlan::acceleratorDevices() const
    {
        std::set<DeviceId> devices;
        for (const auto &request : requests_)
        {
            if (request.device.is_gpu())
                devices.insert(request.device);
        }
        return {devices.begin(), devices.end()};
    }

    MoEExpertOverlayPreparationPlan MoEExpertOverlayPreparationPlan::filteredForRank(
        const OverlayRankPlan &rank_plan) const
    {
        MoEExpertOverlayPreparationPlan filtered;
        std::set<std::tuple<std::string, DeviceId, int, int, WeightResidencyCategory, int, int>> counted_experts;

        for (const auto &request : requests_)
        {
            if (!requestBelongsToRank(request, rank_plan))
                continue;

            auto filtered_request = request;
            filtered_request.residency_category = categoryForFilteredRank(request, rank_plan);
            filtered.requests_.push_back(filtered_request);

            auto &stats = recordRequestStats(filtered.diagnostics_, filtered_request);
            const auto expert_key = std::make_tuple(
                filtered_request.domain_name,
                filtered_request.device,
                filtered_request.participant_index,
                filtered_request.participant_world_rank,
                filtered_request.residency_category,
                filtered_request.layer,
                filtered_request.expert_id);
            if (counted_experts.insert(expert_key).second)
            {
                ++stats.assigned_routed_experts;
                stats.estimated_routed_bytes += filtered_request.estimated_routed_bytes;
            }
        }

        sortDiagnostics(filtered.diagnostics_);
        return filtered;
    }

    bool MoEExpertOverlayPreparationPlan::shouldPrepare(
        DeviceId device,
        int layer,
        int expert_id,
        ExpertGemmRegistry::WeightRole role) const
    {
        return requestFor(device, layer, expert_id, role) != nullptr;
    }

    const MoEExpertOverlayPreparationRequest *MoEExpertOverlayPreparationPlan::requestFor(
        DeviceId device,
        int layer,
        int expert_id,
        ExpertGemmRegistry::WeightRole role) const
    {
        auto it = std::find_if(requests_.begin(), requests_.end(),
                               [&](const auto &request)
                               {
                                   return request.device == device &&
                                          request.layer == layer &&
                                          request.expert_id == expert_id &&
                                          request.role == role;
                               });
        return it == requests_.end() ? nullptr : &(*it);
    }

    const MoEExpertOverlayPreparationRequest *MoEExpertOverlayPreparationPlan::requestForParticipant(
        const std::string &domain_name,
        DeviceId device,
        int participant_world_rank,
        int participant_index,
        int layer,
        int expert_id,
        ExpertGemmRegistry::WeightRole role) const
    {
        auto it = std::find_if(requests_.begin(), requests_.end(),
                               [&](const auto &request)
                               {
                                   return request.domain_name == domain_name &&
                                          request.device == device &&
                                          request.participant_world_rank == participant_world_rank &&
                                          request.participant_index == participant_index &&
                                          request.layer == layer &&
                                          request.expert_id == expert_id &&
                                          request.role == role;
                               });
        return it == requests_.end() ? nullptr : &(*it);
    }

    bool MoEExpertOverlayPreparationPlan::hasAnyRequestForDeviceLayerRole(
        DeviceId device,
        int layer,
        ExpertGemmRegistry::WeightRole role) const
    {
        return std::any_of(requests_.begin(), requests_.end(),
                           [&](const auto &request)
                           {
                               return request.device == device &&
                                      request.layer == layer &&
                                      request.role == role;
                           });
    }

    std::vector<std::string> MoEExpertOverlayPreparationPlan::domainsForDeviceLayerRole(
        DeviceId device,
        int layer,
        ExpertGemmRegistry::WeightRole role) const
    {
        std::vector<std::string> domains;
        for (const auto &request : requests_)
        {
            if (request.device == device &&
                request.layer == layer &&
                request.role == role)
            {
                domains.push_back(request.domain_name);
            }
        }
        std::sort(domains.begin(), domains.end());
        domains.erase(std::unique(domains.begin(), domains.end()), domains.end());
        return domains;
    }

    std::vector<int> MoEExpertOverlayPreparationPlan::expertsForDeviceLayerRole(
        DeviceId device,
        int layer,
        ExpertGemmRegistry::WeightRole role) const
    {
        std::vector<int> experts;
        for (const auto &request : requests_)
        {
            if (request.device == device &&
                request.layer == layer &&
                request.role == role)
            {
                experts.push_back(request.expert_id);
            }
        }
        std::sort(experts.begin(), experts.end());
        experts.erase(std::unique(experts.begin(), experts.end()), experts.end());
        return experts;
    }

    std::vector<int> MoEExpertOverlayPreparationPlan::expertsForDomainDeviceLayerRole(
        const std::string &domain_name,
        DeviceId device,
        int layer,
        ExpertGemmRegistry::WeightRole role) const
    {
        std::vector<int> experts;
        for (const auto &request : requests_)
        {
            if (request.domain_name == domain_name &&
                request.device == device &&
                request.layer == layer &&
                request.role == role)
            {
                experts.push_back(request.expert_id);
            }
        }
        std::sort(experts.begin(), experts.end());
        experts.erase(std::unique(experts.begin(), experts.end()), experts.end());
        return experts;
    }

} // namespace llaminar2
