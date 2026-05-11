#include "MoEExpertOverlayExecutionPlan.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>

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

        int domainOwnerRank(const MoEOverlayRuntimeDomain &domain, int current_world_rank)
        {
            if (domain.owner_rank >= 0)
                return domain.owner_rank;
            if (domain.primary_world_rank_known)
                return domain.primary_world_rank;
            if (!domain.participants.empty() && domain.participants.front().world_rank_known)
                return domain.participants.front().world_rank;
            return current_world_rank;
        }

        bool rankHasCpuParticipant(const MoEOverlayRuntimeDomain &domain, int world_rank)
        {
            return std::any_of(domain.participants.begin(), domain.participants.end(),
                               [&](const auto &participant)
                               {
                                   return participant.address.isCPU() &&
                                          participant.world_rank_known &&
                                          participant.world_rank == world_rank;
                               });
        }

        bool isLocalAcceleratorTPDomain(const MoEOverlayRuntimeDomain &domain)
        {
            if (domain.kind != ExpertDomainKind::LocalTP ||
                domain.compute_kind != ExpertDomainComputeKind::TensorParallelExperts)
            {
                return false;
            }

            return std::any_of(domain.participants.begin(), domain.participants.end(),
                               [](const auto &participant)
                               {
                                   return participant.address.isGPU() &&
                                          participant.locally_addressable &&
                                          participant.local_device.is_gpu();
                               });
        }

        bool isCpuFallbackDomain(
            const MoEOverlayRuntimeDomain &domain,
            const std::set<std::string> &fallback_domains)
        {
            if (fallback_domains.find(domain.name) == fallback_domains.end())
                return false;

            return std::any_of(domain.participants.begin(), domain.participants.end(),
                               [](const auto &participant)
                               {
                                   return participant.address.isCPU();
                               });
        }

        std::set<std::string> fallbackDomainNames(const MoEExpertParallelPlan &plan)
        {
            std::set<std::string> domains;
            for (const auto &tier : plan.routed_tiers)
            {
                if (tier.fallback)
                    domains.insert(tier.domain);
            }
            return domains;
        }

        void addDomainDevices(OverlayRankPlan &rank_plan, const MoEOverlayRuntimeDomain &domain)
        {
            for (const auto &participant : domain.participants)
            {
                if (participant.local_device.is_valid())
                    addUnique(rank_plan.local_devices, participant.local_device);
            }
        }

        OverlayRankRole primaryRoleFor(const OverlayRankPlan &rank_plan)
        {
            if (rank_plan.hasRole(OverlayRankRole::ContinuationRoot))
                return OverlayRankRole::ContinuationRoot;
            if (rank_plan.hasRole(OverlayRankRole::LocalAcceleratorParticipant))
                return OverlayRankRole::LocalAcceleratorParticipant;
            if (rank_plan.hasRole(OverlayRankRole::CpuFallbackParticipant))
                return OverlayRankRole::CpuFallbackParticipant;
            if (rank_plan.hasRole(OverlayRankRole::RemoteExpertParticipant))
                return OverlayRankRole::RemoteExpertParticipant;
            return OverlayRankRole::RelayOnly;
        }
    } // namespace

    const char *toString(OverlayRankRole role)
    {
        switch (role)
        {
        case OverlayRankRole::ContinuationRoot:
            return "ContinuationRoot";
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

    MoEExpertOverlayExecutionPlan buildMoEExpertOverlayExecutionPlan(
        const MoEExpertOverlayRuntimePlan &runtime_plan)
    {
        MoEExpertOverlayExecutionPlan result;
        result.continuation_domain = runtime_plan.sourcePlan().continuation_domain;
        result.shared_expert_domain = runtime_plan.sourcePlan().shared_expert_domain;
        result.domains = runtime_plan.domains();

        OverlayRankPlan rank_plan;
        rank_plan.world_rank = runtime_plan.currentWorldRank();

        const auto &continuation = runtime_plan.continuationDomain();
        const auto &shared = runtime_plan.sharedExpertDomain();
        const int continuation_root_rank = domainOwnerRank(continuation, rank_plan.world_rank);

        rank_plan.builds_root_graph =
            rank_plan.world_rank == continuation_root_rank &&
            continuation.local_reachable_for_mvp &&
            shared.local_reachable_for_mvp;

        if (rank_plan.builds_root_graph)
        {
            addUnique(rank_plan.roles, OverlayRankRole::ContinuationRoot);
            addUnique(rank_plan.owned_domains, continuation.name);
            addDomainDevices(rank_plan, continuation);
        }

        const auto fallback_domains = fallbackDomainNames(runtime_plan.sourcePlan());
        for (const auto &domain : runtime_plan.domains())
        {
            if (isLocalAcceleratorTPDomain(domain) &&
                domainOwnerRank(domain, rank_plan.world_rank) == rank_plan.world_rank)
            {
                addUnique(rank_plan.roles, OverlayRankRole::LocalAcceleratorParticipant);
                addUnique(rank_plan.owned_domains, domain.name);
                addDomainDevices(rank_plan, domain);
            }

            if (!rank_plan.builds_root_graph &&
                isCpuFallbackDomain(domain, fallback_domains) &&
                rankHasCpuParticipant(domain, rank_plan.world_rank))
            {
                addUnique(rank_plan.roles, OverlayRankRole::CpuFallbackParticipant);
                addUnique(rank_plan.owned_domains, domain.name);
                addDomainDevices(rank_plan, domain);
            }
        }

        if (rank_plan.roles.empty())
            addUnique(rank_plan.roles, OverlayRankRole::RelayOnly);

        rank_plan.role = primaryRoleFor(rank_plan);
        rank_plan.loads_tokenizer = rank_plan.builds_root_graph;
        rank_plan.loads_full_model_metadata = !rank_plan.hasRole(OverlayRankRole::RelayOnly);
        rank_plan.loads_root_weights = rank_plan.builds_root_graph;
        rank_plan.loads_expert_weights =
            rank_plan.builds_root_graph ||
            rank_plan.hasRole(OverlayRankRole::LocalAcceleratorParticipant) ||
            rank_plan.hasRole(OverlayRankRole::CpuFallbackParticipant) ||
            rank_plan.hasRole(OverlayRankRole::RemoteExpertParticipant);

        result.current_rank = std::move(rank_plan);
        return result;
    }

    MoEExpertOverlayExecutionPlan resolveMoEExpertOverlayExecutionPlan(
        std::shared_ptr<const MoEExpertParallelPlan> plan,
        int current_world_rank)
    {
        auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(
            std::move(plan),
            MoEExpertOverlayRuntimeResolverOptions{
                .current_world_rank = current_world_rank,
                .validate_mvp_root_reachability = false,
            });
        if (!runtime_plan)
            throw std::invalid_argument("MoEExpertOverlayExecutionPlan requires an enabled tiered overlay plan");
        return buildMoEExpertOverlayExecutionPlan(*runtime_plan);
    }

    std::string MoEExpertOverlayExecutionPlan::diagnostics() const
    {
        std::ostringstream out;
        out << "MoE expert overlay execution plan: rank=" << current_rank.world_rank
            << " primary_role=" << toString(current_rank.role)
            << " builds_root_graph=" << (current_rank.builds_root_graph ? "true" : "false")
            << " continuation_domain=" << continuation_domain
            << " shared_expert_domain=" << shared_expert_domain;

        out << " roles=";
        for (size_t index = 0; index < current_rank.roles.size(); ++index)
        {
            if (index != 0)
                out << ",";
            out << toString(current_rank.roles[index]);
        }

        out << " owned_domains=";
        for (size_t index = 0; index < current_rank.owned_domains.size(); ++index)
        {
            if (index != 0)
                out << ",";
            out << current_rank.owned_domains[index];
        }

        out << " local_devices=";
        for (size_t index = 0; index < current_rank.local_devices.size(); ++index)
        {
            if (index != 0)
                out << ",";
            out << current_rank.local_devices[index].to_string();
        }

        out << " loads_tokenizer=" << (current_rank.loads_tokenizer ? "true" : "false")
            << " loads_full_model_metadata=" << (current_rank.loads_full_model_metadata ? "true" : "false")
            << " loads_root_weights=" << (current_rank.loads_root_weights ? "true" : "false")
            << " loads_expert_weights=" << (current_rank.loads_expert_weights ? "true" : "false");
        return out.str();
    }

} // namespace llaminar2