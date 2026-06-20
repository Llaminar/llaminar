#include "MoEExpertOverlayExecutionPlan.h"

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

        std::vector<int> deterministicParticipantRanks(const ExpertComputeDomain &domain)
        {
            if (!domain.world_ranks.empty())
                return domain.world_ranks;

            if (domain.kind == ExpertDomainKind::NodeLocalTP)
            {
                std::vector<int> ranks;
                ranks.reserve(domain.participants.size());
                for (size_t index = 0; index < domain.participants.size(); ++index)
                    ranks.push_back(static_cast<int>(index));
                return ranks;
            }

            if (domain.owner_rank >= 0)
                return std::vector<int>(domain.participants.size(), domain.owner_rank);

            if (domain.kind == ExpertDomainKind::SingleDevice)
                return std::vector<int>(domain.participants.size(), 0);

            return std::vector<int>(domain.participants.size(), -1);
        }

        int inferredOwnerRank(const ExpertComputeDomain &domain, const std::vector<int> &participant_ranks)
        {
            if (domain.owner_rank >= 0)
                return domain.owner_rank;
            if (!participant_ranks.empty())
                return participant_ranks.front();
            return -1;
        }

        int inferMinimumWorldSize(const MoEExpertParallelPlan &plan, int current_world_rank)
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
                else if (domain.kind == ExpertDomainKind::NodeLocalTP)
                {
                    world_size = std::max(world_size, static_cast<int>(domain.participants.size()));
                }
            }
            return world_size;
        }

        std::unordered_map<std::string, const ExpertComputeDomain *> sourceDomainsByName(
            const MoEExpertParallelPlan &plan)
        {
            std::unordered_map<std::string, const ExpertComputeDomain *> by_name;
            for (const auto &domain : plan.domains)
                by_name.emplace(domain.name, &domain);
            return by_name;
        }

        const ExpertComputeDomain *findSourceDomain(
            const std::unordered_map<std::string, const ExpertComputeDomain *> &by_name,
            const std::string &name)
        {
            auto it = by_name.find(name);
            if (it == by_name.end())
                return nullptr;
            return it->second;
        }

        int domainOwnerRankOrInvalid(
            const std::unordered_map<std::string, const ExpertComputeDomain *> &by_name,
            const std::string &name)
        {
            const auto *domain = findSourceDomain(by_name, name);
            if (!domain)
                return -1;
            return inferredOwnerRank(*domain, deterministicParticipantRanks(*domain));
        }

        std::vector<std::string> validateExecutionTopology(
            const MoEExpertParallelPlan &plan,
            int current_world_rank,
            int world_size)
        {
            std::vector<std::string> errors;
            const auto validation = validateMoEExpertParallelPlan(
                plan,
                MoEExpertParallelValidationOptions{.allow_routed_tensor_parallel_experts = true});
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

                if (domain.kind == ExpertDomainKind::LocalTP && distinct_ranks.size() > 1)
                {
                    errors.push_back("domain '" + domain.name +
                                     "' is LocalTP but maps participants to multiple ranks; use NodeLocalTP for cross-rank domains");
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

        std::set<std::string> routedDomainNames(const MoEExpertParallelPlan &plan)
        {
            std::set<std::string> domains;
            for (const auto &tier : plan.routed_tiers)
                domains.insert(tier.domain);
            return domains;
        }

        struct DomainRoleDescriptor
        {
            const ExpertComputeDomain *source = nullptr;
            const MoEOverlayRuntimeDomain *runtime = nullptr;
            std::vector<int> participant_ranks;
            int owner_rank = -1;
        };

        std::vector<DomainRoleDescriptor> buildDomainRoleDescriptors(
            const MoEExpertParallelPlan &source_plan,
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

            if (descriptor.source->kind == ExpertDomainKind::LocalTP &&
                descriptor.source->compute_kind == ExpertDomainComputeKind::TensorParallelExperts)
            {
                return true;
            }

            return descriptor.source->kind == ExpertDomainKind::SingleDevice &&
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
            if (rank_plan.hasRole(OverlayRankRole::LocalAcceleratorParticipant))
                return OverlayRankRole::LocalAcceleratorParticipant;
            if (rank_plan.hasRole(OverlayRankRole::CpuFallbackParticipant))
                return OverlayRankRole::CpuFallbackParticipant;
            if (rank_plan.hasRole(OverlayRankRole::RemoteExpertParticipant))
                return OverlayRankRole::RemoteExpertParticipant;
            return OverlayRankRole::RelayOnly;
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

    std::optional<std::string> graphNativeMoEOverlayBuildBlocker(
        const MoEExpertOverlayExecutionPlan &execution_plan)
    {
        if (execution_plan.buildsRootGraph())
            return std::nullopt;

        const auto &rank = execution_plan.currentRankPlan();
        return "MoE overlay rank " + std::to_string(rank.world_rank) +
               " has role " + toString(rank.role) +
               ", but production graph-native overlay currently supports only "
               "the root-owned local sparse execution path. Remote warm/cold "
               "participant graph execution must be implemented with matched "
               "MPI sparse dispatch/local-expert/return-reduce stages before "
               "this topology can run.";
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
            if (rank_plan.world_rank == result.continuation_root_rank)
            {
                addUnique(rank_plan.roles, OverlayRankRole::ContinuationRoot);
                addUnique(rank_plan.owned_domains, source_plan.continuation_domain);
                addUnique(rank_plan.owned_domains, result.base_model_domain);
                addUnique(rank_plan.root_weight_domains, result.base_model_domain);
                addUnique(rank_plan.shared_expert_weight_domains, source_plan.shared_expert_domain);
                addDomainDevicesForRank(rank_plan, *continuation, rank_plan.world_rank);
                addDomainDevicesForRank(rank_plan, *base_model, rank_plan.world_rank);
                rank_plan.builds_root_graph = true;
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
                    if (!rank_plan.builds_root_graph)
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
            rank_plan.loads_tokenizer = rank_plan.builds_root_graph;
            rank_plan.loads_worker_tokenizer_state = !rank_plan.builds_root_graph &&
                                                     rank_plan.hasRole(OverlayRankRole::CpuFallbackParticipant);
            rank_plan.loads_full_model_metadata = !rank_plan.hasRole(OverlayRankRole::RelayOnly);
            rank_plan.loads_root_weights = rank_plan.builds_root_graph;
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
        std::shared_ptr<const MoEExpertParallelPlan> plan,
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
        std::shared_ptr<const MoEExpertParallelPlan> plan,
        const MoEExpertOverlayExecutionPlanResolverOptions &options)
    {
        if (!plan || !plan->isTieredOverlay())
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
            out << " builds_root_graph=" << (rank_plan.builds_root_graph ? "true" : "false")
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
