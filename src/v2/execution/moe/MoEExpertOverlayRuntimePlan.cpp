#include "MoEExpertOverlayRuntimePlan.h"
#include "config/CollectiveBackendType.h"
#include "utils/Logger.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        std::string formatValidationErrors(const MoEExpertParallelValidationResult &validation)
        {
            std::ostringstream message;
            message << "Invalid MoE expert overlay runtime plan:";
            for (const auto &error : validation.errors)
                message << "\n - " << error;
            return message.str();
        }

        int participantRankFor(
            const ExpertComputeDomain &domain,
            size_t participant_index,
            int current_world_rank)
        {
            if (participant_index < domain.world_ranks.size())
                return domain.world_ranks[participant_index];
            if (domain.kind == ExpertDomainKind::NodeLocalTP)
                return static_cast<int>(participant_index);
            if (participant_index == 0 && domain.owner_rank >= 0)
                return domain.owner_rank;
            return current_world_rank;
        }

        bool participantRankKnown(const ExpertComputeDomain &domain, size_t participant_index)
        {
            return participant_index < domain.world_ranks.size() ||
                   domain.kind == ExpertDomainKind::NodeLocalTP ||
                   (participant_index == 0 && domain.owner_rank >= 0);
        }

        std::string describeDomainPrimary(const MoEOverlayRuntimeDomain &domain)
        {
            std::ostringstream message;
            message << "domain '" << domain.name << "' primary participant "
                    << domain.primary_participant.toShortString();
            if (domain.primary_device.is_valid())
                message << " (" << domain.primary_device.to_string() << ")";
            if (domain.primary_world_rank_known)
                message << " rank=" << domain.primary_world_rank;
            return message.str();
        }

        bool isCpuNodeLocalFallbackDomain(const MoEOverlayRuntimeDomain &domain)
        {
            if (domain.kind != ExpertDomainKind::NodeLocalTP || domain.participants.empty())
                return false;
            return std::all_of(domain.participants.begin(), domain.participants.end(),
                               [](const auto &participant)
                               {
                                   return participant.address.isCPU();
                               });
        }

        bool isAcceleratorLocalTPTensorParallelDomain(const MoEOverlayRuntimeDomain &domain)
        {
            if (domain.kind != ExpertDomainKind::LocalTP ||
                domain.compute_kind != ExpertDomainComputeKind::TensorParallelExperts ||
                domain.participants.size() < 2)
            {
                return false;
            }

            return std::all_of(domain.participants.begin(), domain.participants.end(),
                               [](const auto &participant)
                               {
                                   return participant.address.isGPU() &&
                                          participant.locally_addressable &&
                                          participant.local_device.is_gpu();
                               });
        }

        bool hasDomainScopedRuntimeSupport(const MoEOverlayRuntimeDomain &domain)
        {
            return isCpuNodeLocalFallbackDomain(domain) ||
                   isAcceleratorLocalTPTensorParallelDomain(domain);
        }

        void validateRootDomainReachable(
            const MoEOverlayRuntimeDomain &domain,
            const char *field_name,
            int current_world_rank)
        {
            if (domain.local_reachable_for_mvp)
                return;

            std::ostringstream message;
            message << "MoE expert overlay " << field_name << " domain '" << domain.name
                    << "' is not locally reachable for the current DeviceGraphExecutor MVP: "
                    << describeDomainPrimary(domain)
                    << "; current_rank=" << current_world_rank
                    << ", primary_is_local=" << (domain.primary_is_local ? "true" : "false")
                    << ", primary_owned_by_current_rank="
                    << (domain.primary_owned_by_current_rank ? "true" : "false");
            throw std::runtime_error(message.str());
        }

        MoEOverlayRuntimeDomain resolveDomain(
            const ExpertComputeDomain &domain,
            int current_world_rank)
        {
            const auto canonical = domain.toExecutionDomainDefinition();
            MoEOverlayRuntimeDomain resolved;
            resolved.name = canonical.name;
            resolved.kind = domain.kind;
            resolved.backend = canonical.backend;
            resolved.compute_kind = domain.compute_kind;
            resolved.owner_rank = canonical.owner_rank.value_or(-1);

            resolved.participants.reserve(canonical.participants.size());
            for (size_t index = 0; index < canonical.participants.size(); ++index)
            {
                MoEOverlayDomainParticipant participant;
                participant.address = canonical.participants[index];
                participant.participant_index = static_cast<int>(index);
                participant.world_rank = participantRankFor(domain, index, current_world_rank);
                participant.world_rank_known = participantRankKnown(domain, index);
                participant.owned_by_current_rank = participant.world_rank == current_world_rank;
                participant.locally_addressable = participant.address.isLocal();
                participant.local_device = participant.locally_addressable
                                               ? participant.address.toLocalDeviceId()
                                               : DeviceId::invalid();
                resolved.participants.push_back(participant);
            }

            if (!resolved.participants.empty())
            {
                const auto &primary = resolved.participants.front();
                resolved.primary_participant = primary.address;
                resolved.primary_device = primary.local_device;
                resolved.primary_world_rank = primary.world_rank;
                resolved.primary_world_rank_known = primary.world_rank_known;
                resolved.primary_is_local = primary.locally_addressable && primary.local_device.is_valid();
                resolved.primary_owned_by_current_rank = primary.owned_by_current_rank;
                resolved.local_reachable_for_mvp = resolved.primary_is_local && resolved.primary_owned_by_current_rank;
            }

            resolved.requires_domain_scoped_collective_context = domain.hasMultipleParticipants();
            resolved.domain_scoped_collective_context_ready =
                resolved.requires_domain_scoped_collective_context && hasDomainScopedRuntimeSupport(resolved);
            resolved.multi_participant_execution_pending =
                domain.hasMultipleParticipants() && !resolved.domain_scoped_collective_context_ready;
            if (resolved.multi_participant_execution_pending)
            {
                const bool tensor_parallel_experts =
                    domain.compute_kind == ExpertDomainComputeKind::TensorParallelExperts;
                std::ostringstream reason;
                reason << "Domain-scoped runtime support is not available for this "
                       << (tensor_parallel_experts ? "TensorParallelExperts" : "multi-participant")
                       << " domain shape. Bridge Phase 5C covers accelerator LocalTP "
                       << "TensorParallelExperts and CPU NodeLocalTP fallback helpers; Bridge Phase 5D "
                       << "still wires the accelerator LocalTP executor into the Qwen graph. "
                       << "Primary-only lowering to " << resolved.primary_device.to_string()
                       << " is no longer used for routed tier work";
                resolved.pending_reason = reason.str();
            }
            return resolved;
        }
    } // namespace

    MoEExpertOverlayRuntimePlan::MoEExpertOverlayRuntimePlan(
        std::shared_ptr<const MoEExpertParallelPlan> source_plan,
        int current_world_rank,
        std::vector<MoEOverlayRuntimeDomain> domains,
        std::vector<MoEOverlayRuntimeTier> routed_tiers)
        : source_plan_(std::move(source_plan)),
          current_world_rank_(current_world_rank),
          domains_(std::move(domains)),
          routed_tiers_(std::move(routed_tiers))
    {
        if (!source_plan_)
            throw std::invalid_argument("MoEExpertOverlayRuntimePlan requires a source plan");

        for (size_t index = 0; index < domains_.size(); ++index)
        {
            auto inserted = domains_by_name_.emplace(domains_[index].name, index).second;
            if (!inserted)
                throw std::invalid_argument("duplicate resolved MoE expert overlay domain: " + domains_[index].name);
        }
    }

    const MoEOverlayRuntimeDomain *MoEExpertOverlayRuntimePlan::domainForName(
        const std::string &domain_name) const
    {
        auto it = domains_by_name_.find(domain_name);
        if (it == domains_by_name_.end())
            return nullptr;
        return &domains_[it->second];
    }

    const MoEOverlayRuntimeDomain &MoEExpertOverlayRuntimePlan::requireDomain(
        const std::string &domain_name,
        const char *context) const
    {
        if (const auto *domain = domainForName(domain_name))
            return *domain;
        throw std::runtime_error(std::string("MoE expert overlay ") + context +
                                 " references unknown resolved domain '" + domain_name + "'");
    }

    const MoEOverlayRuntimeDomain &MoEExpertOverlayRuntimePlan::continuationDomain() const
    {
        return requireDomain(source_plan_->continuation_domain, "continuation_domain");
    }

    const MoEOverlayRuntimeDomain &MoEExpertOverlayRuntimePlan::sharedExpertDomain() const
    {
        return requireDomain(source_plan_->shared_expert_domain, "shared_expert_domain");
    }

    const MoEOverlayRuntimeDomain &MoEExpertOverlayRuntimePlan::domainForTier(size_t tier_index) const
    {
        if (tier_index >= routed_tiers_.size())
        {
            throw std::out_of_range("MoE expert overlay tier index out of range: " +
                                    std::to_string(tier_index));
        }
        return requireDomain(routed_tiers_[tier_index].domain_name, "routed tier");
    }

    DeviceId MoEExpertOverlayRuntimePlan::continuationDevice() const
    {
        return continuationDomain().primary_device;
    }

    DeviceId MoEExpertOverlayRuntimePlan::primaryDeviceForDomain(
        const std::string &domain_name) const
    {
        return requireDomain(domain_name, "domain").primary_device;
    }

    DeviceId MoEExpertOverlayRuntimePlan::sharedExpertDeviceForMVP(int layer_idx) const
    {
        const auto &domain = sharedExpertDomain();
        if (!domain.local_reachable_for_mvp)
        {
            std::ostringstream message;
            message << "MoE expert overlay shared_expert_domain '" << domain.name << "'";
            if (layer_idx >= 0)
                message << " for layer " << layer_idx;
            message << " resolves to unreachable " << describeDomainPrimary(domain)
                    << "; current DeviceGraphExecutor overlay lowering supports only local primary participants owned by this rank";
            throw std::runtime_error(message.str());
        }
        return domain.primary_device;
    }

    DeviceId MoEExpertOverlayRuntimePlan::tierDeviceForMVP(size_t tier_index, int layer_idx) const
    {
        const auto &domain = domainForTier(tier_index);
        if (!domain.local_reachable_for_mvp)
        {
            std::ostringstream message;
            message << "MoE expert overlay tier " << tier_index;
            if (tier_index < routed_tiers_.size() && !routed_tiers_[tier_index].tier.name.empty())
                message << " ('" << routed_tiers_[tier_index].tier.name << "')";
            if (layer_idx >= 0)
                message << " for layer " << layer_idx;
            message << " resolves to unreachable " << describeDomainPrimary(domain)
                    << "; current DeviceGraphExecutor overlay lowering supports only local primary participants owned by this rank";
            throw std::runtime_error(message.str());
        }
        return domain.primary_device;
    }

    std::string MoEExpertOverlayRuntimePlan::diagnostics() const
    {
        std::ostringstream out;
        out << "MoE expert overlay runtime plan: current_rank=" << current_world_rank_
            << " continuation_domain=" << source_plan_->continuation_domain
            << " continuation_device=" << continuationDevice().to_string()
            << " base_model_domain=" << source_plan_->effectiveBaseModelDomain()
            << " shared_expert_domain=" << source_plan_->shared_expert_domain;

        for (const auto &domain : domains_)
        {
            out << "\n  domain " << domain.name
                << ": kind=" << toString(domain.kind)
                << " backend=" << collectiveBackendTypeToString(domain.backend)
                << " compute=" << toString(domain.compute_kind)
                << " participants=" << domain.participants.size()
                << " primary=" << domain.primary_participant.toShortString()
                << " primary_device=" << domain.primary_device.to_string()
                << " owner_rank=" << domain.owner_rank
                << " primary_rank=";
            if (domain.primary_world_rank_known)
                out << domain.primary_world_rank;
            else
                out << "current";
            out << " local_reachable_for_mvp=" << (domain.local_reachable_for_mvp ? "true" : "false")
                << " multi_participant_execution_pending="
                << (domain.multi_participant_execution_pending ? "true" : "false")
                << " collective_context="
                << (domain.domain_scoped_collective_context_ready ? "ready" : (domain.requires_domain_scoped_collective_context ? "pending" : "not_required"));
            if (!domain.pending_reason.empty())
                out << " pending_reason=\"" << domain.pending_reason << "\"";
        }

        for (const auto &tier : routed_tiers_)
        {
            out << "\n  tier[" << tier.tier_index << "] " << tier.tier.name
                << ": domain=" << tier.domain_name
                << " primary_device=" << tier.primary_device.to_string()
                << " fallback=" << (tier.tier.fallback ? "true" : "false")
                << " multi_participant_execution_pending="
                << (tier.multi_participant_execution_pending ? "true" : "false");
        }
        return out.str();
    }

    std::shared_ptr<MoEExpertOverlayRuntimePlan> resolveMoEExpertOverlayRuntimePlan(
        std::shared_ptr<const MoEExpertParallelPlan> plan,
        const MoEExpertOverlayRuntimeResolverOptions &options)
    {
        if (!plan || !plan->isTieredOverlay())
            return nullptr;

        const auto validation = validateMoEExpertParallelPlan(
            *plan,
            MoEExpertParallelValidationOptions{.allow_routed_tensor_parallel_experts = true});
        if (!validation.ok())
            throw std::invalid_argument(formatValidationErrors(validation));

        std::vector<MoEOverlayRuntimeDomain> domains;
        domains.reserve(plan->dense_domains.size() + plan->domains.size());
        auto addDomainIfAbsent = [&](const ExpertComputeDomain &domain)
        {
            const auto exists = std::any_of(domains.begin(), domains.end(), [&](const auto &resolved)
                                            { return resolved.name == domain.name; });
            if (!exists)
                domains.push_back(resolveDomain(domain, options.current_world_rank));
        };
        for (const auto &domain : plan->dense_domains)
            addDomainIfAbsent(ExpertComputeDomain::fromExecutionDomainDefinition(domain));
        for (const auto &domain : plan->domains)
            addDomainIfAbsent(domain);

        std::unordered_map<std::string, const MoEOverlayRuntimeDomain *> domains_by_name;
        for (const auto &domain : domains)
            domains_by_name.emplace(domain.name, &domain);

        auto requireResolvedDomain = [&](const std::string &domain_name, const char *field_name) -> const MoEOverlayRuntimeDomain &
        {
            auto it = domains_by_name.find(domain_name);
            if (it == domains_by_name.end())
                throw std::runtime_error(std::string("MoE expert overlay ") + field_name +
                                         " references unknown domain '" + domain_name + "'");
            return *it->second;
        };

        const auto &continuation_domain = requireResolvedDomain(plan->continuation_domain, "continuation_domain");
        const auto &shared_domain = requireResolvedDomain(plan->shared_expert_domain, "shared_expert_domain");
        if (options.validate_mvp_root_reachability)
        {
            validateRootDomainReachable(continuation_domain, "continuation", options.current_world_rank);
            validateRootDomainReachable(shared_domain, "shared expert", options.current_world_rank);
        }

        std::vector<MoEOverlayRuntimeTier> routed_tiers;
        routed_tiers.reserve(plan->routed_tiers.size());
        for (size_t tier_index = 0; tier_index < plan->routed_tiers.size(); ++tier_index)
        {
            const auto &tier = plan->routed_tiers[tier_index];
            const auto &domain = requireResolvedDomain(tier.domain, "routed tier");

            MoEOverlayRuntimeTier resolved_tier;
            resolved_tier.tier_index = static_cast<int>(tier_index);
            resolved_tier.tier = tier;
            resolved_tier.domain_name = tier.domain;
            resolved_tier.primary_device = domain.primary_device;
            resolved_tier.local_reachable_for_mvp = domain.local_reachable_for_mvp;
            resolved_tier.multi_participant_execution_pending = domain.multi_participant_execution_pending;
            routed_tiers.push_back(std::move(resolved_tier));
        }

        auto runtime_plan = std::make_shared<MoEExpertOverlayRuntimePlan>(
            std::move(plan), options.current_world_rank, std::move(domains), std::move(routed_tiers));

        LOG_INFO("[MoEExpertOverlayRuntimePlan] Resolved overlay topology: continuation_domain="
                 << runtime_plan->sourcePlan().continuation_domain
                 << " continuation_device=" << runtime_plan->continuationDevice().to_string()
                 << " domains=" << runtime_plan->domains().size()
                 << " routed_tiers=" << runtime_plan->routedTiers().size());
        LOG_DEBUG("[MoEExpertOverlayRuntimePlan] " << runtime_plan->diagnostics());
        for (const auto &domain : runtime_plan->domains())
        {
            if (domain.multi_participant_execution_pending)
            {
                LOG_WARN("[MoEExpertOverlayRuntimePlan] Domain '" << domain.name
                                                                  << "' requests " << toString(domain.compute_kind)
                                                                  << " over " << domain.participants.size()
                                                                  << " participants; " << domain.pending_reason);
            }
        }

        return runtime_plan;
    }

} // namespace llaminar2
