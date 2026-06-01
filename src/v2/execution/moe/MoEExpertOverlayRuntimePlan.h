/**
 * @file MoEExpertOverlayRuntimePlan.h
 * @brief Resolved runtime descriptors for same-layer MoE expert overlay domains.
 *
 * The configuration-time MoEExpertParallelPlan names domains and tiers. This
 * runtime plan resolves those names to explicit rank/device descriptors and
 * records the current MVP lowering contract for multi-participant domains.
 */

#pragma once

#include "MoEExpertParallelPlan.h"
#include "backends/DeviceId.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    struct MoEOverlayDomainParticipant
    {
        GlobalDeviceAddress address;
        int participant_index = -1;
        int world_rank = -1;
        bool world_rank_known = false;
        bool owned_by_current_rank = false;
        bool locally_addressable = false;
        DeviceId local_device = DeviceId::invalid();
    };

    struct MoEOverlayRuntimeDomain
    {
        std::string name;
        ExpertDomainKind kind = ExpertDomainKind::SingleDevice;
        CollectiveBackendType backend = CollectiveBackendType::AUTO;
        ExpertDomainComputeKind compute_kind = ExpertDomainComputeKind::ReplicatedExperts;

        std::vector<MoEOverlayDomainParticipant> participants;
        GlobalDeviceAddress primary_participant;
        DeviceId primary_device = DeviceId::invalid();
        int primary_world_rank = -1;
        bool primary_world_rank_known = false;
        int owner_rank = -1;
        bool primary_is_local = false;
        bool primary_owned_by_current_rank = false;
        bool local_reachable_for_mvp = false;

        bool requires_domain_scoped_collective_context = false;
        bool domain_scoped_collective_context_ready = false;
        bool multi_participant_execution_pending = false;
        std::string pending_reason;

        int routed_tier_count = 0;
        bool routed_rebalance_controller_eligible = false;
        std::string rebalance_domain_id;
    };

    struct MoEOverlayRuntimeTier
    {
        int tier_index = -1;
        ExpertRoutedTier tier;
        std::string domain_name;
        DeviceId primary_device = DeviceId::invalid();
        bool local_reachable_for_mvp = false;
        bool multi_participant_execution_pending = false;
    };

    struct MoEExpertOverlayRuntimeResolverOptions
    {
        int current_world_rank = 0;
        bool validate_mvp_root_reachability = true;
    };

    class MoEExpertOverlayRuntimePlan
    {
    public:
        MoEExpertOverlayRuntimePlan(
            std::shared_ptr<const MoEExpertParallelPlan> source_plan,
            int current_world_rank,
            std::vector<MoEOverlayRuntimeDomain> domains,
            std::vector<MoEOverlayRuntimeTier> routed_tiers);

        const MoEExpertParallelPlan &sourcePlan() const { return *source_plan_; }
        std::shared_ptr<const MoEExpertParallelPlan> sourcePlanPtr() const { return source_plan_; }
        int currentWorldRank() const { return current_world_rank_; }

        const std::vector<MoEOverlayRuntimeDomain> &domains() const { return domains_; }
        const std::vector<MoEOverlayRuntimeTier> &routedTiers() const { return routed_tiers_; }

        const MoEOverlayRuntimeDomain *domainForName(const std::string &domain_name) const;
        const MoEOverlayRuntimeDomain &continuationDomain() const;
        const MoEOverlayRuntimeDomain &sharedExpertDomain() const;
        const MoEOverlayRuntimeDomain &domainForTier(size_t tier_index) const;

        DeviceId continuationDevice() const;
        DeviceId primaryDeviceForDomain(const std::string &domain_name) const;
        DeviceId sharedExpertDeviceForMVP(int layer_idx = -1) const;
        DeviceId tierDeviceForMVP(size_t tier_index, int layer_idx = -1) const;

        std::string diagnostics() const;

    private:
        const MoEOverlayRuntimeDomain &requireDomain(const std::string &domain_name, const char *context) const;

        std::shared_ptr<const MoEExpertParallelPlan> source_plan_;
        int current_world_rank_ = 0;
        std::vector<MoEOverlayRuntimeDomain> domains_;
        std::vector<MoEOverlayRuntimeTier> routed_tiers_;
        std::unordered_map<std::string, size_t> domains_by_name_;
    };

    std::shared_ptr<MoEExpertOverlayRuntimePlan> resolveMoEExpertOverlayRuntimePlan(
        std::shared_ptr<const MoEExpertParallelPlan> plan,
        const MoEExpertOverlayRuntimeResolverOptions &options = {});

} // namespace llaminar2
