#pragma once

#include "MoEExpertOverlayRuntimePlan.h"
#include "loaders/ExpertGemmRegistry.h"
#include "loaders/WeightIdentity.h"

#include <cstddef>
#include <string>
#include <vector>

namespace llaminar2
{
    struct OverlayRankPlan;

    struct MoEExpertOverlayPreparationRequest
    {
        int layer = -1;
        int expert_id = -1;
        ExpertGemmRegistry::WeightRole role = ExpertGemmRegistry::WeightRole::GATE;
        int tier_index = -1;
        std::string tier_name;
        std::string domain_name;
        DeviceId device = DeviceId::invalid();
        int participant_index = -1;
        int participant_world_rank = -1;
        bool participant_world_rank_known = false;
        int owner_world_rank = -1;
        WeightResidencyCategory residency_category = WeightResidencyCategory::Unspecified;
        ExpertResidencyPolicy residency_policy = ExpertResidencyPolicy::Disabled;
        size_t estimated_routed_bytes = 0;
        size_t memory_budget_bytes = 0;
        bool fallback = false;
    };

    struct MoEExpertOverlayDomainPreparationStats
    {
        std::string domain_name;
        DeviceId device = DeviceId::invalid();
        int participant_index = -1;
        int participant_world_rank = -1;
        bool participant_world_rank_known = false;
        int owner_world_rank = -1;
        WeightResidencyCategory residency_category = WeightResidencyCategory::Unspecified;
        ExpertResidencyPolicy residency_policy = ExpertResidencyPolicy::Disabled;
        bool accelerator = false;
        bool fallback = false;
        size_t memory_budget_bytes = 0;
        size_t assigned_routed_experts = 0;
        size_t planned_engine_count = 0;
        size_t estimated_routed_bytes = 0;
    };

    struct MoEExpertOverlayPreparationDiagnostics
    {
        std::vector<MoEExpertOverlayDomainPreparationStats> domains;

        const MoEExpertOverlayDomainPreparationStats *domainStats(
            const std::string &domain_name,
            DeviceId device) const;
        const MoEExpertOverlayDomainPreparationStats *domainStats(
            const std::string &domain_name,
            DeviceId device,
            int participant_world_rank,
            int participant_index) const;
        std::string render() const;
    };

    class MoEExpertOverlayPreparationPlan
    {
    public:
        static MoEExpertOverlayPreparationPlan build(
            const MoEExpertOverlayRuntimePlan &runtime_plan,
            size_t routed_expert_bytes_per_expert = 0);

        const std::vector<MoEExpertOverlayPreparationRequest> &requests() const
        {
            return requests_;
        }

        const MoEExpertOverlayPreparationDiagnostics &diagnostics() const
        {
            return diagnostics_;
        }

        bool empty() const { return requests_.empty(); }
        bool hasRequestsForDevice(DeviceId device) const;
        bool hasAcceleratorRequests() const;
        bool hasCpuRoutedAssignments() const;
        std::vector<DeviceId> acceleratorDevices() const;
        MoEExpertOverlayPreparationPlan filteredForRank(const OverlayRankPlan &rank_plan) const;

        bool shouldPrepare(
            DeviceId device,
            int layer,
            int expert_id,
            ExpertGemmRegistry::WeightRole role) const;

        const MoEExpertOverlayPreparationRequest *requestFor(
            DeviceId device,
            int layer,
            int expert_id,
            ExpertGemmRegistry::WeightRole role) const;

        const MoEExpertOverlayPreparationRequest *requestForParticipant(
            const std::string &domain_name,
            DeviceId device,
            int participant_world_rank,
            int participant_index,
            int layer,
            int expert_id,
            ExpertGemmRegistry::WeightRole role) const;

        bool hasAnyRequestForDeviceLayerRole(
            DeviceId device,
            int layer,
            ExpertGemmRegistry::WeightRole role) const;

        std::vector<std::string> domainsForDeviceLayerRole(
            DeviceId device,
            int layer,
            ExpertGemmRegistry::WeightRole role) const;

        std::vector<int> expertsForDeviceLayerRole(
            DeviceId device,
            int layer,
            ExpertGemmRegistry::WeightRole role) const;

        std::vector<int> expertsForDomainDeviceLayerRole(
            const std::string &domain_name,
            DeviceId device,
            int layer,
            ExpertGemmRegistry::WeightRole role) const;

    private:
        std::vector<MoEExpertOverlayPreparationRequest> requests_;
        MoEExpertOverlayPreparationDiagnostics diagnostics_;
    };

} // namespace llaminar2
