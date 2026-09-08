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
        RoutedExpertResidencyPolicy residency_policy = RoutedExpertResidencyPolicy::Disabled;
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
        RoutedExpertResidencyPolicy residency_policy = RoutedExpertResidencyPolicy::Disabled;
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

        /**
         * @brief Restrict preparation ownership to one graph participant device.
         *
         * A per-device graph runner may prepare only the routed experts assigned
         * to its own device.  Keeping this restriction in the immutable plan
         * makes it impossible for one LocalTP runner to repack another runner's
         * experts through WeightManager's process-wide caches.  The returned
         * diagnostics are rebuilt from the retained requests, so logs describe
         * the exact work owned by the caller rather than the rank-wide plan.
         *
         * @param device Device owned by the graph runner performing preparation.
         * @return A plan containing only requests whose device equals @p device.
         */
        MoEExpertOverlayPreparationPlan filteredForDevice(DeviceId device) const;

        /**
         * @brief Restrict preparation to an explicit graph execution device set.
         *
         * A heterogeneous continuation root can execute its captured GPU graph
         * and one or more colocated CPU sparse endpoints.  Those endpoints are
         * one graph ownership unit even though their prepared engines have
         * different physical devices.  Callers provide the complete typed set;
         * no domain name or implicit host-fallback rule is reconstructed here.
         *
         * @param devices Exact devices whose rank-local requests are retained.
         * @return A plan with rebuilt diagnostics for only those devices.
         * @throws std::invalid_argument when @p devices is empty or contains an
         *         invalid or duplicate device.
         */
        MoEExpertOverlayPreparationPlan filteredForDevices(
            const std::vector<DeviceId> &devices) const;

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
