/**
 * @file MoEExpertOverlayPreparationPlan.h
 * @brief Immutable participant-local preparation requests derived from overlay ownership.
 *
 * Logical placement is global; preparation is rank/device local and projection
 * specific. Requests retain both identities before any allocation or packing.
 * Filtering cannot reassign owners or drop source/slice identity. Diagnostics
 * describe requested work, never replace PhysicalMemoryAuthority admission.
 */
#pragma once

#include "MoEExpertOverlayRuntimePlan.h"
#include "MoEExpertProjectionOwnership.h"
#include "loaders/ExpertGemmRegistry.h"
#include "loaders/WeightIdentity.h"
#include "loaders/WeightPlan.h"

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
    struct OverlayRankPlan;
    struct MoEExpertOwnerParticipant;
    class IModelLoader;

    /** @brief One immutable-at-publication expert/projection materialization request. */
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
        /** Exact source/partition identity; absence means an ordinary complete expert. */
        std::optional<MoEExpertProjectionOwnership> projection_ownership;
    };

    /** @brief Passive preparation counts for one exact rank/device/domain endpoint. */
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
        size_t assigned_routed_experts = 0; ///< Gate/up owner count, not fixed down slices.
        size_t planned_engine_count = 0;
        size_t estimated_routed_bytes = 0;
    };

    /** @brief Setup-only summaries with no live capacity or placement authority. */
    struct MoEExpertOverlayPreparationDiagnostics
    {
        std::vector<MoEExpertOverlayDomainPreparationStats> domains;

        /** @return First matching domain/device summary, or null if absent. */
        const MoEExpertOverlayDomainPreparationStats *domainStats(
            const std::string &domain_name,
            DeviceId device) const;
        /** @return Exact participant summary, including rank identity, or null. */
        const MoEExpertOverlayDomainPreparationStats *domainStats(
            const std::string &domain_name,
            DeviceId device,
            int participant_world_rank,
            int participant_index) const;
        /** @return Human-readable setup counts, not a physical capacity certificate. */
        std::string render() const;
    };

    /** @brief Compile a frozen owner plan into exact local projection requests. */
    class MoEExpertOverlayPreparationPlan
    {
    public:
        /**
         * @brief Compile ordinary complete-expert preparation from the existing owner map.
         * @param runtime_plan Frozen topology and initial global expert ownership.
         * @param routed_expert_bytes_per_expert Optional diagnostic whole-expert estimate.
         * @return Local requests; remote experts remain the owning rank's obligation.
         */
        static MoEExpertOverlayPreparationPlan build(
            const MoEExpertOverlayRuntimePlan &runtime_plan,
            size_t routed_expert_bytes_per_expert = 0);

        /**
         * @brief Prepare exactly the physical mode declared by the frozen plan.
         * @param runtime_plan Canonical topology, ownership and compute-mode choice.
         * @param loader Authoritative source directory for projection geometry.
         * @param routed_expert_bytes_per_expert Optional whole-expert diagnostic estimate.
         * @return Whole-expert or gate/up-owned requests without changing policy.
         * @throws std::invalid_argument For inconsistent or unsupported geometry.
         *
         * The mode is part of the plan, not a debug flag or a caller-selected
         * packing shortcut. Keeping both modes behind this entrypoint makes
         * source loading and final materialization consume the same A/B choice.
         */
        static MoEExpertOverlayPreparationPlan build(
            const MoEExpertOverlayRuntimePlan &runtime_plan,
            const IModelLoader &loader,
            size_t routed_expert_bytes_per_expert = 0);

        /**
         * @brief Compile the exact source selections consumed by these requests.
         * @param loader Authoritative GGUF directory; no tensor payload is loaded here.
         * @param model_id Model lifetime owning the resulting frozen bindings.
         * @return Rank/device-scoped source requirements, with independent IDs per role.
         * @throws std::invalid_argument for missing or incompatible source geometry.
         *
         * Callers first filter this immutable plan to their graph's endpoints.
         * Loading and packing then consume the same requests, including down
         * slices on endpoints that own no gate/up experts. Expert-axis loading
         * retains full N/K on the host; MoEExpertSourceView selects the exact
         * output interval before GPU preparation. The source slice must never
         * claim an N partition that the loader has not physically performed.
         */
        WeightPlan sourceWeightPlan(const IModelLoader &loader, ModelContextId model_id) const;

        /** @return Stable requests in layer/expert/role/participant construction order. */
        const std::vector<MoEExpertOverlayPreparationRequest> &requests() const
        {
            return requests_;
        }

        /** @return Passive summaries rebuilt after every filtering operation. */
        const MoEExpertOverlayPreparationDiagnostics &diagnostics() const
        {
            return diagnostics_;
        }

        /** @return Whether this rank has no materialization work. */
        bool empty() const { return requests_.empty(); }
        /** @return Whether the exact device has any projection request. */
        bool hasRequestsForDevice(DeviceId device) const;
        /** @return Whether any request targets a CUDA or ROCm device. */
        bool hasAcceleratorRequests() const;
        /** @return Whether any local request targets CPU execution. */
        bool hasCpuRoutedAssignments() const;
        /** @return Sorted unique accelerator devices with preparation work. */
        std::vector<DeviceId> acceleratorDevices() const;
        /** @brief Retain this rank's requests and rebuild its passive diagnostics. */
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

        /** @return Whether this exact local expert projection is requested. */
        bool shouldPrepare(
            DeviceId device,
            int layer,
            int expert_id,
            ExpertGemmRegistry::WeightRole role) const;

        /** @return Matching device-local request, or null; scoped users use the participant query. */
        const MoEExpertOverlayPreparationRequest *requestFor(
            DeviceId device,
            int layer,
            int expert_id,
            ExpertGemmRegistry::WeightRole role) const;

        /** @return Exact domain/rank/device/expert/projection request, or null. */
        const MoEExpertOverlayPreparationRequest *requestForParticipant(
            const std::string &domain_name,
            DeviceId device,
            int participant_world_rank,
            int participant_index,
            int layer,
            int expert_id,
            ExpertGemmRegistry::WeightRole role) const;

        /**
         * @brief Resolve the frozen projection layout independently of movable residents.
         * @param participant Exact owner-map endpoint, not a physical ordinal used as a slice ID.
         * @param layer Source model layer required by the runtime consumer.
         * @return Source-authenticated layout shared by every request for this participant/layer.
         * @throws std::invalid_argument For absent, whole-expert, or inconsistent requests.
         *
         * Every projected participant requests down slices, including a zero
         * gate/up owner. This query therefore never guesses a layout from a
         * resident payload or silently interprets missing preparation as whole
         * experts. It performs no allocation or new placement decision.
         */
        MoEExpertProjectionOwnership requireProjectionOwnershipForParticipant(
            const MoEExpertOwnerParticipant &participant, int layer) const;

        /** @return Whether the named layer/projection needs any local source tensor. */
        bool hasAnyRequestForDeviceLayerRole(
            DeviceId device,
            int layer,
            ExpertGemmRegistry::WeightRole role) const;

        /** @return Sorted unique domains consuming one local source projection. */
        std::vector<std::string> domainsForDeviceLayerRole(
            DeviceId device,
            int layer,
            ExpertGemmRegistry::WeightRole role) const;

        /** @return Sorted unique physical residents for this projection, not whole experts. */
        std::vector<int> expertsForDeviceLayerRole(
            DeviceId device,
            int layer,
            ExpertGemmRegistry::WeightRole role) const;

        /** @return Sorted unique projection residents restricted to the exact domain. */
        std::vector<int> expertsForDomainDeviceLayerRole(
            const std::string &domain_name,
            DeviceId device,
            int layer,
            ExpertGemmRegistry::WeightRole role) const;

    private:
        /**
         * @brief Compile only an explicitly selected gate/up-owned physical mode.
         * @param runtime_plan Frozen GateUpOwnedDownColumns plan for one homogeneous GPU domain.
         * @param loader Canonical directory authenticating each source layer independently.
         * @return Exact movable gate/up and fixed full-K down-slice requests.
         * @throws std::invalid_argument For partial, heterogeneous or incompatible ownership.
         *
         * Private by design: no caller may turn an Apportioned control plan into
         * this mode merely by choosing a different preparation function. The
         * public builder dispatches from the sole frozen compute policy.
         */
        static MoEExpertOverlayPreparationPlan buildProjectionPartitioned(
            const MoEExpertOverlayRuntimePlan &runtime_plan,
            const IModelLoader &loader);

        /** @brief Shared compiler for complete and explicitly partitioned projection requests. */
        static MoEExpertOverlayPreparationPlan compile(
            const MoEExpertOverlayRuntimePlan &runtime_plan,
            size_t routed_expert_bytes_per_expert,
            const std::map<int, MoEExpertProjectionOwnership::Geometry> &projection_geometries);

        std::vector<MoEExpertOverlayPreparationRequest> requests_;
        MoEExpertOverlayPreparationDiagnostics diagnostics_;
    };

} // namespace llaminar2
