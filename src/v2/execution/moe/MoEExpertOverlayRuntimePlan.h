/**
 * @file MoEExpertOverlayRuntimePlan.h
 * @brief Resolved runtime descriptors for same-layer MoE expert overlay domains.
 *
 * The configuration-time MoERoutedExpertPlacementPlan names domains and tiers. This
 * runtime plan resolves those names to explicit rank/device descriptors. It
 * does not claim that collective contexts have been materialized. Dense
 * continuation validation and sparse expert-follower execution have distinct
 * contracts; the latter is assigned by MoEExpertOverlayExecutionPlan.
 */

#pragma once

#include "MoERoutedExpertPlacementPlan.h"
#include "backends/DeviceId.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    /** @brief One configured endpoint and its exact process-local ownership projection. */
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

    /** @brief Immutable domain geometry, not a mutable collective-readiness ledger. */
    struct MoEOverlayRuntimeDomain
    {
        std::string name;
        ExecutionDomainScope scope = ExecutionDomainScope::SINGLE;
        CollectiveBackendType backend = CollectiveBackendType::AUTO;
        RoutedExpertComputePolicy routed_compute_policy =
            RoutedExpertComputePolicy::Apportioned;
        RoutedExpertPhasePolicy routed_phase_policy =
            RoutedExpertPhasePolicy::Uniform;
        RoutedExpertAssignmentPolicy routed_decode_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        RoutedExpertAssignmentPolicy routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;

        std::vector<MoEOverlayDomainParticipant> participants;
        GlobalDeviceAddress primary_participant;
        DeviceId primary_device = DeviceId::invalid();
        int primary_world_rank = -1;
        bool primary_world_rank_known = false;
        int owner_rank = -1;
        bool primary_is_local = false;
        bool primary_owned_by_current_rank = false;
        bool local_reachable_for_mvp = false;

        int routed_tier_count = 0;
        bool routed_rebalance_controller_eligible = false;
        std::string rebalance_domain_id;

        /**
         * @brief Validate this domain specifically as a dense continuation.
         * @throws std::runtime_error if its multi-participant dense collective
         *         shape has no installed graph-native implementation.
         *
         * Sparse expert followers are not dense TP participants. Call this
         * only at continuation graph admission, never while resolving routed
         * descriptors or to advertise sparse execution support. Actual graph
         * construction still owns collective context creation and validation.
         */
        void validateContinuationCollectiveRuntime() const;
    };

    /** @brief One priority tier referring to its resolved domain and local primary. */
    struct MoEOverlayRuntimeTier
    {
        int tier_index = -1;
        RoutedExpertTier tier;
        std::string domain_name;
        DeviceId primary_device = DeviceId::invalid();
        bool local_reachable_for_mvp = false;
    };

    /** @brief Rank identity and optional local-root requirement for descriptor resolution. */
    struct MoEExpertOverlayRuntimeResolverOptions
    {
        int current_world_rank = 0;
        bool validate_mvp_root_reachability = true;
    };

    /** @brief Named ownership descriptors retained by participant-local graph builders. */
    class MoEExpertOverlayRuntimePlan
    {
    public:
        /** @brief Retain the source plan and index unique resolved domains; reject null/duplicates. */
        MoEExpertOverlayRuntimePlan(
            std::shared_ptr<const MoERoutedExpertPlacementPlan> source_plan,
            int current_world_rank,
            std::vector<MoEOverlayRuntimeDomain> domains,
            std::vector<MoEOverlayRuntimeTier> routed_tiers);

        /** @return The immutable configuration from which these descriptors were resolved. */
        const MoERoutedExpertPlacementPlan &sourcePlan() const { return *source_plan_; }
        /** @return Shared ownership for downstream graph and weight-preparation lifetimes. */
        std::shared_ptr<const MoERoutedExpertPlacementPlan> sourcePlanPtr() const { return source_plan_; }
        /** @return MPI execution rank used to resolve local device addresses. */
        int currentWorldRank() const { return current_world_rank_; }

        /** @return Complete ordered descriptors, including remote participants. */
        const std::vector<MoEOverlayRuntimeDomain> &domains() const { return domains_; }
        /** @return All routed priority tiers in configuration order. */
        const std::vector<MoEOverlayRuntimeTier> &routedTiers() const { return routed_tiers_; }

        /** @return Named descriptor, or null when the domain is absent. */
        const MoEOverlayRuntimeDomain *domainForName(const std::string &domain_name) const;
        /** @return Required continuation descriptor; missing references throw. */
        const MoEOverlayRuntimeDomain &continuationDomain() const;
        /** @return Required shared-expert descriptor; missing references throw. */
        const MoEOverlayRuntimeDomain &sharedExpertDomain() const;
        /** @return A routed tier's domain; invalid indices/references throw. */
        const MoEOverlayRuntimeDomain &domainForTier(size_t tier_index) const;

        /** @return Continuation primary when locally addressable, otherwise invalid. */
        DeviceId continuationDevice() const;
        /** @return A required domain's primary; remote primaries remain invalid locally. */
        DeviceId primaryDeviceForDomain(const std::string &domain_name) const;
        /** @brief Require the shared-expert primary to be local; layer index contextualizes errors. */
        DeviceId sharedExpertDeviceForMVP(int layer_idx = -1) const;
        /** @brief Require a tier's primary to be local; tier/layer identify failures. */
        DeviceId tierDeviceForMVP(size_t tier_index, int layer_idx = -1) const;

        /** @return Ownership and policy diagnostics without invented runtime readiness. */
        std::string diagnostics() const;

    private:
        /** @brief Resolve a mandatory name or throw with the consuming role's context. */
        const MoEOverlayRuntimeDomain &requireDomain(const std::string &domain_name, const char *context) const;

        std::shared_ptr<const MoERoutedExpertPlacementPlan> source_plan_;
        int current_world_rank_ = 0;
        std::vector<MoEOverlayRuntimeDomain> domains_;
        std::vector<MoEOverlayRuntimeTier> routed_tiers_;
        std::unordered_map<std::string, size_t> domains_by_name_;
    };

    /**
     * @brief Resolve validated placement into exact local/remote descriptors.
     * @return Null for a non-overlay plan; otherwise a retained descriptor owner.
     * @throws std::invalid_argument for malformed placement.
     * @throws std::runtime_error for missing domains or an unsatisfied local-root requirement.
     */
    std::shared_ptr<MoEExpertOverlayRuntimePlan> resolveMoEExpertOverlayRuntimePlan(
        std::shared_ptr<const MoERoutedExpertPlacementPlan> plan,
        const MoEExpertOverlayRuntimeResolverOptions &options = {});

} // namespace llaminar2
