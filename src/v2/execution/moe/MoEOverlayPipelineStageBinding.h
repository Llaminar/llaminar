/**
 * @file MoEOverlayPipelineStageBinding.h
 * @brief Checked handoff of one pipeline stage's expert runtime to its graphs.
 *
 * PP composes different layer owners. Sibling TP graphs within one stage share
 * one residency authority and prepared-bank registry; different stages cannot
 * inherit the parent's model-wide runtime. This setup-only value retains those
 * owners together and authenticates their layer and participant identities.
 * It allocates no model storage and owns no physical-memory accounting.
 */
#pragma once

#include "execution/factory/FactoryPPStageConfig.h"
#include "MoERoutedExpertPlacementPlanner.h"
#include <memory>
#include <span>
#include <vector>

namespace llaminar2
{
    class MoEOverlayResidencyAuthority;
    class MoEOverlayParticipantResidencyRegistry;
    class IMPIContext;
    class MoEOverlayRankBatchTransportRegistry;
    class MoEOverlayNodeLocalDeviceControllerFabric;

    /**
     * @brief Retained, scope-checked owners for one PP stage's expert execution.
     *
     * Construction derives placement from the authority's initial snapshot.
     * A caller cannot independently install a plan from one stage and live
     * residency from another. Route exchange remains child-RankOrchestrator
     * owned and is constructed from that child's exact participant topology.
     */
    class MoEOverlayPipelineStageBinding final
    {
    public:
        /** @brief Existing runtime owners supplied by orchestration admission. */
        struct Owners
        {
            std::shared_ptr<MoEOverlayResidencyAuthority> authority;
            std::shared_ptr<MoEOverlayParticipantResidencyRegistry> residency;
            std::shared_ptr<DecodeExpertHistogram> histogram;
            std::shared_ptr<IMPIContext> mpi;
            std::shared_ptr<MoEOverlayRankBatchTransportRegistry> rank_transport;
            std::shared_ptr<MoEOverlayNodeLocalDeviceControllerFabric> device_fabric;
        };

        /**
         * @brief Seal exact stage geometry and the live owners admitted for it.
         * @param stage Main-model layer interval and global edge roles.
         * @param metadata Model-resolved routed capacity, including terminal MTP.
         * @param participants Ordered continuation TP devices, or one PP device.
         * @param owners Existing authority and process-local runtime lifetimes.
         * @return Immutable stage binding; no GPU or prepared weight is allocated.
         * @throws std::invalid_argument for missing or mismatched owners/geometry.
         */
        [[nodiscard]] static std::shared_ptr<const MoEOverlayPipelineStageBinding> seal(
            FactoryPPStageConfig stage, MoERoutedExpertModelMetadata metadata,
            std::vector<GlobalDeviceAddress> participants, Owners owners);

        /**
         * @brief Reject installing a sealed runtime into a different PP stage.
         * @param stage Exact destination main-layer interval and edge roles.
         * @param participants Destination devices in canonical TP order.
         * @param weights Exact admitted tensor-parallel workshare declaration.
         * @param backend Collective implementation bound to that declaration.
         * @param mode Requested maintenance mode, which must retain its authority.
         * @throws std::invalid_argument for any changed scope, role or policy.
         */
        void requireDestination(const FactoryPPStageConfig &stage,
            std::span<const GlobalDeviceAddress> participants,
            std::span<const float> weights, CollectiveBackendType backend,
            MoERebalanceRuntimeMode mode) const;

        /** @return The sole initial placement published by this stage's authority. */
        [[nodiscard]] const std::shared_ptr<const MoERoutedExpertPlacementPlan> &plan() const noexcept
        { return plan_; }
        /** @return Exact runtime owners retained through child graph destruction. */
        [[nodiscard]] const Owners &owners() const noexcept { return owners_; }
        /** @return Model-resolved compact routed capacity, never allocation accounting. */
        [[nodiscard]] const MoERoutedExpertModelMetadata &metadata() const noexcept { return metadata_; }

    private:
        /** @brief Only seal() publishes a fully checked binding. */
        MoEOverlayPipelineStageBinding(FactoryPPStageConfig stage,
            MoERoutedExpertModelMetadata metadata,
            std::vector<GlobalDeviceAddress> participants, Owners owners,
            std::shared_ptr<const MoERoutedExpertPlacementPlan> plan);

        FactoryPPStageConfig stage_;
        MoERoutedExpertModelMetadata metadata_;
        std::vector<GlobalDeviceAddress> participants_;
        Owners owners_;
        std::shared_ptr<const MoERoutedExpertPlacementPlan> plan_;
    };
}
