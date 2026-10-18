/**
 * @file MoEOverlayResidencySetup.h
 * @brief Checked construction of one admitted expert authority's runtime owners.
 *
 * Main-model and pipeline-stage setup use this same boundary. All layer IDs
 * remain model-global while histograms and prepared-bank registries retain only
 * their owned interval. The result publishes its owners together after complete
 * validation and never introduces a second admission or live memory ledger.
 */
#pragma once

#include "MoEOverlayCapacityAdmission.h"
#include "MoERoutedExpertPlacementPlanner.h"
#include <memory>

namespace llaminar2
{
    class ModelLoader;
    class MoEOverlayResidencyAuthority;
    class MoEOverlayParticipantResidencyRegistry;
    class MoEOverlayInferenceInterferenceProbe;

    /** @brief Borrowed immutable setup inputs; all referents outlive create(). */
    struct MoEOverlayResidencySetupInput
    {
        const MoERoutedExpertPlacementPlan &plan;
        const MoERoutedExpertModelMetadata &metadata;
        const ModelLoader &loader;
        const MoERebalanceRuntimeConfig &rebalance;
        const MoEOverlayCapacityAdmissionPolicy &storage_policy;
        const MoEOverlayResolvedCapacityPlan &capacity;
        std::shared_ptr<PhysicalMemoryAuthority> memory;
        int world_rank = 0;
        int world_size = 1;
    };

    /**
     * @brief Sealed runtime owner set for one exact expert placement interval.
     *
     * This object may be copied to retain lifetimes, but its owners cannot be
     * independently replaced. Prepared payloads arrive later through the existing
     * registry publication contract. No device context or model tensor is created.
     */
    class MoEOverlayResidencySetup final
    {
    public:
        /**
         * @brief Build histogram, residency publication and empty prepared endpoints.
         * @param input Frozen placements and their original physical admission.
         * @return Complete owner set with exact global coordinates and source policy.
         * @throws std::invalid_argument for mismatched scope, rank, model or certificate.
         * @throws std::logic_error for incomplete Dynamic demand/economy ownership.
         *
         * Host Dynamic transaction banks claim their admitted PMA storage here.
         * GPU-owned policy remains on device; this boundary does not manufacture
         * a host planner or a movement service for native GPU domains.
         */
        static MoEOverlayResidencySetup create(const MoEOverlayResidencySetupInput &input);

        /** @return Routing evidence whose lifetime encloses its authority's raw reference. */
        const std::shared_ptr<DecodeExpertHistogram> &histogram() const noexcept { return histogram_; }
        /** @return Sole epoch and expert-owner publication authority for this interval. */
        const std::shared_ptr<MoEOverlayResidencyAuthority> &authority() const noexcept { return authority_; }
        /** @return Every prepared endpoint physically owned by this rank. */
        const std::shared_ptr<MoEOverlayParticipantResidencyRegistry> &residency() const noexcept { return residency_; }
        /** @return Dynamic interference observation, absent for Off/Observe policy. */
        const std::shared_ptr<MoEOverlayInferenceInterferenceProbe> &interference() const noexcept { return interference_; }

    private:
        /** @brief Only create() can publish a completely validated owner set. */
        MoEOverlayResidencySetup() = default;
        // Destruction retires prepared endpoints and the authority before the
        // histogram it borrows. Shared consumers must preserve the same edge.
        std::shared_ptr<DecodeExpertHistogram> histogram_;
        std::shared_ptr<MoEOverlayResidencyAuthority> authority_;
        std::shared_ptr<MoEOverlayParticipantResidencyRegistry> residency_;
        std::shared_ptr<MoEOverlayInferenceInterferenceProbe> interference_;
    };
}
