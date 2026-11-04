/**
 * @file MoEOverlayJointPlacementCandidateTestAccess.h
 * @brief Explicit test boundary for the actual private joint placement producer.
 *
 * Tests author only tier assignments through the same closed authority method.
 * They cannot replace a source definition, forge a Snapshot or publish a bank.
 */
#pragma once
#include "execution/moe/MoEOverlayResidencyAuthority.h"
namespace llaminar2::test
{
    struct MoEOverlayJointPlacementCandidateTestAccess
    {
        /**
         * @brief Exercise genuine unpublished placement construction without enabling maintenance.
         * @param authority Actual authority retaining original source and current placement.
         * @param placements Complete original-layer tier assignments.
         * @return Checked immutable candidate, never physical readiness or publication.
         */
        static std::shared_ptr<const MoEOverlayResidencySnapshot>
        prepare(const MoEOverlayResidencyAuthority &authority,
                std::vector<RoutedExpertLayerPlacement> placements)
        {
            const auto live = authority.snapshot();
            return authority.prepareCPUJointPlacementCandidate(live, live->epoch,
                                                               std::move(placements));
        }
        /**
         * @brief Prove stale or foreign expected-state rejection through the actual producer.
         * @param authority Actual source authority.
         * @param expected_live Caller-retained expected publication.
         * @param expected_epoch Caller-retained expected epoch.
         * @param placements Complete proposed tier assignments.
         * @return Genuine candidate only when the expected publication is still exact.
         */
        static std::shared_ptr<const MoEOverlayResidencySnapshot>
        prepareExpected(const MoEOverlayResidencyAuthority &authority,
                        const std::shared_ptr<const MoEOverlayResidencySnapshot> &expected_live,
                        uint64_t expected_epoch, std::vector<RoutedExpertLayerPlacement> placements)
        {
            return authority.prepareCPUJointPlacementCandidate(expected_live, expected_epoch,
                                                               std::move(placements));
        }
    };
} // namespace llaminar2::test
