/**
 * @file MoEOverlayParticipantMigrationTestAccess.h
 * @brief Explicit recorded component construction; never original model-source readiness.
 *
 * Existing controller/consensus tests own recorded engine/operation states,
 * including inert GPU identities. This private access preserves that boundary
 * instead of fabricating ModelContext IDs or weakening public factory admission.
 */
#pragma once
#include "execution/moe/MoEOverlayParticipantMigration.h"
namespace llaminar2::test
{
    struct MoEOverlayParticipantMigrationTestAccess final
    {
        /**
         * @brief Construct only the existing component protocol under recorded inputs.
         * @param config Test-owned bank and operation state; no native model is certified.
         * @return Factory retaining the same component operations and bank lifecycle.
         */
        static MoEOverlayParticipantPreparedWaveFactory
        recordedComponentFactory(MoEOverlayParticipantPreparedWaveFactory::Config config)
        {
            return MoEOverlayParticipantPreparedWaveFactory(
                std::move(config),
                MoEOverlayParticipantPreparedWaveFactory::RecordedComponentSource{});
        }
    };
} // namespace llaminar2::test
