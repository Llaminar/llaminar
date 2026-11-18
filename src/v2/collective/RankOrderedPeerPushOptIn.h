/**
 * @file RankOrderedPeerPushOptIn.h
 * @brief Process-lifetime policy for the original ROCm peer transport.
 */
#pragma once

#include "backends/PeerAccessCoverage.h"
#include "utils/DebugEnv.h"
#include <optional>
#include <string_view>

namespace llaminar2
{
    /**
     * @brief Require the driver-observed complete directed clique before native initialization.
     * @param coverage Observation for the exact ordered device cohort.
     * @throws std::invalid_argument If any directed edge or observation is missing.
     */
    inline void requireRankOrderedPeerPushClique(std::optional<PeerAccessCoverage> coverage)
    {
        if (coverage != PeerAccessCoverage::Complete)
            throw std::invalid_argument("RCCL FP32 direct peer requires a complete observed directed P2P clique");
    }

    /**
     * @brief Determine whether all required runtime peer properties were qualified for this ISA.
     * @param target Actual enumerated HIP target including optional feature suffixes.
     * @return Whether this ISA is admitted to the opt-in transport.
     */
    [[nodiscard]] constexpr bool rankOrderedPeerPushQualifiedISA(std::string_view target) noexcept
    {
        return target.substr(0, target.find(':')) == "gfx906";
    }

    /**
     * @brief Freeze the peer selection before original communicator quotation.
     * @return Whether explicit startup policy enables the admitted small route.
     * @details Quotation and graph recording read the same immutable selection.
     * A later environment change cannot mutate an original communicator or its captures.
     */
    inline bool rankOrderedPeerPushStartupOptIn()
    {
        static const bool selected = DebugEnv::isTruthyEnv("LLAMINAR_RCCL_RANK_ORDERED_PUSH");
        return selected;
    }
}
