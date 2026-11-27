/**
 * @file RankOrderedPeerPushOptIn.h
 * @brief Startup and per-communicator policy for the original ROCm peer transport.
 */
#pragma once

#include "backends/PeerAccessCoverage.h"
#include "collective/RankOrderedNativeResources.h"
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

    /** @brief Immutable process request; automatic selection is evaluated for each original communicator. */
    enum class RankOrderedPeerPushRequest { Automatic, Disabled, Required };

    /**
     * @param owner Native transport ownership of the actual model's LocalTP context.
     * @return Frozen request used to authenticate original communicator creation and pooling.
     */
    inline RankOrderedPeerPushRequest rankOrderedPeerPushRequest(RankOrderedQuotePolicy owner)
    {
        static const auto request = [] {
            const auto *value = DebugEnv::envValue("LLAMINAR_RCCL_RANK_ORDERED_PUSH");
            if (!value) return RankOrderedPeerPushRequest::Automatic;
            if (DebugEnv::isTruthyEnvValue(value)) return RankOrderedPeerPushRequest::Required;
            if (DebugEnv::isFalseyEnvValue(value)) return RankOrderedPeerPushRequest::Disabled;
            throw std::invalid_argument("LLAMINAR_RCCL_RANK_ORDERED_PUSH requires a boolean value");
        }();
        return request == RankOrderedPeerPushRequest::Automatic && owner == RankOrderedQuotePolicy::ExplicitOnly
            ? RankOrderedPeerPushRequest::Disabled : request;
    }

    /**
     * @brief Resolve process/owner policy for CUDA direct P2P peer-push collectives.
     * @param owner Native transport ownership of the actual model's LocalTP context.
     * @return Frozen request: Automatic when MeasuredDomain, Disabled when ExplicitOnly,
     *         overridable via LLAMINAR_CUDA_PEER_PUSH=0|1.
     */
    inline RankOrderedPeerPushRequest cudaPeerPushRequest(RankOrderedQuotePolicy owner)
    {
        static const auto request = [] {
            const auto *value = DebugEnv::envValue("LLAMINAR_CUDA_PEER_PUSH");
            if (!value) return RankOrderedPeerPushRequest::Automatic;
            if (DebugEnv::isTruthyEnvValue(value)) return RankOrderedPeerPushRequest::Required;
            if (DebugEnv::isFalseyEnvValue(value)) return RankOrderedPeerPushRequest::Disabled;
            throw std::invalid_argument("LLAMINAR_CUDA_PEER_PUSH requires a boolean value");
        }();
        return request == RankOrderedPeerPushRequest::Automatic && owner == RankOrderedQuotePolicy::ExplicitOnly
            ? RankOrderedPeerPushRequest::Disabled : request;
    }

    /**
     * @brief Select the measured rank-local transport before quoting any device storage.
     * @param request Frozen startup override.
     * @param coverage Directed peer data for exactly these participants.
     * @param qualified_isa Whether every member supports the native program.
     * @param measured_cohort Whether this is the benchmarked four-MI50 arrangement.
     * @return Whether to quote and eventually prepare direct-push resources.
     * @throws std::invalid_argument If an explicitly required transport is unavailable.
     */
    inline bool selectRankOrderedPeerPush(RankOrderedPeerPushRequest request,
        std::optional<PeerAccessCoverage> coverage, bool qualified_isa, bool measured_cohort)
    {
        if (request == RankOrderedPeerPushRequest::Required) {
            if (!qualified_isa)
                throw std::invalid_argument("RCCL FP32 direct peer requires qualified gfx906 participants");
            requireRankOrderedPeerPushClique(coverage);
            return true;
        }
        return request == RankOrderedPeerPushRequest::Automatic && measured_cohort &&
               qualified_isa && coverage == PeerAccessCoverage::Complete;
    }
}
