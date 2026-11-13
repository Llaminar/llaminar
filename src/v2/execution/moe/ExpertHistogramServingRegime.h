/**
 * @file ExpertHistogramServingRegime.h
 * @brief One retained MTP phase-economy policy for whole-model and stage owners.
 *
 * Positive-depth MTP keeps serial decode graph-reachable for short tails, but
 * only depth-zero adaptive policy prices it as recurring service. Stage offsets
 * affect topology coordinates, never this common request-policy projection.
 */
#pragma once
#include "DecodeExpertHistogram.h"
#include "execution/config/RuntimeConfig.h"

namespace llaminar2
{
    /**
     * @brief Resolve recurring service phases from the retained MTP policy.
     * @param mtp Exact admitted policy, not a current request's transient depth.
     * @return Serial, strictly positive MTP, or adaptive serial/MTP economy.
     */
    [[nodiscard]] inline ExpertHistogramServingRegime expertHistogramServingRegime(
        const MTPRuntimeConfig &mtp) noexcept
    {
        if (!mtp.enabled) return ExpertHistogramServingRegime::Serial;
        if (mtp.depth_policy.mode != MTPDepthPolicyMode::Fixed && mtp.depth_policy.min_depth == 0)
            return ExpertHistogramServingRegime::AdaptiveSerialOrMTP;
        return ExpertHistogramServingRegime::PositiveDepthMTP;
    }
}
