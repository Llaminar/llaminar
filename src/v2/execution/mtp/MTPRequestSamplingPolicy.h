/**
 * @file MTPRequestSamplingPolicy.h
 * @brief Resolve MTP execution semantics from the admitted request sampling law.
 *
 * Verification capability belongs to the retained/request policy. The sampler
 * determines which specialization executes: greedy requests do not build
 * probability distributions and must not inherit stochastic depth economics.
 * This projection is ephemeral; it never mutates graph capacity, placement or
 * the authored policy, so successive greedy/stochastic requests share one
 * retained graph family without a second configuration authority.
 */
#pragma once

#include "../config/RuntimeConfig.h"
#include "../../utils/Sampler.h"

namespace llaminar2
{
    /**
     * @brief Compose an active request with its exact sampling specialization.
     * @param retained Immutable physical graph/weight envelope.
     * @param requested Request-selectable verification and depth intent.
     * @param sampling Admitted sampler, including argmax-equivalent top-k rules.
     * @return Execution view; stochastic capability specializes only for argmax.
     *
     * An explicitly greedy-only policy remains greedy even for an incompatible
     * sampler. The runner rejects that pair before decode instead of upgrading
     * an operator's explicit policy or silently bypassing MTP.
     */
    [[nodiscard]] inline MTPRuntimeConfig resolveMTPSamplingRequestConfig(
        const MTPRuntimeConfig &retained,
        const MTPRequestPolicy &requested,
        const SamplingParams &sampling)
    {
        auto active = composeMTPRequestConfig(retained, requested);
        if (active.verify_mode == MTPVerifyMode::SpeculativeSampling &&
            sampling.is_greedy())
        {
            active.verify_mode = MTPVerifyMode::Greedy;
        }
        return active;
    }
}
