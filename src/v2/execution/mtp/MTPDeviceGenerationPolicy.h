/**
 * @file MTPDeviceGenerationPolicy.h
 * @brief Converts request-level MTP policy into the fixed device-controller ABI.
 *
 * MTP request policy is authored once by the orchestration layer and admitted
 * into a retained graph family.  This header owns the only conversion from the
 * host-facing runtime configuration (including floating-point thresholds) to
 * the integer-only policy consumed by CUDA, ROCm, and CPU controller tests.
 * Keeping that conversion here prevents setup graphs and live request code
 * from independently deriving subtly different depth bounds.
 */

#pragma once

#include "../config/RuntimeConfig.h"
#include "../../kernels/common/SamplingMath.h"

#include <cmath>

namespace llaminar2
{
    /**
     * @brief Seal one request's MTP depth policy into the device controller ABI.
     * @param mtp Complete active request configuration.
     * @return Fixed-width policy suitable for request admission.
     *
     * Floating-point rates are converted exactly once at the host admission
     * boundary.  Captured kernels subsequently compare integer PPM counters in
     * a fixed arithmetic order. Invalid host values deliberately produce an
     * invalid device policy; this function never clamps or substitutes a
     * different execution mode.
     */
    [[nodiscard]] inline sampling_math::DeviceGenerationPolicy
    resolveMTPDeviceGenerationDepthPolicy(const MTPRuntimeConfig &mtp)
    {
        using sampling_math::DeviceGenerationPolicy;
        using sampling_math::DeviceGenerationPolicyMode;

        const auto rate_to_ppm = [](double rate) -> int
        {
            if (!std::isfinite(rate) || rate < 0.0 || rate > 1.0)
                return -1;
            return static_cast<int>(std::llround(
                rate * DeviceGenerationPolicy::kRateScale));
        };

        DeviceGenerationPolicy policy;
        const MTPDepthPolicyConfig depth = resolveMTPDepthPolicyConfig(mtp);
        switch (mtp.depth_policy.mode)
        {
        case MTPDepthPolicyMode::Fixed:
            policy.mode = DeviceGenerationPolicyMode::Fixed;
            policy.minimum_depth = mtp.draft_tokens;
            policy.maximum_depth = mtp.draft_tokens;
            policy.initial_depth = mtp.draft_tokens;
            break;
        case MTPDepthPolicyMode::Observe:
            policy.mode = DeviceGenerationPolicyMode::Observe;
            policy.minimum_depth = depth.min_depth;
            policy.maximum_depth =
                resolveMTPMaximumExecutionDraftDepth(mtp);
            policy.initial_depth = resolveMTPDepthPolicyInitialDepth(
                depth,
                mtp.draft_tokens,
                mtp.verify_mode);
            break;
        case MTPDepthPolicyMode::Dynamic:
            policy.mode = DeviceGenerationPolicyMode::Dynamic;
            policy.minimum_depth = depth.min_depth;
            policy.maximum_depth =
                resolveMTPMaximumExecutionDraftDepth(mtp);
            policy.initial_depth = resolveMTPDepthPolicyInitialDepth(
                depth,
                mtp.draft_tokens,
                mtp.verify_mode);
            break;
        default:
            // Zero depth is rejected by DeviceGenerationPolicy::valid().
            policy = DeviceGenerationPolicy::fixed(0);
            break;
        }

        policy.window_size = depth.window_size;
        policy.minimum_samples = depth.min_samples;
        policy.cooldown_steps = depth.cooldown_steps;
        policy.promote_consecutive_windows =
            depth.promote_consecutive_windows;
        policy.promote_full_accept_rate_ppm =
            rate_to_ppm(depth.promote_full_accept_rate);
        policy.demote_zero_accept_rate_ppm =
            rate_to_ppm(*depth.demote_zero_accept_rate);
        policy.demote_acceptance_rate_ppm =
            rate_to_ppm(depth.demote_acceptance_rate);
        return policy;
    }
}
