/**
 * @file MTPDepthLearnedPolicy.h
 * @brief One generated-rule interpreter for CPU and device-resident MTP.
 *
 * The trainer owns the rule rows. Admission supplies immutable backend/model/
 * sampling keys; the live controller supplies integer window observations.
 * No clocks, host polling, tensor transfers or additional allocations occur.
 */
#pragma once

#include "../config/MTPDepthPolicyTypes.h"
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MTP_POLICY_HD __host__ __device__ inline
#else
#define LLAMINAR_MTP_POLICY_HD inline
#endif

namespace llaminar2
{
    /** @brief Offline trainer ABI; rule order defines first-match precedence. */
    struct MTPGeneratedDepthPolicyRule
    {
        MTPVerifyMode verify_mode;
        MTPDepthPolicyBackend backend;
        MTPDepthPolicyModelClass model_class;
        int current_depth;
        double min_acceptance_rate;
        double max_acceptance_rate;
        double max_zero_accept_rate;
        double min_full_accept_rate;
        int depth_delta;
        const char *label;
    };

    /**
     * @brief Offline-measured admission choice, separate from live transitions.
     *
     * Several depths can have profitable hold intervals. Their numeric order
     * does not rank request-wide economics. The trainer chooses this warm start
     * from measured training throughput; admission publishes it through the
     * existing host/device depth contract without changing adaptive capacity.
     */
    struct MTPGeneratedDepthPolicyStartup
    {
        MTPVerifyMode verify_mode;
        MTPDepthPolicyBackend backend;
        MTPDepthPolicyModelClass model_class;
        int initial_depth;
    };

#include "MTPDepthPolicyGenerated.inc"

    /** @brief Immutable learned-policy identity, carried in the resident row. */
    struct MTPLearnedDepthPolicyContext
    {
        bool enabled = false;
        MTPDepthPolicyBackend backend = MTPDepthPolicyBackend::Any;
        MTPDepthPolicyModelClass model_class = MTPDepthPolicyModelClass::Any;
        MTPVerifyMode verify_mode = MTPVerifyMode::Greedy;

        /** @return Whether every key is a supported value, even when disabled. */
        LLAMINAR_MTP_POLICY_HD bool valid() const
        {
            return (backend == MTPDepthPolicyBackend::Any ||
                    backend == MTPDepthPolicyBackend::CPU ||
                    backend == MTPDepthPolicyBackend::CUDA ||
                    backend == MTPDepthPolicyBackend::ROCm) &&
                   (model_class == MTPDepthPolicyModelClass::Any ||
                    model_class == MTPDepthPolicyModelClass::Dense ||
                    model_class == MTPDepthPolicyModelClass::MoE) &&
                   (verify_mode == MTPVerifyMode::Greedy ||
                    verify_mode == MTPVerifyMode::SpeculativeSampling);
        }
    };

    /** @brief Integer observations; zero-denominator windows never match. */
    struct MTPLearnedDepthWindow
    {
        uint64_t attempted = 0;
        uint64_t accepted = 0;
        uint64_t runs = 0;
        uint64_t zero_accepts = 0;
        uint64_t full_accepts = 0;
    };

    /** @brief A hold is a real match, distinct from no learned recommendation. */
    struct MTPLearnedDepthMatch
    {
        bool matched = false;
        int depth_delta = 0;
    };

    /** @brief Convert the trainer's six-decimal rate to an exact integer key. */
    LLAMINAR_MTP_POLICY_HD constexpr uint64_t mtpLearnedRatePPM(double rate)
    {
        return static_cast<uint64_t>(rate * 1'000'000.0 + 0.5);
    }

    /**
     * @brief Interpret generated rows in their original order on either owner.
     * @tparam Index Compile-time row ordinal, starting at zero.
     * @param context Immutable admitted policy identity.
     * @param depth Current live speculative depth.
     * @param window Completed verifier observations at that depth.
     * @return First learned match, including delta-zero holds.
     *
     * Compile-time traversal embeds scalar constants in GPU instructions. It
     * avoids a host table pointer or a per-thread array that could spill. Both
     * owners compare integer cross-products against the trainer's PPM rates;
     * neither rounds an observed acceptance ratio at a threshold boundary.
     */
    template <size_t Index = 0>
    LLAMINAR_MTP_POLICY_HD MTPLearnedDepthMatch matchMTPLearnedDepthPolicy(
        const MTPLearnedDepthPolicyContext &context, int depth,
        const MTPLearnedDepthWindow &window)
    {
        if (!context.enabled || window.attempted == 0 || window.runs == 0)
            return {};
        if constexpr (Index < sizeof(kMTPGeneratedDepthPolicyRules) /
                                  sizeof(kMTPGeneratedDepthPolicyRules[0]))
        {
            constexpr auto rule = kMTPGeneratedDepthPolicyRules[Index];
            if (rule.verify_mode == context.verify_mode &&
                (rule.backend == MTPDepthPolicyBackend::Any || rule.backend == context.backend) &&
                (rule.model_class == MTPDepthPolicyModelClass::Any || rule.model_class == context.model_class) &&
                rule.current_depth == depth &&
                window.accepted * 1'000'000 >= window.attempted * mtpLearnedRatePPM(rule.min_acceptance_rate) &&
                window.accepted * 1'000'000 <= window.attempted * mtpLearnedRatePPM(rule.max_acceptance_rate) &&
                window.zero_accepts * 1'000'000 <= window.runs * mtpLearnedRatePPM(rule.max_zero_accept_rate) &&
                window.full_accepts * 1'000'000 >= window.runs * mtpLearnedRatePPM(rule.min_full_accept_rate))
            {
                return {.matched = true, .depth_delta = rule.depth_delta};
            }
            return matchMTPLearnedDepthPolicy<Index + 1>(context, depth, window);
        }
        return {};
    }
}

#undef LLAMINAR_MTP_POLICY_HD
