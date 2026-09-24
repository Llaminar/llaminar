/**
 * @file MTPDepthDefaults.h
 * @brief Canonical hardware-profile defaults for MTP request admission.
 *
 * Topology planning selects a profile from the complete continuation domain.
 * These immutable defaults are not a second depth controller: admission seals
 * the selected numbers into the existing device policy, and live decisions
 * remain device-owned. Explicit request thresholds always take precedence.
 */
#pragma once

#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Largest draft depth supported by the production MTP controller.
     *
     * An automatic adaptive policy is intentionally unconstrained within the
     * complete supported controller range.  Keep the device ABI tied to this
     * value rather than maintaining a second literal in kernel code.
     */
    inline constexpr int kMaximumSupportedMTPDraftDepth = 15;

    /**
     * @brief Canonical ceiling for an adaptive policy without an override.
     * @return The complete supported draft range's inclusive upper bound.
     */
    [[nodiscard]] inline constexpr int defaultMTPAdaptiveMaximumDraftDepth() noexcept
    {
        return kMaximumSupportedMTPDraftDepth;
    }

    /** @brief Measured card profiles; Portable covers uncharacterized domains. */
    enum class MTPDepthDefaultsProfile
    {
        Portable,
        CUDARTX3090,
        ROCmMI50,
    };

    /** @brief Stable diagnostic name for a topology-selected profile. */
    [[nodiscard]] inline const char *mtpDepthDefaultsProfileToString(
        MTPDepthDefaultsProfile profile)
    {
        switch (profile)
        {
        case MTPDepthDefaultsProfile::Portable: return "portable";
        case MTPDepthDefaultsProfile::CUDARTX3090: return "cuda-rtx3090";
        case MTPDepthDefaultsProfile::ROCmMI50: return "rocm-mi50";
        }
        throw std::invalid_argument("Invalid MTP hardware-default profile");
    }

    /**
     * @brief Default rejection-window fraction that permits depth demotion.
     * @param profile Immutable continuation-domain hardware classification.
     * @return Threshold measured without changing depth capacity or arithmetic.
     *
     * The MI50 profile postpones premature short-window demotion. RTX3090's
     * measured winner retains the portable threshold. Other hardware is not
     * silently classified as either measured card merely by GPU vendor.
     */
    [[nodiscard]] inline double defaultMTPZeroAcceptDemotionRate(
        MTPDepthDefaultsProfile profile)
    {
        switch (profile)
        {
        case MTPDepthDefaultsProfile::Portable:
        case MTPDepthDefaultsProfile::CUDARTX3090: return 0.30;
        case MTPDepthDefaultsProfile::ROCmMI50: return 0.45;
        }
        throw std::invalid_argument("Invalid MTP hardware-default profile");
    }
}
