/**
 * @file MoEGroupedPlanDemand.h
 * @brief Serial-visible demand ownership for a complete grouped expert plan.
 *
 * Grouped math is also used for one-row decode, static execution and MTP
 * sidecars. None of those is ordinary prefill merely because it does not
 * retain speculative routes. This immutable graph policy names the one
 * permitted history side effect; it never owns or mirrors device counters.
 */
#pragma once

#include <cstdint>

namespace llaminar2
{
    /** @brief History publication performed while publishing a complete plan. */
    enum class MoEGroupedPlanDemand : uint8_t
    {
        OrdinaryPrefill = 0, ///< Count serial-visible prefill routes exactly once.
        DeferredAcceptedRows = 1, ///< Retain final assignments for the existing MTP commit.
        None = 2, ///< Static/sidecar work, or decode already counted by its router.
    };

    /** @brief Reject corrupt or unresolved policy before recording any GPU work. */
    constexpr bool validMoEGroupedPlanDemand(MoEGroupedPlanDemand demand) noexcept
    {
        return demand == MoEGroupedPlanDemand::OrdinaryPrefill ||
               demand == MoEGroupedPlanDemand::DeferredAcceptedRows ||
               demand == MoEGroupedPlanDemand::None;
    }
} // namespace llaminar2
