/**
 * @file CollectiveRuntimeLifecycle.h
 * @brief Typed retirement evidence for GPU collective runtime generations.
 *
 * Native device reset invalidates every communicator, stream, and event that
 * belongs to the retired CUDA/HIP primary context.  Collective backends may
 * deliberately retain inactive coordinators between runner instances, so an
 * allocation-only preflight cannot prove that reset is safe.  This header
 * defines the small typed receipt used by the process collective authority and
 * TransferEngine to make that otherwise-hidden ownership edge explicit.
 */

#pragma once

#include "backends/DeviceId.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace llaminar2
{

    /** Exact outcome of retiring collective resources before native reset. */
    enum class CollectiveRuntimeRetirementState : std::uint8_t
    {
        /** Every matching inactive owner was destroyed; reset may proceed. */
        Complete,
        /** A live backend still owns a matching collective coordinator. */
        ActiveOwner,
        /** The request did not name one concrete GPU runtime. */
        InvalidDevice,
    };

    /**
     * @brief Immutable proof that collective runtime ownership was resolved.
     *
     * `retired_inactive_owners` counts whole coordinator owners, not devices:
     * one multi-GPU communicator clique is one lifecycle owner even when the
     * requested device is only one of its participants.  `active_owners`
     * prevents TransferEngine from resetting beneath a still-live runner.
     */
    struct CollectiveRuntimeRetirementReceipt
    {
        DeviceId device = DeviceId::invalid(); ///< Requested physical GPU.
        CollectiveRuntimeRetirementState state =
            CollectiveRuntimeRetirementState::InvalidDevice;
        std::size_t retired_inactive_owners = 0u; ///< Destroyed pooled owners.
        std::size_t active_owners = 0u; ///< Live owners blocking retirement.
        std::string diagnostic; ///< Stable failure context when incomplete.

        /** @return Whether native reset may now retire this device runtime. */
        [[nodiscard]] bool complete() const noexcept
        {
            return state == CollectiveRuntimeRetirementState::Complete &&
                   active_owners == 0u;
        }
    };

} // namespace llaminar2
