/**
 * @file ServingRowState.h
 * @brief Typed per-row serving state record (design §6 per-row ABI substrate).
 *
 * One immutable-admission record which is the single host-side identity for a
 * slot: typed slot + 64-bit generation (never reused across runtime
 * replacement), admission kind (new request vs. continued response),
 * immutable policy identity, response limit, intrinsic RNG coordinates and
 * the sealed window plan. This record is the substrate the executor publishes
 * into device-owned per-row tables; no second identity ledger may exist.
 */

#pragma once

#include "execution/config/ServingCapacity.h"
#include "execution/serving/RequestIntrinsicCoordinates.h"

#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Per-row admission kind (design §6). */
    enum class AdmissionKind : std::uint8_t
    {
        NewRequest = 0,     ///< First admission of this request.
        ContinueResponse,   ///< Same admitted request continuing after publication swap.
    };

    /**
     * @brief Immutable host-side per-row identity/ownership record.
     *
     * slot + generation name the physical slot; generation must not be reused
     * across runtime replacement (a replacement is a new generation, never the
     * same). Only the intrinsic window may reference committed count — load
     * or batch occupancy never enters the record.
     */
    struct ServingRowState
    {
        std::uint32_t slot = 0;
        std::uint64_t generation = 0;
        AdmissionKind kind = AdmissionKind::NewRequest;
        std::uint64_t policy_identity = 0; ///< Immutable admitted policy digest.
        std::uint64_t response_limit = 0;  ///< Immutable response cap.
        std::uint64_t committed = 0;       ///< Committed generated-token count c.
        DrawCoordinates rng{};             ///< (seed, logical position, purpose) — RS coords only.
        IntrinsicWindowPlan window{};      ///< Sealed quantum + response limit.

        /** @brief Distinct identity even when a generation changed on the same slot. */
        [[nodiscard]] bool samePhysicalSlotIdentity(const ServingRowState &other) const noexcept
        {
            return slot == other.slot && generation == other.generation;
        }
    };

    /**
     * @brief A continuation may only reference an already-admitted new request.
     *
     * @throws std::logic_error When ContinueResponse claims a fresh row (no
     *         predecessor) or when the predecessor is the same generation used
     *         for a different policy identity.
     */
    inline void admitContinuation(ServingRowState &row, const ServingRowState &predecessor)
    {
        if (row.kind != AdmissionKind::ContinueResponse)
            throw std::logic_error("Continuation admission must be marked ContinueResponse");
        // §14: A retains ContinueResponse across publication swap; C is a NewRequest.
        // A continuation is the SAME requested lifetime: exact slot/generation and
        // the full immutable policy/RNG identity are preserved, and successive
        // continuations are allowed. It requires a NON-TERMINAL predecessor (a
        // finished response is a new request, not a continuation), and it never
        // changes the immutable response limit or seed.
        if (predecessor.committed >= predecessor.response_limit)
            throw std::logic_error(
                "Continuation requires a NON-TERMINAL predecessor "
                "(a finished response is a new request, not a continuation)");
        if (row.slot != predecessor.slot || row.generation != predecessor.generation)
            throw std::logic_error("Continuation must keep the exact slot/generation identity");
        if (row.policy_identity != predecessor.policy_identity)
            throw std::logic_error("Continuation must keep the immutable policy identity");
        if (row.rng.seed != predecessor.rng.seed ||
            row.rng.purpose != predecessor.rng.purpose)
            throw std::logic_error("Continuation must keep the immutable request RNG identity");
        if (row.rng.logical_position < predecessor.rng.logical_position ||
            row.rng.logical_position > predecessor.committed + 1)
            throw std::logic_error(
                "Continuation logical RNG position must be monotonic within the "
                "committed frontier (no rewind, no arbitrary forward jumps)");
        if (row.response_limit != predecessor.response_limit)
            throw std::logic_error("Continuation must keep the immutable response limit");
        if (row.window.quantum != predecessor.window.quantum)
            throw std::logic_error("Continuation must keep the sealed window quantum");
        if (row.window.response_limit != predecessor.window.response_limit)
            throw std::logic_error("Continuation must keep the sealed window capacity");
        // The committed count carries the predecessor's current count.
        row.committed = predecessor.committed;
    }
} // namespace llaminar2
