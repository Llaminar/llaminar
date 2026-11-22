/**
 * @file MTPRowPolicy.h
 * @brief Typed per-request MTP row policy and verifier publication contract (design §6, §8).
 *
 * Depth policy, learner state identity and verifier/publication/rollback
 * ownership are request-intrinsic and immutable after admission. A verifier
 * publication is exactly-once per logical position: the same position must
 * never be published twice, and a rollback only ever restores an earlier
 * committed position. These typed states make the ownership rules
 * unrepresentable rather than caller-discipline comments.
 */

#pragma once

#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Immutable per-request MTP depth policy. */
    enum class MTPDepthPolicy : std::uint8_t
    {
        Dynamic = 0, ///< Depth chosen by the request's own policy/observations.
        Fixed,       ///< Sealed fixed depth.
    };

    /** @brief Per-request MTP row policy (immutable after admission). */
    struct MTPRowPolicy
    {
        std::uint8_t max_depth = 0;                  ///< Admitted maximum depth (>= 1).
        MTPDepthPolicy depth_policy = MTPDepthPolicy::Dynamic;
        std::uint64_t learner_state_identity = 0;    ///< Immutable learner state digest.
        std::uint64_t sidecar_identity = 0;          ///< Immutable sidecar (MTP head) digest.
    };

    /** @brief Verifier/rollback ownership per logical position (exactly-once). */
    struct MTPVerifierPosition
    {
        std::uint64_t position = 0;
        bool accepted = false;  ///< Commit published exactly once.
        bool rolled_back = false;
        std::uint64_t rollback_target = 0; ///< Recorded rollback destination (authenticated).
    };

    /**
     * @brief Exactly-once verifier publication per logical position.
     *
     * @param pos The position being accepted.
     * @throws std::logic_error When the position was already accepted or rolled
     *         back (no double publication, no publish-after-rollback).
     */
    inline void acceptPosition(MTPVerifierPosition &pos)
    {
        if (pos.accepted)
            throw std::logic_error("Verifier position already accepted (exactly once)");
        if (pos.rolled_back)
            throw std::logic_error("Verifier position rolled back; cannot publish");
        pos.accepted = true;
    }

    /**
     * @brief Rollback restores an earlier committed position only.
     *
     * @throws std::logic_error When the position is currently accepted (a
     *         rollback reviews an unaccepted frontier), or the target exceeds
     *         the current position.
     */
    inline void rollbackTo(MTPVerifierPosition &current, std::uint64_t target)
    {
        if (current.accepted)
            throw std::logic_error("Cannot roll back an already-accepted position");
        if (current.rolled_back)
            throw std::logic_error("Rollback is once per position");
        if (target >= current.position)
            throw std::logic_error("Rollback target must be strictly earlier");
        current.rolled_back = true;
        current.rollback_target = target; // recorded + authenticated
        current.position = target;        // rollback MOVES the frontier (never ignored)
    }
} // namespace llaminar2
