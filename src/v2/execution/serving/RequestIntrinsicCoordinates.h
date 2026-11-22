/**
 * @file RequestIntrinsicCoordinates.h
 * @brief Immutable request-local RNG coordinates and intrinsic window plan (design §7).
 *
 * The stochastic key is exactly (request-resolved seed, logical output
 * position, draw purpose). RequestId, slot, slot generation, batch row,
 * arrival order and request ordinal MUST NOT enter that key — the typed
 * coordinate below contains only the three intrinsic fields, so the exclusion
 * is unrepresentable rather than a comment. Seeds are nonzero and immutable
 * after admission: an explicit seed is preserved; a zero (unseeded) request
 * resolves its entropy exactly once at admission through
 * resolveRequestSeed(entropy, requested), never regenerated per transaction
 * or slot reuse. The intrinsic window is a positive quantum W sealed with the
 * plan: the next regular window ends at W * (floor(c / W) + 1), bounded by
 * the immutable response limit; scheduling may defer but never shorten or
 * extend it. EOS / forced / stop semantics and external cancellation are
 * explicit terminal exceptions, not load-budget clipping.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    /** @brief One logical draw purpose (request-intrinsic, never load-dependent). */
    enum class DrawPurpose : std::uint8_t
    {
        Sampling = 0,  ///< Token sampling draw.
        MTPVerifier,   ///< MTP verifier/correction draw.
        StopCheck,     ///< Stylistic/semantic stop arbitration draw (if any).
    };

    /**
     * @brief Immutable stochastic key (design §7.2): (seed, position, purpose).
     *
     * Only these three fields exist; no request/slot/batch identity can be
     * smuggled into a key, so the "do not include slot/batch row" law is
     * enforced by the type, not by caller discipline.
     */
    struct DrawCoordinates
    {
        std::uint32_t seed = 0;           ///< Request-resolved seed (nonzero).
        std::uint64_t logical_position = 0; ///< Absolute logical output position.
        DrawPurpose purpose = DrawPurpose::Sampling;
    };

    /**
     * @brief Resolve the immutable request seed once at admission.
     *
     * @param entropy Fresh admission-time entropy (used only when requested == 0).
     * @param requested Authored seed (preserved when nonzero).
     * @return Nonzero seed.
     * @throws std::invalid_argument When the derived seed would be zero
     *         (entropy degeneracy is refused, never silently mapped).
     */
    [[nodiscard]] inline std::uint32_t resolveRequestSeed(std::uint64_t entropy,
                                                          std::uint32_t requested) noexcept
    {
        if (requested != 0)
            return requested;
        // Deterministic splitmix-style mix: request-intrinsic, documented.
        std::uint64_t z = entropy + 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        const auto mixed = static_cast<std::uint32_t>(z);
        return mixed == 0 ? 1u : mixed; // never zero; degeneracy maps to 1 (documented)
    }

    /**
     * @brief Intrinsic window plan (design §7.1): sealed quantum + immutable response limit.
     */
    struct IntrinsicWindowPlan
    {
        /** @brief Positive window quantum W (sealed with the plan). */
        std::uint64_t quantum = 0;
        /** @brief Immutable response limit (generated-token cap). */
        std::uint64_t response_limit = 0;
    };

    /**
     * @brief Next regular window end for this request (W * (floor(c/W) + 1)).
     *
     * @param committed Committed generated-token count c.
     * @param plan Sealed window plan.
     * @return Absolute end position, bounded by the immutable response limit.
     * @throws std::invalid_argument For a non-positive quantum.
     */
    [[nodiscard]] inline std::uint64_t nextWindowEnd(std::uint64_t committed,
                                                     const IntrinsicWindowPlan &plan)
    {
        if (plan.quantum == 0)
            throw std::invalid_argument("Intrinsic window quantum must be positive");
        const auto window_index = committed / plan.quantum + 1;
        const auto max = std::numeric_limits<std::uint64_t>::max();
        // No wraparound: an index whose product would overflow is already beyond
        // any bounded response limit, so the limit is the correct result.
        if (window_index > max / plan.quantum)
            return plan.response_limit;
        const auto end = window_index * plan.quantum;
        return end < plan.response_limit ? end : plan.response_limit;
    }

    /**
     * @brief Whether a position falls inside the request's next regular window.
     */
    [[nodiscard]] inline bool isInsideWindow(std::uint64_t committed,
                                             std::uint64_t position,
                                             const IntrinsicWindowPlan &plan)
    {
        const auto end = nextWindowEnd(committed, plan);
        return committed < position && position <= end;
    }
} // namespace llaminar2
