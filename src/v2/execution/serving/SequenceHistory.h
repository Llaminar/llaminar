/**
 * @file SequenceHistory.h
 * @brief Per-request sequence-history policy + bounded history window (design §6, §7.1).
 *
 * DRY/sequence-history semantics are per-request and never shared: an
 * unrelated request must not observe another request's emitted history. The
 * window is bounded by the request's intrinsic response limit; inserting a
 * token advances the intrinsic logical position. Emitting twice at the same
 * position is refused (exactly-once history insertion), tied to the §10.3
 * outcome rules.
 */

#pragma once

#include "execution/serving/RequestIntrinsicCoordinates.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace llaminar2
{
    /** @brief Request-local bounded sequence history. */
    struct SequenceHistory
    {
        std::vector<std::uint32_t> tokens; ///< Emitted request token ids (immutable per output).
        IntrinsicWindowPlan window{};      ///< Sealed window/response limit.

        /** @brief Append one emitted token (exactly once per logical position). */
        void emitOne(std::uint32_t token, std::uint64_t logical_position)
        {
            if (tokens.size() >= window.response_limit)
                throw std::logic_error("Response limit reached; cannot emit");
            // Logical positions are strictly increasing (no rewrite/replay).
            if (logical_position != tokens.size())
                throw std::logic_error("Logical position must equal emitted history length");
            tokens.push_back(token);
        }

        [[nodiscard]] std::uint64_t committed() const noexcept
        {
            return static_cast<std::uint64_t>(tokens.size());
        }
    };
} // namespace llaminar2
