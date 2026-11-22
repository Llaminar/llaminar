/**
 * @file OutputGrantBudget.h
 * @brief Per-request output grant budget with admission-time sealing (design §3.1, §5.2).
 *
 * A request's response limit and window plan are sealed at admission; each
 * output window consumes exactly one grant, and a grant never exceeds the
 * remaining budget. This decouples admission from publication: a response
 * stream can allocate storage for a window only when its grant is usable,
 * never from load-derived arithmetic.
 */

#pragma once

#include "execution/serving/WindowGrant.h"

#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Sealed per-request output budget. */
    struct OutputGrantBudget
    {
        std::uint64_t response_limit = 0;

        [[nodiscard]] std::uint64_t remaining(std::uint64_t committed) const noexcept
        {
            return committed < response_limit ? response_limit - committed : 0;
        }

        /** @brief Validate a window grant against the remaining budget.
         *
         * The FINAL window may be PARTIAL: whatever remains is granted and the
         * engine bounds production at the response limit. Only a zero-remaining
         * budget (or a non-positive quantum) refuses.
         */
        void requireFits(const WindowGrant &grant, std::uint64_t committed) const
        {
            if (grant.grant_id == 0)
                throw std::logic_error("Budget validation requires an issued grant");
            if (grant.window.quantum == 0)
                throw std::logic_error("Grant window quantum must be positive");
            if (grant.serving_instance == 0 || grant.destination_reservation == 0)
                throw std::logic_error("Budget validation requires instance/backed grant");
            if (grant.window.response_limit != response_limit)
                throw std::logic_error("Grant window must respect the sealed response limit");
            if (remaining(committed) == 0)
                throw std::logic_error("Grant window exceeds remaining output budget (none left)");
        }
    };
} // namespace llaminar2
