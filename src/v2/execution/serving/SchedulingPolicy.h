/**
 * @file SchedulingPolicy.h
 * @brief Host-side one-window scheduling selection policy (design §5.2, §11).
 *
 * The first policy gives granted decode windows regular service and schedules
 * one canonical prefill chunk with anti-starvation. Selection is a PURE
 * function of admitted per-row facts only: saturating age (completed serving
 * operations since admission/last service) with a stable admission-order
 * tie-break. Host wall-clock, timestamps, load and batch occupancy never
 * enter — the selection signature makes that unrepresentable. A row without a
 * usable grant is never selected; rows without grants never block healthy
 * rows. Age saturation never wraps a waiting request back to zero.
 */

#pragma once

#include "execution/serving/WindowGrant.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace llaminar2
{
    /** @brief One eligible row's selection facts (age + stable ordinal). */
    struct SelectionRow
    {
        std::uint32_t slot = 0;
        std::uint64_t generation = 0;
        std::uint64_t age = 0;             ///< Saturating completed-serving-operation count.
        std::uint64_t admission_ordinal = 0; ///< Stable admission-order tie-break.
        bool has_usable_grant = false;     ///< Only granted rows are selectable.
    };

    /**
     * @brief Select the next row: highest age, lowest admission ordinal on ties.
     *
     * @param rows Non-granted rows are never selected; an empty eligible set
     *             returns std::nullopt.
     * @return Index of the selected row, or std::nullopt.
     */
    [[nodiscard]] inline std::optional<std::size_t> selectNextGrantedRow(
        const std::vector<SelectionRow> &rows)
    {
        std::optional<std::size_t> best;
        for (std::size_t i = 0; i < rows.size(); ++i)
        {
            const auto &row = rows[i];
            if (!row.has_usable_grant)
                continue;
            if (!best.has_value())
            {
                best = i;
                continue;
            }
            const auto &cur = rows[*best];
            if (row.age > cur.age ||
                (row.age == cur.age && row.admission_ordinal < cur.admission_ordinal))
                best = i;
        }
        return best;
    }
} // namespace llaminar2
