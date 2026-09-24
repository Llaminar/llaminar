/**
 * @file MoEOverlayHostDispatchRows.h
 * @brief Validated linear traversal of a host-owned sparse dispatch descriptor.
 *
 * The router publishes token-major entries and increasing selected token rows.
 * Host transports must consume that order directly rather than search every
 * route again for every token. This borrowed view validates coverage before any
 * packet is published and introduces no index allocation or shadow route state.
 */
#pragma once

#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"

#include <span>
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Immutable row spans over one graph-owned dispatch descriptor.
     *
     * The caller must retain the descriptor unchanged for this view's short
     * synchronous lifetime. No span escapes through production transport.
     */
    class MoEOverlayHostDispatchRows final
    {
    public:
        /**
         * @brief Validate token-major ordering and complete entry coverage.
         * @param tier Descriptor published by MoEExpertDispatchStage.
         * @param logical_rows Live sequence length, excluding bucket padding.
         * @throws std::invalid_argument For unordered, missing, or invalid rows.
         */
        MoEOverlayHostDispatchRows(const MoEExpertTierDispatch &tier, int logical_rows)
            : rows_(tier.token_rows), entries_(tier.entries)
        {
            if (logical_rows <= 0)
                throw std::invalid_argument("Host sparse dispatch needs a positive sequence length");
            int previous = -1;
            size_t cursor = 0;
            for (int row : rows_)
            {
                if (row <= previous || row >= logical_rows)
                    throw std::invalid_argument("Host sparse dispatch rows must be increasing live token indices");
                while (cursor < entries_.size() && entries_[cursor].token_row == row)
                    ++cursor;
                if (cursor < entries_.size() && entries_[cursor].token_row < row)
                    throw std::invalid_argument("Host sparse dispatch entries are not token-major or omit a selected row");
                previous = row;
            }
            if (cursor != entries_.size())
                throw std::invalid_argument("Host sparse dispatch rows do not cover every route entry");
        }

        /**
         * @brief Visit each selected row and its contiguous routes exactly once.
         * @param visit Visitor receiving the original row and borrowed entry span.
         * @return False if a visitor rejects its row; otherwise true.
         *
         * Entry order is unchanged, including route-slot order and participant
         * filtering by the consumer. Empty selected rows have an empty span.
         */
        template <typename Visitor>
        bool forEachRow(const Visitor &visit) const
        {
            size_t cursor = 0;
            for (int row : rows_)
            {
                const size_t begin = cursor;
                while (cursor < entries_.size() && entries_[cursor].token_row == row)
                    ++cursor;
                if (!visit(row, entries_.subspan(begin, cursor - begin)))
                    return false;
            }
            return true;
        }

    private:
        std::span<const int> rows_;
        std::span<const MoEExpertDispatchEntry> entries_;
    };
}
