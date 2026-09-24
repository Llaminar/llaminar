/**
 * @file MTPSidecarCaptureLayout.h
 * @brief Defines total cache and device-input-slot ownership for captured MTP sidecars.
 *
 * Graph-captured MTP sidecars retain the address of their condition-token input.
 * Full sidecars, chained sidecars, and KV-only catch-up sidecars therefore need
 * distinct persistent slices of the arena-owned condition-token buffer. KV-only
 * catch-up graphs also need one graph cache for every supported flattened row
 * count because row count is part of the captured graph shape.
 *
 * This layout is the sole authority for both identities. Keeping cache indices,
 * condition-token slots, and total allocation size in one value object prevents
 * the producer graph and its input publication slot from drifting apart.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Semantic execution role of a depth-zero MTP sidecar graph.
     */
    enum class MTPSidecarCaptureRole : uint8_t
    {
        Full,    ///< Runs the complete MTP sidecar and publishes logits.
        Chained, ///< Runs a complete sidecar from a prior sidecar hidden state.
        KVOnly,  ///< Publishes shifted MTP KV rows without terminal logits.
    };

    /**
     * @brief Immutable address layout for MTP sidecar graph caches and token slots.
     *
     * Slots zero through two belong to the full, chained, and one-row KV-only
     * device-token graphs. Multi-row KV-only graphs use contiguous slots starting
     * at three, with row count two at slot three. The matching cache directory is
     * zero-based at row count two. Thus the following identities always hold:
     *
     * @code
     * cache_index(rows) = rows - 2
     * token_slot(rows)  = 3 + cache_index(rows)
     * @endcode
     *
     * The maximum is supplied by runtime planning. It is a configured capacity,
     * not an architectural MTP-depth limit.
     */
    class MTPSidecarCaptureLayout final
    {
    public:
        /**
         * @brief Construct a layout for all flattened row counts through a maximum.
         *
         * @param maximum_rows Largest flattened sidecar row count that can execute.
         * @throws std::invalid_argument When @p maximum_rows is not positive.
         */
        explicit MTPSidecarCaptureLayout(int maximum_rows)
            : maximum_rows_(maximum_rows)
        {
            if (maximum_rows_ <= 0)
            {
                throw std::invalid_argument(
                    "MTP sidecar capture layout requires a positive row capacity");
            }
        }

        /// Return the largest flattened row count represented by this layout.
        [[nodiscard]] int maximumRows() const noexcept { return maximum_rows_; }

        /**
         * @brief Return the number of multi-row KV-only graph caches to preallocate.
         *
         * Row count one has a dedicated cache. The directory represented by this
         * count owns rows `[2, maximumRows()]`.
         */
        [[nodiscard]] size_t kvOnlyBatchCacheCount() const noexcept
        {
            return static_cast<size_t>(maximum_rows_ - 1);
        }

        /**
         * @brief Resolve the zero-based KV-only batch cache index for a row count.
         *
         * @param total_rows Flattened sidecar row count; must be at least two.
         * @throws std::out_of_range When the row count is outside this layout.
         */
        [[nodiscard]] size_t kvOnlyBatchCacheIndex(int total_rows) const
        {
            validateRows(total_rows);
            if (total_rows < 2)
            {
                throw std::out_of_range(
                    "One-row MTP KV sidecars use their dedicated graph cache");
            }
            return static_cast<size_t>(total_rows - 2);
        }

        /**
         * @brief Resolve the persistent device condition-token slot for a graph role.
         *
         * @param role Semantic sidecar role.
         * @param total_rows Flattened graph row count. For full and chained roles
         *        this validates capacity; KV-only uses it to select a shape slot.
         * @throws std::out_of_range When the row count is outside this layout.
         */
        [[nodiscard]] int conditionTokenSlot(
            MTPSidecarCaptureRole role,
            int total_rows) const
        {
            validateRows(total_rows);
            switch (role)
            {
            case MTPSidecarCaptureRole::Full:
                return 0;
            case MTPSidecarCaptureRole::Chained:
                return 1;
            case MTPSidecarCaptureRole::KVOnly:
                return total_rows == 1
                           ? 2
                           : 3 + static_cast<int>(
                                     kvOnlyBatchCacheIndex(total_rows));
            }

            throw std::logic_error("Unknown MTP sidecar capture role");
        }

        /**
         * @brief Return the exact number of persistent condition-token slots.
         *
         * Three dedicated role slots are followed by one slot for every multi-row
         * KV-only graph shape. Callers multiply this count by `maximumRows()` to
         * allocate the arena buffer once, before graph capture.
         */
        [[nodiscard]] int conditionTokenSlotCount() const noexcept
        {
            return 3 + static_cast<int>(kvOnlyBatchCacheCount());
        }

    private:
        void validateRows(int total_rows) const
        {
            if (total_rows <= 0 || total_rows > maximum_rows_)
            {
                throw std::out_of_range(
                    "MTP sidecar row count is outside the configured capture layout");
            }
        }

        int maximum_rows_;
    };
} // namespace llaminar2
