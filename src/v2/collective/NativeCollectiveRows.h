/**
 * @file NativeCollectiveRows.h
 * @brief Immutable row geometry for a captured native GPU collective.
 *
 * Storage retains its admitted rank-bank stride while the device authority
 * publishes the live prefix for each replay. Neither graph construction nor
 * transport submission reads that count. This is a borrowed operand, not a
 * second execution-state or allocation owner. The producer must precede the
 * complete native batch and its count must remain immutable through that batch.
 */
#pragma once

#include "kernels/common/DeviceRowRange.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Native operation over equal-prefix, communicator-ordered row banks. */
    enum class NativeRowCollective : std::uint8_t
    {
        AllGather,
        AllReduce,
        ReduceScatter
    };

    /**
     * @brief Checked element geometry without a mutable host count shadow.
     *
     * All-gather and reduce-scatter retain capacity-sized rank strides, but
     * communicate only the live rows in each stride. All-reduce uses one bank.
     * Unequal ownership and ragged requests need their own compact protocol;
     * neither can be represented by passing their first request's row count.
     */
    class NativeCollectiveRows final
    {
    public:
        /**
         * @brief Freeze complete rows and their exact element width.
         * @param rows Fixed rows or the canonical device-owned query prefix.
         * @param elements_per_row Exact scalar width, not backing allocation capacity.
         * @throws std::invalid_argument For a sliced range or an empty/oversized row.
         * @throws std::overflow_error For an unrepresentable bank extent.
         *
         * The native ABI accepts a whole prefix. A sliced DeviceRowRange has
         * different origin semantics and must never silently lose its offset.
         */
        NativeCollectiveRows(DeviceRowRange rows, std::size_t elements_per_row)
            : rows_(rows), elements_per_row_(elements_per_row)
        {
            if (rows.physicalRows() != rows.capacity() || elements_per_row == 0 ||
                elements_per_row > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("Native collective requires a whole row prefix and a nonempty UINT32 row width");
            if (elements_per_row > std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(rows.capacity()))
                throw std::overflow_error("Native collective row bank overflows size_t");
        }

        /** @return Immutable borrowed geometry; its count is never read on the host. */
        [[nodiscard]] const DeviceRowRange &rows() const noexcept { return rows_; }
        /** @return Scalar elements in one live row. */
        [[nodiscard]] std::size_t elementsPerRow() const noexcept { return elements_per_row_; }
        /** @return Admitted bank stride, not the number of elements communicated. */
        [[nodiscard]] std::size_t bankElements() const noexcept
        { return static_cast<std::size_t>(rows_.capacity()) * elements_per_row_; }

        /**
         * @brief Bind an optional persistent device-only useful-byte receipt.
         * @param counter Aligned device UINT64 retained through graph retirement.
         * @return A new immutable declaration; arithmetic and row authority are unchanged.
         * @throws std::invalid_argument For a null/misaligned counter or fixed rows.
         *
         * This opt-in diagnostic observes native payload, never schedules work
         * or creates a host count authority. Production declarations omit it.
         */
        [[nodiscard]] NativeCollectiveRows withPayloadReceipt(unsigned long long *counter) const
        {
            if (!counter || !rows_.countOwner() ||
                reinterpret_cast<std::uintptr_t>(counter) % alignof(unsigned long long))
                throw std::invalid_argument("Native payload receipt requires counted rows and an aligned device counter");
            auto observed = *this;
            observed.payload_receipt_ = counter;
            return observed;
        }

        /** @return Optional device-only observation counter, never a row count authority. */
        [[nodiscard]] unsigned long long *payloadReceipt() const noexcept { return payload_receipt_; }

        /**
         * @brief Check the native byte ABI before recording any operation.
         * @param element_bytes Exact scalar storage width (1, 2, 4 or 8 bytes).
         * @param banks Number of physical rank banks in the largest operand.
         * @return Whether row bytes and every retained bank fit their ABI limits.
         *
         * Byte bounds are geometry validation only. PMA and the graph arena
         * still own materialization and capacity admission.
         */
        [[nodiscard]] bool byteGeometryValid(std::size_t element_bytes, int banks) const noexcept
        {
            if ((element_bytes != 1 && element_bytes != 2 && element_bytes != 4 && element_bytes != 8) || banks <= 0)
                return false;
            const auto maximum = std::numeric_limits<std::size_t>::max();
            return elements_per_row_ <= std::numeric_limits<std::uint32_t>::max() / element_bytes &&
                bankElements() <= maximum / element_bytes / static_cast<std::size_t>(banks);
        }

    private:
        DeviceRowRange rows_; ///< Borrowed device authority, or exact fixed rows.
        std::size_t elements_per_row_; ///< Immutable native scalar geometry.
        unsigned long long *payload_receipt_ = nullptr; ///< Optional borrowed diagnostic storage, not execution state.
    };
}
