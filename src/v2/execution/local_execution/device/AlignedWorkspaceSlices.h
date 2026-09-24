/**
 * @file AlignedWorkspaceSlices.h
 * @brief Checked non-owning partitions of an already admitted scratch buffer.
 *
 * Graph families merge maximum byte capacities from stages with different
 * concurrency. Dividing those bytes directly by a stage's stream count can
 * create misaligned addresses even when the allocation itself is aligned.
 * This value type divides alignment units instead, retaining disjoint slices
 * and leaving any remainder unused. It allocates nothing and is not a memory
 * ledger: allocation/admission remains with PhysicalMemoryAuthority and the
 * workspace owner. Persistent metadata leases are a different lifetime; these
 * slices describe scratch whose consumers join before another stage uses it.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Immutable aligned geometry for one concurrent scratch fan-out. */
    class AlignedWorkspaceSlices final
    {
    public:
        /**
         * @brief Partition existing capacity without rounding beyond its end.
         * @param total_bytes Exact available buffer capacity.
         * @param count Positive number of concurrently active slices.
         * @param alignment Required power-of-two alignment of each slice.
         * @throws std::invalid_argument For invalid geometry or empty slices.
         */
        AlignedWorkspaceSlices(
            std::size_t total_bytes, std::size_t count, std::size_t alignment)
            : count_(count), alignment_(alignment)
        {
            if (!count || !alignment || (alignment & (alignment - 1)))
                throw std::invalid_argument("Aligned workspace slices require positive count and power-of-two alignment");
            // Divide first; neither count*alignment nor align-up arithmetic
            // may overflow for a malformed or extreme capacity declaration.
            stride_ = (total_bytes / alignment / count) * alignment;
            if (!stride_)
                throw std::invalid_argument("Aligned workspace slice capacity is empty");
        }

        /** @return Exact usable byte capacity of every aligned slice. */
        std::size_t strideBytes() const noexcept { return stride_; }

        /**
         * @brief Resolve one aligned slice and reject an oversized payload.
         * @param base Exact beginning of the owner's bound buffer.
         * @param index Zero-based stream/slice ordinal within this geometry.
         * @param payload_bytes Required live payload, excluding unused slack.
         * @return Non-owning view of the complete isolated slice.
         * @throws std::invalid_argument For null/misaligned storage, bad index,
         *         or a payload that cannot fit in the slice.
         */
        std::span<std::byte> slice(
            void *base, std::size_t index, std::size_t payload_bytes) const
        {
            if (!base || reinterpret_cast<std::uintptr_t>(base) % alignment_ ||
                index >= count_ || !payload_bytes || payload_bytes > stride_)
                throw std::invalid_argument("Aligned workspace slice has invalid storage, index, or payload capacity");
            // index*stride is bounded by the admitted total from construction;
            // payload size never changes the stride of a concurrently live lane.
            return {static_cast<std::byte *>(base) + index * stride_, stride_};
        }

    private:
        std::size_t count_; ///< Immutable number of simultaneous scratch users.
        std::size_t alignment_; ///< Alignment retained by base and every offset.
        std::size_t stride_ = 0; ///< Validated byte capacity of each isolated lane.
    };
}
