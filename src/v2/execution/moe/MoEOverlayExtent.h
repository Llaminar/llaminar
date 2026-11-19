/**
 * @file MoEOverlayExtent.h
 * @brief Checked immutable array extents shared by mapped CPU/CUDA/HIP records.
 *
 * Every offset is relative to the admitted storage root passed by its owner.
 * Subtraction-based bounds checks reject multiplication and addition overflow
 * before an address is formed. This helper owns neither storage nor live state.
 */
#pragma once
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_EXTENT_HD __host__ __device__
#else
#define LLAMINAR_MOE_EXTENT_HD
#endif

namespace llaminar2
{
    /**
     * @brief Validate a complete typed array inside one admitted byte extent.
     * @param bytes Exact immutable storage extent.
     * @param offset Array offset relative to that storage root.
     * @param count Exact element count, never allocation capacity used as work.
     * @return True only for an aligned, nonoverflowing contained array.
     */
    template <typename T>
    LLAMINAR_MOE_EXTENT_HD constexpr bool moeOverlayExtentContains(
        std::uint64_t bytes, std::uint64_t offset, std::uint64_t count) noexcept
    {
        return offset % alignof(T) == 0u && offset <= bytes &&
               count <= (bytes - offset) / sizeof(T);
    }

    /**
     * @brief Resolve an array only after authenticating its complete range.
     * @param base Admitted host or exact device alias of the storage root.
     * @param bytes Immutable extent supplied by that root's owner.
     * @param offset Root-relative byte offset.
     * @param count Exact array cardinality.
     * @return The typed alias, or null for any malformed range/root.
     */
    template <typename T>
    LLAMINAR_MOE_EXTENT_HD T *moeOverlayExtentArray(
        void *base, std::uint64_t bytes, std::uint64_t offset,
        std::uint64_t count) noexcept
    {
        if (!base || !moeOverlayExtentContains<T>(bytes, offset, count))
            return nullptr;
        return reinterpret_cast<T *>(static_cast<unsigned char *>(base) + offset);
    }

    /** @brief Resolve an immutable array with the same checked range contract.
     * @param base Immutable admitted root. @param bytes Admitted extent.
     * @param offset Root-relative byte offset. @param count Exact cardinality.
     * @return Immutable typed alias, or null for a malformed range. */
    template <typename T>
    LLAMINAR_MOE_EXTENT_HD const T *moeOverlayExtentArray(
        const void *base, std::uint64_t bytes, std::uint64_t offset,
        std::uint64_t count) noexcept
    {
        if (!base || !moeOverlayExtentContains<T>(bytes, offset, count))
            return nullptr;
        return reinterpret_cast<const T *>(static_cast<const unsigned char *>(base) + offset);
    }
}
#undef LLAMINAR_MOE_EXTENT_HD
