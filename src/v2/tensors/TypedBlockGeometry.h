/**
 * @file TypedBlockGeometry.h
 * @brief Shared checked element geometry for complete typed residual blocks.
 *
 * Q8/Q16 residual APIs publish complete 32-element metadata/value blocks.
 * Reject malformed geometry before ISA dispatch or mutation; truncating count/32
 * would silently leave part of a live output unauthored. This is shape validation,
 * not allocation/admission accounting, and zero is a valid empty operation.
 */
#pragma once
#include <cstddef>
#include <stdexcept>

namespace llaminar2::simd
{
    /** @brief Require the exact full-block contract used by typed residual APIs.
     * @param count Logical element count, including zero for an empty operation.
     * @throws std::invalid_argument Unless the count is a multiple of 32.
     */
    inline void requireCompleteTypedBlocks(std::size_t count)
    {
        if (count % 32 != 0)
            throw std::invalid_argument("Typed block operation requires a multiple of 32 elements");
    }
} // namespace llaminar2::simd
