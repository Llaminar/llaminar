/**
 * @file CPUWeightPreparationMemory.h
 * @brief CPU source-native preparation demand contributed to physical admission.
 *
 * This is an allocation-shape description, not a capacity decision or a live
 * ledger. NativeVNNI's consumed cache footprint is smaller than its allocation
 * footprint for layouts retaining a scalar payload. Preparation also overlaps
 * final storage with format-specific temporary matrices. Callers contribute
 * both named demands to PhysicalMemoryAuthority before invoking the packer.
 */
#pragma once

#include <cstddef>
#include <string_view>

namespace llaminar2
{
    /** @brief Distinct lifetimes of final prepared weights and setup scratch. */
    struct CPUWeightPreparationMemory final
    {
        size_t persistent_bytes; ///< Retained until the prepared engine retires.
        size_t temporary_bytes;  ///< Retired when native preparation completes.

        /**
         * @brief Describe the canonical unrotated CPU expert preparation path.
         * @param format Exact native source name, including F32/F16/BF16.
         * @param n Complete matrix output rows before 64-column packing padding.
         * @param k Source-codebook-aligned inner dimension.
         * @return Exact payload allocations, excluding caller-owned source bytes.
         * @throws std::invalid_argument for unsupported formats or geometry.
         * @throws std::overflow_error for unrepresentable allocation geometry.
         *
         * The source-native observation path does not enable activation rotation.
         * Rotated preparation has a different scratch contract and must not be
         * admitted with this value. No ISA, thread count or device capacity is
         * guessed here; these packing buffers are independent of worker count.
         */
        static CPUWeightPreparationMemory sourceNative(std::string_view format, size_t n, size_t k);
    };
}
