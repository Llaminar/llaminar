/**
 * @file CPUWeightPreparationMemory.h
 * @brief CPU source-native preparation demand contributed to physical admission.
 *
 * This is an allocation-shape description, not a capacity decision or a live
 * ledger. The native packer retains only its final interleaved root; source
 * payloads used to build it retire inside the synchronous preparation call.
 * Allocator-rounded retained bytes and logical engine bytes are distinct.
 * Callers contribute retained and setup-peak demands before preparation.
 */
#pragma once

#include <cstddef>
#include <string_view>

namespace llaminar2
{
    /** @brief Distinct lifetimes of final prepared weights and setup scratch. */
    struct CPUWeightPreparationMemory final
    {
        size_t persistent_bytes; ///< Exact local physical root, retained by actual backing.
        size_t temporary_bytes;  ///< Exact simultaneous setup vectors, retired before synchronous return.
        size_t persistent_payload_bytes; ///< Logical engine payload, excluding allocator rounding.

        /**
         * @brief Describe the canonical unrotated CPU expert preparation path.
         * @param format Exact native source name, including F32/F16/BF16.
         * @param n Complete matrix output rows before 64-column packing padding.
         * @param k Source-codebook-aligned inner dimension.
         * @return Exact local retained allocation, setup peak and logical payload;
         *         caller-owned source bytes are excluded.
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
