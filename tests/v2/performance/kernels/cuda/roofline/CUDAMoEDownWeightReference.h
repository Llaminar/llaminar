/**
 * @file CUDAMoEDownWeightReference.h
 * @brief Diagnostic CUDA reader for the weight planes of a captured MoE Down leaf.
 *
 * The fixture borrows the native kernel's original argument storage and owns
 * the capture lifetime and checksum observation.
 */
#pragma once

#include <cstdint>

namespace llaminar2::test
{
    /** @brief Independent access patterns used to check the reference checksum. */
    enum class CUDAWeightPacketAccess : std::uint8_t
    {
        ByteOracle,
        Aligned,
    };

    /**
     * @brief Resolve a reader for the original captured Down argument list.
     * @param codebook Exact physical decoder observed in the prepared fixture.
     * @param access Scalar byte oracle or widest natural packet loads.
     * @return CUDA kernel entry, or null for an unsupported physical decoder.
     */
    [[nodiscard]] const void *cudaMoEDownWeightReferenceEntry(
        std::uint8_t codebook, CUDAWeightPacketAccess access) noexcept;
}
