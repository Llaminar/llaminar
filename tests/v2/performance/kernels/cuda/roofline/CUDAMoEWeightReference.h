/**
 * @file CUDAMoEWeightReference.h
 * @brief Diagnostic CUDA readers for captured routed MoE projection weight planes.
 *
 * The fixture borrows the native kernel's original argument storage and owns
 * the capture lifetime and checksum observation.
 */
#pragma once

#include <cstdint>

namespace llaminar2::test
{
    /** @brief Original kernel argument layout and work order of the reference. */
    enum class CUDAMoEWeightPhase : std::uint8_t
    {
        VerifierDown,
        VerifierGateUp,
        DecodeGateUp,
        DecodeDown,
    };

    /** @brief Independent access patterns used to check the reference checksum. */
    enum class CUDAWeightPacketAccess : std::uint8_t
    {
        ByteOracle,
        Aligned,
    };

    /**
     * @brief Resolve a reader for a captured projection argument list.
     * @param phase Captured projection and its physical launch order.
     * @param codebook Exact physical decoder observed in the prepared fixture.
     * @param access Scalar byte oracle or widest natural packet loads.
     * @return CUDA kernel entry, or null for an unsupported physical decoder.
     */
    [[nodiscard]] const void *cudaMoEWeightReferenceEntry(
        CUDAMoEWeightPhase phase, std::uint8_t codebook,
        CUDAWeightPacketAccess access) noexcept;
}
