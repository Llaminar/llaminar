/**
 * @file ROCmExpertTierWeightKernels.h
 * @brief Explicit-stream HIP conversion for heterogeneous expert tiers.
 */

#pragma once

#include "execution/moe/ExpertTierWeightDeviceLayout.h"

#include <cstddef>
#include <cstdint>

namespace llaminar2
{
    /** @brief Root views and live K32/N intersection for a coded HIP peer copy. */
    struct ROCmCodedPeerProjectionRectangle
    {
        unsigned rows, blocks, payload_bytes;
        std::uint8_t codebook;
        unsigned src_n, dst_n, src_row, dst_row, src_block, dst_block;
        ExpertTierGpuConstProjectionView source;
        ExpertTierGpuMutableProjectionView destination;
    };

    /** @brief Copy the exact live rectangle, preserving adjacent destination code bits. */
    [[nodiscard]] bool launchROCmCodedPeerProjectionRectangle(ROCmCodedPeerProjectionRectangle rectangle,
                                                             void *stream) noexcept;

    /**
     * @brief Convert common GPU packed bytes into final CPU NativeQuantised units.
     * @param source Read-only separated ROCm projection.
     * @param layout Immutable cross-tier format and geometry.
     * @param first_unit First global CPU unit represented by this chunk.
     * @param unit_count Number of complete consecutive units to produce.
     * @param device_cpu_chunk Preallocated HIP staging destination.
     * @param device_cpu_chunk_capacity Capacity of the staging destination.
     * @param stream Exact non-null HIP producer stream.
     * @return Whether validation and asynchronous kernel launch succeeded.
     */
    [[nodiscard]] bool launchGpuToCpuExpertTierChunkROCm(
        const ExpertTierGpuConstProjectionView &source,
        const ExpertTierWeightDeviceLayout &layout,
        std::uint32_t first_unit,
        std::uint32_t unit_count,
        std::uint8_t *device_cpu_chunk,
        std::size_t device_cpu_chunk_capacity,
        void *stream) noexcept;

    /**
     * @brief Convert streamed CPU NativeQuantised units into common GPU packed data.
     * @param device_cpu_chunk Preallocated HIP staging source.
     * @param device_cpu_chunk_bytes Valid bytes in the source chunk.
     * @param layout Immutable cross-tier format and geometry.
     * @param first_unit First global CPU unit represented by this chunk.
     * @param unit_count Number of complete consecutive units to consume.
     * @param destination Writable separated ROCm projection.
     * @param stream Exact non-null HIP consumer stream.
     * @return Whether validation and asynchronous kernel launch succeeded.
     */
    [[nodiscard]] bool launchCpuToGpuExpertTierChunkROCm(
        const std::uint8_t *device_cpu_chunk,
        std::size_t device_cpu_chunk_bytes,
        const ExpertTierWeightDeviceLayout &layout,
        std::uint32_t first_unit,
        std::uint32_t unit_count,
        const ExpertTierGpuMutableProjectionView &destination,
        void *stream) noexcept;
} // namespace llaminar2
