/**
 * @file CUDAExpertTierWeightKernels.h
 * @brief Explicit-stream CUDA conversion for heterogeneous expert tiers.
 */

#pragma once

#include "execution/moe/ExpertTierWeightDeviceLayout.h"

#include <cstddef>
#include <cstdint>

namespace llaminar2
{
    /**
     * @brief Convert common GPU packed bytes into final CPU NativeVNNI units.
     *
     * The destination is device staging memory whose bytes are already in the
     * final CPU execution representation. The call allocates nothing, never
     * synchronizes, and rejects a null/default stream.
     *
     * @param source Read-only separated CUDA projection.
     * @param layout Immutable cross-tier format and geometry.
     * @param first_unit First global CPU unit represented by this chunk.
     * @param unit_count Number of complete consecutive units to produce.
     * @param device_cpu_chunk Preallocated CUDA staging destination.
     * @param device_cpu_chunk_capacity Capacity of the staging destination.
     * @param stream Exact non-null CUDA producer stream.
     */
    [[nodiscard]] bool launchGpuToCpuExpertTierChunkCUDA(
        const ExpertTierGpuConstProjectionView &source,
        const ExpertTierWeightDeviceLayout &layout,
        std::uint32_t first_unit,
        std::uint32_t unit_count,
        std::uint8_t *device_cpu_chunk,
        std::size_t device_cpu_chunk_capacity,
        void *stream) noexcept;

    /**
     * @brief Convert streamed CPU NativeVNNI units into common GPU packed data.
     *
     * The source is device staging memory filled from an incoming CPU stream;
     * the destination is the preallocated final CUDA projection. The call is
     * asynchronous and uses only the exact caller-provided stream.
     *
     * @param device_cpu_chunk Preallocated CUDA staging source.
     * @param device_cpu_chunk_bytes Valid bytes in the source chunk.
     * @param layout Immutable cross-tier format and geometry.
     * @param first_unit First global CPU unit represented by this chunk.
     * @param unit_count Number of complete consecutive units to consume.
     * @param destination Writable separated CUDA projection.
     * @param stream Exact non-null CUDA consumer stream.
     */
    [[nodiscard]] bool launchCpuToGpuExpertTierChunkCUDA(
        const std::uint8_t *device_cpu_chunk,
        std::size_t device_cpu_chunk_bytes,
        const ExpertTierWeightDeviceLayout &layout,
        std::uint32_t first_unit,
        std::uint32_t unit_count,
        const ExpertTierGpuMutableProjectionView &destination,
        void *stream) noexcept;
} // namespace llaminar2
