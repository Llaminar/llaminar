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
    /**
     * @brief Publish host IQ codebook tables into this kernel translation unit.
     * @param device_id ROCm device whose constant memory receives the tables.
     * @param iq3s_grid Host array of 512 packed IQ3_S entries.
     * @param iq3xxs_grid Host array of 256 packed IQ3_XXS entries.
     * @param iq2s_grid Host array of 1024 packed IQ2_S entries.
     * @param iq2xs_grid Host array of 512 packed IQ2_XS entries.
     * @param iq2xxs_grid Host array of 256 packed IQ2_XXS entries.
     * @param iq1s_grid Host array of 2048 packed IQ1_S/IQ1_M entries.
     * @return Whether device selection and all six synchronous publications
     *         succeeded.
     *
     * HIP does not enable relocatable device code for `rocm_backend`, so each
     * kernel translation unit owns the constant symbols it reads. The central
     * ROCm weight-packer initialization authority calls this once per device,
     * before an IQ source can be streamed or a graph can be captured.
     */
    [[nodiscard]] bool initializeExpertTierIQGridTablesROCm(
        int device_id,
        const void *iq3s_grid,
        const void *iq3xxs_grid,
        const void *iq2s_grid,
        const void *iq2xs_grid,
        const void *iq2xxs_grid,
        const void *iq1s_grid) noexcept;

    /**
     * @brief Convert common GPU packed bytes into final CPU NativeVNNI units.
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
     * @brief Convert streamed CPU NativeVNNI units into common GPU packed data.
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
