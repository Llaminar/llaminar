/**
 * @file GPUPreparedWeightPoolLayout.h
 * @brief Physical representation admitted for a prepared GPU weight pool.
 *
 * Immutable static sources retain their original canonical device layout.
 * Movable expert slots reserve the source/migration union instead. Planning
 * and preparation share this type so neither silently prices the other layout.
 */
#pragma once

#include "tensors/NativeVnniFormatInfo.h"
#include <stdexcept>

namespace llaminar2
{
    /** @brief Pool stride policy, independent of the initial execution codebook. */
    enum class GPUPreparedWeightPoolLayout
    {
        SourceNative, ///< Exact canonical representation of the immutable source.
        MigrationReusable, ///< Canonical initial/migrated representation union.
    };

    /**
     * @brief Resolve the physical planes shared by admission and pool binding.
     * @param source Original GGUF codebook metadata, before device normalization.
     * @param layout Immutable source storage or the admitted migration union.
     * @return Exact payload stride and presence of the minima metadata planes.
     * @throws std::invalid_argument If the caller supplies an unknown layout.
     */
    [[nodiscard]] inline NativeVnniReusableDeviceAllocationFormat
    gpuPreparedWeightPoolAllocation(const NativeVnniFormatInfo &source,
        GPUPreparedWeightPoolLayout layout)
    {
        switch (layout)
        {
        case GPUPreparedWeightPoolLayout::SourceNative:
            return {static_cast<uint8_t>(source.payload_bytes),
                canonicalDeviceVnniFormat(source).is_asymmetric, source.has_emins};
        case GPUPreparedWeightPoolLayout::MigrationReusable:
            return reusableDeviceVnniAllocationFormat(source);
        default:
            throw std::invalid_argument("Unknown GPU prepared weight pool layout");
        }
    }
}
