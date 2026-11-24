/**
 * @file GPUPreparedWeightPoolLayout.h
 * @brief Physical representation admitted for a prepared GPU weight pool.
 *
 * Immutable static sources retain their original canonical device layout.
 * Movable expert slots reserve the source/migration union instead. Planning
 * and preparation share this type so neither silently prices the other layout.
 */
#pragma once

#include "backends/DeviceId.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedEncoding.h"
#include "tensors/NativeQuantisedFormatInfo.h"
#include "tensors/NativeQuantisedScalePlane.h"
#include <stdexcept>

namespace llaminar2
{
    /** @brief Pool stride policy, independent of the initial execution codebook. */
    enum class GPUPreparedWeightPoolLayout
    {
        SourceNative, ///< Exact canonical representation of the immutable source.
        MigrationReusable, ///< Canonical physical decoder with planes sealed for tier arrivals.
    };

    /**
     * @brief Resolve the physical planes shared by admission and pool binding.
     * @param source Original GGUF codebook metadata, before device normalization.
     * @param layout Immutable source storage or the admitted migration union.
     * @return Exact payload stride and presence of the minima metadata planes.
     * @throws std::invalid_argument If the caller supplies an unknown layout.
     */
    [[nodiscard]] inline NativeQuantisedReusableDeviceAllocationFormat
    gpuPreparedWeightPoolAllocation(const NativeQuantisedFormatInfo &source,
        GPUPreparedWeightPoolLayout layout)
    {
        switch (layout)
        {
        case GPUPreparedWeightPoolLayout::SourceNative:
            return {static_cast<uint8_t>(source.payload_bytes),
                canonicalDeviceVnniFormat(source).is_asymmetric, source.has_emins};
        case GPUPreparedWeightPoolLayout::MigrationReusable:
            if (cpu::native_quantised::migrationUsesSuperblockCodes(source))
                return {static_cast<uint8_t>(source.payload_bytes), true, false}; // One superblock plane, no staged minima.
            return reusableDeviceVnniAllocationFormat(source);
        default:
            throw std::invalid_argument("Unknown GPU prepared weight pool layout");
        }
    }

    /**
     * @brief Whether one prepared GPU matrix stores superblock-coded metadata.
     *
     * CUDA and ROCm immutable source matrices use the native codec whenever supported.
     * Reusable CUDA and ROCm migration slots use it when the CPU retains original codes.
     *
     * @param source Original GGUF codebook metadata.
     * @param layout Pool stride policy of the matrix.
     * @param device Device owning the prepared matrix.
     */
    [[nodiscard]] inline bool gpuPreparedSuperblockCoded(const NativeQuantisedFormatInfo &source,
        GPUPreparedWeightPoolLayout layout, DeviceId device)
    {
        if (layout == GPUPreparedWeightPoolLayout::MigrationReusable &&
            cpu::native_quantised::migrationUsesSuperblockCodes(source))
            return device.is_cuda() || device.is_rocm();
        return layout == GPUPreparedWeightPoolLayout::SourceNative && (device.is_cuda() || device.is_rocm()) &&
               source.is_superblock &&
               nativeQuantisedSuperblockCodable(canonicalDeviceVnniCodebookId(source));
    }
}
