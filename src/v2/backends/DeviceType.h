/**
 * @file DeviceType.h
 * @brief Standalone device type enum to avoid circular dependencies
 *
 * This header provides the DeviceType enum used by:
 * - DeviceId.h for type-safe device identification
 * - DeviceContext.h for device execution contexts
 * - KernelFactory.h for kernel dispatch
 */

#pragma once

namespace llaminar2
{
    /**
     * @brief Device type for heterogeneous compute dispatch
     */
    enum class DeviceType
    {
        CPU,    ///< CPU execution (OneDNN, MKL, etc.)
        CUDA,   ///< NVIDIA CUDA GPU
        ROCm,   ///< AMD ROCm/HIP GPU
        Vulkan, ///< Vulkan compute shaders
        Metal   ///< Apple Metal Performance Shaders
    };

} // namespace llaminar2
