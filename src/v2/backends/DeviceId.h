/**
 * @file DeviceId.h
 * @brief Type-safe device identification for heterogeneous compute
 *
 * Replaces magic integer device indices with explicit type+ordinal.
 * - DeviceId::cpu() for CPU execution
 * - DeviceId::cuda(0) for first CUDA GPU
 * - DeviceId::rocm(1) for second ROCm GPU
 */

#pragma once

#include "DeviceType.h" // Shared DeviceType enum
#include <cassert>
#include <string>
#include <stdexcept>
#include <ostream>

namespace llaminar2
{
    /**
     * @brief Type-safe device identifier
     *
     * Eliminates the legacy convention where device_idx=0 meant CPU and
     * device_idx>=1 meant GPU ordinal (device_idx-1). Now explicit:
     * - DeviceId::cpu() - CPU execution
     * - DeviceId::cuda(n) - CUDA GPU n (0-indexed)
     * - DeviceId::rocm(n) - ROCm GPU n (0-indexed)
     */
    struct DeviceId
    {
        DeviceType type;
        int ordinal; // GPU ordinal (0-based), 0 for CPU

        // Factory methods for clarity
        static DeviceId cpu() { return {DeviceType::CPU, 0}; }
        static DeviceId cuda(int gpu_ordinal) { return {DeviceType::CUDA, gpu_ordinal}; }
        static DeviceId rocm(int gpu_ordinal) { return {DeviceType::ROCm, gpu_ordinal}; }

        /// Invalid/unset device marker (ordinal = -1)
        static DeviceId invalid() { return {DeviceType::CPU, -1}; }

        // Predicates
        bool is_cpu() const { return type == DeviceType::CPU && ordinal >= 0; }
        bool is_cuda() const { return type == DeviceType::CUDA; }
        bool is_rocm() const { return type == DeviceType::ROCm; }
        bool is_gpu() const { return type != DeviceType::CPU && ordinal >= 0; }
        bool is_valid() const { return ordinal >= 0; }

        // Get CUDA device ordinal - asserts if not CUDA
        int cuda_ordinal() const
        {
            assert(type == DeviceType::CUDA && "cuda_ordinal() called on non-CUDA device");
            return ordinal;
        }

        // Get ROCm device ordinal - asserts if not ROCm
        int rocm_ordinal() const
        {
            assert(type == DeviceType::ROCm && "rocm_ordinal() called on non-ROCm device");
            return ordinal;
        }

        // Get GPU ordinal for any GPU type (CUDA or ROCm)
        int gpu_ordinal() const
        {
            assert(is_gpu() && "gpu_ordinal() called on CPU device");
            return ordinal;
        }

        /**
         * @brief Get device index for kernel API calls
         *
         * Returns the appropriate device index for CUDA/ROCm API calls:
         * - CPU: returns -1 (convention for CPU execution)
         * - CUDA/ROCm: returns the 0-based GPU ordinal
         */
        int toKernelDeviceIndex() const
        {
            if (is_cpu())
                return -1;
            return ordinal; // Direct GPU ordinal for cudaSetDevice/hipSetDevice
        }

        // String representation for logging
        std::string to_string() const
        {
            switch (type)
            {
            case DeviceType::CPU:
                return "CPU";
            case DeviceType::CUDA:
                return "CUDA:" + std::to_string(ordinal);
            case DeviceType::ROCm:
                return "ROCm:" + std::to_string(ordinal);
            default:
                return "Unknown";
            }
        }

        // Equality comparison
        bool operator==(const DeviceId &other) const
        {
            return type == other.type && ordinal == other.ordinal;
        }
        bool operator!=(const DeviceId &other) const { return !(*this == other); }

        // =========================================================================
        // String Conversion (for logging)
        // =========================================================================

        /**
         * @brief Get string representation of the device
         * @return String like "CPU", "CUDA:0", "ROCm:1", etc.
         */
        std::string toString() const
        {
            switch (type)
            {
            case DeviceType::CPU:
                return "CPU";
            case DeviceType::CUDA:
                return "CUDA:" + std::to_string(ordinal);
            case DeviceType::ROCm:
                return "ROCm:" + std::to_string(ordinal);
            default:
                return "Unknown";
            }
        }
    };

    /**
     * @brief Stream output operator for DeviceId (enables LOG_* macros)
     */
    inline std::ostream &operator<<(std::ostream &os, const DeviceId &device)
    {
        return os << device.toString();
    }

} // namespace llaminar2

// Hash specialization for using DeviceId as unordered_map key
namespace std
{
    template <>
    struct hash<llaminar2::DeviceId>
    {
        size_t operator()(const llaminar2::DeviceId &device) const noexcept
        {
            // Combine device type and ordinal for hash
            size_t h1 = std::hash<int>{}(static_cast<int>(device.type));
            size_t h2 = std::hash<int>{}(device.ordinal);
            return h1 ^ (h2 << 1);
        }
    };
} // namespace std
