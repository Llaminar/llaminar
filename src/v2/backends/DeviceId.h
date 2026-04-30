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
#include <optional>
#include <utility>

namespace llaminar2
{
    /**
     * @brief Descriptive hardware architecture metadata for a DeviceId.
     *
     * This metadata is intentionally NOT part of DeviceId identity. Equality,
     * ordering, and hashing for DeviceId continue to use only (type, ordinal).
     */
    struct DeviceArchInfo
    {
        DeviceType type = DeviceType::CPU;
        int ordinal = -1;
        std::string device_name;

        // CUDA architecture information.
        int cc_major = 0;
        int cc_minor = 0;
        int sm = 0;
        std::string sm_string;

        // ROCm architecture information.
        std::string gcn_arch_name;

        // Common capability flags used by runtime dispatch.
        bool supports_dp4a = false;
        bool supports_wmma = false;
        bool supports_mfma = false;
        bool supports_int8_tensor_cores = false;
        bool supports_native_vnni = false;

        // Optional topology/execution details.
        int multiprocessor_count = 0; // SM count for CUDA, CU count for ROCm.
        int warp_or_wave_size = 0;

        bool valid() const { return ordinal >= 0; }

        std::string arch_string() const
        {
            if (!gcn_arch_name.empty())
            {
                return gcn_arch_name;
            }
            if (!sm_string.empty())
            {
                return sm_string;
            }
            return "unknown";
        }
    };

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
        int ordinal; // GPU ordinal (0-based), 0 for CPU, -1 for invalid
        std::optional<DeviceArchInfo> arch; // Descriptive metadata, not identity.

        // =========================================================================
        // Default constructor creates INVALID device (breaks silent CPU fallback)
        // =========================================================================

        /// Default constructor creates an INVALID device
        /// This ensures uninitialized DeviceId variables cause loud failures
        DeviceId() : type(DeviceType::CPU), ordinal(-1) {}

        /// Explicit construction (use factory methods instead)
        DeviceId(DeviceType t, int ord) : type(t), ordinal(ord) {}

        /// Explicit construction with descriptive architecture metadata.
        DeviceId(DeviceType t, int ord, DeviceArchInfo arch_info)
            : type(t), ordinal(ord), arch(std::move(arch_info)) {}

        // Factory methods for clarity (preferred way to create DeviceIds)
        static DeviceId cpu() { return {DeviceType::CPU, 0}; }
        static DeviceId cuda(int gpu_ordinal) { return {DeviceType::CUDA, gpu_ordinal}; }
        static DeviceId rocm(int gpu_ordinal) { return {DeviceType::ROCm, gpu_ordinal}; }

        /// Invalid/unset device marker (ordinal = -1)
        static DeviceId invalid() { return {}; } // Same as default constructor

        // Predicates
        bool is_cpu() const { return type == DeviceType::CPU && ordinal >= 0; }
        bool is_cuda() const { return type == DeviceType::CUDA && ordinal >= 0; }
        bool is_rocm() const { return type == DeviceType::ROCm && ordinal >= 0; }
        bool is_gpu() const { return (type == DeviceType::CUDA || type == DeviceType::ROCm) && ordinal >= 0; }
        bool is_valid() const { return ordinal >= 0; }
        bool has_arch_info() const { return arch.has_value(); }
        const DeviceArchInfo *arch_info() const { return arch ? &(*arch) : nullptr; }

        DeviceId with_arch_info(DeviceArchInfo arch_info) const
        {
            DeviceId copy = *this;
            copy.arch = std::move(arch_info);
            return copy;
        }

        // Get CUDA device ordinal - throws if not valid CUDA
        int cuda_ordinal() const
        {
            if (!is_valid())
            {
                throw std::runtime_error("cuda_ordinal() called on INVALID DeviceId - was the device properly initialized?");
            }
            if (type != DeviceType::CUDA)
            {
                throw std::runtime_error("cuda_ordinal() called on non-CUDA device: " + to_string());
            }
            return ordinal;
        }

        // Get ROCm device ordinal - throws if not valid ROCm
        int rocm_ordinal() const
        {
            if (!is_valid())
            {
                throw std::runtime_error("rocm_ordinal() called on INVALID DeviceId - was the device properly initialized?");
            }
            if (type != DeviceType::ROCm)
            {
                throw std::runtime_error("rocm_ordinal() called on non-ROCm device: " + to_string());
            }
            return ordinal;
        }

        // Get GPU ordinal for any GPU type (CUDA or ROCm)
        int gpu_ordinal() const
        {
            if (!is_valid())
            {
                throw std::runtime_error("gpu_ordinal() called on INVALID DeviceId - was the device properly initialized?");
            }
            if (!is_gpu())
            {
                throw std::runtime_error("gpu_ordinal() called on CPU device");
            }
            return ordinal;
        }

        /**
         * @brief Get device index for kernel API calls
         *
         * Returns the appropriate device index for CUDA/ROCm API calls:
         * - CPU: returns -1 (convention for CPU execution)
         * - CUDA/ROCm: returns the 0-based GPU ordinal
         * @throws std::runtime_error if DeviceId is invalid
         */
        int toKernelDeviceIndex() const
        {
            if (!is_valid())
            {
                throw std::runtime_error("toKernelDeviceIndex() called on INVALID DeviceId - was the device properly initialized?");
            }
            if (is_cpu())
                return -1;
            return ordinal; // Direct GPU ordinal for cudaSetDevice/hipSetDevice
        }

        // String representation for logging (safe to call on invalid)
        std::string to_string() const
        {
            if (!is_valid())
            {
                return "INVALID";
            }
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

        // Ordering comparison (for std::map)
        bool operator<(const DeviceId &other) const
        {
            if (type != other.type)
                return static_cast<int>(type) < static_cast<int>(other.type);
            return ordinal < other.ordinal;
        }

        // =========================================================================
        // String Conversion (for logging)
        // =========================================================================

        /**
         * @brief Get string representation of the device
         * @return String like "CPU", "CUDA:0", "ROCm:1", "INVALID"
         */
        std::string toString() const
        {
            if (!is_valid())
            {
                return "INVALID";
            }
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
