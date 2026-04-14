/**
 * @file BackendSelector.h
 * @brief Automatic backend selection based on device topology
 *
 * Encapsulates the rules for selecting optimal collective backends
 * for different device-pair combinations in PP and TP scenarios.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "OrchestrationConfig.h"
#include "../backends/DeviceId.h"
#include <vector>

namespace llaminar2
{

    /**
     * @brief Automatic backend selection based on device topology
     *
     * Provides static methods to select optimal collective backends
     * for PP transfers and TP allreduce operations.
     */
    class BackendSelector
    {
    public:
        // =========================================================================
        // PP Transfer Backend Selection
        // =========================================================================

        /**
         * @brief Select optimal backend for PP activation transfer
         *
         * Rules:
         * - CUDA → CUDA: NCCL (low latency, high bandwidth)
         * - ROCm → ROCm: RCCL (AMD equivalent)
         * - CUDA ↔ ROCm (same NUMA): HOST (host-staged cross-vendor)
         * - CUDA ↔ ROCm (cross NUMA): HOST (host-staged cross-vendor)
         * - GPU → CPU or CPU → GPU: HOST (staging through host memory)
         * - CPU → CPU: HOST
         *
         * @param src Source device
         * @param dst Destination device
         * @return Selected backend type (never AUTO)
         */
        static CollectiveBackendType selectForTransfer(DeviceId src, DeviceId dst);

        /**
         * @brief Select optimal backend for PP transfer between device types
         *
         * Same as selectForTransfer but takes DeviceType directly.
         *
         * @param src_type Source device type
         * @param dst_type Destination device type
         * @return Selected backend type (never AUTO)
         */
        static CollectiveBackendType selectForTransfer(DeviceType src_type, DeviceType dst_type);

        // =========================================================================
        // TP Domain Backend Selection
        // =========================================================================

        /**
         * @brief Select optimal backend for TP allreduce within a domain
         *
         * Rules:
         * - All CUDA: NCCL
         * - All ROCm: RCCL
         * - Mixed CUDA+ROCm (2 devices, same NUMA): HOST
         * - Mixed CUDA+ROCm (2 devices, cross NUMA): HOST
         * - Mixed CUDA+ROCm (3+ devices): HETEROGENEOUS
         * - All CPU: MPI (or HOST for single-node)
         * - Single device: HOST (no collective needed)
         *
         * @param devices Devices in the TP domain
         * @return Selected backend type (never AUTO)
         */
        static CollectiveBackendType selectForTPDomain(const std::vector<DeviceId> &devices);

        // =========================================================================
        // Backend Availability
        // =========================================================================

        /**
         * @brief Check if a backend is available at compile time
         *
         * Checks if the required library (NCCL, RCCL) was compiled in.
         *
         * @param backend Backend type to check
         * @return true if backend was compiled in
         */
        static bool isBackendCompiled(CollectiveBackendType backend);

        /**
         * @brief Check if a backend is usable with given devices
         *
         * Verifies that the backend is compiled and the devices support it.
         * For example, NCCL requires all devices to be CUDA.
         *
         * @param backend Backend type to check
         * @param devices Devices that would use this backend
         * @return true if backend can be used with these devices
         */
        static bool isBackendUsable(CollectiveBackendType backend,
                                    const std::vector<DeviceId> &devices);

        // =========================================================================
        // Utility
        // =========================================================================

        /**
         * @brief Resolve AUTO backend to a concrete type
         *
         * If backend is AUTO, selects based on devices. Otherwise returns as-is.
         *
         * @param backend Backend type (may be AUTO)
         * @param devices Devices to consider for AUTO resolution
         * @return Concrete backend type (never AUTO)
         */
        static CollectiveBackendType resolve(CollectiveBackendType backend,
                                             const std::vector<DeviceId> &devices);

        /**
         * @brief Check if all devices are the same type
         * @param devices Device list
         * @return true if all devices have the same DeviceType
         */
        static bool isHomogeneous(const std::vector<DeviceId> &devices);

        /**
         * @brief Check if any device is a GPU (CUDA or ROCm)
         * @param devices Device list
         * @return true if at least one device is a GPU
         */
        static bool hasGPU(const std::vector<DeviceId> &devices);

        /**
         * @brief Check if devices span multiple GPU vendors (CUDA and ROCm)
         * @param devices Device list
         * @return true if both CUDA and ROCm devices present
         */
        static bool isCrossVendor(const std::vector<DeviceId> &devices);
    };

} // namespace llaminar2
