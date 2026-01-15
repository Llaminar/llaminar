/**
 * @file BackendManager.h
 * @brief Global backend accessor (Phase 6: Heterogeneous Multi-GPU + CPU)
 *
 * **Purpose**: Provides accessors for all compute backends (CPU, CUDA, ROCm).
 * Supports heterogeneous multi-GPU where both NVIDIA and AMD GPUs coexist,
 * plus CPU backend for unified memory management.
 *
 * **Thread Safety**: Initialization is thread-safe via call_once.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "IBackend.h"
#include "DeviceId.h"
#include "ComputeBackend.h" // For ComputeBackendType

namespace llaminar2
{

    /**
     * @brief Get the global GPU backend instance (legacy - prefers CUDA over ROCm)
     *
     * Returns:
     * - CUDABackend* if HAVE_CUDA is defined
     * - ROCmBackend* if HAVE_ROCM is defined (and no CUDA)
     * - nullptr if CPU-only build
     *
     * @return IBackend* or nullptr
     *
     * @note First call initializes the backend (lazy init)
     * @note Thread-safe via std::call_once
     * @deprecated Use getBackendForDevice() for heterogeneous multi-GPU
     */
    IBackend *getGPUBackend();

    /**
     * @brief Get CUDA backend specifically
     *
     * @return CUDABackend* or nullptr if CUDA not available
     */
    IBackend *getCUDABackend();

    /**
     * @brief Get ROCm backend specifically
     *
     * @return ROCmBackend* or nullptr if ROCm not available
     */
    IBackend *getROCmBackend();

    /**
     * @brief Get backend for a specific device type
     *
     * @param type The backend type (GPU_CUDA, GPU_ROCM, etc.)
     * @return IBackend* for the device type, or nullptr if unavailable
     *
     * **Phase 6 Usage** (heterogeneous multi-GPU):
     * ```cpp
     * // Allocate memory on the right GPU based on device type
     * const auto& device = DeviceManager::instance().devices()[device_idx];
     * IBackend* backend = getBackendForDeviceType(device.type);
     * if (backend) {
     *     void* ptr = backend->allocate(bytes, device.device_id);
     * }
     * ```
     */
    IBackend *getBackendForDeviceType(ComputeBackendType type);

    /**
     * @brief Check if GPU backend is available
     * @return true if getGPUBackend() != nullptr
     */
    bool hasGPUBackend();

    /**
     * @brief Check if CUDA backend is available
     * @return true if getCUDABackend() != nullptr
     */
    bool hasCUDABackend();

    /**
     * @brief Check if ROCm backend is available
     * @return true if getROCmBackend() != nullptr
     */
    bool hasROCmBackend();

    // ====================================================================
    // CPU Backend (Phase 0.5: CPU/NUMA Backend)
    // ====================================================================

    /**
     * @brief Initialize CPU backend for this MPI rank
     * @param local_numa_node NUMA node from MPITopology::placement().numa_node
     *
     * Call once during MPI initialization:
     *   BackendManager::initCPUBackend(topo.placement().numa_node);
     *
     * @note Thread-safe via std::call_once
     */
    void initCPUBackend(int local_numa_node);

    /**
     * @brief Get CPU backend
     * @return CPUBackend* or nullptr if not initialized
     *
     * @note Returns nullptr if initCPUBackend() hasn't been called
     */
    IBackend *getCPUBackend();

    /**
     * @brief Check if CPU backend is available
     * @return true if getCPUBackend() != nullptr
     */
    bool hasCPUBackend();

    // ====================================================================
    // Unified Backend Accessor
    // ====================================================================

    /**
     * @brief Unified backend accessor for any device type
     * @param device DeviceId (CPU, CUDA, or ROCm)
     * @return IBackend* for the device, or nullptr if unavailable
     *
     * **Usage**:
     * ```cpp
     * IBackend* backend = getBackendFor(DeviceId::cpu());
     * size_t free = backend->deviceMemoryFree(0);
     *
     * IBackend* cuda = getBackendFor(DeviceId::cuda(0));
     * void* ptr = cuda->allocate(1024, 0);
     * ```
     *
     * @note For CPU devices, returns getCPUBackend()
     * @note For CUDA devices, returns getCUDABackend()
     * @note For ROCm devices, returns getROCmBackend()
     */
    IBackend *getBackendFor(DeviceId device);

} // namespace llaminar2
