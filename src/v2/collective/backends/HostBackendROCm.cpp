/**
 * @file HostBackendROCm.cpp
 * @brief ROCm-specific helper functions for HostBackend
 *
 * Isolated HIP runtime calls in separate compilation unit to avoid
 * conflicts with CUDA headers.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <hip/hip_runtime.h>
#include <mutex>

namespace llaminar2 {
namespace host_backend_detail {

// Global mutex for serializing hipHostRegister/hipHostUnregister calls.
// Concurrent hipHostRegister from multiple threads can corrupt KFD page tables
// (observed as "Memory access fault by GPU node-N" on unused ROCm devices).
// The KFD driver updates GPU page tables for the current device, and concurrent
// updates from different threads are not safe.
static std::mutex s_hip_host_register_mutex;

bool hipCopyToHost(void* host_dst, const void* device_src, int device_ordinal, size_t bytes)
{
    hipError_t err = hipSetDevice(device_ordinal);
    if (err != hipSuccess)
    {
        return false;
    }
    
    err = hipMemcpy(host_dst, device_src, bytes, hipMemcpyDeviceToHost);
    return (err == hipSuccess);
}

bool hipCopyFromHost(void* device_dst, const void* host_src, int device_ordinal, size_t bytes)
{
    hipError_t err = hipSetDevice(device_ordinal);
    if (err != hipSuccess)
    {
        return false;
    }
    
    err = hipMemcpy(device_dst, host_src, bytes, hipMemcpyHostToDevice);
    return (err == hipSuccess);
}

bool hipHostRegisterBuffer(void* ptr, size_t size)
{
    // Use hipHostRegisterDefault (current device only) instead of hipHostRegisterPortable
    // (all devices). hipHostRegisterPortable modifies page tables of ALL GPUs in the system,
    // including unused ones (e.g., ROCm:2 when only ROCm:0+1 are used for TP). Concurrent
    // hipHostRegister calls with Portable flag can corrupt KFD page tables, causing
    // "Memory access fault by GPU node-N" on the unused device.
    //
    // hipHostRegisterDefault only updates the current device's page tables, which is
    // sufficient since each tensor is per-device and only DMA'd from its owning device.
    //
    // Additionally, serialize all registration calls via mutex because the KFD driver's
    // internal page table update logic is not guaranteed thread-safe for concurrent calls.
    std::lock_guard<std::mutex> lock(s_hip_host_register_mutex);
    hipError_t err = hipHostRegister(ptr, size, hipHostRegisterDefault);
    return (err == hipSuccess);
}

void hipHostUnregisterBuffer(void* ptr)
{
    std::lock_guard<std::mutex> lock(s_hip_host_register_mutex);
    (void)hipHostUnregister(ptr);  // Ignore return value in cleanup
}

} // namespace host_backend_detail
} // namespace llaminar2
