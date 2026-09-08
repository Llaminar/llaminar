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

bool hipCopyToHost(void* host_dst, const void* device_src, int device_ordinal, size_t bytes, void* stream)
{
    hipError_t err = hipSetDevice(device_ordinal);
    if (err != hipSuccess)
    {
        return false;
    }
    
    hipStream_t s = static_cast<hipStream_t>(stream);
    err = hipMemcpyAsync(host_dst, device_src, bytes, hipMemcpyDeviceToHost, s);
    if (err != hipSuccess)
        return false;
    err = hipStreamSynchronize(s);
    return (err == hipSuccess);
}

bool hipCopyFromHost(void* device_dst, const void* host_src, int device_ordinal, size_t bytes, void* stream)
{
    hipError_t err = hipSetDevice(device_ordinal);
    if (err != hipSuccess)
    {
        return false;
    }
    
    hipStream_t s = static_cast<hipStream_t>(stream);
    err = hipMemcpyAsync(device_dst, host_src, bytes, hipMemcpyHostToDevice, s);
    if (err != hipSuccess)
        return false;
    err = hipStreamSynchronize(s);
    return (err == hipSuccess);
}

bool hipHostRegisterBuffer(void* ptr, size_t size, int device_ordinal)
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
    int previous_device = -1;
    if (!ptr || size == 0 || device_ordinal < 0 ||
        hipGetDevice(&previous_device) != hipSuccess ||
        hipSetDevice(device_ordinal) != hipSuccess)
        return false;

    const hipError_t error =
        hipHostRegister(ptr, size, hipHostRegisterDefault);
    if (previous_device != device_ordinal &&
        hipSetDevice(previous_device) != hipSuccess)
    {
        if (error == hipSuccess)
        {
            (void)hipSetDevice(device_ordinal);
            (void)hipHostUnregister(ptr);
        }
        return false;
    }
    return error == hipSuccess;
}

bool hipHostUnregisterBuffer(void* ptr, int device_ordinal)
{
    std::lock_guard<std::mutex> lock(s_hip_host_register_mutex);
    int previous_device = -1;
    if (!ptr || device_ordinal < 0 ||
        hipGetDevice(&previous_device) != hipSuccess ||
        hipSetDevice(device_ordinal) != hipSuccess)
        return false;

    const hipError_t error = hipHostUnregister(ptr);
    const bool restored = previous_device == device_ordinal ||
                          hipSetDevice(previous_device) == hipSuccess;
    return error == hipSuccess && restored;
}

} // namespace host_backend_detail
} // namespace llaminar2
