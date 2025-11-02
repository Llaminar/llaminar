/**
 * @file ROCmBackend.cpp
 * @brief ROCm/HIP backend implementation with hip_runtime.h
 *
 * **Purpose**: Implements IBackend for AMD GPUs. This .cpp file is the ONLY
 * compilation unit that includes hip_runtime.h, preventing header conflicts.
 *
 * @author David Sanftenberg
 */

#include "ROCmBackend.h"
#include "../../utils/Logger.h"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <sstream>

namespace llaminar2
{

    // ====================================================================
    // Constructor / Destructor
    // ====================================================================

    ROCmBackend::ROCmBackend()
        : device_count_(0)
    {
        hipError_t err = hipGetDeviceCount(&device_count_);
        if (err != hipSuccess)
        {
            device_count_ = 0;
            // Log warning but don't throw - allow CPU-only execution
        }
    }

    ROCmBackend::~ROCmBackend()
    {
        // hipDeviceReset() intentionally omitted - managed by HIP runtime
    }

    // ====================================================================
    // Memory Transfer Operations
    // ====================================================================

    bool ROCmBackend::deviceToHost(void *dst, const void *src, size_t bytes, int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        hipError_t err_set = hipSetDevice(device_id);
        if (err_set != hipSuccess)
        {
            return false;
        }

        hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
        return (err == hipSuccess);
    }

    bool ROCmBackend::hostToDevice(void *dst, const void *src, size_t bytes, int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        hipError_t err_set = hipSetDevice(device_id);
        if (err_set != hipSuccess)
        {
            return false;
        }

        hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
        return (err == hipSuccess);
    }

    bool ROCmBackend::synchronize(int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        hipError_t err_set = hipSetDevice(device_id);
        if (err_set != hipSuccess)
        {
            return false;
        }

        hipError_t err = hipDeviceSynchronize();
        return (err == hipSuccess);
    }

    bool ROCmBackend::setDevice(int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        hipError_t err = hipSetDevice(device_id);
        return (err == hipSuccess);
    }

    // ====================================================================
    // Memory Allocation Operations
    // ====================================================================

    void *ROCmBackend::allocate(size_t bytes, int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_ERROR("[ROCmBackend] Invalid device ID " << device_id << " (max: " << device_count_ - 1 << ")");
            return nullptr;
        }

        // Set device before allocation
        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend] Failed to set device " << device_id << ": " << hipGetErrorString(err));
            return nullptr;
        }

        void *ptr = nullptr;
        err = hipMalloc(&ptr, bytes);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend] hipMalloc failed for " << bytes << " bytes on device "
                                                            << device_id << ": " << hipGetErrorString(err));
            return nullptr;
        }

        return ptr;
    }

    void ROCmBackend::free(void *ptr, int device_id)
    {
        if (ptr == nullptr)
        {
            return; // Freeing nullptr is a no-op
        }

        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_ERROR("[ROCmBackend] Invalid device ID " << device_id << " for hipFree");
            return;
        }

        // Set device before freeing
        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend] Failed to set device " << device_id << " before hipFree: "
                                                            << hipGetErrorString(err));
            return;
        }

        err = hipFree(ptr);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend] hipFree failed: " << hipGetErrorString(err));
        }
    }

    // ====================================================================
    // Device Query Operations
    // ====================================================================

    int ROCmBackend::deviceCount() const
    {
        return device_count_;
    }

    std::string ROCmBackend::backendName() const
    {
        return "ROCm";
    }

    std::string ROCmBackend::deviceName(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return "Invalid Device";
        }

        hipDeviceProp_t prop;
        hipError_t err = hipGetDeviceProperties(&prop, device_id);
        if (err != hipSuccess)
        {
            return "Unknown Device";
        }

        return std::string(prop.name);
    }

    size_t ROCmBackend::deviceMemoryTotal(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return 0;
        }

        hipDeviceProp_t prop;
        hipError_t err = hipGetDeviceProperties(&prop, device_id);
        if (err != hipSuccess)
        {
            return 0;
        }

        return prop.totalGlobalMem;
    }

    size_t ROCmBackend::deviceMemoryFree(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return 0;
        }

        hipError_t err_set = hipSetDevice(device_id);
        if (err_set != hipSuccess)
        {
            return 0;
        }

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        hipError_t err = hipMemGetInfo(&free_bytes, &total_bytes);
        if (err != hipSuccess)
        {
            return 0;
        }

        return free_bytes;
    }

    // ====================================================================
    // Capability Queries
    // ====================================================================

    bool ROCmBackend::supportsBF16(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        hipDeviceProp_t prop;
        hipError_t err = hipGetDeviceProperties(&prop, device_id);
        if (err != hipSuccess)
        {
            return false;
        }

        // BF16 support on AMD GPUs:
        // - MI200 series (gfx90a): Full BF16 support
        // - MI100 (gfx908): Limited BF16 support
        // GCN architecture ID (gcnArch) is deprecated in newer ROCm versions
        // Use compute capability via prop.major/minor or architecture string

        // Conservative check: Assume MI200+ (gfx90a and later) for full BF16
        // This is a heuristic - may need refinement based on actual hardware
        std::string arch_name(prop.gcnArchName);
        return (arch_name.find("gfx90a") != std::string::npos ||
                arch_name.find("gfx940") != std::string::npos ||
                arch_name.find("gfx941") != std::string::npos ||
                arch_name.find("gfx942") != std::string::npos);
    }

    bool ROCmBackend::supportsFP16(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        hipDeviceProp_t prop;
        hipError_t err = hipGetDeviceProperties(&prop, device_id);
        if (err != hipSuccess)
        {
            return false;
        }

        // FP16 support widely available on AMD GPUs (Vega and later)
        // gfx900 (Vega 10) and later all support FP16
        std::string arch_name(prop.gcnArchName);
        return (arch_name.find("gfx9") != std::string::npos ||
                arch_name.find("gfx10") != std::string::npos ||
                arch_name.find("gfx11") != std::string::npos);
    }

    bool ROCmBackend::supportsINT8(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        hipDeviceProp_t prop;
        hipError_t err = hipGetDeviceProperties(&prop, device_id);
        if (err != hipSuccess)
        {
            return false;
        }

        // INT8 support widely available on modern AMD GPUs
        // Conservatively assume gfx9 and later (Vega+)
        std::string arch_name(prop.gcnArchName);
        return (arch_name.find("gfx9") != std::string::npos ||
                arch_name.find("gfx10") != std::string::npos ||
                arch_name.find("gfx11") != std::string::npos);
    }

    // ====================================================================
    // Compute Operations
    // ====================================================================

    bool ROCmBackend::gemmIQ4NL(
        const void *A_device,
        const void *B_device,
        void *C_device,
        int m,
        int n,
        int k,
        int device_id)
    {
        // TODO: Implement ROCm/HIP version of IQ4_NL GEMM kernel
        // For now, return false to indicate not implemented
        (void)A_device;
        (void)B_device;
        (void)C_device;
        (void)m;
        (void)n;
        (void)k;
        (void)device_id;

        return false;
    }

} // namespace llaminar2
