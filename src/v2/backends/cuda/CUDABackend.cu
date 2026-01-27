/**
 * @file CUDABackend.cu
 * @brief CUDA backend implementation with cuda_runtime.h
 *
 * **Purpose**: Implements IBackend for NVIDIA GPUs. This .cu file is the ONLY
 * compilation unit that includes cuda_runtime.h, preventing header conflicts.
 *
 * @author David Sanftenberg
 */

#include "CUDABackend.h"
#include "../../utils/Logger.h"
#include <cuda_runtime.h>
#include <cuda.h> // For cuCtxSetCurrent, cuDevicePrimaryCtxRetain
#include <stdexcept>
#include <sstream>

namespace llaminar2
{

    // ====================================================================
    // Constructor / Destructor
    // ====================================================================

    CUDABackend::CUDABackend()
        : device_count_(0)
    {
        cudaError_t err = cudaGetDeviceCount(&device_count_);
        if (err != cudaSuccess)
        {
            device_count_ = 0;
            // Log warning but don't throw - allow CPU-only execution
        }
    }

    CUDABackend::~CUDABackend()
    {
        // cudaDeviceReset() intentionally omitted - managed by CUDA runtime
    }

    // ====================================================================
    // Memory Transfer Operations
    // ====================================================================

    bool CUDABackend::deviceToHost(void *dst, const void *src, size_t bytes, int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        // Use setDevice() which handles both runtime and driver API context
        if (!setDevice(device_id))
        {
            return false;
        }

        cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
        return (err == cudaSuccess);
    }

    bool CUDABackend::hostToDevice(void *dst, const void *src, size_t bytes, int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        // Use setDevice() which handles both runtime and driver API context
        if (!setDevice(device_id))
        {
            return false;
        }

        cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
        return (err == cudaSuccess);
    }

    bool CUDABackend::synchronize(int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        // Use setDevice() which handles both runtime and driver API context
        if (!setDevice(device_id))
        {
            return false;
        }

        cudaError_t err = cudaDeviceSynchronize();
        return (err == cudaSuccess);
    }

    bool CUDABackend::streamSynchronize(int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        // Use setDevice() which handles both runtime and driver API context
        if (!setDevice(device_id))
        {
            return false;
        }

        // Synchronize only the default stream (nullptr), not all streams
        cudaError_t err = cudaStreamSynchronize(nullptr);
        return (err == cudaSuccess);
    }

    // ====================================================================
    // Event Operations (Fine-grained Synchronization)
    // ====================================================================

    void *CUDABackend::createEvent(int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return nullptr;
        }

        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            return nullptr;
        }

        cudaEvent_t event;
        err = cudaEventCreate(&event);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::createEvent] cudaEventCreate failed: " << cudaGetErrorString(err));
            return nullptr;
        }

        return reinterpret_cast<void *>(event);
    }

    void CUDABackend::destroyEvent(void *event, int device_id)
    {
        if (!event || device_id >= device_count_ || device_id < 0)
        {
            return;
        }

        cudaSetDevice(device_id);
        cudaEvent_t cuda_event = reinterpret_cast<cudaEvent_t>(event);
        cudaEventDestroy(cuda_event);
    }

    bool CUDABackend::recordEvent(void *event, int device_id)
    {
        if (!event || device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            return false;
        }

        cudaEvent_t cuda_event = reinterpret_cast<cudaEvent_t>(event);
        // Record on default stream (0)
        err = cudaEventRecord(cuda_event, 0);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::recordEvent] cudaEventRecord failed: " << cudaGetErrorString(err));
            return false;
        }

        return true;
    }

    bool CUDABackend::waitForEvent(void *event, int device_id)
    {
        if (!event || device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        // Use setDevice() which handles both runtime and driver API context
        if (!setDevice(device_id))
        {
            LOG_ERROR("[CUDABackend::waitForEvent] setDevice(" << device_id << ") failed");
            return false;
        }

        cudaEvent_t cuda_event = reinterpret_cast<cudaEvent_t>(event);
        cudaError_t err = cudaEventSynchronize(cuda_event);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::waitForEvent] cudaEventSynchronize failed: " << cudaGetErrorString(err)
                                                                                  << " (device=" << device_id << ", event=" << event << ")");
            return false;
        }

        return true;
    }

    bool CUDABackend::setDevice(int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        // First, set the runtime API device
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            return false;
        }

        // Also set the driver API context for cross-thread compatibility.
        // This is crucial when CUDA is called from a thread that primarily
        // uses another GPU runtime (e.g., HIP/ROCm). The cudaSetDevice alone
        // may not properly establish the context for driver API operations
        // or for events created on other threads.
        CUdevice cu_device;
        CUresult cu_err = cuDeviceGet(&cu_device, device_id);
        if (cu_err != CUDA_SUCCESS)
        {
            // Fall through - runtime API is set, driver API failed but may work
            LOG_WARN("[CUDABackend::setDevice] cuDeviceGet failed for device " << device_id << " error=" << cu_err);
            return true; // cudaSetDevice succeeded, return true
        }

        CUcontext ctx;
        cu_err = cuDevicePrimaryCtxRetain(&ctx, cu_device);
        if (cu_err != CUDA_SUCCESS)
        {
            LOG_WARN("[CUDABackend::setDevice] cuDevicePrimaryCtxRetain failed for device " << device_id << " error=" << cu_err);
            return true; // cudaSetDevice succeeded
        }

        cu_err = cuCtxSetCurrent(ctx);
        if (cu_err != CUDA_SUCCESS)
        {
            LOG_WARN("[CUDABackend::setDevice] cuCtxSetCurrent failed for device " << device_id << " error=" << cu_err);
        }
        else
        {
            LOG_DEBUG("[CUDABackend::setDevice] Successfully set context for device " << device_id);
        }

        return true;
    }

    // ====================================================================
    // Memory Allocation Operations
    // ====================================================================

    void *CUDABackend::allocate(size_t bytes, int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_ERROR("[CUDABackend] Invalid device ID " << device_id << " (max: " << device_count_ - 1 << ")");
            return nullptr;
        }

        // Set device before allocation
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend] Failed to set device " << device_id << ": " << cudaGetErrorString(err));
            return nullptr;
        }

        void *ptr = nullptr;
        err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend] cudaMalloc failed for " << bytes << " bytes on device "
                                                             << device_id << ": " << cudaGetErrorString(err));
            return nullptr;
        }

        return ptr;
    }

    void CUDABackend::free(void *ptr, int device_id)
    {
        if (ptr == nullptr)
        {
            return; // Freeing nullptr is a no-op
        }

        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_ERROR("[CUDABackend] Invalid device ID " << device_id << " for cudaFree");
            return;
        }

        // Set device before freeing
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend] Failed to set device " << device_id << " before cudaFree: "
                                                            << cudaGetErrorString(err));
            return;
        }

        err = cudaFree(ptr);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend] cudaFree failed for ptr=" << std::hex << ptr << std::dec 
                      << " on device " << device_id << ": " << cudaGetErrorString(err));
        }
    }

    bool CUDABackend::memset(void *ptr, int value, size_t bytes, int device_id)
    {
        if (ptr == nullptr || bytes == 0)
        {
            return true; // No-op for null pointer or zero bytes
        }

        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_ERROR("[CUDABackend] Invalid device ID " << device_id << " for cudaMemset");
            return false;
        }

        // Set device before memset
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend] Failed to set device " << device_id << " before cudaMemset: "
                                                            << cudaGetErrorString(err));
            return false;
        }

        err = cudaMemset(ptr, value, bytes);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend] cudaMemset failed: " << cudaGetErrorString(err));
            return false;
        }

        return true;
    }

    void *CUDABackend::allocateMapped(size_t bytes, int device_id, void **device_ptr)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_ERROR("[CUDABackend] Invalid device ID " << device_id << " for allocateMapped");
            if (device_ptr)
                *device_ptr = nullptr;
            return nullptr;
        }

        // Set device before allocation
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend] Failed to set device " << device_id << ": " << cudaGetErrorString(err));
            if (device_ptr)
                *device_ptr = nullptr;
            return nullptr;
        }

        // Allocate mapped host memory (GPU can write directly to this via PCIe)
        void *host_ptr = nullptr;
        err = cudaHostAlloc(&host_ptr, bytes, cudaHostAllocMapped | cudaHostAllocWriteCombined);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend] cudaHostAlloc(Mapped) failed for " << bytes << " bytes on device "
                                                                        << device_id << ": " << cudaGetErrorString(err));
            if (device_ptr)
                *device_ptr = nullptr;
            return nullptr;
        }

        // Get the device-visible pointer for this mapped host memory
        if (device_ptr)
        {
            err = cudaHostGetDevicePointer(device_ptr, host_ptr, 0);
            if (err != cudaSuccess)
            {
                LOG_ERROR("[CUDABackend] cudaHostGetDevicePointer failed: " << cudaGetErrorString(err));
                cudaFreeHost(host_ptr);
                *device_ptr = nullptr;
                return nullptr;
            }
            LOG_DEBUG("[CUDABackend] allocateMapped: " << bytes << " bytes, host_ptr=" << host_ptr
                                                       << ", device_ptr=" << *device_ptr);
        }

        return host_ptr;
    }

    void CUDABackend::freeMapped(void *host_ptr, int device_id)
    {
        if (host_ptr == nullptr)
        {
            return; // Freeing nullptr is a no-op
        }

        // cudaFreeHost doesn't require setting device, but we do it for consistency
        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_WARN("[CUDABackend] Invalid device ID " << device_id << " for freeMapped, attempting anyway");
        }
        else
        {
            cudaSetDevice(device_id); // Best effort
        }

        cudaError_t err = cudaFreeHost(host_ptr);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend] cudaFreeHost failed: " << cudaGetErrorString(err));
        }
    }

    // ====================================================================
    // Device Query Operations
    // ====================================================================

    int CUDABackend::deviceCount() const
    {
        return device_count_;
    }

    std::string CUDABackend::backendName() const
    {
        return "CUDA";
    }

    std::string CUDABackend::deviceName(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return "Invalid Device";
        }

        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
        if (err != cudaSuccess)
        {
            return "Unknown Device";
        }

        return std::string(prop.name);
    }

    size_t CUDABackend::deviceMemoryTotal(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return 0;
        }

        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
        if (err != cudaSuccess)
        {
            return 0;
        }

        return prop.totalGlobalMem;
    }

    size_t CUDABackend::deviceMemoryFree(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return 0;
        }

        cudaError_t err_set = cudaSetDevice(device_id);
        if (err_set != cudaSuccess)
        {
            return 0;
        }

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err != cudaSuccess)
        {
            return 0;
        }

        return free_bytes;
    }

    // ====================================================================
    // Capability Queries
    // ====================================================================

    bool CUDABackend::supportsBF16(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
        if (err != cudaSuccess)
        {
            return false;
        }

        // BF16 support requires compute capability >= 8.0 (Ampere and later)
        int compute_capability = prop.major * 10 + prop.minor;
        return compute_capability >= 80;
    }

    bool CUDABackend::supportsFP16(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
        if (err != cudaSuccess)
        {
            return false;
        }

        // FP16 support requires compute capability >= 5.3 (Maxwell and later)
        int compute_capability = prop.major * 10 + prop.minor;
        return compute_capability >= 53;
    }

    bool CUDABackend::supportsINT8(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
        if (err != cudaSuccess)
        {
            return false;
        }

        // INT8 support requires compute capability >= 6.1 (Pascal and later)
        int compute_capability = prop.major * 10 + prop.minor;
        return compute_capability >= 61;
    }

    // ====================================================================
    // Compute Operations
    // ====================================================================

    bool CUDABackend::gemmIQ4NL(
        const void * /*A_device*/,
        const void * /*B_device*/,
        void * /*C_device*/,
        int /*m*/,
        int /*n*/,
        int /*k*/,
        int /*device_id*/)
    {
        // IQ4_NL GEMM via backend is deprecated - use kernel interface directly
        LOG_ERROR("CUDABackend::gemmIQ4NL is deprecated and no longer implemented");
        return false;
    }

} // namespace llaminar2
