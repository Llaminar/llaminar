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
#include "../../kernels/cuda/IQ4_NL_BlockDecoder.h"
#include "../../kernels/cuda/CudaGemmVariantsBaseline.h"
#include "../../kernels/cuda/CudaGemmAutoTuner.h"
#include <cuda_runtime.h>
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

        cudaError_t err_set = cudaSetDevice(device_id);
        if (err_set != cudaSuccess)
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

        cudaError_t err_set = cudaSetDevice(device_id);
        if (err_set != cudaSuccess)
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

        cudaError_t err_set = cudaSetDevice(device_id);
        if (err_set != cudaSuccess)
        {
            return false;
        }

        cudaError_t err = cudaDeviceSynchronize();
        return (err == cudaSuccess);
    }

    bool CUDABackend::setDevice(int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        cudaError_t err = cudaSetDevice(device_id);
        return (err == cudaSuccess);
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
            LOG_ERROR("[CUDABackend] cudaFree failed: " << cudaGetErrorString(err));
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
        const void *A_device,
        const void *B_device,
        void *C_device,
        int m,
        int n,
        int k,
        int device_id)
    {
        // Validate device ID
        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_ERROR("CUDABackend::gemmIQ4NL - Invalid device ID: " << device_id);
            return false;
        }

        // Set device
        cudaError_t err_set = cudaSetDevice(device_id);
        if (err_set != cudaSuccess)
        {
            LOG_ERROR("CUDABackend::gemmIQ4NL - Failed to set device " << device_id
                                                                       << ": " << cudaGetErrorString(err_set));
            return false;
        }

        // Cast to typed pointers
        const float *A = static_cast<const float *>(A_device);
        const cuda::IQ4_NLBlock *B = static_cast<const cuda::IQ4_NLBlock *>(B_device);
        float *C = static_cast<float *>(C_device);

        // Get optimal tile configuration from auto-tuner
        auto &autotuner = cuda::CudaGemmAutoTuner::instance();
        auto config = autotuner.getOptimalConfig(m, n, k);

        // Launch optimized variant kernel (uses default CUDA stream)
        cudaError_t err = cuda::launchIQ4NLGemmVariant(
            A, B, C, m, n, k, config, 0);

        if (err != cudaSuccess)
        {
            LOG_ERROR("CUDABackend::gemmIQ4NL - Kernel launch failed: "
                      << cudaGetErrorString(err));
            return false;
        }

        // Synchronize to catch kernel execution errors
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            LOG_ERROR("CUDABackend::gemmIQ4NL - Kernel execution failed: "
                      << cudaGetErrorString(err));
            return false;
        }

        return true;
    }

} // namespace llaminar2
