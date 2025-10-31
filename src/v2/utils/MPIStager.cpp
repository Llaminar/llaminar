/**
 * @file MPIStager.cpp
 * @brief MPI host staging implementation
 *
 * @author David Sanftenberg
 */

#include "MPIStager.h"
#include "Logger.h"
#include <cstring> // memcpy
#include <stdexcept>

// Conditional GPU includes (only when backends enabled)
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace llaminar2
{

    // ========================================================================
    // Public API: FP32 staging
    // ========================================================================

    std::vector<float> MPIStager::toHost(const TensorBase *tensor)
    {
        if (!tensor)
        {
            throw std::invalid_argument("[MPIStager] toHost: null tensor");
        }

        // Calculate element count from shape
        const auto &shape = tensor->shape();
        size_t numel = 1;
        for (auto dim : shape)
        {
            numel *= dim;
        }

        std::vector<float> host_buffer(numel);

        int device_id = tensor->device_index();

        if (device_id < 0)
        {
            // CPU tensor - direct memcpy
            std::memcpy(host_buffer.data(), tensor->data(), numel * sizeof(float));
            LOG_TRACE("[MPIStager] toHost: CPU tensor, direct copy (" << numel << " elements)");
        }
        else
        {
            // GPU tensor - device-to-host transfer
            LOG_DEBUG("[MPIStager] toHost: GPU tensor (device " << device_id << "), staging " << numel << " elements");
            synchronizeDevice(device_id);
            deviceToHost(host_buffer.data(), tensor->data(), numel, device_id);
        }

        return host_buffer;
    }

    void MPIStager::toDevice(const std::vector<float> &host_buffer, TensorBase *tensor)
    {
        if (!tensor)
        {
            throw std::invalid_argument("[MPIStager] toDevice: null tensor");
        }

        // Calculate element count from shape
        const auto &shape = tensor->shape();
        size_t numel = 1;
        for (auto dim : shape)
        {
            numel *= dim;
        }

        if (host_buffer.size() != numel)
        {
            throw std::invalid_argument("[MPIStager] toDevice: buffer size mismatch (host="
                                        + std::to_string(host_buffer.size()) + ", tensor=" + std::to_string(numel) + ")");
        }

        int device_id = tensor->device_index();

        if (device_id < 0)
        {
            // CPU tensor - direct memcpy
            std::memcpy(tensor->mutable_data(), host_buffer.data(), numel * sizeof(float));
            LOG_TRACE("[MPIStager] toDevice: CPU tensor, direct copy (" << numel << " elements)");
        }
        else
        {
            // GPU tensor - host-to-device transfer
            LOG_DEBUG("[MPIStager] toDevice: GPU tensor (device " << device_id << "), staging " << numel << " elements");
            hostToDevice(tensor->mutable_data(), host_buffer.data(), numel, device_id);
            synchronizeDevice(device_id);
        }
    }

    bool MPIStager::requiresStaging(const TensorBase *tensor)
    {
        return tensor && tensor->device_index() >= 0;
    }

    void MPIStager::synchronizeDevice(int device_id)
    {
        if (device_id < 0)
        {
            return; // CPU device - no sync needed
        }

#ifdef HAVE_CUDA
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            LOG_ERROR("[MPIStager] CUDA synchronize failed: " << cudaGetErrorString(err));
            throw std::runtime_error("CUDA synchronize failed");
        }
        LOG_TRACE("[MPIStager] CUDA device synchronized");
#elif defined(HAVE_ROCM)
        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess)
        {
            LOG_ERROR("[MPIStager] HIP synchronize failed: " << hipGetErrorString(err));
            throw std::runtime_error("HIP synchronize failed");
        }
        LOG_TRACE("[MPIStager] HIP device synchronized");
#else
        // No GPU backends compiled - should not reach here
        LOG_WARN("[MPIStager] synchronizeDevice called but no GPU backends available (device_id=" << device_id << ")");
#endif
    }

    // ========================================================================
    // BF16 staging (future - currently stubs)
    // ========================================================================

    std::vector<float> MPIStager::toHostBF16(const TensorBase *tensor)
    {
        // TODO: Implement BF16→FP32 conversion for MPI operations
        LOG_ERROR("[MPIStager] toHostBF16 not yet implemented");
        throw std::runtime_error("BF16 staging not implemented");
    }

    void MPIStager::toDeviceBF16(const std::vector<float> &host_buffer, TensorBase *tensor)
    {
        // TODO: Implement FP32→BF16 conversion after MPI operations
        LOG_ERROR("[MPIStager] toDeviceBF16 not yet implemented");
        throw std::runtime_error("BF16 staging not implemented");
    }

    // ========================================================================
    // Private: GPU memcpy wrappers
    // ========================================================================

    void MPIStager::deviceToHost(float *dst, const float *src, size_t count, int device_id)
    {
#ifdef HAVE_CUDA
        cudaError_t err = cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[MPIStager] CUDA D2H memcpy failed: " << cudaGetErrorString(err));
            throw std::runtime_error("CUDA D2H memcpy failed");
        }
        LOG_TRACE("[MPIStager] CUDA D2H: copied " << count << " floats (" << (count * sizeof(float) / 1024.0 / 1024.0) << " MB)");
#elif defined(HAVE_ROCM)
        hipError_t err = hipMemcpy(dst, src, count * sizeof(float), hipMemcpyDeviceToHost);
        if (err != hipSuccess)
        {
            LOG_ERROR("[MPIStager] HIP D2H memcpy failed: " << hipGetErrorString(err));
            throw std::runtime_error("HIP D2H memcpy failed");
        }
        LOG_TRACE("[MPIStager] HIP D2H: copied " << count << " floats (" << (count * sizeof(float) / 1024.0 / 1024.0) << " MB)");
#else
        // No GPU backends - this should not be reached
        (void)device_id; // Silence unused warning
        LOG_ERROR("[MPIStager] deviceToHost called but no GPU backends available");
        throw std::runtime_error("No GPU backends available for staging");
#endif
    }

    void MPIStager::hostToDevice(float *dst, const float *src, size_t count, int device_id)
    {
#ifdef HAVE_CUDA
        cudaError_t err = cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[MPIStager] CUDA H2D memcpy failed: " << cudaGetErrorString(err));
            throw std::runtime_error("CUDA H2D memcpy failed");
        }
        LOG_TRACE("[MPIStager] CUDA H2D: copied " << count << " floats (" << (count * sizeof(float) / 1024.0 / 1024.0) << " MB)");
#elif defined(HAVE_ROCM)
        hipError_t err = hipMemcpy(dst, src, count * sizeof(float), hipMemcpyHostToDevice);
        if (err != hipSuccess)
        {
            LOG_ERROR("[MPIStager] HIP H2D memcpy failed: " << hipGetErrorString(err));
            throw std::runtime_error("HIP H2D memcpy failed");
        }
        LOG_TRACE("[MPIStager] HIP H2D: copied " << count << " floats (" << (count * sizeof(float) / 1024.0 / 1024.0) << " MB)");
#else
        // No GPU backends - this should not be reached
        (void)device_id; // Silence unused warning
        LOG_ERROR("[MPIStager] hostToDevice called but no GPU backends available");
        throw std::runtime_error("No GPU backends available for staging");
#endif
    }

} // namespace llaminar2
