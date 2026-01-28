/**
 * @file MPIStager.cpp
 * @brief MPI host staging implementation
 *
 * **Phase 3 Update**: Refactored to use IBackend interface instead of direct
 * CUDA/ROCm headers. Eliminates header conflicts between cuda_runtime.h and
 * hip_runtime.h.
 *
 * @author David Sanftenberg
 */

#include "MPIStager.h"
#include "Logger.h"
#include <cstring> // memcpy
#include <stdexcept>

// Backend interface (no GPU headers exposed)
#include "../backends/IBackend.h"

// Conditional backend includes (separate compilation units)
#ifdef HAVE_CUDA
#include "../backends/cuda/CUDABackend.h"
#endif

#ifdef HAVE_ROCM
#include "../backends/rocm/ROCmBackend.h"
#endif

namespace llaminar2
{
    // ========================================================================
    // Global backend instance (initialized at first use)
    // ========================================================================

    namespace
    {
        IBackend *g_gpu_backend = nullptr;
        bool g_backend_initialized = false;

        IBackend *getBackend()
        {
            if (!g_backend_initialized)
            {
#ifdef HAVE_CUDA
                g_gpu_backend = new CUDABackend();
                LOG_INFO("[MPIStager] Initialized CUDA backend (" << g_gpu_backend->deviceCount() << " devices)");
#elif defined(HAVE_ROCM)
                g_gpu_backend = new ROCmBackend();
                LOG_INFO("[MPIStager] Initialized ROCm backend (" << g_gpu_backend->deviceCount() << " devices)");
#else
                g_gpu_backend = nullptr;
                LOG_DEBUG("[MPIStager] No GPU backend available (CPU-only mode)");
#endif
                g_backend_initialized = true;
            }
            return g_gpu_backend;
        }
    } // anonymous namespace

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

        DeviceId home_device = tensor->home_device();

        if (home_device.is_cpu())
        {
            // CPU tensor - direct memcpy
            std::memcpy(host_buffer.data(), tensor->data(), numel * sizeof(float));
            LOG_TRACE("[MPIStager] toHost: CPU tensor, direct copy (" << numel << " elements)");
        }
        else
        {
            // GPU tensor - device-to-host transfer
            // CRITICAL: Use active_data_ptr() to get the GPU pointer directly,
            // NOT data() which would sync GPU→Host and return the host pointer!
            int device_id = home_device.toKernelDeviceIndex();
            const float *gpu_ptr = static_cast<const float *>(tensor->active_data_ptr());
            LOG_DEBUG("[MPIStager] toHost: GPU tensor (" << home_device.toString() << "), staging " << numel << " elements");
            synchronizeDevice(device_id);
            deviceToHost(host_buffer.data(), gpu_ptr, numel, device_id);
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
            throw std::invalid_argument("[MPIStager] toDevice: buffer size mismatch (host=" + std::to_string(host_buffer.size()) + ", tensor=" + std::to_string(numel) + ")");
        }

        DeviceId home_device = tensor->home_device();

        if (home_device.is_cpu())
        {
            // CPU tensor - direct memcpy
            std::memcpy(tensor->mutable_data(), host_buffer.data(), numel * sizeof(float));
            LOG_TRACE("[MPIStager] toDevice: CPU tensor, direct copy (" << numel << " elements)");
        }
        else
        {
            // GPU tensor - host-to-device transfer
            // CRITICAL: Use active_mutable_data_ptr() to get the GPU pointer directly,
            // NOT mutable_data() which would return the host pointer!
            int device_id = home_device.toKernelDeviceIndex();
            float *gpu_ptr = static_cast<float *>(tensor->active_mutable_data_ptr());
            LOG_DEBUG("[MPIStager] toDevice: GPU tensor (" << home_device.toString() << "), staging " << numel << " elements");
            hostToDevice(gpu_ptr, host_buffer.data(), numel, device_id);
            synchronizeDevice(device_id);
            // Mark device data as authoritative after H2D transfer (with event for fine-grained sync)
            tensor->mark_device_dirty_with_event();
        }
    }

    bool MPIStager::requiresStaging(const TensorBase *tensor)
    {
        // GPU tensors require staging (host<->device transfer) for MPI
        return tensor && tensor->home_device().is_gpu();
    }

    void MPIStager::synchronizeDevice(int device_id)
    {
        if (device_id < 0)
        {
            return; // CPU device - no sync needed
        }

        IBackend *backend = getBackend();
        if (!backend)
        {
            LOG_WARN("[MPIStager] synchronizeDevice called but no GPU backend available (device_id=" << device_id << ")");
            return;
        }

        if (!backend->synchronize(device_id))
        {
            LOG_ERROR("[MPIStager] " << backend->backendName() << " synchronize failed (device " << device_id << ")");
            throw std::runtime_error("GPU synchronize failed");
        }

        LOG_TRACE("[MPIStager] " << backend->backendName() << " device " << device_id << " synchronized");
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
    // Private: GPU memcpy wrappers (using backend interface)
    // ========================================================================

    void MPIStager::deviceToHost(float *dst, const float *src, size_t count, int device_id)
    {
        IBackend *backend = getBackend();
        if (!backend)
        {
            LOG_ERROR("[MPIStager] deviceToHost called but no GPU backend available");
            throw std::runtime_error("No GPU backend available for staging");
        }

        size_t bytes = count * sizeof(float);
        if (!backend->deviceToHost(dst, src, bytes, device_id))
        {
            LOG_ERROR("[MPIStager] " << backend->backendName() << " D2H memcpy failed (device " << device_id << ")");
            throw std::runtime_error("GPU D2H memcpy failed");
        }

        LOG_TRACE("[MPIStager] " << backend->backendName() << " D2H: copied " << count << " floats ("
                                 << (bytes / 1024.0 / 1024.0) << " MB)");
    }

    void MPIStager::hostToDevice(float *dst, const float *src, size_t count, int device_id)
    {
        IBackend *backend = getBackend();
        if (!backend)
        {
            LOG_ERROR("[MPIStager] hostToDevice called but no GPU backend available");
            throw std::runtime_error("No GPU backend available for staging");
        }

        size_t bytes = count * sizeof(float);
        if (!backend->hostToDevice(dst, src, bytes, device_id))
        {
            LOG_ERROR("[MPIStager] " << backend->backendName() << " H2D memcpy failed (device " << device_id << ")");
            throw std::runtime_error("GPU H2D memcpy failed");
        }

        LOG_TRACE("[MPIStager] " << backend->backendName() << " H2D: copied " << count << " floats ("
                                 << (bytes / 1024.0 / 1024.0) << " MB)");
    }

} // namespace llaminar2
