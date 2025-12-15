/**
 * @file DeviceContext.cpp
 * @brief Implementation of device execution contexts
 * @author David Sanftenberg
 * @date December 2025
 */

#include "DeviceContext.h"
#include "../utils/Logger.h"
#include <omp.h>
#include <cstring>
#include <cstdlib>

#ifdef __linux__
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

namespace llaminar2
{

    // =============================================================================
    // Factory
    // =============================================================================

    std::unique_ptr<IDeviceContext> IDeviceContext::create(int device_idx, int num_threads)
    {
        auto &dm = DeviceManager::instance();
        const auto &devices = dm.devices();

        if (device_idx < 0 || static_cast<size_t>(device_idx) >= devices.size())
        {
            LOG_ERROR("IDeviceContext::create: invalid device_idx " << device_idx);
            return nullptr;
        }

        const auto &device = devices[device_idx];

        switch (device.type)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            return std::make_unique<CPUDeviceContext>(device_idx, num_threads);

        case ComputeBackendType::GPU_CUDA:
#ifdef HAVE_CUDA
            return std::make_unique<CUDADeviceContext>(device_idx, device.device_id);
#else
            LOG_ERROR("CUDA support not compiled in");
            return nullptr;
#endif

        case ComputeBackendType::GPU_ROCM:
#ifdef HAVE_ROCM
            return std::make_unique<ROCmDeviceContext>(device_idx, device.device_id);
#else
            LOG_ERROR("ROCm support not compiled in");
            return nullptr;
#endif

        default:
            LOG_ERROR("Unsupported device type: " << static_cast<int>(device.type));
            return nullptr;
        }
    }

    // =============================================================================
    // CPUDeviceContext
    // =============================================================================

    CPUDeviceContext::CPUDeviceContext(int device_idx, int num_threads)
        : device_idx_(device_idx), num_threads_(num_threads > 0 ? num_threads : omp_get_max_threads())
    {
        LOG_DEBUG("CPUDeviceContext created: device_idx=" << device_idx_
                                                          << ", threads=" << num_threads_);
    }

    CPUDeviceContext::~CPUDeviceContext()
    {
        // Workspace freed automatically via vector destructor
    }

    std::string CPUDeviceContext::deviceName() const
    {
        return "CPU (OpenMP, " + std::to_string(num_threads_) + " threads)";
    }

    void CPUDeviceContext::synchronize()
    {
        // CPU operations are synchronous - nothing to do
    }

    void CPUDeviceContext::barrier()
    {
        if (omp_in_parallel())
        {
#pragma omp barrier
        }
    }

    void *CPUDeviceContext::allocate(size_t bytes)
    {
        if (bytes == 0)
            return nullptr;

        // Use aligned allocation for SIMD efficiency
        constexpr size_t alignment = 64; // Cache line alignment
        void *ptr = nullptr;

#ifdef _WIN32
        ptr = _aligned_malloc(bytes, alignment);
#else
        if (posix_memalign(&ptr, alignment, bytes) != 0)
        {
            ptr = nullptr;
        }
#endif

        if (!ptr)
        {
            LOG_ERROR("CPUDeviceContext::allocate failed for " << bytes << " bytes");
        }

        return ptr;
    }

    void CPUDeviceContext::free(void *ptr)
    {
        if (!ptr)
            return;

#ifdef _WIN32
        _aligned_free(ptr);
#else
        ::free(ptr);
#endif
    }

    void *CPUDeviceContext::getWorkspace(size_t bytes)
    {
        if (bytes > workspace_size_)
        {
            // Grow workspace (with some extra room)
            size_t new_size = bytes + bytes / 4; // 25% headroom
            workspace_.resize(new_size);
            workspace_size_ = new_size;
            LOG_DEBUG("CPUDeviceContext: workspace grown to " << new_size << " bytes");
        }

        return workspace_.data();
    }

    size_t CPUDeviceContext::availableMemory() const
    {
#ifdef __linux__
        struct sysinfo info;
        if (sysinfo(&info) == 0)
        {
            return info.freeram * info.mem_unit;
        }
#endif
        // Fallback: assume 4GB available
        return 4ULL * 1024 * 1024 * 1024;
    }

    size_t CPUDeviceContext::totalMemory() const
    {
#ifdef __linux__
        struct sysinfo info;
        if (sysinfo(&info) == 0)
        {
            return info.totalram * info.mem_unit;
        }
#endif
        // Fallback: assume 16GB total
        return 16ULL * 1024 * 1024 * 1024;
    }

    bool CPUDeviceContext::copyToDevice(void *dst, const void *src, size_t bytes)
    {
        if (!dst || !src || bytes == 0)
            return false;
        std::memcpy(dst, src, bytes);
        return true;
    }

    bool CPUDeviceContext::copyToHost(void *dst, const void *src, size_t bytes)
    {
        if (!dst || !src || bytes == 0)
            return false;
        std::memcpy(dst, src, bytes);
        return true;
    }

    bool CPUDeviceContext::copyFromDevice(void *dst, const void *src, size_t bytes,
                                          IDeviceContext *src_ctx)
    {
        if (!dst || !src || bytes == 0)
            return false;

        if (src_ctx && src_ctx->isGPU())
        {
            // Source is GPU - need to copy through host
            return src_ctx->copyToHost(dst, src, bytes);
        }

        // Source is CPU - direct memcpy
        std::memcpy(dst, src, bytes);
        return true;
    }

    void CPUDeviceContext::runParallel(std::function<void(int, int)> work)
    {
        if (inParallelRegion())
        {
            // Already in parallel region - just call with current thread info
            work(omp_get_thread_num(), omp_get_num_threads());
        }
        else
        {
// Create new parallel region
#pragma omp parallel num_threads(num_threads_)
            {
                work(omp_get_thread_num(), omp_get_num_threads());
            }
        }
    }

    void CPUDeviceContext::runFor(size_t start, size_t end, std::function<void(size_t)> work)
    {
        if (start >= end)
            return;

        if (inParallelRegion())
        {
// Use worksharing in existing region
#pragma omp for schedule(static)
            for (size_t i = start; i < end; ++i)
            {
                work(i);
            }
        }
        else
        {
// Create parallel for
#pragma omp parallel for schedule(static) num_threads(num_threads_)
            for (size_t i = start; i < end; ++i)
            {
                work(i);
            }
        }
    }

    bool CPUDeviceContext::inParallelRegion() const
    {
        return omp_in_parallel() != 0;
    }

    // =============================================================================
    // IGPUDeviceContext (Base for CUDA and ROCm)
    // =============================================================================

    IGPUDeviceContext::IGPUDeviceContext(int device_idx, int gpu_device_id, ComputeBackendType backend_type)
        : device_idx_(device_idx), gpu_device_id_(gpu_device_id), backend_type_(backend_type)
    {
        LOG_DEBUG("IGPUDeviceContext created: device_idx=" << device_idx_
                                                           << ", gpu_device_id=" << gpu_device_id_
                                                           << ", backend=" << static_cast<int>(backend_type_));
    }

    IGPUDeviceContext::~IGPUDeviceContext()
    {
        // Workspace cleanup is handled by derived class destructors
    }

    void *IGPUDeviceContext::getWorkspace(size_t bytes)
    {
        if (bytes > workspace_size_)
        {
            // Free old workspace
            if (workspace_)
            {
                free(workspace_);
                workspace_ = nullptr;
            }

            // Allocate new workspace with headroom
            size_t new_size = bytes + bytes / 4; // 25% headroom
            workspace_ = allocate(new_size);
            if (workspace_)
            {
                workspace_size_ = new_size;
                LOG_DEBUG("IGPUDeviceContext[" << gpu_device_id_ << "]: workspace grown to " << new_size << " bytes");
            }
            else
            {
                workspace_size_ = 0;
                LOG_ERROR("IGPUDeviceContext: failed to allocate workspace of " << new_size << " bytes");
            }
        }

        return workspace_;
    }

    void IGPUDeviceContext::runParallel(std::function<void(int, int)> work)
    {
        // GPU parallelism is implicit in kernel launches
        // Just call the work function as single "thread"
        work(0, 1);
    }

    void IGPUDeviceContext::runFor(size_t start, size_t end, std::function<void(size_t)> work)
    {
        // GPU parallelism is implicit in kernel launches
        // Just run sequentially (kernels handle parallelism)
        for (size_t i = start; i < end; ++i)
        {
            work(i);
        }
    }

    // =============================================================================
    // CUDADeviceContext
    // =============================================================================

#ifdef HAVE_CUDA

} // temporarily close namespace llaminar2 for include

#include "../backends/cuda/CUDABackend.h"

namespace llaminar2
{ // reopen namespace

    // Static helper to get backend singleton
    static CUDABackend &getCUDABackend()
    {
        static CUDABackend instance;
        return instance;
    }

    CUDADeviceContext::CUDADeviceContext(int device_idx, int cuda_device_id)
        : IGPUDeviceContext(device_idx, cuda_device_id, ComputeBackendType::GPU_CUDA)
    {
        // Set device to ensure proper context
        auto &backend = getCUDABackend();
        if (!backend.setDevice(cuda_device_id))
        {
            LOG_ERROR("CUDADeviceContext: failed to set CUDA device " << cuda_device_id);
        }
        LOG_INFO("CUDADeviceContext created: " << backend.deviceName(cuda_device_id));
    }

    CUDADeviceContext::~CUDADeviceContext()
    {
        auto &backend = getCUDABackend();
        backend.setDevice(gpu_device_id_);

        // Free workspace on this device
        if (workspace_)
        {
            backend.free(workspace_, gpu_device_id_);
            workspace_ = nullptr;
            workspace_size_ = 0;
        }
    }

    std::string CUDADeviceContext::deviceName() const
    {
        return getCUDABackend().deviceName(gpu_device_id_);
    }

    void CUDADeviceContext::synchronize()
    {
        getCUDABackend().synchronize(gpu_device_id_);
    }

    void CUDADeviceContext::barrier()
    {
        // For GPU, barrier is same as synchronize
        synchronize();
    }

    void *CUDADeviceContext::allocate(size_t bytes)
    {
        if (bytes == 0)
            return nullptr;
        return getCUDABackend().allocate(bytes, gpu_device_id_);
    }

    void CUDADeviceContext::free(void *ptr)
    {
        if (ptr)
        {
            getCUDABackend().free(ptr, gpu_device_id_);
        }
    }

    size_t CUDADeviceContext::availableMemory() const
    {
        return getCUDABackend().deviceMemoryFree(gpu_device_id_);
    }

    size_t CUDADeviceContext::totalMemory() const
    {
        return getCUDABackend().deviceMemoryTotal(gpu_device_id_);
    }

    bool CUDADeviceContext::copyToDevice(void *dst, const void *src, size_t bytes)
    {
        if (!dst || !src || bytes == 0)
            return false;
        return getCUDABackend().hostToDevice(dst, src, bytes, gpu_device_id_);
    }

    bool CUDADeviceContext::copyToHost(void *dst, const void *src, size_t bytes)
    {
        if (!dst || !src || bytes == 0)
            return false;
        return getCUDABackend().deviceToHost(dst, src, bytes, gpu_device_id_);
    }

    bool CUDADeviceContext::copyFromDevice(void *dst, const void *src, size_t bytes,
                                           IDeviceContext *src_ctx)
    {
        if (!dst || !src || bytes == 0)
            return false;

        if (src_ctx && src_ctx->isGPU())
        {
            // GPU-to-GPU copy
            // For now, go through host (TODO: use peer-to-peer or unified memory)
            std::vector<char> temp(bytes);
            if (!src_ctx->copyToHost(temp.data(), src, bytes))
            {
                return false;
            }
            return copyToDevice(dst, temp.data(), bytes);
        }

        // CPU-to-GPU copy
        return copyToDevice(dst, src, bytes);
    }

#endif // HAVE_CUDA

    // =============================================================================
    // ROCmDeviceContext
    // =============================================================================

#ifdef HAVE_ROCM

} // temporarily close namespace llaminar2 for include

#include "../backends/rocm/ROCmBackend.h"

namespace llaminar2
{ // reopen namespace

    // Static helper to get backend singleton
    static ROCmBackend &getROCmBackend()
    {
        static ROCmBackend instance;
        return instance;
    }

    ROCmDeviceContext::ROCmDeviceContext(int device_idx, int hip_device_id)
        : IGPUDeviceContext(device_idx, hip_device_id, ComputeBackendType::GPU_ROCM)
    {
        // Set device to ensure proper context
        auto &backend = getROCmBackend();
        if (!backend.setDevice(hip_device_id))
        {
            LOG_ERROR("ROCmDeviceContext: failed to set HIP device " << hip_device_id);
        }
        LOG_INFO("ROCmDeviceContext created: " << backend.deviceName(hip_device_id));
    }

    ROCmDeviceContext::~ROCmDeviceContext()
    {
        auto &backend = getROCmBackend();
        backend.setDevice(gpu_device_id_);

        // Free workspace on this device
        if (workspace_)
        {
            backend.free(workspace_, gpu_device_id_);
            workspace_ = nullptr;
            workspace_size_ = 0;
        }
    }

    std::string ROCmDeviceContext::deviceName() const
    {
        return getROCmBackend().deviceName(gpu_device_id_);
    }

    void ROCmDeviceContext::synchronize()
    {
        getROCmBackend().synchronize(gpu_device_id_);
    }

    void ROCmDeviceContext::barrier()
    {
        // For GPU, barrier is same as synchronize
        synchronize();
    }

    void *ROCmDeviceContext::allocate(size_t bytes)
    {
        if (bytes == 0)
            return nullptr;
        return getROCmBackend().allocate(bytes, gpu_device_id_);
    }

    void ROCmDeviceContext::free(void *ptr)
    {
        if (ptr)
        {
            getROCmBackend().free(ptr, gpu_device_id_);
        }
    }

    size_t ROCmDeviceContext::availableMemory() const
    {
        return getROCmBackend().deviceMemoryFree(gpu_device_id_);
    }

    size_t ROCmDeviceContext::totalMemory() const
    {
        return getROCmBackend().deviceMemoryTotal(gpu_device_id_);
    }

    bool ROCmDeviceContext::copyToDevice(void *dst, const void *src, size_t bytes)
    {
        if (!dst || !src || bytes == 0)
            return false;
        return getROCmBackend().hostToDevice(dst, src, bytes, gpu_device_id_);
    }

    bool ROCmDeviceContext::copyToHost(void *dst, const void *src, size_t bytes)
    {
        if (!dst || !src || bytes == 0)
            return false;
        return getROCmBackend().deviceToHost(dst, src, bytes, gpu_device_id_);
    }

    bool ROCmDeviceContext::copyFromDevice(void *dst, const void *src, size_t bytes,
                                           IDeviceContext *src_ctx)
    {
        if (!dst || !src || bytes == 0)
            return false;

        if (src_ctx && src_ctx->isGPU())
        {
            // GPU-to-GPU copy
            // For now, go through host (TODO: use peer-to-peer or unified memory)
            std::vector<char> temp(bytes);
            if (!src_ctx->copyToHost(temp.data(), src, bytes))
            {
                return false;
            }
            return copyToDevice(dst, temp.data(), bytes);
        }

        // CPU-to-GPU copy
        return copyToDevice(dst, src, bytes);
    }

#endif // HAVE_ROCM

} // namespace llaminar2
