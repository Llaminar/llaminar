/**
 * @file BackendManager.cpp
 * @brief Global backend accessor implementation (Phase 6: Heterogeneous Multi-GPU + CPU)
 *
 * Supports CPU, CUDA, and ROCm backends simultaneously for heterogeneous compute.
 *
 * @author David Sanftenberg
 */

#include "BackendManager.h"
#include "CPUBackend.h"
#include "../utils/Logger.h"

#ifdef HAVE_CUDA
#include "cuda/CUDABackend.h"
#endif

#ifdef HAVE_ROCM
#include "rocm/ROCmBackend.h"
#include "../kernels/rocm/gemm/HipBLASGemmKernel.h" // For registerHipBLASGemmKernelFactory
#endif

#include <mutex>
#include <atomic>
#include <stdexcept>

namespace llaminar2
{

    namespace
    {
        // Phase 6: Support CPU, CUDA, and ROCm backends simultaneously
        std::atomic<IBackend *> g_cpu_backend{nullptr};
        IBackend *g_cuda_backend = nullptr;
        IBackend *g_rocm_backend = nullptr;

        std::once_flag g_cuda_init_flag;
        std::once_flag g_rocm_init_flag;

        // Each MPI process owns one immutable rank-local CPU backend identity.
        std::mutex g_cpu_init_mutex;

        void initCUDABackend()
        {
#ifdef HAVE_CUDA
            g_cuda_backend = new CUDABackend();
            LOG_DEBUG("[BackendManager] Initialized CUDA backend (" << g_cuda_backend->deviceCount() << " devices)");
#else
            g_cuda_backend = nullptr;
            LOG_DEBUG("[BackendManager] CUDA backend not available (HAVE_CUDA not defined)");
#endif
        }

        void initROCmBackend()
        {
#ifdef HAVE_ROCM
            g_rocm_backend = new ROCmBackend();
            LOG_DEBUG("[BackendManager] Initialized ROCm backend (" << g_rocm_backend->deviceCount() << " devices)");

            // Register hipBLAS GEMM kernel factory for DeviceKernelCache
            rocm::registerHipBLASGemmKernelFactory();
#else
            g_rocm_backend = nullptr;
            LOG_DEBUG("[BackendManager] ROCm backend not available (HAVE_ROCM not defined)");
#endif
        }
    } // anonymous namespace

    IBackend *getCUDABackend()
    {
        std::call_once(g_cuda_init_flag, initCUDABackend);
        return g_cuda_backend;
    }

    IBackend *getROCmBackend()
    {
        std::call_once(g_rocm_init_flag, initROCmBackend);
        return g_rocm_backend;
    }

    IBackend *getGPUBackend()
    {
        // Legacy function - prefer CUDA, fall back to ROCm
        IBackend *cuda = getCUDABackend();
        if (cuda)
            return cuda;
        return getROCmBackend();
    }

    IBackend *getBackendForDeviceType(ComputeBackendType type)
    {
        switch (type)
        {
        case ComputeBackendType::GPU_CUDA:
            return getCUDABackend();
        case ComputeBackendType::GPU_ROCM:
            return getROCmBackend();
        default:
            return nullptr; // CPU or unknown types don't use IBackend
        }
    }

    bool hasGPUBackend()
    {
        return getGPUBackend() != nullptr;
    }

    bool hasCUDABackend()
    {
        return getCUDABackend() != nullptr;
    }

    bool hasROCmBackend()
    {
        return getROCmBackend() != nullptr;
    }

    // ====================================================================
    // CPU Backend
    // ====================================================================

    void initCPUBackend(int local_numa_node)
    {
        if (local_numa_node < -1)
        {
            throw std::invalid_argument(
                "CPU backend NUMA node must be -1 (aggregate) or non-negative");
        }

        std::lock_guard<std::mutex> lock(g_cpu_init_mutex);
        if (IBackend *existing = g_cpu_backend.load(std::memory_order_acquire))
        {
            auto *cpu = dynamic_cast<CPUBackend *>(existing);
            if (!cpu || cpu->numaNode() != local_numa_node)
            {
                throw std::logic_error(
                    "CPU backend was already initialized for a different NUMA domain");
            }
            return;
        }

        auto *backend = new CPUBackend(local_numa_node);
        g_cpu_backend.store(backend, std::memory_order_release);
        LOG_DEBUG("[BackendManager] Initialized CPU backend (NUMA node: "
                  << local_numa_node << ", memory: "
                  << (backend->deviceMemoryTotal(0) / (1024 * 1024)) << " MB)");
    }

    IBackend *getCPUBackend()
    {
        return g_cpu_backend.load(std::memory_order_acquire);
    }

    int cpuBackendNUMANode()
    {
        auto *backend = dynamic_cast<CPUBackend *>(getCPUBackend());
        if (!backend)
            return -1;
        return backend->numaNode();
    }

    bool hasCPUBackend()
    {
        return g_cpu_backend.load(std::memory_order_acquire) != nullptr;
    }

    // ====================================================================
    // Unified Backend Accessor
    // ====================================================================

    IBackend *getBackendFor(DeviceId device)
    {
        switch (device.type)
        {
        case DeviceType::CPU:
            return getCPUBackend();
        case DeviceType::CUDA:
            return getCUDABackend();
        case DeviceType::ROCm:
            return getROCmBackend();
        default:
            LOG_ERROR("[BackendManager] Unknown device type: " << device.toString());
            return nullptr;
        }
    }

} // namespace llaminar2
