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
#include "../kernels/rocm/HipBLASGemmKernel.h" // For registerHipBLASGemmKernelFactory
#include "../kernels/rocm/ROCmQuantisedGemmKernel.h" // For rocmQuantGemm_warmupKernels
#endif

#include <mutex>
#include <atomic>

namespace llaminar2
{

    namespace
    {
        // Phase 6: Support CPU, CUDA, and ROCm backends simultaneously
        IBackend *g_cpu_backend = nullptr;
        IBackend *g_cuda_backend = nullptr;
        IBackend *g_rocm_backend = nullptr;

        std::once_flag g_cpu_init_flag;
        std::once_flag g_cuda_init_flag;
        std::once_flag g_rocm_init_flag;

        // CPU backend requires explicit initialization with NUMA node
        std::atomic<int> g_cpu_numa_node{-1};
        std::atomic<bool> g_cpu_init_requested{false};

        void initCPUBackendImpl()
        {
            int numa_node = g_cpu_numa_node.load();
            g_cpu_backend = new CPUBackend(numa_node);
            LOG_INFO("[BackendManager] Initialized CPU backend (NUMA node: "
                     << numa_node << ", memory: "
                     << (g_cpu_backend->deviceMemoryTotal(0) / (1024 * 1024)) << " MB)");
        }

        void initCUDABackend()
        {
#ifdef HAVE_CUDA
            g_cuda_backend = new CUDABackend();
            LOG_INFO("[BackendManager] Initialized CUDA backend (" << g_cuda_backend->deviceCount() << " devices)");
#else
            g_cuda_backend = nullptr;
            LOG_DEBUG("[BackendManager] CUDA backend not available (HAVE_CUDA not defined)");
#endif
        }

        void initROCmBackend()
        {
#ifdef HAVE_ROCM
            g_rocm_backend = new ROCmBackend();
            LOG_INFO("[BackendManager] Initialized ROCm backend (" << g_rocm_backend->deviceCount() << " devices)");

            // Register hipBLAS GEMM kernel factory for DeviceKernelCache
            rocm::registerHipBLASGemmKernelFactory();
            
            // Pre-initialize CK INT8 GEMM kernels to avoid first-call latency
            // This takes ~15-20 seconds on gfx906 but happens at startup, not during inference
            rocmQuantGemm_warmupKernels();
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
        g_cpu_numa_node.store(local_numa_node);
        g_cpu_init_requested.store(true);
        std::call_once(g_cpu_init_flag, initCPUBackendImpl);
    }

    IBackend *getCPUBackend()
    {
        if (!g_cpu_init_requested.load())
        {
            // Auto-initialize with NUMA node -1 (system-wide) if not explicitly initialized
            initCPUBackend(-1);
        }
        return g_cpu_backend;
    }

    bool hasCPUBackend()
    {
        return getCPUBackend() != nullptr;
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
