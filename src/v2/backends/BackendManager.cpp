/**
 * @file BackendManager.cpp
 * @brief Global GPU backend accessor implementation
 *
 * @author David Sanftenberg
 */

#include "BackendManager.h"
#include "../utils/Logger.h"

#ifdef HAVE_CUDA
#include "cuda/CUDABackend.h"
#endif

#ifdef HAVE_ROCM
#include "rocm/ROCmBackend.h"
#endif

#include <mutex>

namespace llaminar2
{

    namespace
    {
        IBackend *g_gpu_backend = nullptr;
        std::once_flag g_backend_init_flag;

        void initBackend()
        {
#ifdef HAVE_CUDA
            g_gpu_backend = new CUDABackend();
            LOG_INFO("[BackendManager] Initialized CUDA backend (" << g_gpu_backend->deviceCount() << " devices)");
#elif defined(HAVE_ROCM)
            g_gpu_backend = new ROCmBackend();
            LOG_INFO("[BackendManager] Initialized ROCm backend (" << g_gpu_backend->deviceCount() << " devices)");
#else
            g_gpu_backend = nullptr;
            LOG_DEBUG("[BackendManager] No GPU backend available (CPU-only build)");
#endif
        }
    } // anonymous namespace

    IBackend *getGPUBackend()
    {
        std::call_once(g_backend_init_flag, initBackend);
        return g_gpu_backend;
    }

    bool hasGPUBackend()
    {
        return getGPUBackend() != nullptr;
    }

} // namespace llaminar2
