/**
 * @file GPUDeviceContextPool.cpp
 * @brief Implementation of GPUDeviceContextPool singleton
 *
 * This file does NOT include any CUDA/HIP headers directly. Instead, it uses
 * factory functions registered by cuda_backend and rocm_backend libraries.
 * This keeps GPUDeviceContextPool in llaminar2_core without GPU dependencies.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "GPUDeviceContextPool.h"
#include "../utils/Logger.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace llaminar2
{

    // =============================================================================
    // Singleton Instance
    // =============================================================================

    GPUDeviceContextPool &GPUDeviceContextPool::instance()
    {
        // Meyers singleton - thread-safe in C++11 and later
        static GPUDeviceContextPool pool;
        return pool;
    }

    // =============================================================================
    // Destructor
    // =============================================================================

    GPUDeviceContextPool::~GPUDeviceContextPool()
    {
        shutdown();
    }

    // =============================================================================
    // Factory Registration
    // =============================================================================

    void GPUDeviceContextPool::registerNvidiaFactory(GPUContextFactory factory, int device_count)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        nvidia_factory_ = std::move(factory);
        cuda_device_count_ = device_count;

        LOG_INFO("[GPUDeviceContextPool] Registered NVIDIA factory with "
                 << device_count << " devices available");
    }

    void GPUDeviceContextPool::registerAMDFactory(GPUContextFactory factory, int device_count)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        amd_factory_ = std::move(factory);
        rocm_device_count_ = device_count;

        LOG_INFO("[GPUDeviceContextPool] Registered AMD factory with "
                 << device_count << " devices available");
    }

    // =============================================================================
    // Context Access
    // =============================================================================

    IWorkerGPUContext &GPUDeviceContextPool::getNvidiaContext(int device_ordinal)
    {
#ifdef HAVE_CUDA
        // Auto-register factory if not yet registered (same pattern as getContext(DeviceId))
        if (!hasNvidiaSupport())
        {
            ensureNvidiaFactoryRegistered();
        }
#endif

        std::lock_guard<std::mutex> lock(mutex_);

        if (!nvidia_factory_)
        {
            throw std::runtime_error(
                "[GPUDeviceContextPool] NVIDIA factory not registered. "
                "Ensure cuda_backend is linked and initialized.");
        }

        if (cuda_device_count_ == 0)
        {
            throw std::runtime_error("[GPUDeviceContextPool] No CUDA devices available");
        }

        if (device_ordinal < 0 || device_ordinal >= cuda_device_count_)
        {
            throw std::runtime_error("[GPUDeviceContextPool] Invalid CUDA device ordinal " + std::to_string(device_ordinal) + " (valid range: 0-" + std::to_string(cuda_device_count_ - 1) + ")");
        }

        // Check if context already exists
        auto it = nvidia_contexts_.find(device_ordinal);
        if (it != nvidia_contexts_.end())
        {
            return *it->second;
        }

        // Create new context via factory (lazy initialization)
        LOG_DEBUG("[GPUDeviceContextPool] Creating NvidiaDeviceContext for device " << device_ordinal);

        auto context = nvidia_factory_(device_ordinal);
        IWorkerGPUContext &ctx_ref = *context;
        nvidia_contexts_[device_ordinal] = std::move(context);

        return ctx_ref;
    }

    IWorkerGPUContext &GPUDeviceContextPool::getAMDContext(int device_ordinal)
    {
#ifdef HAVE_ROCM
        // Auto-register factory if not yet registered (same pattern as getContext(DeviceId))
        if (!hasAMDSupport())
        {
            ensureAMDFactoryRegistered();
        }
#endif

        std::lock_guard<std::mutex> lock(mutex_);

        if (!amd_factory_)
        {
            throw std::runtime_error(
                "[GPUDeviceContextPool] AMD factory not registered. "
                "Ensure rocm_backend is linked and initialized.");
        }

        if (rocm_device_count_ == 0)
        {
            throw std::runtime_error("[GPUDeviceContextPool] No ROCm devices available");
        }

        if (device_ordinal < 0 || device_ordinal >= rocm_device_count_)
        {
            throw std::runtime_error("[GPUDeviceContextPool] Invalid ROCm device ordinal " + std::to_string(device_ordinal) + " (valid range: 0-" + std::to_string(rocm_device_count_ - 1) + ")");
        }

        // Check if context already exists
        auto it = amd_contexts_.find(device_ordinal);
        if (it != amd_contexts_.end())
        {
            return *it->second;
        }

        // Create new context via factory (lazy initialization)
        LOG_DEBUG("[GPUDeviceContextPool] Creating AMDDeviceContext for device " << device_ordinal);

        auto context = amd_factory_(device_ordinal);
        IWorkerGPUContext &ctx_ref = *context;
        amd_contexts_[device_ordinal] = std::move(context);

        return ctx_ref;
    }

    IWorkerGPUContext &GPUDeviceContextPool::getContext(const std::string &device_type, int device_ordinal)
    {
        // Normalize device type string to lowercase for comparison
        std::string type_lower = device_type;
        std::transform(type_lower.begin(), type_lower.end(), type_lower.begin(),
                       [](unsigned char c)
                       { return std::tolower(c); });

        if (type_lower == "cuda" || type_lower == "nvidia")
        {
            return getNvidiaContext(device_ordinal);
        }
        else if (type_lower == "rocm" || type_lower == "hip" || type_lower == "amd")
        {
            return getAMDContext(device_ordinal);
        }
        else
        {
            throw std::invalid_argument("[GPUDeviceContextPool] Unknown device type: '" + device_type + "' (expected: cuda, nvidia, rocm, hip, amd)");
        }
    }

    IWorkerGPUContext &GPUDeviceContextPool::getContext(const DeviceId &device)
    {
        if (!device.is_gpu())
        {
            throw std::invalid_argument("[GPUDeviceContextPool] Device is not a GPU: '" + device.to_string() + "'");
        }

        if (device.is_cuda())
        {
#ifdef HAVE_CUDA
            if (!hasNvidiaSupport())
            {
                ensureNvidiaFactoryRegistered();
            }
            return getNvidiaContext(device.cuda_ordinal());
#else
            throw std::runtime_error("[GPUDeviceContextPool] CUDA support not compiled in");
#endif
        }

        if (device.is_rocm())
        {
#ifdef HAVE_ROCM
            if (!hasAMDSupport())
            {
                ensureAMDFactoryRegistered();
            }
            return getAMDContext(device.rocm_ordinal());
#else
            throw std::runtime_error("[GPUDeviceContextPool] ROCm support not compiled in");
#endif
        }

        throw std::invalid_argument("[GPUDeviceContextPool] Unsupported GPU device: '" + device.to_string() + "'");
    }

    // =============================================================================
    // Availability Queries
    // =============================================================================

    bool GPUDeviceContextPool::hasNvidiaSupport() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return nvidia_factory_ && cuda_device_count_ > 0;
    }

    bool GPUDeviceContextPool::hasAMDSupport() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return amd_factory_ && rocm_device_count_ > 0;
    }

    int GPUDeviceContextPool::nvidiaDeviceCount() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return cuda_device_count_;
    }

    int GPUDeviceContextPool::amdDeviceCount() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return rocm_device_count_;
    }

    // =============================================================================
    // Lifecycle Management
    // =============================================================================

    void GPUDeviceContextPool::shutdown()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        size_t nvidia_count = nvidia_contexts_.size();
        size_t amd_count = amd_contexts_.size();

        // Clear all contexts (destructors handle cleanup)
        nvidia_contexts_.clear();
        amd_contexts_.clear();

        LOG_DEBUG("[GPUDeviceContextPool] Shutdown cleared " << nvidia_count
                                                             << " NVIDIA and " << amd_count
                                                             << " AMD contexts");
    }

} // namespace llaminar2
