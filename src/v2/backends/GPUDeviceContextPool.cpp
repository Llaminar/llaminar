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
#include <limits>
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

        LOG_DEBUG("[GPUDeviceContextPool] Registered NVIDIA factory with "
                  << device_count << " devices available");
    }

    void GPUDeviceContextPool::registerAMDFactory(GPUContextFactory factory, int device_count)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        amd_factory_ = std::move(factory);
        rocm_device_count_ = device_count;

        LOG_DEBUG("[GPUDeviceContextPool] Registered AMD factory with "
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

        if (retiring_nvidia_contexts_.contains(device_ordinal))
        {
            throw std::logic_error(
                "[GPUDeviceContextPool] CUDA context acquisition raced exclusive generation retirement for device " +
                std::to_string(device_ordinal));
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
        if (next_generation_ == std::numeric_limits<std::uint64_t>::max())
        {
            nvidia_contexts_.erase(device_ordinal);
            throw std::overflow_error(
                "[GPUDeviceContextPool] Context generation identity exhausted");
        }
        nvidia_generations_[device_ordinal] = next_generation_++;

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

        if (retiring_amd_contexts_.contains(device_ordinal))
        {
            throw std::logic_error(
                "[GPUDeviceContextPool] ROCm context acquisition raced exclusive generation retirement for device " +
                std::to_string(device_ordinal));
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
        if (next_generation_ == std::numeric_limits<std::uint64_t>::max())
        {
            amd_contexts_.erase(device_ordinal);
            throw std::overflow_error(
                "[GPUDeviceContextPool] Context generation identity exhausted");
        }
        amd_generations_[device_ordinal] = next_generation_++;

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
        decltype(nvidia_contexts_) retired_nvidia_contexts;
        decltype(amd_contexts_) retired_amd_contexts;
        size_t nvidia_count = 0u;
        size_t amd_count = 0u;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!retiring_nvidia_contexts_.empty() ||
                !retiring_amd_contexts_.empty())
            {
                throw std::logic_error(
                    "[GPUDeviceContextPool] Shutdown cannot overlap an exclusive runtime-generation retirement");
            }

            nvidia_count = nvidia_contexts_.size();
            amd_count = amd_contexts_.size();
            retired_nvidia_contexts.swap(nvidia_contexts_);
            retired_amd_contexts.swap(amd_contexts_);
            nvidia_generations_.clear();
            amd_generations_.clear();
        }

        /* Worker teardown joins threads and enters CUDA/HIP to destroy exact
         * streams, events, and library handles. Keep those operations outside
         * the pool mutex so unrelated device acquisition cannot deadlock on a
         * backend destructor that re-enters infrastructure. */
        retired_nvidia_contexts.clear();
        retired_amd_contexts.clear();

        LOG_DEBUG("[GPUDeviceContextPool] Shutdown cleared " << nvidia_count
                                                             << " NVIDIA and " << amd_count
                                                             << " AMD contexts");
    }

    GPUDeviceContextGenerationRetirementReceipt
    GPUDeviceContextPool::retireExclusiveGeneration(DeviceId device)
    {
        auto scope = beginExclusiveGenerationRetirement(device);
        return scope.receipt();
    }

    ExclusiveGPUDeviceGenerationRetirement
    GPUDeviceContextPool::beginExclusiveGenerationRetirement(DeviceId device)
    {
        if (!device.is_gpu())
        {
            throw std::invalid_argument(
                "[GPUDeviceContextPool] Exclusive generation retirement requires an exact GPU device");
        }

        std::unique_ptr<IWorkerGPUContext> retired_context;
        std::uint64_t retired_generation = 0u;
        const int ordinal = device.gpu_ordinal();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto &contexts = device.is_cuda() ? nvidia_contexts_ : amd_contexts_;
            auto &generations =
                device.is_cuda() ? nvidia_generations_ : amd_generations_;
            auto &retiring = device.is_cuda()
                                 ? retiring_nvidia_contexts_
                                 : retiring_amd_contexts_;

            if (retiring.contains(ordinal))
            {
                throw std::logic_error(
                    "[GPUDeviceContextPool] Device context generation is already retiring for " +
                    device.toString());
            }

            /* Install exclusion even when no worker was materialized. The
             * backend runtime generation still exists and must not race a lazy
             * context acquisition during reset. */
            retiring.insert(ordinal);

            const auto context_it = contexts.find(ordinal);
            if (context_it == contexts.end())
            {
                return ExclusiveGPUDeviceGenerationRetirement(
                    this,
                    GPUDeviceContextGenerationRetirementReceipt{
                        .device = device,
                    });
            }

            const auto generation_it = generations.find(ordinal);
            if (generation_it == generations.end() ||
                generation_it->second == 0u)
            {
                retiring.erase(ordinal);
                throw std::logic_error(
                    "[GPUDeviceContextPool] Live context has no generation identity for " +
                    device.toString());
            }

            retired_generation = generation_it->second;
            retired_context = std::move(context_it->second);
            contexts.erase(context_it);
            generations.erase(generation_it);
        }

        /*
         * Context destruction joins the worker and destroys every exact stream,
         * event, and library handle. Do it outside mutex_: cleanup may enter a
         * backend runtime and must not serialize unrelated device acquisition.
         */
        try
        {
            retired_context.reset();
        }
        catch (...)
        {
            finishExclusiveGenerationRetirement(device);
            throw;
        }

        LOG_INFO(
            "[GPUDeviceContextPool] Retired exclusive context generation device="
            << device.toString() << " generation=" << retired_generation);
        return ExclusiveGPUDeviceGenerationRetirement(
            this,
            GPUDeviceContextGenerationRetirementReceipt{
                .device = device,
                .retired_generation = retired_generation,
            });
    }

    void GPUDeviceContextPool::finishExclusiveGenerationRetirement(
        DeviceId device) noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto &retiring = device.is_cuda()
                             ? retiring_nvidia_contexts_
                             : retiring_amd_contexts_;
        if (retiring.erase(device.gpu_ordinal()) != 1u)
        {
            LOG_ERROR(
                "[GPUDeviceContextPool] Exclusive retirement lost its marker for "
                << device.toString());
            std::terminate();
        }
    }

    ExclusiveGPUDeviceGenerationRetirement::~ExclusiveGPUDeviceGenerationRetirement()
    {
        release();
    }

    ExclusiveGPUDeviceGenerationRetirement::ExclusiveGPUDeviceGenerationRetirement(
        ExclusiveGPUDeviceGenerationRetirement &&other) noexcept
        : pool_(other.pool_), receipt_(other.receipt_)
    {
        other.pool_ = nullptr;
        other.receipt_ = {};
    }

    ExclusiveGPUDeviceGenerationRetirement &
    ExclusiveGPUDeviceGenerationRetirement::operator=(
        ExclusiveGPUDeviceGenerationRetirement &&other) noexcept
    {
        if (this == &other)
            return *this;
        release();
        pool_ = other.pool_;
        receipt_ = other.receipt_;
        other.pool_ = nullptr;
        other.receipt_ = {};
        return *this;
    }

    void ExclusiveGPUDeviceGenerationRetirement::release() noexcept
    {
        if (!pool_)
            return;
        pool_->finishExclusiveGenerationRetirement(receipt_.device);
        pool_ = nullptr;
        receipt_ = {};
    }

} // namespace llaminar2
