/**
 * @file GPUDeviceProcessLifetime.cpp
 * @brief Retire canonical GPU contexts before main returns, not from exit order.
 *
 * This control authority holds no memory quantities or second context registry.
 * It closes native admission before asking GPUDeviceContextPool to release its
 * exact existing generations. Late vendor/device runtime initialization cannot
 * move that lexical retirement behind runtime process-exit callbacks.
 */
#include "GPUDeviceProcessLifetime.h"
#include "GPUDeviceContextPool.h"
#include "../collective/BackendRouter.h"
#include "../kernels/KernelFactory.h"
#include "../utils/Logger.h"
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <thread>

namespace llaminar2
{
    namespace
    {
        /** @brief One process's native admission state, unrelated to byte accounting. */
        enum class AdmissionPhase { Unowned, Active, Retiring, Retired };
        /** @brief Serialize lifecycle changes while native retirement runs outside this lock. */
        struct ProcessAuthority
        {
            std::mutex mutex;
            std::condition_variable complete;
            AdmissionPhase phase = AdmissionPhase::Unowned;
            const GPUDeviceProcessLifetime *owner = nullptr;
            std::size_t native_contexts = 0; ///< Lifecycle borrowers, never memory quantities.
            std::thread::id retiring_thread{};
        };
        /** @return The process-local lifecycle authority, with no GPU initialization. */
        ProcessAuthority &authority()
        {
            static ProcessAuthority state;
            return state;
        }
    }

    GPUDeviceProcessLifetime::GPUDeviceProcessLifetime()
    {
        // The pool is core-only. Materialize it before admission without touching
        // a vendor factory, SDK, stream, driver, or physical inventory.
        (void)GPUDeviceContextPool::instance();
        auto &state = authority();
        std::lock_guard lock(state.mutex);
        if (state.phase != AdmissionPhase::Unowned)
            throw std::logic_error("GPU process lifetime requires one entry owner and cannot reopen retired admission");
        state.owner = this;
        state.phase = AdmissionPhase::Active;
    }

    GPUDeviceProcessLifetime::NativeContextLease::~NativeContextLease() noexcept
    {
        if (active_) GPUDeviceProcessLifetime::releaseNativeContext();
    }

    GPUDeviceProcessLifetime::NativeContextLease GPUDeviceProcessLifetime::acquireNativeContext()
    {
        auto &state = authority();
        std::lock_guard lock(state.mutex);
        if (state.phase != AdmissionPhase::Active)
            throw std::logic_error("Native GPU context admission requires a live process-entry owner");
        if (state.native_contexts == std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("Native GPU context lifetime count exhausted");
        ++state.native_contexts;
        return NativeContextLease(true);
    }

    void GPUDeviceProcessLifetime::releaseNativeContext() noexcept
    {
        auto &state = authority();
        std::lock_guard lock(state.mutex);
        if (state.native_contexts == 0 || state.phase == AdmissionPhase::Unowned ||
            state.phase == AdmissionPhase::Retired)
        {
            LOG_ERROR("[GPUDeviceProcessLifetime] Native context released outside its exact live process lifetime");
            std::terminate();
        }
        --state.native_contexts;
    }

    GPUDeviceProcessLifetime::~GPUDeviceProcessLifetime() noexcept { retire(); }
    void GPUDeviceProcessLifetime::retire() noexcept { retireOwner(this); }
    void GPUDeviceProcessLifetime::retireActive() noexcept { retireOwner(nullptr); }

    void GPUDeviceProcessLifetime::requireActive()
    {
        auto &state = authority();
        std::lock_guard lock(state.mutex);
        if (state.phase != AdmissionPhase::Active)
            throw std::logic_error("Native GPU admission requires an active GPUDeviceProcessLifetime in the process entry frame");
    }

    void GPUDeviceProcessLifetime::retireOwner(const GPUDeviceProcessLifetime *expected) noexcept
    {
        auto &state = authority();
        std::unique_lock lock(state.mutex);
        if (state.phase == AdmissionPhase::Unowned || state.phase == AdmissionPhase::Retired)
            return;
        if (expected && state.owner != expected)
        {
            LOG_ERROR("[GPUDeviceProcessLifetime] Retirement requested by a foreign process owner");
            std::terminate();
        }
        if (state.phase == AdmissionPhase::Retiring)
        {
            if (state.retiring_thread == std::this_thread::get_id())
            {
                LOG_ERROR("[GPUDeviceProcessLifetime] Recursive terminal retirement cannot prove completion");
                std::terminate();
            }
            state.complete.wait(lock, [&] { return state.phase == AdmissionPhase::Retired; });
            return;
        }
        state.phase = AdmissionPhase::Retiring;
        state.retiring_thread = std::this_thread::get_id();
        lock.unlock();
        try
        {
            // No admission-state lock is held while worker joins/native calls run.
            // Pool shutdown itself rejects retained graph/submission owners before
            // removing a generation, and native release failures remain fatal.
            GlobalBackendRouter::shutdown();
            llaminar::v2::kernels::KernelFactory::clearCache();
            GPUDeviceContextPool::instance().shutdown();
        }
        catch (const std::exception &error)
        {
            LOG_ERROR("[GPUDeviceProcessLifetime] Terminal GPU retirement failed: " << error.what());
            std::terminate();
        }
        catch (...)
        {
            LOG_ERROR("[GPUDeviceProcessLifetime] Terminal GPU retirement failed with an unknown exception");
            std::terminate();
        }
        lock.lock();
        if (state.native_contexts != 0)
        {
            LOG_ERROR("[GPUDeviceProcessLifetime] Terminal retirement still has "
                << state.native_contexts << " direct or incompletely constructed native context owners");
            std::terminate();
        }
        state.owner = nullptr;
        state.retiring_thread = {};
        state.phase = AdmissionPhase::Retired;
        lock.unlock();
        state.complete.notify_all();
    }
}
