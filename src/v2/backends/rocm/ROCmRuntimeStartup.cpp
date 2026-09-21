/**
 * @file ROCmRuntimeStartup.cpp
 * @brief Prepare the ROCr pinned-host allocation ABI before its first initialization.
 *
 * This narrow infrastructure boundary owns HSA_USERPTR_FOR_PAGED_MEM. It is not
 * a Llaminar debug knob: ROCr snapshots it once at initialization, so lazy setup
 * at the first large allocation is already too late. The immutable preparation
 * result is checked by every public runtime-admission boundary. CPU-only startup
 * performs no HIP call, HSA initialization, GPU enumeration or allocation.
 */
#include "ROCmRuntimeStartup.h"
#include "utils/DebugEnv.h"

#include <hsa/hsa.h>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        constexpr const char *kBackingVariable = "HSA_USERPTR_FOR_PAGED_MEM";

        /** @brief Immutable preparation outcome; errors are raised only on ROCm use. */
        enum class Preparation
        {
            Unprepared,
            Ready,
            ConflictingBacking,
            RuntimeAlreadyInitialized,
            RuntimeProbeFailed,
            EnvironmentWriteFailed,
        };

        /**
         * @brief Configure the vendor's one-shot ABI without initializing hardware.
         * @return A complete preparation outcome, never a best-effort warning.
         */
        Preparation prepareHostBacking()
        {
            // Use the central typed parser without eagerly constructing the
            // entire mutable debug/profiling configuration during library load.
            // Only this immutable vendor setup belongs before main().
            const BackendStartupConfig startup;
            const auto &requested = startup.rocm_userptr_for_paged_mem;
            auto runtime = ROCmRuntimeState::Uninitialized;
            if (!requested)
            {
                // HIP links this runtime, but obtaining its symbol and asking
                // whether HSA is open does not open it. In particular, avoid
                // hipGetDeviceCount here: it would consume the old policy.
                const auto query = reinterpret_cast<decltype(&hsa_system_get_info)>(
                    dlsym(RTLD_DEFAULT, "hsa_system_get_info"));
                if (!query)
                    return Preparation::RuntimeProbeFailed;
                std::uint16_t major = 0;
                const auto status = query(HSA_SYSTEM_INFO_VERSION_MAJOR, &major);
                if (status == HSA_STATUS_SUCCESS)
                    runtime = ROCmRuntimeState::Initialized;
                else if (status != HSA_STATUS_ERROR_NOT_INITIALIZED)
                    return Preparation::RuntimeProbeFailed;
            }
            const auto action = selectROCmHostBackingAction(
                requested ? std::optional<std::string_view>{*requested} : std::nullopt,
                runtime);
            switch (action)
            {
            case ROCmHostBackingAction::InstallDriverBacking:
                // The process is still in library initialization, before user
                // threads or native HIP clients can initialize the runtime.
                return setenv(kBackingVariable, "0", 0) == 0
                    ? Preparation::Ready : Preparation::EnvironmentWriteFailed;
            case ROCmHostBackingAction::DriverBackingConfigured:
                return Preparation::Ready;
            case ROCmHostBackingAction::RejectUserPointerBacking:
                return Preparation::ConflictingBacking;
            case ROCmHostBackingAction::RejectLatePreparation:
                return Preparation::RuntimeAlreadyInitialized;
            }
            return Preparation::RuntimeProbeFailed;
        }

        // A lazy singleton would miss callers that legitimately enter HIP
        // directly before constructing a Llaminar backend. Prepare just this
        // environment contract eagerly; hardware initialization remains lazy.
        const Preparation preparation = prepareHostBacking();
    }

    void requireROCmRuntimeStartup()
    {
        switch (preparation)
        {
        case Preparation::Unprepared:
            throw std::runtime_error("ROCm entered before its library startup policy was prepared");
        case Preparation::Ready:
            return;
        case Preparation::ConflictingBacking:
            throw std::runtime_error(
                "ROCm requires driver-owned pinned host memory: unset "
                "HSA_USERPTR_FOR_PAGED_MEM or set it to 0 before starting Llaminar");
        case Preparation::RuntimeAlreadyInitialized:
            throw std::runtime_error(
                "ROCm was initialized before Llaminar could prepare driver-owned "
                "host memory; set HSA_USERPTR_FOR_PAGED_MEM=0 before initializing HIP/HSA");
        case Preparation::RuntimeProbeFailed:
            throw std::runtime_error("Cannot inspect the ROCr startup state for pinned-host backing");
        case Preparation::EnvironmentWriteFailed:
            throw std::runtime_error("Cannot prepare HSA_USERPTR_FOR_PAGED_MEM=0 for ROCr");
        }
        throw std::runtime_error("Invalid ROCm host-memory startup state");
    }
}
