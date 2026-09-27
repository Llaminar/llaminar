/**
 * @file ROCmRuntimeStartup.cpp
 * @brief Prepare explicit ROCr allocation/registration ownership before initialization.
 *
 * This infrastructure boundary owns HSA_USERPTR_FOR_PAGED_MEM and HSA_USE_SVM.
 * They are not debug knobs: ROCr snapshots them at initialization, so lazy setup
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
        constexpr const char *kRegistrationVariable = "HSA_USE_SVM";

        /** @brief Immutable preparation outcome; errors are raised only on ROCm use. */
        enum class Preparation
        {
            Unprepared,
            Ready,
            ConflictingBacking,
            ConflictingRegistration,
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
            const auto &registration = startup.rocm_use_svm;
            auto runtime = ROCmRuntimeState::Uninitialized;
            if (!requested || !registration)
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
            const auto action = selectROCmHostMemoryAction(
                requested ? std::optional<std::string_view>{*requested} : std::nullopt,
                registration ? std::optional<std::string_view>{*registration} : std::nullopt,
                runtime);
            switch (action)
            {
            case ROCmHostMemoryAction::InstallExplicitOwnership:
                // The process is still in library initialization, before user
                // threads or native HIP clients can initialize the runtime.
                // Native allocations use driver-owned GTT pages. Existing
                // caller pages get a real pinned buffer-object lifetime, not
                // HMM's deferred per-page mapping/retirement. Install both
                // halves once, before any device or worker is constructed.
                return setenv(kBackingVariable, "0", 0) == 0 &&
                       setenv(kRegistrationVariable, "0", 0) == 0
                    ? Preparation::Ready : Preparation::EnvironmentWriteFailed;
            case ROCmHostMemoryAction::ExplicitOwnershipConfigured:
                return Preparation::Ready;
            case ROCmHostMemoryAction::RejectUserPointerBacking:
                return Preparation::ConflictingBacking;
            case ROCmHostMemoryAction::RejectSvmRegistration:
                return Preparation::ConflictingRegistration;
            case ROCmHostMemoryAction::RejectLatePreparation:
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
        case Preparation::ConflictingRegistration:
            throw std::runtime_error(
                "ROCm requires explicit pinned host registration, not HMM/SVM ranges: "
                "unset HSA_USE_SVM or set it to 0 before starting Llaminar");
        case Preparation::RuntimeAlreadyInitialized:
            throw std::runtime_error(
                "ROCm was initialized before Llaminar could prepare driver-owned "
                "host memory; set HSA_USERPTR_FOR_PAGED_MEM=0 and HSA_USE_SVM=0 "
                "before initializing HIP/HSA");
        case Preparation::RuntimeProbeFailed:
            throw std::runtime_error("Cannot inspect the ROCr startup state for pinned-host backing");
        case Preparation::EnvironmentWriteFailed:
            throw std::runtime_error(
                "Cannot prepare HSA_USERPTR_FOR_PAGED_MEM=0 and HSA_USE_SVM=0 for ROCr");
        }
        throw std::runtime_error("Invalid ROCm host-memory startup state");
    }
}
