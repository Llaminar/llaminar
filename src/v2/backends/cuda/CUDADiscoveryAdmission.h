/**
 * @file CUDADiscoveryAdmission.h
 * @brief Typed admission policy for native CUDA discovery on optional-driver members.
 *
 * Full-backend binaries must also discover and serve on CPU-only cluster
 * members, while an explicit CUDA request keeps its precise device diagnostic.
 * This policy owns exactly that boundary: a positively distinguished absence
 * of the optional NVIDIA driver publishes an empty visibility set, and every
 * present-but-unusable installation (an incompatible or broken driver, a
 * stub resolved by a defective search path, a failed native query) stays a
 * fatal discovery fault with its original cause. The classification is pure
 * and device-free: the native adapter in CUDAHardwareMetadata.cu supplies
 * the observed status, and CUDADriverApi supplies the loader binding result.
 */
#pragma once
#include <string>
#include <string_view>

namespace llaminar2
{
    /** @brief One exact native loader observation for a requested object. */
    struct CUDADriverLoaderOutcome
    {
        bool loadable = false; ///< Whether the loader completely loaded the object.
        std::string diagnostic; ///< The loader's exact dlerror text, empty when none.
    };

    /**
     * @brief Probe the authoritative binding's loader path for one requested object.
     * @param request Exact object name or path the binding would request.
     * @return The loader's own outcome; never an exception or a second opinion.
     * @note This is the single native probe implementation: the production
     *       driver-presence authority and its device-free proof both call it.
     *       It uses the authoritative binding's exact flags so presence answers
     *       the same loader and search-path question, never a lazier one, and it
     *       never pins the library, resolves further symbols or touches a bound
     *       table.
     */
    CUDADriverLoaderOutcome probeNativeLibraryBinding(std::string_view request) noexcept;
    /** @brief Exact loader outcome for the optional native driver library. */
    enum class CUDADriverBindingStatus
    {
        NotInstalled, ///< The loader positively found no such library: genuine driver absence.
        PresentButUnbindable, ///< The library exists but cannot bind (dependency, symbol, format): a defect.
        Loadable, ///< The library loaded completely: a real installation is present.
    };

    /** @brief Observed native device-count query outcome, before any policy. */
    enum class CUDADeviceCountStatus
    {
        Success, ///< The query answered with a count (including zero visible devices).
        NoDevice, ///< The runtime positively reports no CUDA-capable device.
        InsufficientDriver, ///< The runtime cannot use the installed driver; ambiguous between absence and incompatibility.
        SystemDriverMismatch, ///< The runtime reports a driver/library mismatch: a broken installation.
        QueryFailure, ///< Any other native failure.
    };

    /** @brief Admission decision for one observed discovery outcome. */
    enum class CUDADiscoveryDisposition
    {
        EnumerateVisibility, ///< Trust the count and enumerate the visible set.
        EmptyVisibility, ///< Publish a complete empty visibility set for this member.
        FatalQueryFailure, ///< Preserve the original fault; never publish a partial or absent set.
    };

    /**
     * @brief Classify one exact loader outcome for the optional native driver.
     * @param loadable Whether dlopen completely loaded the requested library.
     * @param requestedSoname The exact SONAME the loader was asked for.
     * @param loaderDiagnostic The loader's diagnostic for a failed load, empty when none.
     * @return The typed binding status.
     *
     * Only a complete, exact match of the loader's own non-installation
     * diagnostic, naming the requested SONAME itself with nothing additional,
     * establishes genuine absence. A present library that cannot bind for any
     * reason (an unopened dependency, an unresolved symbol, an invalid format,
     * an access failure), an unknown or localized diagnostic, an empty request
     * and any trailing contradictory text all mean the requested library was
     * found or the outcome cannot be positively established: never absence.
     */
    [[nodiscard]] constexpr CUDADriverBindingStatus classifyCUDADriverBinding(
        bool loadable, std::string_view requestedSoname, std::string_view loaderDiagnostic) noexcept
    {
        if (loadable)
            return CUDADriverBindingStatus::Loadable;
        if (requestedSoname.empty())
            return CUDADriverBindingStatus::PresentButUnbindable;
        constexpr std::string_view absentTail = ": cannot open shared object file: No such file or directory";
        return loaderDiagnostic.size() == requestedSoname.size() + absentTail.size() &&
               loaderDiagnostic.substr(0, requestedSoname.size()) == requestedSoname &&
               loaderDiagnostic.substr(requestedSoname.size()) == absentTail
                   ? CUDADriverBindingStatus::NotInstalled
                   : CUDADriverBindingStatus::PresentButUnbindable;
    }

    /**
     * @brief Admit one observed device-count outcome for an optional-driver member.
     * @param status Native query outcome; never a policy guess.
     * @param binding Exact loader binding status for the driver library.
     * @return The admission decision.
     *
     * A positively driverless member (no loadable driver library anywhere) has
     * no CUDA capability at all, so an insufficient-driver runtime answer is a
     * complete empty visibility set, not a fault. With any present driver
     * installation the same runtime answer means the driver cannot serve this
     * runtime: the original fault is preserved, as are system driver mismatches
     * and every other query failure.
     */
    [[nodiscard]] constexpr CUDADiscoveryDisposition classifyCUDADiscovery(
        CUDADeviceCountStatus status, CUDADriverBindingStatus binding) noexcept
    {
        switch (status)
        {
        case CUDADeviceCountStatus::Success:
            return CUDADiscoveryDisposition::EnumerateVisibility;
        case CUDADeviceCountStatus::NoDevice:
            return CUDADiscoveryDisposition::EmptyVisibility;
        case CUDADeviceCountStatus::InsufficientDriver:
            return binding == CUDADriverBindingStatus::NotInstalled
                       ? CUDADiscoveryDisposition::EmptyVisibility
                       : CUDADiscoveryDisposition::FatalQueryFailure;
        case CUDADeviceCountStatus::SystemDriverMismatch:
        case CUDADeviceCountStatus::QueryFailure:
            return CUDADiscoveryDisposition::FatalQueryFailure;
        }
        return CUDADiscoveryDisposition::FatalQueryFailure;
    }
}
