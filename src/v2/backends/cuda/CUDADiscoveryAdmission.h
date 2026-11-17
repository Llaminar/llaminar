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
#include <string_view>

namespace llaminar2
{
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
     * Only the requested library itself being absent from every search path is
     * genuine non-installation. The loader names the object it could not open,
     * so any other diagnostic (an unopened dependency, an unresolved symbol, an
     * invalid file format or an access failure) means the requested library was
     * found but cannot serve: a present, broken installation, never absence.
     */
    [[nodiscard]] constexpr CUDADriverBindingStatus classifyCUDADriverBinding(
        bool loadable, std::string_view requestedSoname, std::string_view loaderDiagnostic) noexcept
    {
        if (loadable)
            return CUDADriverBindingStatus::Loadable;
        constexpr std::string_view absentTail = ": cannot open shared object file: No such file or directory";
        return requestedSoname.size() + absentTail.size() <= loaderDiagnostic.size() &&
               loaderDiagnostic.substr(0, requestedSoname.size()) == requestedSoname &&
               loaderDiagnostic.substr(requestedSoname.size(), absentTail.size()) == absentTail
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
