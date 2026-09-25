/**
 * @file ROCmRuntimeStartup.h
 * @brief One process-wide contract for explicit ROCm host-memory ownership.
 *
 * ROCr's default USERPTR implementation of hipHostMalloc asks Linux to fault
 * huge pages while pinning them. Under fragmented model-sized allocations this
 * can spend minutes in direct compaction. Llaminar requires KFD/GTT-owned host
 * backing while retaining HIP's allocation, coherence and release contracts.
 * Registered caller-owned pages must likewise use explicit KFD buffer-object
 * pin/unpin ownership, not HMM/SVM range registration. The latter repeatedly
 * remaps per-page DMA addresses across GPUs and can overflow Vega's ATS
 * interrupt ring with Intel IOMMU invalidations. Its range lifetime also ends
 * at virtual-memory invalidation, not necessarily hipHostUnregister().
 *
 * Llaminar owns residency and event ordering through TransferEngine; it does
 * not use demand-paged managed memory. These two vendor startup settings select
 * that explicit allocation/registration ABI without adding copies, changing
 * graph execution, or disabling asynchronous transfers.
 */
#pragma once

#include <optional>
#include <string_view>

namespace llaminar2
{
    /** @brief Observed HSA lifecycle; querying it must not initialize a device. */
    enum class ROCmRuntimeState { Uninitialized, Initialized };

    /** @brief Total startup decision, including failures that cannot be repaired late. */
    enum class ROCmHostMemoryAction
    {
        InstallExplicitOwnership,
        ExplicitOwnershipConfigured,
        RejectUserPointerBacking,
        RejectSvmRegistration,
        RejectLatePreparation,
    };

    /**
     * @brief Select the complete native pinned-host allocation/registration ABI.
     * @param userptr Exact HSA_USERPTR_FOR_PAGED_MEM value, or unspecified.
     * @param svm Exact HSA_USE_SVM value, or unspecified.
     * @param runtime HSA lifecycle before preparing Llaminar's backend.
     * @return One total action; reject conflicts before modifying either value.
     *
     * An initialized runtime is acceptable only when both settings were already
     * explicit. Installing the missing half after ROCr snapshots its environment
     * would publish a policy that the live driver did not actually adopt.
     */
    [[nodiscard]] inline ROCmHostMemoryAction selectROCmHostMemoryAction(
        std::optional<std::string_view> userptr,
        std::optional<std::string_view> svm,
        ROCmRuntimeState runtime)
    {
        if (userptr && *userptr != "0")
            return ROCmHostMemoryAction::RejectUserPointerBacking;
        if (svm && *svm != "0")
            return ROCmHostMemoryAction::RejectSvmRegistration;
        if (userptr && svm)
            return ROCmHostMemoryAction::ExplicitOwnershipConfigured;
        return runtime == ROCmRuntimeState::Uninitialized
            ? ROCmHostMemoryAction::InstallExplicitOwnership
            : ROCmHostMemoryAction::RejectLatePreparation;
    }

    /**
     * @brief Validate early host-backing preparation before entering HIP.
     *
     * Library initialization prepares only the vendor ABI environment; it never
     * enumerates devices or opens an HSA runtime. This covers native HIP clients
     * linked to Llaminar as well as the CLI. Backend entrypoints surface any
     * conflict here, leaving CPU-only processes independent of ROCm availability.
     * Repeated calls do not mutate the environment or runtime.
     * @throws std::runtime_error If backing was conflicting, late, or unconfigurable.
     */
    void requireROCmRuntimeStartup();
}
