/**
 * @file ROCmRuntimeStartup.h
 * @brief One process-wide contract for native ROCm pinned-host backing.
 *
 * ROCr's default USERPTR implementation of hipHostMalloc asks Linux to fault
 * huge pages while pinning them. Under fragmented model-sized allocations this
 * can spend minutes in direct compaction. Llaminar requires KFD/GTT-owned host
 * backing while retaining HIP's allocation, coherence and release contracts.
 * This is a driver-initialization policy, not an inference transport selector.
 */
#pragma once

#include <optional>
#include <string_view>

namespace llaminar2
{
    /** @brief Observed HSA lifecycle; querying it must not initialize a device. */
    enum class ROCmRuntimeState { Uninitialized, Initialized };

    /** @brief Total startup decision, including failures that cannot be repaired late. */
    enum class ROCmHostBackingAction
    {
        InstallDriverBacking,
        DriverBackingConfigured,
        RejectUserPointerBacking,
        RejectLatePreparation,
    };

    /**
     * @brief Select the only supported native pinned-host backing policy.
     * @param requested Exact upstream HSA variable, absent when not specified.
     * @param runtime HSA lifecycle before preparing Llaminar's backend.
     * @return Explicit action; no initialized runtime is silently reconfigured.
     */
    [[nodiscard]] inline ROCmHostBackingAction selectROCmHostBackingAction(
        std::optional<std::string_view> requested, ROCmRuntimeState runtime)
    {
        if (requested.has_value())
            return *requested == "0" ? ROCmHostBackingAction::DriverBackingConfigured
                                      : ROCmHostBackingAction::RejectUserPointerBacking;
        return runtime == ROCmRuntimeState::Uninitialized
            ? ROCmHostBackingAction::InstallDriverBacking
            : ROCmHostBackingAction::RejectLatePreparation;
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
