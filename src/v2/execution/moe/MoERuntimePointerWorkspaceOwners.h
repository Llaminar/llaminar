/**
 * @file MoERuntimePointerWorkspaceOwners.h
 * @brief Exclusive graph-lifetime ownership for grouped-MoE pointer tables.
 *
 * Grouped CUDA and ROCm kernels pass arrays of device scratch pointers to
 * captured decode kernels. Weight descriptor tables are immutable and may be
 * deduplicated across graphs, but scratch pointers belong to one graph-local
 * stage. This helper gives those two kinds of state independent authorities:
 * descriptor publications can remain shared while mutable pointer arrays hold
 * an exclusive RAII workspace lease until their graph owner is destroyed.
 */

#pragma once

#include "MoEWorkspaceRequirements.h"
#include "../local_execution/device/DeviceWorkspaceManager.h"
#include "../../utils/Logger.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <unordered_map>

namespace llaminar2
{
    /** @brief Physical grouped-decode pointer-array buffer being leased. */
    enum class MoERuntimePointerArrayRole
    {
        GateUp,
        Down,
    };

    /** @brief Whether a missing owner slot may be acquired at this boundary. */
    enum class MoERuntimePointerWorkspaceAccess
    {
        WarmupMayAcquire,
        CaptureExistingOnly,
    };

    /**
     * @brief Own exclusive mutable pointer-array slots for one MoE kernel object.
     *
     * A logical key combines a descriptor-table slot with the decode route
     * scope. The key is local to this owner; the returned physical slot comes
     * from the workspace-wide RAII registry and therefore cannot alias another
     * graph-local kernel, even when both kernels adopted the same immutable
     * descriptor publication.
     */
    class MoERuntimePointerWorkspaceOwners
    {
    public:
        /**
         * @brief Resolve or acquire one exclusive physical pointer-table slot.
         * @param workspace Workspace whose registry owns the physical table.
         * @param role Gate/up output pointers or down-input pointers.
         * @param descriptor_slot Immutable descriptor publication slot used only
         *        as part of this owner's logical lookup key.
         * @param scope_slot Decode route scope in `[0, kRuntimePointerWorkspaceScopes)`.
         * @param access Warmup may acquire; capture may only consume an existing lease.
         * @param context Operation name included in fatal diagnostics.
         * @return Exclusive physical workspace slot, or no value on contract failure.
         */
        std::optional<std::size_t> resolve(
            DeviceWorkspaceManager *workspace,
            MoERuntimePointerArrayRole role,
            std::size_t descriptor_slot,
            std::size_t scope_slot,
            MoERuntimePointerWorkspaceAccess access,
            const char *context)
        {
            if (!workspace ||
                descriptor_slot >= static_cast<std::size_t>(
                    MoEWorkspaceBuffers::kRuntimePointerTableSlots) ||
                scope_slot >= static_cast<std::size_t>(
                    MoEWorkspaceBuffers::kRuntimePointerWorkspaceScopes))
            {
                LOG_ERROR("[MoERuntimePointerWorkspaceOwners] Invalid "
                          << (context ? context : "pointer-table owner")
                          << " request: workspace=" << static_cast<void *>(workspace)
                          << " descriptor_slot=" << descriptor_slot
                          << " scope=" << scope_slot);
                return std::nullopt;
            }

            const std::size_t logical_key =
                scope_slot * static_cast<std::size_t>(
                                 MoEWorkspaceBuffers::kRuntimePointerTableSlots) +
                descriptor_slot;
            auto &owners = ownersFor(role);
            auto existing = owners.find(logical_key);
            if (existing != owners.end())
                return existing->second->slot();

            /*
             * Capturing code must perform lookup only. Reserving a registry
             * lease can allocate host metadata and, more importantly, means
             * warmup failed to publish the exact address the graph will embed.
             */
            if (access ==
                MoERuntimePointerWorkspaceAccess::CaptureExistingOnly)
            {
                LOG_ERROR("[MoERuntimePointerWorkspaceOwners] "
                          << (context ? context : "pointer-table owner")
                          << " entered graph capture without an exclusive warmup lease"
                          << " descriptor_slot=" << descriptor_slot
                          << " scope=" << scope_slot);
                return std::nullopt;
            }

            auto lease = workspace->acquirePersistentSlot(
                domainFor(role),
                static_cast<std::size_t>(
                    MoEWorkspaceBuffers::kRuntimePointerWorkspaceEntries));
            if (!lease)
            {
                LOG_ERROR("[MoERuntimePointerWorkspaceOwners] "
                          << (context ? context : "pointer-table owner")
                          << " exhausted exclusive graph pointer-table slots");
                return std::nullopt;
            }

            const std::size_t physical_slot = lease->slot();
            const auto [inserted, did_insert] =
                owners.emplace(logical_key, std::move(lease));
            if (!did_insert || !inserted->second ||
                inserted->second->slot() != physical_slot)
            {
                LOG_ERROR("[MoERuntimePointerWorkspaceOwners] Failed to retain "
                          << (context ? context : "pointer-table owner")
                          << " lease");
                return std::nullopt;
            }
            return physical_slot;
        }

        /**
         * @brief Release every pointer-table lease held for the current workspace.
         *
         * Call only when the kernel unbinds or changes workspace. Any graph that
         * embeds these addresses must have been retired by that lifecycle edge.
         */
        void reset() noexcept
        {
            gate_up_owners_.clear();
            down_owners_.clear();
        }

    private:
        using OwnerMap = std::unordered_map<
            std::size_t,
            std::shared_ptr<PersistentWorkspaceSlotLease>>;

        /** @brief Select this kernel's host map for one physical pointer role. */
        OwnerMap &ownersFor(MoERuntimePointerArrayRole role) noexcept
        {
            return role == MoERuntimePointerArrayRole::GateUp
                       ? gate_up_owners_
                       : down_owners_;
        }

        /** @brief Select the workspace-wide exclusivity domain for one buffer role. */
        static const char *domainFor(
            MoERuntimePointerArrayRole role) noexcept
        {
            return role == MoERuntimePointerArrayRole::GateUp
                       ? "moe_runtime_gateup_pointer_array_owners"
                       : "moe_runtime_down_pointer_array_owners";
        }

        OwnerMap gate_up_owners_;
        OwnerMap down_owners_;
    };
} // namespace llaminar2
