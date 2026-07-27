#pragma once

/**
 * @file DeviceMoETransferSlotDirectory.h
 * @brief Stable device-side directory for graph-captured MoE expert arrivals.
 */

#include "../../backends/DeviceId.h"
#include "DeviceMoERebalanceController.h"
#include "GpuExpertSlotPool.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{
    class IBackend;
    class LoadOrchestrator;

    /**
     * @brief Owns stable device allocations used to receive graph-side MoE arrivals.
     *
     * A decode-maintenance directory is shared by every MoE layer in a model.
     * Its allocations must therefore describe capacity, not one layer's active
     * codebook. The active format is carried by each arriving directory entry
     * and retargeted onto these stable pointers by the unpack kernel.
     */
    class DeviceMoETransferSlotDirectory
    {
    public:
        using ProjectionSpec = GpuExpertSlotPool::ProjectionSpec;

        /**
         * @brief Allocation and wire requirements distilled from model layer formats.
         */
        struct FormatProfile
        {
            std::vector<ProjectionSpec> allocation_specs;
            size_t max_wire_payload_bytes = 0;
        };

        /**
         * @brief Merge exact per-layer formats into one reusable allocation profile.
         *
         * The returned allocation takes the component-wise maximum payload and
         * auxiliary-array capacity for each projection. The wire size is the
         * largest complete expert actually observed in any one layer, avoiding
         * the waste of transmitting an impossible combination of per-projection
         * maxima.
         *
         * @throws std::invalid_argument when the profile is empty, lacks a
         *         gate/up/down projection, or mixes incompatible matrix geometry.
         */
        static FormatProfile profileForLayerFormats(
            const std::vector<std::vector<ProjectionSpec>> &layer_formats);

        /**
         * @brief Describe the persistent and transactional capacity of one directory.
         *
         * Transfer-backed MoE placement has two distinct storage obligations:
         *
         * 1. `active_slots` retain experts published by earlier maintenance waves.
         * 2. `staging_slots` receive complete future waves before inactive runtime
         *    banks are rebuilt and atomically published.
         *
         * Conflating those obligations makes a full directory impossible to
         * rebalance: the copy must complete before publication, but there is no
         * unclaimed destination into which it can write. Keeping the terms
         * explicit makes the no-overwrite-before-publication rule structural.
         */
        struct BufferedCapacity
        {
            uint32_t active_slots = 0;
            uint32_t staging_slots = 0;
            uint32_t total_slots = 0;
        };

        /**
         * @brief Compute capacity for persistent experts plus buffered arrival waves.
         *
         * @param requested_active_slots Maximum number of experts that may remain
         *        published concurrently across the controller's layer window.
         * @param transfer_wave_slots Maximum payload-bearing arrivals in one wave.
         * @param transfer_buffer_count Number of complete arrival waves that may
         *        be staged without overwriting active or in-flight publications.
         * @return The explicit active, staging, and total slot counts.
         *
         * @throws std::invalid_argument when any input is zero or the resulting
         *         directory exceeds the device kernel's transfer-slot address space.
         */
        static BufferedCapacity planBufferedCapacity(
            uint64_t requested_active_slots,
            uint32_t transfer_wave_slots,
            uint32_t transfer_buffer_count);

        static std::shared_ptr<DeviceMoETransferSlotDirectory> create(
            IBackend *backend,
            DeviceId device,
            int device_ordinal,
            uint32_t participant_id,
            uint32_t slot_count,
            FormatProfile format_profile,
            size_t vram_safety_margin_bytes);

        ~DeviceMoETransferSlotDirectory();

        DeviceMoETransferSlotDirectory(const DeviceMoETransferSlotDirectory &) = delete;
        DeviceMoETransferSlotDirectory &operator=(const DeviceMoETransferSlotDirectory &) = delete;

        DeviceMoEExpertDirectoryEntry *deviceEntries() const { return device_entries_; }
        uint32_t slotCount() const { return slot_count_; }
        const std::vector<DeviceMoEExpertDirectoryEntry> &hostEntriesForTest() const { return host_entries_; }
        size_t plannedBytes() const { return planned_bytes_; }
        size_t wirePayloadBytes() const { return wire_payload_bytes_; }
        size_t slotStorageCapacityBytes() const { return slot_storage_capacity_bytes_; }
        /**
         * @brief Materialize the live payload descriptor owned by a transfer slot.
         *
         * Prefix-cache MoE runtime restore stores the logical expert id and
         * stable local slot id, while the directory remains the first-class owner
         * of the VRAM payload buffers.  This method rebuilds a
         * DeviceMoEExpertDescriptor from the directory's slot allocation without
         * exposing the directory entry array or requiring a host copy from the
         * device-side status table.  The caller supplies the logical expert id
         * because a transfer slot may be reused for different experts over time.
         *
         * @return true when @p slot_index names an allocated slot with ready
         *         gate/up/down NativeVNNI payload descriptors.
         */
        bool descriptorForSlot(uint32_t slot_index,
                               uint32_t logical_expert,
                               DeviceMoEExpertDescriptor &out) const;

    private:
        DeviceMoETransferSlotDirectory(
            IBackend *backend,
            DeviceId device,
            int device_ordinal,
            uint32_t participant_id,
            uint32_t slot_count,
            std::vector<ProjectionSpec> specs,
            std::shared_ptr<LoadOrchestrator> orchestrator,
            DeviceMoEExpertDirectoryEntry *device_entries,
            std::vector<DeviceMoEExpertDirectoryEntry> host_entries,
            size_t planned_bytes,
            size_t wire_payload_bytes,
            size_t slot_storage_capacity_bytes);

        static std::string slotName(uint32_t slot_index, const std::string &label);

        IBackend *backend_ = nullptr;
        DeviceId device_;
        int device_ordinal_ = -1;
        uint32_t participant_id_ = 0;
        uint32_t slot_count_ = 0;
        std::vector<ProjectionSpec> specs_;
        std::shared_ptr<LoadOrchestrator> orchestrator_;
        DeviceMoEExpertDirectoryEntry *device_entries_ = nullptr;
        std::vector<DeviceMoEExpertDirectoryEntry> host_entries_;
        size_t planned_bytes_ = 0;
        size_t wire_payload_bytes_ = 0;
        size_t slot_storage_capacity_bytes_ = 0;
    };

} // namespace llaminar2
