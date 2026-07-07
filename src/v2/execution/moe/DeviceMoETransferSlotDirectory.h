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

    class DeviceMoETransferSlotDirectory
    {
    public:
        using ProjectionSpec = GpuExpertSlotPool::ProjectionSpec;

        static std::shared_ptr<DeviceMoETransferSlotDirectory> create(
            IBackend *backend,
            DeviceId device,
            int device_ordinal,
            uint32_t participant_id,
            uint32_t slot_count,
            std::vector<ProjectionSpec> specs,
            size_t vram_safety_margin_bytes);

        ~DeviceMoETransferSlotDirectory();

        DeviceMoETransferSlotDirectory(const DeviceMoETransferSlotDirectory &) = delete;
        DeviceMoETransferSlotDirectory &operator=(const DeviceMoETransferSlotDirectory &) = delete;

        DeviceMoEExpertDirectoryEntry *deviceEntries() const { return device_entries_; }
        uint32_t slotCount() const { return slot_count_; }
        const std::vector<DeviceMoEExpertDirectoryEntry> &hostEntriesForTest() const { return host_entries_; }
        size_t plannedBytes() const { return planned_bytes_; }
        size_t slotPayloadBytes() const { return slot_payload_bytes_; }
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
            size_t slot_payload_bytes);

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
        size_t slot_payload_bytes_ = 0;
    };

} // namespace llaminar2
