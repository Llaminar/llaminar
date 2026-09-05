#pragma once

/**
 * @file DeviceMoETransferSlotDirectory.h
 * @brief Stable device-side directory for graph-captured MoE expert arrivals.
 */

#include "../../backends/DeviceId.h"
#include "../../planning/PhysicalMemoryAuthority.h"
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
    struct MoEOverlayLayerWeightManifest;

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
         * @brief Exact physical allocation owned by one transfer directory.
         *
         * Payload bytes are the single allocator-owned `WeightVRAMPool`
         * region. Descriptor bytes cover the mutable table and immutable
         * request-reset baseline. Keeping both terms together makes setup
         * admission and runtime claims consume one byte-identical contract.
         */
        struct AllocationBOM
        {
            size_t payload_bytes = 0;
            size_t descriptor_bytes = 0;
            size_t total_bytes = 0;
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
         * @brief Derive the reusable device profile from model-authenticated metadata.
         *
         * Every quantized source codebook is converted through the same
         * NativeVNNI execution/allocation catalog used by CUDA and ROCm weight
         * preparation. This setup-time form lets physical-memory admission run
         * before prepared engines and their device pointers exist.
         *
         * @param layer_weight_manifest Contiguous gate/up/down model manifest.
         * @return Cross-layer allocation and wire-capacity union.
         * @throws std::invalid_argument for floating, malformed, or
         *         uncatalogued projection formats.
         */
        static FormatProfile profileForLayerWeightManifest(
            const std::vector<MoEOverlayLayerWeightManifest> &
                layer_weight_manifest);

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
         * @brief Compute durable transfer-slot demand for the runtime domain.
         *
         * Layer windows are scheduling cursors, not storage ownership domains.
         * Current-batch LLEP, prefix rehydration, and Dynamic durable residency
         * maintenance all publish descriptors into the same runtime table and
         * must therefore resolve one directory with one physical slot identity
         * space. LLEP assignments are request-transient while Dynamic updates
         * survive into future requests; sharing transfer storage does not merge
         * those lifecycles. Basing
         * capacity on `layer_window_start` or `layer_window_count` lets two
         * graphs for the same table create differently sized directories and
         * later reinterpret a valid slot ID against the wrong allocation.
         *
         * Reserve at least one durable lane for every runtime layer even when
         * hot-replica caching is disabled, because Dynamic ownership transfer
         * can retain one remote-origin expert per layer.
         *
         * @param config Device rebalance geometry for the complete runtime domain.
         * @return Number of persistent active slots required before staging.
         */
        static uint64_t persistentActiveSlotDemand(
            const DeviceMoERebalanceConfig &config) noexcept;

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

        /**
         * @brief Resolve the complete active/staging capacity from runtime policy.
         * @param config Complete device-side rebalance domain geometry.
         * @param minimum_active_slots Additional non-durable operation demand,
         *        such as one current-batch LLEP payload window.
         * @param transfer_wave_slots Maximum experts prepared by one wave.
         * @param transfer_buffer_count Independently retained arrival waves.
         * @return Exact capacity shared by admission and graph materialization.
         * @throws std::invalid_argument when the resulting capacity is empty
         *         or exceeds the device transfer-slot ABI.
         */
        static BufferedCapacity planRuntimeCapacity(
            const DeviceMoERebalanceConfig &config,
            uint64_t minimum_active_slots,
            uint32_t transfer_wave_slots,
            uint32_t transfer_buffer_count);

        /**
         * @brief Price one directory through the concrete allocator contracts.
         *
         * The calculation uses `WeightVRAMPool` planning rather than copying
         * its 256-byte region-alignment arithmetic. No backend or device is
         * touched, so this method is safe during pure capacity admission.
         *
         * @param capacity Typed active/staging directory capacity.
         * @param format_profile Exact reusable projection profile.
         * @return Payload, descriptor, and total physical bytes.
         * @throws std::invalid_argument for incoherent capacity or format data.
         * @throws std::overflow_error when the complete allocation cannot be
         *         represented by `size_t`.
         */
        static AllocationBOM allocationBOM(
            BufferedCapacity capacity,
            const FormatProfile &format_profile);

        /**
         * @brief Materialize one production directory against admitted VRAM.
         * @param backend Exact CUDA or ROCm allocation/copy authority.
         * @param device Physical GPU identity charged by memory admission.
         * @param device_ordinal Backend-local ordinal matching @p device.
         * @param participant_id Stable routed-expert participant identity.
         * @param capacity Exact active and arrival-wave slot capacity.
         * @param format_profile Allocation union for every runtime layer.
         * @param memory_authority Canonical admitted physical-memory ledger.
         * @return Model-lifetime owner of all payload and descriptor storage.
         * @throws std::invalid_argument for incomplete topology or admission.
         * @throws std::runtime_error when device materialization fails.
         */
        static std::shared_ptr<DeviceMoETransferSlotDirectory> create(
            IBackend *backend,
            DeviceId device,
            int device_ordinal,
            uint32_t participant_id,
            BufferedCapacity capacity,
            FormatProfile format_profile,
            std::shared_ptr<PhysicalMemoryAuthority> memory_authority);

        /**
         * @brief Allocate an unadmitted directory at a named device-test boundary.
         * @param backend Exact test backend.
         * @param device Physical test GPU.
         * @param device_ordinal Backend-local ordinal matching @p device.
         * @param participant_id Test participant identity.
         * @param slot_count Total fixture slots; active/staging semantics are
         *        intentionally absent from this test-only boundary.
         * @param format_profile Exact test allocation profile.
         * @return Test-owned directory using the unadmitted allocation token.
         */
        static std::shared_ptr<DeviceMoETransferSlotDirectory> createForTest(
            IBackend *backend,
            DeviceId device,
            int device_ordinal,
            uint32_t participant_id,
            uint32_t slot_count,
            FormatProfile format_profile);

        /** @brief Release descriptor arrays before their payload-pool owner. */
        ~DeviceMoETransferSlotDirectory();

        DeviceMoETransferSlotDirectory(const DeviceMoETransferSlotDirectory &) = delete;
        DeviceMoETransferSlotDirectory &operator=(const DeviceMoETransferSlotDirectory &) = delete;

        DeviceMoEExpertDirectoryEntry *deviceEntries() const { return device_entries_; }
        /**
         * @brief Return the exact GPU that physically owns every slot allocation.
         *
         * The logical rebalance domain is shared by all TP participants, but a
         * transfer directory is not: its payload pointers are dereferenced by
         * one participant's local grouped kernels.  Exposing this immutable
         * identity lets graph caches reject an accidentally shared directory
         * before a peer-memory pointer reaches captured inference.
         */
        DeviceId device() const noexcept { return device_; }

        /**
         * @brief Return the backend ordinal used for all directory allocations.
         */
        int deviceOrdinal() const noexcept { return device_ordinal_; }

        /**
         * @brief Return the TP participant that owns the directory.
         */
        uint32_t participantId() const noexcept { return participant_id_; }

        /**
         * @brief Require this directory to match one physical graph participant.
         *
         * A mismatch is a construction error, not a recoverable cache miss.
         * Continuing would make grouped decode read expert weights through
         * peer memory, which is numerically valid on permissive topologies but
         * catastrophically uneconomical and invalid for graph ownership.
         *
         * @throws std::logic_error when any physical owner field differs.
         */
        void requirePhysicalOwner(
            DeviceId expected_device,
            int expected_device_ordinal,
            uint32_t expected_participant_id) const;

        uint32_t slotCount() const { return slot_count_; }
        /** @return Typed active/staging capacity admitted for production use. */
        BufferedCapacity capacity() const noexcept { return capacity_; }
        const std::vector<DeviceMoEExpertDirectoryEntry> &hostEntriesForTest() const { return host_entries_; }
        size_t plannedBytes() const { return planned_bytes_; }
        size_t wirePayloadBytes() const { return wire_payload_bytes_; }
        size_t slotStorageCapacityBytes() const { return slot_storage_capacity_bytes_; }
        /**
         * @brief Retire every request-owned logical occupant on an explicit stream.
         *
         * The directory's payload allocations and descriptor pointers are
         * model-lifetime state, while `(layer, expert, generation, residency)`
         * are request-owned publications. Model setup stores one immutable
         * device baseline containing the former and no logical occupants.
         * Request reset restores that baseline with a single D2D copy, after
         * prior maintenance and inference producers have joined the reset
         * stream. No host mirror, allocation, synchronization, or payload-byte
         * clearing occurs in this path.
         *
         * @throws std::invalid_argument for a null stream.
         * @throws std::runtime_error when the backend rejects the D2D enqueue.
         */
        void resetRequestPublications(void *stream);
        /**
         * @brief Materialize the live payload descriptor owned by a transfer slot.
         *
         * The directory remains the first-class owner of rolling VRAM payload
         * buffers. This method rebuilds a DeviceMoEExpertDescriptor from one
         * graph-local slot allocation without exposing the directory entry array
         * or copying a device-side status table to the host. The caller supplies
         * the logical expert id because a transfer slot may be reused for
         * different experts over time. Portable prefix state never persists this
         * graph-lifetime slot index.
         *
         * @return true when @p slot_index names an allocated slot with ready
         *         gate/up/down NativeVNNI payload descriptors.
         */
        bool descriptorForSlot(uint32_t slot_index,
                               uint32_t logical_expert,
                               DeviceMoEExpertDescriptor &out) const;

    private:
        /** @brief Retain every already-materialized allocation and its ledger lease. */
        DeviceMoETransferSlotDirectory(
            IBackend *backend,
            DeviceId device,
            int device_ordinal,
            uint32_t participant_id,
            BufferedCapacity capacity,
            std::vector<ProjectionSpec> specs,
            std::shared_ptr<LoadOrchestrator> orchestrator,
            DeviceMoEExpertDirectoryEntry *device_entries,
            DeviceMoEExpertDirectoryEntry *device_baseline_entries,
            std::vector<DeviceMoEExpertDirectoryEntry> host_entries,
            std::optional<PhysicalMemoryAllocationLease>
                directory_entries_lease,
            size_t planned_bytes,
            size_t wire_payload_bytes,
            size_t slot_storage_capacity_bytes);

        /**
         * @brief Shared implementation for admitted and named test setup.
         * @param explicit_test_allocation Whether the named test-only
         *        unadmitted allocation contract is active.
         * @return Fully uploaded stable directory.
         */
        static std::shared_ptr<DeviceMoETransferSlotDirectory> createImpl(
            IBackend *backend,
            DeviceId device,
            int device_ordinal,
            uint32_t participant_id,
            BufferedCapacity capacity,
            FormatProfile format_profile,
            std::shared_ptr<PhysicalMemoryAuthority> memory_authority,
            bool explicit_test_allocation);

        /** @return Stable allocator key for one slot/projection pair. */
        static std::string slotName(
            uint32_t slot_index,
            const std::string &label);

        IBackend *backend_ = nullptr;
        DeviceId device_;
        int device_ordinal_ = -1;
        uint32_t participant_id_ = 0;
        uint32_t slot_count_ = 0;
        BufferedCapacity capacity_;
        std::vector<ProjectionSpec> specs_;
        std::shared_ptr<LoadOrchestrator> orchestrator_;
        DeviceMoEExpertDirectoryEntry *device_entries_ = nullptr;
        DeviceMoEExpertDirectoryEntry *device_baseline_entries_ = nullptr;
        std::vector<DeviceMoEExpertDirectoryEntry> host_entries_;
        /** Live authority claim paired with both device descriptor arrays. */
        std::optional<PhysicalMemoryAllocationLease>
            directory_entries_lease_;
        size_t planned_bytes_ = 0;
        size_t wire_payload_bytes_ = 0;
        size_t slot_storage_capacity_bytes_ = 0;
    };

} // namespace llaminar2
