#pragma once

/**
 * @file DeviceMoETransferSlotDirectory.h
 * @brief Stable device-side directory for graph-captured MoE expert arrivals.
 *
 * One immutable projection set drives both admission and physical storage.
 * Complete-expert and gate/up-only arrivals cannot reuse each other's slots;
 * fixed down-column banks never enter the movement directory.
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
    struct MoEOverlayLayerWeightManifest;
    struct PrefixCacheRuntimeConfig;
    enum class RoutedExpertAssignmentPolicy : uint8_t;

    /**
     * @brief Owns stable device allocations used to receive graph-side MoE arrivals.
     *
     * Graphs share a stable slot index space. Each layer may lease only the
     * range belonging to its exact movable projection layout tuple.
     */
    class DeviceMoETransferSlotDirectory
    {
    public:
        using ProjectionSpec = GpuExpertSlotPool::ProjectionSpec;

        /** @brief One exact complete movable projection tuple and its member layers. */
        struct SlotFamily
        {
            std::vector<ProjectionSpec> projections;
            uint32_t layer_count = 0;
        };

        /** @brief Exact families in first-use order and their runtime layer membership. */
        struct FormatProfile
        {
            std::vector<SlotFamily> families;
            std::vector<uint32_t> layer_families;
            size_t max_wire_payload_bytes = 0;
            DeviceMoEProjectionSet projection_set = DeviceMoEProjectionSet::CompleteExpert;
        };

        /** @brief Allocator quote for a family's payload, live entries and reset entries. */
        struct FamilyAllocationBOM
        {
            DeviceMoETransferSlotRange range;
            size_t payload_bytes = 0;
            size_t descriptor_bytes = 0;
            /** @return Total physical bytes owned by the family. */
            size_t totalBytes() const noexcept { return payload_bytes + descriptor_bytes; }
        };

        /** @brief Exact physical quote with one allocation line per slot family. */
        struct AllocationBOM
        {
            std::vector<FamilyAllocationBOM> families;
            size_t payload_bytes = 0;
            size_t descriptor_bytes = 0;
            size_t layer_range_bytes = 0;
            size_t total_bytes = 0;
            uint32_t active_slots = 0;
            uint32_t total_slots = 0;
        };

        /**
         * @brief Group ordered runtime layers by their exact complete projection tuple.
         * @param layer_formats One movable tuple per runtime layer, in layer order.
         * @param projection_set Frozen projection ownership contract.
         * @throws std::invalid_argument for malformed or unsupported projection tuples.
         */
        static FormatProfile profileForLayerFormats(
            const std::vector<std::vector<ProjectionSpec>> &layer_formats,
            DeviceMoEProjectionSet projection_set = DeviceMoEProjectionSet::CompleteExpert);

        /**
         * @brief Derive the reusable device profile from model-authenticated metadata.
         *
         * Every quantized source codebook is converted through the same
         * NativeQuantised execution/allocation catalog used by CUDA and ROCm weight
         * preparation. This setup-time form lets physical-memory admission run
         * before prepared engines and their device pointers exist.
         *
         * @param layer_weight_manifest Contiguous gate/up/down manifest for this stage's owned global layers.
         * @param projection_set The frozen domain's movable projections only.
         * @return Exact layer families and maximum live wire footprint.
         * @throws std::invalid_argument for malformed, mixed projection, or
         *         uncatalogued projection formats.
         */
        static FormatProfile profileForLayerWeightManifest(
            const std::vector<MoEOverlayLayerWeightManifest> &
                layer_weight_manifest,
            DeviceMoEProjectionSet projection_set = DeviceMoEProjectionSet::CompleteExpert);

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
            uint32_t active_slots_per_layer = 0; ///< Runtime durable lane policy.
            uint32_t minimum_active_slots = 0; ///< Non-durable demand per family.
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
         * @throws std::invalid_argument for incoherent capacity or format data,
         *         a family whose projections differ from the profile's
         *         projection set, or a wire capacity that fits no family slot.
         * @throws std::overflow_error when the complete allocation cannot be
         *         represented by `size_t`.
         */
        static AllocationBOM allocationBOM(
            BufferedCapacity capacity,
            const FormatProfile &format_profile);

        /** @return Per-family live demand for the admitted native prefill/prefix transport. */
        static uint32_t runtimeMinimumActiveSlots(bool native_gpu_transport,
            RoutedExpertAssignmentPolicy prefill_assignment, const PrefixCacheRuntimeConfig &prefix_cache,
            uint32_t compact_payload_slots) noexcept;

        /** @return Complete capture identity, including exact layouts, membership and range policy. */
        static std::string captureIdentity(const FormatProfile &profile, BufferedCapacity capacity);

        /** @brief Reject a second layout for the same retained runtime domain. */
        void requireCaptureIdentity(const FormatProfile &profile, BufferedCapacity capacity) const;

        /**
         * @brief Materialize one production directory against admitted VRAM.
         * @param backend Exact CUDA or ROCm allocation/copy authority.
         * @param device Physical GPU identity charged by memory admission.
         * @param device_ordinal Backend-local ordinal matching @p device.
         * @param participant_id Stable routed-expert participant identity.
         * @param capacity Exact active and arrival-wave slot capacity.
         * @param format_profile Exact slot families for every runtime layer.
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
        /** @return Active claims admitted across all families. */
        uint32_t activeSlotCount() const noexcept { return active_slot_count_; }
        /** @return Immutable range table indexed by runtime layer, owned by this device. */
        const DeviceMoETransferSlotRange *deviceLayerRanges() const noexcept { return device_layer_ranges_; }
        /** @return Complete physical quote verified during materialization. */
        const AllocationBOM &materializedBOM() const noexcept { return bom_; }
        const std::vector<DeviceMoEExpertDirectoryEntry> &hostEntriesForTest() const { return host_entries_; }
        size_t plannedBytes() const { return planned_bytes_; }
        size_t wirePayloadBytes() const { return wire_payload_bytes_; }
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
         *         gate/up/down NativeQuantised payload descriptors.
         */
        bool descriptorForSlot(uint32_t slot_index,
                               uint32_t layer,
                               uint32_t logical_expert,
                               DeviceMoEExpertDescriptor &out) const;

    private:
        /** @brief Retain every already-materialized allocation and its ledger lease. */
        DeviceMoETransferSlotDirectory(
            IBackend *backend,
            DeviceId device,
            int device_ordinal,
            uint32_t participant_id,
            AllocationBOM bom,
            FormatProfile profile,
            std::string capture_identity,
            std::vector<PhysicalMemoryAllocationLease> payload_leases,
            std::vector<std::unique_ptr<WeightVRAMPool>> pools,
            DeviceMoEExpertDirectoryEntry *device_entries,
            DeviceMoEExpertDirectoryEntry *device_baseline_entries,
            DeviceMoETransferSlotRange *device_layer_ranges,
            std::vector<DeviceMoEExpertDirectoryEntry> host_entries,
            std::optional<PhysicalMemoryAllocationLease>
                directory_entries_lease);

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

        IBackend *backend_ = nullptr;
        DeviceId device_;
        int device_ordinal_ = -1;
        uint32_t participant_id_ = 0;
        uint32_t slot_count_ = 0;
        uint32_t active_slot_count_ = 0;
        AllocationBOM bom_;
        FormatProfile profile_;
        std::string capture_identity_;
        std::vector<PhysicalMemoryAllocationLease> payload_leases_;
        std::vector<std::unique_ptr<WeightVRAMPool>> pools_;
        DeviceMoEExpertDirectoryEntry *device_entries_ = nullptr;
        DeviceMoEExpertDirectoryEntry *device_baseline_entries_ = nullptr;
        DeviceMoETransferSlotRange *device_layer_ranges_ = nullptr;
        std::vector<DeviceMoEExpertDirectoryEntry> host_entries_;
        /** Live authority claim paired with both device descriptor arrays. */
        std::optional<PhysicalMemoryAllocationLease>
            directory_entries_lease_;
        size_t planned_bytes_ = 0;
        size_t wire_payload_bytes_ = 0;
    };

} // namespace llaminar2
