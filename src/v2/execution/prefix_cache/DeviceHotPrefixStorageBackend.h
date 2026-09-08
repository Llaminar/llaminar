/**
 * @file DeviceHotPrefixStorageBackend.h
 * @brief Pure-device ownership for optional hot prefix-cache replicas.
 *
 * A device-hot block is populated either directly from serialized GPU staging
 * on the harvest stream or by one asynchronous H2D promotion after a lower-tier
 * hit. Once resident, restore reads it directly without a host-visible replay.
 * Pinned RAM and disk remain durable capacity tiers; this backend owns only an
 * acceleration replica.
 */

#pragma once

#include "execution/prefix_cache/PrefixStorageBackend.h"

#include "backends/DeviceId.h"

#include <memory>
#include <string>

namespace llaminar2
{
    class PhysicalMemoryAuthority;

    /**
     * @brief Pre-capture VRAM arena for optional accelerator-hot prefix replicas.
     *
     * The configured hot-tier budget is physically reserved once, before any
     * native serving executable is captured. Prefix harvest and promotion only
     * lease stable slots from that arena; they never call the backend allocator
     * in the inference path. A slot is recycled only after every handle alias
     * has retired, and its next producer stream waits on the previous payload's
     * exact readiness event before overwriting the bytes.
     */
    class DeviceHotPrefixStorageBackend : public IPrefixStorageBackend
    {
    public:
        /** Destroy the arena after all outstanding slot leases retire. */
        ~DeviceHotPrefixStorageBackend() override;

        /**
         * @brief Reserve a bounded device-hot arena before graph capture.
         * @param device Accelerator that owns the arena.
         * @param budget_bytes Configured maximum VRAM capacity.
         * @param slot_bytes Maximum serialized device payload for one block.
         * @param memory_authority Rank-local authority that admitted this arena.
         * @param error Optional diagnostic populated on admission/allocation failure.
         * @return A valid backend, or nullptr without retaining a partial arena.
         */
        [[nodiscard]] static std::shared_ptr<DeviceHotPrefixStorageBackend>
        create(
            DeviceId device,
            size_t budget_bytes,
            size_t slot_bytes,
            std::shared_ptr<PhysicalMemoryAuthority> memory_authority,
            std::string *error = nullptr);

        /** @return true when an immediately reusable slot can hold @p bytes. */
        bool canStore(size_t bytes) const override;

        /**
         * @brief Reject the generic allocation surface.
         *
         * Device-hot handles require an exact RAM archive layout and producer
         * stream, so callers must use @ref allocateDeviceBlock.
         */
        PrefixBlockHandle allocate(const PrefixCacheKey &key,
                                   const PrefixPayloadLayout &layout) override;

        /**
         * @brief Retire one key's arena lease without synchronizing the host.
         * @param handle Installed device-hot handle to retire.
         * @return true exactly once for an active key owned by this arena.
         */
        bool release(const PrefixBlockHandle &handle) override;

        /**
         * @brief Lease an archive-shaped block from the preallocated VRAM arena.
         *
         * No allocation occurs here. When the selected slot was used by an
         * earlier block, this method queues an event wait on @p producer_stream
         * before exposing the slot for overwrite. The orchestrator then fills
         * every section and publishes the new payload readiness event.
         *
         * @param ram_archive Durable RAM block defining the exact device layout.
         * @param producer_stream Non-null stream that will fill the returned slot.
         * @param device_handle Receives aliased section owners into the arena.
         * @param error Optional failure diagnostic.
         * @return true when one slot was leased and fully described.
         */
        bool allocateDeviceBlock(
            const PrefixBlockHandle &ram_archive,
            void *producer_stream,
            PrefixBlockHandle *device_handle,
            std::string *error = nullptr);

        /** @return Configured upper bound supplied by topology policy. */
        size_t budgetBytes() const;
        /** @return Bytes charged to currently installed hot handles. */
        size_t usedBytes() const;
        /** @return Bytes physically reserved before graph capture. */
        size_t reservedBytes() const;
        /** @return Maximum serialized device payload accepted by one slot. */
        size_t slotBytes() const;
        /** @return Number of graph-stable slots in the arena. */
        size_t slotCount() const;
        /** @return Accelerator that owns the arena. */
        DeviceId device() const;

        /**
         * @brief Test immutable capacity without considering current occupancy.
         * @param bytes Charged block bytes.
         * @return true when one configured arena slot can represent the block.
         */
        bool capacityEligible(size_t bytes) const;

    private:
        struct Impl;

        /** Construct only through @ref create so invalid arenas cannot escape. */
        DeviceHotPrefixStorageBackend();

        std::unique_ptr<Impl> impl_;
    };

} // namespace llaminar2
