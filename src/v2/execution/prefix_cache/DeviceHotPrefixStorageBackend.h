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

#include <string>
#include <unordered_map>

namespace llaminar2
{

    class DeviceHotPrefixStorageBackend : public IPrefixStorageBackend
    {
    public:
        explicit DeviceHotPrefixStorageBackend(size_t budget_bytes);
        DeviceHotPrefixStorageBackend(DeviceId device, size_t budget_bytes);

        bool canStore(size_t bytes) const override;
        PrefixBlockHandle allocate(const PrefixCacheKey &key,
                                   const PrefixPayloadLayout &layout) override;
        bool release(const PrefixBlockHandle &handle) override;

        /**
         * @brief Allocate a persistent VRAM replica with archive-identical layout.
         *
         * This method allocates only. The orchestrator fills every section by
         * D2D copies on its explicit archive stream and publishes one readiness
         * event after the final copy.
         */
        bool allocateDeviceBlock(
            const PrefixBlockHandle &ram_archive,
            PrefixBlockHandle *device_handle,
            std::string *error = nullptr);

        size_t budgetBytes() const { return budget_bytes_; }
        size_t usedBytes() const { return used_bytes_; }
        DeviceId device() const { return device_; }

    private:
        DeviceId device_ = DeviceId::cpu();
        size_t budget_bytes_ = 0;
        size_t used_bytes_ = 0;
        std::unordered_map<PrefixCacheKey, size_t, PrefixCacheKeyHasher> allocations_;
    };

} // namespace llaminar2
