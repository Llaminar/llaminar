/**
 * @file RamPrefixStorageBackend.h
 * @brief Capacity-tier prefix storage in pageable CPU or pinned GPU-host RAM.
 */

#pragma once

#include "execution/prefix_cache/PrefixStorageBackend.h"
#include "backends/DeviceId.h"

#include <unordered_map>

namespace llaminar2
{

    class RamPrefixStorageBackend : public IPrefixStorageBackend
    {
    public:
        explicit RamPrefixStorageBackend(size_t budget_bytes);
        RamPrefixStorageBackend(DeviceId producer_device, size_t budget_bytes);

        bool canStore(size_t bytes) const override;
        PrefixBlockHandle allocate(const PrefixCacheKey &key,
                                   const PrefixPayloadLayout &layout) override;
        bool release(const PrefixBlockHandle &handle) override;

        size_t budgetBytes() const { return budget_bytes_; }
        size_t usedBytes() const { return used_bytes_; }
        size_t allocationCount() const { return allocations_.size(); }

    private:
        /**
         * @brief Allocate one zeroed payload section in the correct RAM class.
         *
         * GPU archives hard-require backend-pinned memory. There is no pageable
         * fallback because that would silently turn supposedly asynchronous
         * DMA into a blocking runtime staging operation.
         */
        bool allocateSection(
            size_t bytes,
            std::shared_ptr<std::vector<uint8_t>> *pageable_owner,
            std::shared_ptr<void> *pinned_owner,
            void **payload,
            const std::shared_ptr<PrefixPayloadReadiness> &readiness) const;

        DeviceId producer_device_ = DeviceId::cpu();
        size_t budget_bytes_ = 0;
        size_t used_bytes_ = 0;
        std::unordered_map<PrefixCacheKey, size_t, PrefixCacheKeyHasher> allocations_;
    };

} // namespace llaminar2
