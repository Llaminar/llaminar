#pragma once

#include "execution/prefix_cache/PrefixStorageBackend.h"

#include <unordered_map>

namespace llaminar2
{

    class RamPrefixStorageBackend : public IPrefixStorageBackend
    {
    public:
        explicit RamPrefixStorageBackend(size_t budget_bytes);

        bool canStore(size_t bytes) const override;
        PrefixBlockHandle allocate(const PrefixCacheKey &key,
                                   const PrefixPayloadLayout &layout) override;
        bool release(const PrefixBlockHandle &handle) override;
        bool hydrateToRam(const PrefixBlockHandle &handle,
                          PrefixBlockHandle *ram_handle) override;

        size_t budgetBytes() const { return budget_bytes_; }
        size_t usedBytes() const { return used_bytes_; }
        size_t allocationCount() const { return allocations_.size(); }

    private:
        size_t budget_bytes_ = 0;
        size_t used_bytes_ = 0;
        std::unordered_map<PrefixCacheKey, size_t, PrefixCacheKeyHasher> allocations_;
    };

} // namespace llaminar2
