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
        bool hydrateToRam(const PrefixBlockHandle &handle,
                          PrefixBlockHandle *ram_handle) override;

        bool promoteFromRam(const PrefixBlockHandle &ram_handle,
                            PrefixBlockHandle *device_handle,
                            std::string *error = nullptr);
        bool hydrateToRamBackend(const PrefixBlockHandle &handle,
                                 IPrefixStorageBackend &ram_backend,
                                 PrefixBlockHandle *ram_handle,
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
