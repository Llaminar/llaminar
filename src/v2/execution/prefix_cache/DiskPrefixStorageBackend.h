#pragma once

#include "execution/prefix_cache/PrefixStorageBackend.h"

#include <filesystem>
#include <string>

namespace llaminar2
{

    class DiskPrefixStorageBackend : public IPrefixStorageBackend
    {
    public:
        DiskPrefixStorageBackend(std::filesystem::path root_dir, size_t budget_bytes);

        bool canStore(size_t bytes) const override;
        PrefixBlockHandle allocate(const PrefixCacheKey &key,
                                   const PrefixPayloadLayout &layout) override;
        bool release(const PrefixBlockHandle &handle) override;
        bool hydrateToRam(const PrefixBlockHandle &handle,
                          PrefixBlockHandle *ram_handle) override;

        bool writeBlock(const PrefixBlockHandle &handle, std::string *error = nullptr);
        bool readBlock(const PrefixCacheKey &key,
                       const PrefixPayloadLayout &layout,
                       PrefixBlockHandle *ram_handle,
                       std::string *error = nullptr) const;
        bool readBlockIntoRamBackend(const PrefixCacheKey &key,
                                     const PrefixPayloadLayout &layout,
                                     IPrefixStorageBackend &ram_backend,
                                     PrefixBlockHandle *ram_handle,
                                     std::string *error = nullptr) const;

        const std::filesystem::path &rootDir() const { return root_dir_; }
        size_t budgetBytes() const { return budget_bytes_; }
        std::filesystem::path blockMetaPath(const PrefixCacheKey &key) const;
        std::filesystem::path blockPayloadPath(const PrefixCacheKey &key, const char *suffix) const;

    private:
        std::filesystem::path root_dir_;
        size_t budget_bytes_ = 0;
    };

} // namespace llaminar2
