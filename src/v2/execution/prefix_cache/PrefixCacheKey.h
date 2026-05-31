#pragma once

#include "execution/prefix_cache/BlockHash.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{

    struct PrefixCacheKey
    {
        uint64_t fingerprint = 0;
        uint64_t parent_hash = 0;
        uint64_t token_hash = 0;
        int block_index = 0;
        int token_start = 0;
        int token_count = 0;

        bool operator==(const PrefixCacheKey &other) const;
        bool operator!=(const PrefixCacheKey &other) const { return !(*this == other); }

        uint64_t stableHash() const;
        std::string toHex() const;
        bool valid() const { return fingerprint != 0 && token_count >= 0 && block_index >= 0; }
    };

    struct PrefixCacheKeyHasher
    {
        size_t operator()(const PrefixCacheKey &key) const
        {
            return static_cast<size_t>(key.stableHash());
        }
    };

    PrefixCacheKey makePrefixCacheKey(
        uint64_t fingerprint,
        uint64_t parent_hash,
        int block_index,
        int token_start,
        const std::vector<int32_t> &tokens);

} // namespace llaminar2
