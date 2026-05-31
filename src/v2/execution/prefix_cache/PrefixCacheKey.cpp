#include "execution/prefix_cache/PrefixCacheKey.h"

#include <sstream>

namespace llaminar2
{

    bool PrefixCacheKey::operator==(const PrefixCacheKey &other) const
    {
        return fingerprint == other.fingerprint &&
               parent_hash == other.parent_hash &&
               token_hash == other.token_hash &&
               block_index == other.block_index &&
               token_start == other.token_start &&
               token_count == other.token_count;
    }

    uint64_t PrefixCacheKey::stableHash() const
    {
        uint64_t hash = 1469598103934665603ull;
        hash = combinePrefixHash(hash, fingerprint);
        hash = combinePrefixHash(hash, parent_hash);
        hash = combinePrefixHash(hash, token_hash);
        hash = combinePrefixHash(hash, static_cast<uint64_t>(block_index));
        hash = combinePrefixHash(hash, static_cast<uint64_t>(token_start));
        hash = combinePrefixHash(hash, static_cast<uint64_t>(token_count));
        return hash;
    }

    std::string PrefixCacheKey::toHex() const
    {
        std::ostringstream out;
        out << prefixHashHex(stableHash())
            << "-"
            << prefixHashHex(fingerprint)
            << "-"
            << prefixHashHex(token_hash)
            << "-"
            << block_index;
        return out.str();
    }

    PrefixCacheKey makePrefixCacheKey(
        uint64_t fingerprint,
        uint64_t parent_hash,
        int block_index,
        int token_start,
        const std::vector<int32_t> &tokens)
    {
        PrefixCacheKey key;
        key.fingerprint = fingerprint;
        key.parent_hash = parent_hash;
        key.token_hash = hashPrefixTokens(tokens);
        key.block_index = block_index;
        key.token_start = token_start;
        key.token_count = static_cast<int>(tokens.size());
        return key;
    }

} // namespace llaminar2
