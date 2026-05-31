#include "execution/prefix_cache/BlockHash.h"

#include <iomanip>
#include <sstream>

namespace llaminar2
{
    namespace
    {
        constexpr uint64_t kFnvPrime = 1099511628211ull;
    }

    uint64_t hashPrefixBytes(const void *data, size_t bytes, uint64_t seed)
    {
        uint64_t hash = seed;
        const auto *raw = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < bytes; ++i)
        {
            hash ^= static_cast<uint64_t>(raw[i]);
            hash *= kFnvPrime;
        }
        return hash;
    }

    uint64_t hashPrefixTokens(const std::vector<int32_t> &tokens, uint64_t seed)
    {
        return hashPrefixBytes(tokens.data(), tokens.size() * sizeof(int32_t), seed);
    }

    uint64_t combinePrefixHash(uint64_t current, uint64_t value)
    {
        current ^= value + 0x9e3779b97f4a7c15ull + (current << 6) + (current >> 2);
        current *= kFnvPrime;
        return current;
    }

    std::string prefixHashHex(uint64_t value)
    {
        std::ostringstream out;
        out << std::hex << std::setfill('0') << std::setw(16) << value;
        return out.str();
    }

} // namespace llaminar2
