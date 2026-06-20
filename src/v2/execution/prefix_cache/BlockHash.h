#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{

    uint64_t hashPrefixBytes(const void *data, size_t bytes, uint64_t seed = 1469598103934665603ull);
    uint64_t hashPrefixTokens(const std::vector<int32_t> &tokens, uint64_t seed = 1469598103934665603ull);
    uint64_t combinePrefixHash(uint64_t current, uint64_t value);
    std::string prefixHashHex(uint64_t value);

} // namespace llaminar2
