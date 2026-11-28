/**
 * @file RetentionIntentResolver.cpp
 * @brief §8.4 standard-field retention intent resolution.
 */
#include "execution/prefix_cache/RetentionIntentResolver.h"

#include "execution/prefix_cache/BlockHash.h"

#include <cstring>
#include <stdexcept>

namespace llaminar2
{
namespace
{
bool validClass(RetentionClass cls)
{
    return cls == RetentionClass::InMemory || cls == RetentionClass::Default ||
        cls == RetentionClass::TieredDay;
}
} // namespace

RetentionIntentResolver::RetentionIntentResolver(std::optional<RetentionClass> ceiling)
    : ceiling_(ceiling)
{
    if (ceiling.has_value() && !validClass(*ceiling))
        throw std::invalid_argument(
            "retention classes ceiling must be an explicit valid retention class");
}

RetentionIntentResolution RetentionIntentResolver::resolve(
    std::string_view cache_retention, std::string_view cache_key) const
{
    std::string refusal;
    auto intent = RetentionClassMap::forValue(cache_retention, &refusal);
    if (!intent.has_value())
        return RetentionIntentResolution{std::nullopt, std::nullopt, refusal};
    // Operator ceiling (P0 §4): a class rank GREATER than the ceiling is
    // refused with a precise diagnostic — never silently demoted to a lower
    // class (that would defeat the operator's capacity promise).
    if (ceiling_.has_value() &&
        static_cast<std::uint32_t>(intent->cls) > static_cast<std::uint32_t>(*ceiling_))
        return RetentionIntentResolution{std::nullopt, std::nullopt,
            "requested retention class '" + std::string(cache_retention) +
            "' exceeds the configured retention classes ceiling"};
    std::optional<std::uint64_t> ns;
    if (!cache_key.empty())
        ns = hashPrefixBytes(cache_key.data(), cache_key.size());
    return RetentionIntentResolution{intent, ns, {}};
}
} // namespace llaminar2
