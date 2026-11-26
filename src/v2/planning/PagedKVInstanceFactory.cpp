/**
 * @file PagedKVInstanceFactory.cpp
 * @brief Tier-intent-to-instance factory implementation.
 */
#include "PagedKVInstanceFactory.h"

namespace llaminar2
{
PagedKVInstanceOutcome admitConfiguredPagedKVInstance(
    std::shared_ptr<PhysicalMemoryAuthority> authority,
    const PagedKVTierConfig &tiers, const PagedKVPoolBudget &remaining,
    const PagedKVInstancePolicy &policy, DeviceId device,
    CPUStoragePlacement placement)
{
    std::string error;
    if (!tiers.valid(&error))
        throw std::invalid_argument(error);
    auto plan = planPagedKVInstance(tiers, remaining, policy, device, placement);
    if (!plan)
        return PagedKVInstanceOutcome{std::nullopt,
            "the remaining budget cannot hold a pooled instance with a legal whole-frame pool"};
    auto instance = PagedKVInstance::create(std::move(authority), *plan, device, placement);
    if (!instance)
        return PagedKVInstanceOutcome{std::nullopt,
            "the physical authority refused the complete pooled instance BOM"};
    return PagedKVInstanceOutcome{std::move(*instance), {}};
}
} // namespace llaminar2
